"""Live perception + MPPI planning loop.

Three cadences in a single process:
    A. train_step              every step                  (fast)
    B. replan + 1 agent step   every REPLAN_INTERVAL       (MPPI is cheap)
    C. refresh buffer + maps   every cfg.hash_buffer_refresh_interval (bottleneck)

Bootstrap: spin a full circle observing each frame, then cold-train for
BOOTSTRAP_TRAIN_STEPS so the first map is meaningful before the first plan.
"""

import argparse
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
os.environ['GLOG_minloglevel'] = '2'
os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['HABITAT_SIM_LOG'] = 'quiet'

import gc
import json
import time
from collections import deque

import numpy as np
import quaternion  # registers numpy.quaternion dtype used by habitat-sim
import torch
import imageio

from src.config import Config
from src.habitat.habitat_utils import (
    get_scene_bounds_from_pathfinder,
    init_simulator,
    spawn_agent_at_random_navpoint,
)
from src.habitat.sim_interface import SimInterface
from src.perception.perception_stack import PerceptionStack
from src.planning.mppi import MPPIPlanner
from src.planning.utils import normalize_sim


REPLAN_INTERVAL = 100
SPIN_FRAMES = 36                       # 36 * 10° = full circle
SPIN_OMEGA = np.deg2rad(10) / 0.1      # 10° per 0.1s sim step
BOOTSTRAP_TRAIN_STEPS = 2000


def get_agent_heading(agent) -> float:
    """Agent yaw in MPPI's grid frame: atan2(forward_z_world, forward_x_world)."""
    rot = agent.get_state().rotation
    R = quaternion.as_rotation_matrix(rot)
    forward_world = R @ np.array([0.0, 0.0, -1.0])  # Habitat agent forward is local -Z
    return float(np.arctan2(forward_world[2], forward_world[0]))


def world_to_grid(x_world: float, z_world: float, ref_grid, res: float):
    z_idx = int((z_world - ref_grid.min_z) / res)
    x_idx = int((x_world - ref_grid.min_x) / res)
    z_idx = max(0, min(z_idx, ref_grid.num_z - 1))
    x_idx = max(0, min(x_idx, ref_grid.num_x - 1))
    return (z_idx, x_idx)


def _to_numpy(maybe_tensor):
    if maybe_tensor is None:
        return None
    if isinstance(maybe_tensor, torch.Tensor):
        return maybe_tensor.detach().cpu().numpy()
    return maybe_tensor


def plan_one_action(perception, sim_iface, mppi, cfg, action_queue: deque, k_max: int = 5, progress: float = 0.0):
    """Run MPPI from current pose and return (action, ref_traj, opt_traj).

    action is [v_mps, w_rad_per_s] or None if idling.
    ref_traj / opt_traj are grid-coord lists [(z_idx, x_idx), ...]; None when
    falling back to the queue or idling (no new plan was computed).

    On a successful plan the full MPPI control sequence replaces the queue.
    If every A* goal is unreachable the queue is consumed so the agent keeps
    moving.
    """
    bev_sim = _to_numpy(perception.similarity_grid.get_2d_map(min_y=0.1, max_y=1.5))
    bev_epi = _to_numpy(perception.ugrid.get_2d_map(type='epistemic'))
    bev_occ = _to_numpy(perception.occupancy_grid.get_2d_map())

    if bev_sim is None or bev_epi is None or bev_occ is None:
        print("  plan: missing map(s); idling")
        return (action_queue.popleft() if action_queue else None), None, None, None

    pos = sim_iface.agent_position
    heading = get_agent_heading(sim_iface.agent)
    start_grid = world_to_grid(pos[0], pos[2], perception.similarity_grid, cfg.voxel_resolution)

    # Approach points around the single highest-similarity peak, ordered by
    # proximity. Target objects (e.g. a bed) are marked as obstacles in the
    # occupancy grid, so the peak itself is usually unreachable — we A* to
    # the closest free cell next to it.
    # Closest free cell to the highest-similarity peak. Target objects (e.g.
    # a bed) are marked as obstacles in the occupancy grid, so the sim peak
    # itself is usually unreachable — we just need to get close.
    approach_points = mppi.get_goals_near_highest_sim(bev_sim, bev_occ, max_candidates=1)
    if not approach_points:
        print("  plan: no observed cells, can't pick a goal")
        return (action_queue.popleft() if action_queue else None), None, None, None
    goal = approach_points[0]

    opt_path, U_opt, _seen, mppi_score = mppi.optimize_trajectory(
        start_grid, goal, bev_epi, bev_occ,
        initial_heading=heading,
        intrinsics=sim_iface.intrinsics,
        sensor_height=cfg.sensor_height,
        progress=progress,
    )
    if U_opt is None or U_opt.shape[0] == 0:
        # No safe MPPI plan this replan — spin in place to scan for new
        # options. Rotating doesn't translate the agent, so it's always safe
        # in a free start cell, and the heading change usually reveals new
        # geometry that makes the next replan more likely to find a safe plan.
        print("  plan: MPPI returned no safe control sequence; spinning in place")
        action_queue.clear()
        return [0.0, cfg.mppi_w_sign * cfg.mppi_max_w_rps], None, None, None

    # Successful plan: replace queue with full MPPI control sequence.
    # MPPI U units: v in [grid cells / mppi_step], w in [rad / mppi_step].
    # sim_iface.step expects [m/s, rad/s].
    action_queue.clear()
    for i in range(U_opt.shape[0]):
        v_mps = float(U_opt[i, 0]) * cfg.voxel_resolution / cfg.mppi_dt
        w_rps = cfg.mppi_w_sign * float(U_opt[i, 1]) / cfg.mppi_dt
        action_queue.append([v_mps, w_rps])
    return action_queue.popleft(), None, opt_path, mppi_score


def run(cfg: Config, save_enabled: bool = True) -> None:
    # 1. Init simulator
    sim, agent = init_simulator(
        cfg.scene_path, resolution=cfg.img_width, fov_deg=cfg.fov, sensor_height=cfg.sensor_height,
    )
    spawn_agent_at_random_navpoint(sim, agent)
    scene_bounds = get_scene_bounds_from_pathfinder(sim)

    sim_iface = SimInterface(cfg, sim, agent)
    perception = PerceptionStack(cfg, scene_bounds)
    perception.target_query = cfg.target_query

    mppi = MPPIPlanner(cfg, device=cfg.device)
    action_queue: deque = deque()

    # 2. Bootstrap A: spin in a full circle, observe each frame
    print(f"[bootstrap A] spinning {SPIN_FRAMES} frames (~360°)...")
    for f in range(SPIN_FRAMES):
        sim_iface.step([0.0, SPIN_OMEGA])
        rgb, depth, c2w = perception.observe(sim_iface)
        perception.update_replay_buffer(rgb, depth, c2w, sim_iface.intrinsics)
        perception.update_occupancy(depth, c2w, sim_iface.intrinsics)
        
        if save_enabled:
            # Save periodic bootstrap views
            rgb_dir = os.path.join(cfg.output_dir, "rgbs")
            os.makedirs(rgb_dir, exist_ok=True)
        #     rgb_img = (rgb.numpy() * 255).astype(np.uint8)
        #     imageio.imwrite(os.path.join(rgb_dir, f"bootstrap_{f:03d}.png"), rgb_img)
            
        torch.cuda.empty_cache()

        print(f"  bootstrap frame {f + 1}/{SPIN_FRAMES}")
    super_pts, super_feats = perception.make_super_batch()

    # 3. Bootstrap B: cold-train so first BEV maps have signal
    print(f"[bootstrap B] cold-training {BOOTSTRAP_TRAIN_STEPS} steps...")
    t0 = time.time()
    for s in range(BOOTSTRAP_TRAIN_STEPS):
        loss = perception.train_step(super_pts, super_feats)
        if s % 200 == 0:
            print(f"  bootstrap step {s:04d} | loss {loss:.5f} | t {time.time()-t0:.1f}s")

    # 4. Bootstrap C: build first maps, save grid extent for visualize.py
    print("[bootstrap C] computing first BEV maps...")
    perception.update_maps(step=0, save_enabled=save_enabled)
    if save_enabled:
        # Save RGB
        step = 0
        rgb_dir = os.path.join(cfg.output_dir, "rgbs")
        os.makedirs(rgb_dir, exist_ok=True)
        rgb_img = (rgb.numpy() * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(rgb_dir, f"rgb_{step:03d}.png"), rgb_img)
        
        # Save 2D Sim Map
        perception.save_2d_similarity(step, depth, c2w, sim_iface.intrinsics)

    extent_path = os.path.join(cfg.output_dir, "grid_extent.json")
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(extent_path, 'w') as f:
        json.dump({
            'min_x': perception.similarity_grid.min_x,
            'max_x': perception.similarity_grid.max_x,
            'min_z': perception.similarity_grid.min_z,
            'max_z': perception.similarity_grid.max_z,
            'voxel_resolution': cfg.voxel_resolution,
        }, f)

    traj_log_path = os.path.join(cfg.output_dir, "traj_log.jsonl")
    open(traj_log_path, 'w').close()  # truncate / create fresh

    # 5. Main loop
    print(f"[main] running for {cfg.iterations} iterations "
          f"(replan every {REPLAN_INTERVAL}, refresh every {cfg.hash_buffer_refresh_interval})")
    start_time = time.time()

    for step in range(1, cfg.iterations + 1):
        # A. train every step
        loss = perception.train_step(super_pts, super_feats)

        # B. replan + 1 agent step
        if step % REPLAN_INTERVAL == 0:
            t_plan = time.time()
            pos = sim_iface.agent_position.copy()
            heading = get_agent_heading(sim_iface.agent)
            progress = step / max(1, cfg.iterations)
            action, ref_traj, opt_traj, score = plan_one_action(
                perception, sim_iface, mppi, cfg, action_queue, progress=progress
            )
            if action is not None:
                sim_iface.step(action, dt=cfg.mppi_dt)
                print(f"step {step}: plan {time.time()-t_plan:.2f}s | "
                      f"action [v={action[0]:+.3f} m/s, w={action[1]:+.3f} rad/s] | position: {pos[0]:.2f}, {pos[2]:.2f}")
            else:
                print(f"step {step}: plan {time.time()-t_plan:.2f}s | no action")

            with open(traj_log_path, 'a') as _f:
                # Convert score to cost: higher score is better, so cost is -score.
                cost = -float(score) if score is not None else None
                _f.write(json.dumps({
                    'step': step,
                    'pos': [float(pos[0]), float(pos[2])],
                    'heading': float(heading),
                    'action': [float(action[0]), float(action[1])] if action is not None else [0.0, 0.0],
                    'ref_traj': [[int(p[0]), int(p[1])] for p in ref_traj] if ref_traj else [],
                    'opt_traj': [[int(p[0]), int(p[1])] for p in opt_traj] if opt_traj else [],
                    'mppi_cost': cost,
                }) + '\n')

        # C. refresh buffer + maps (slowest cadence — bottleneck)
        if step % cfg.hash_buffer_refresh_interval == 0:
            t_refresh = time.time()
            rgb, depth, c2w = perception.observe(sim_iface)
            perception.update_replay_buffer(rgb, depth, c2w, sim_iface.intrinsics)
            perception.update_occupancy(depth, c2w, sim_iface.intrinsics)

            if save_enabled:
                # Save RGB
                rgb_dir = os.path.join(cfg.output_dir, "rgbs")
                os.makedirs(rgb_dir, exist_ok=True)
                rgb_img = (rgb.numpy() * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(rgb_dir, f"rgb_{step:03d}.png"), rgb_img)
                
                # Save 2D Sim Map
                perception.save_2d_similarity(step, depth, c2w, sim_iface.intrinsics)

            if super_pts is not None:
                del super_pts, super_feats
                torch.cuda.empty_cache()
            super_pts, super_feats = perception.make_super_batch()

            perception.update_maps(step=step, save_enabled=save_enabled)
            gc.collect()
            torch.cuda.empty_cache()
            print(f"step {step}: refresh+maps {time.time()-t_refresh:.2f}s | "
                  f"history {perception.buffer_size}/{perception.buffer_capacity} pts")

        # if step % 100 == 0:
        #     print(f"  step {step:05d} | loss {loss:.5f} | t {time.time()-start_time:.1f}s")

    perception.save_models()
    sim.close()
    print("[main] done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None, help="Override target query")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device index")
    parser.add_argument("--no-save", action="store_true", default=False,
                        help="Skip saving BEV maps to disk during the live loop")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    cfg = Config("config/config.yaml")
    if args.query is not None:
        cfg.target_query = args.query

    run(cfg, save_enabled=not args.no_save)
