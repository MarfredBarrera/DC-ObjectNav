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
    spawn_agent_at_pos
)
from src.habitat.sim_interface import SimInterface
from src.perception.obj_detection import make_detector
from src.perception.perception_stack import PerceptionStack
from src.perception.utils import unprojection
from src.planning.mppi import MPPIPlanner
from src.planning.utils import normalize_sim
from visualize import render_navigation


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


def bev_cells_from_sim_pixels(perception, cfg, intrinsics, rgb, depth, c2w,
                              target_query, top_frac=0.05, min_pixels=50):
    """BEV cells from the top-K% most target-similar pixels in the current frame.

    Robust alternative to `bev_cells_from_det_box`. At distance the detector
    tends to bracket a generous region of wall/floor around the target, and
    unprojecting *every* in-box pixel smears those background cells into the
    goal region. Per-pixel CLIP similarity is much sharper: pixels actually
    on the object score higher than adjacent background. We unproject only
    the top-K% (constrained to valid depth) and use those BEV cells as the
    goal-search region.

    `top_frac`: fraction of valid pixels to keep. 0.05 = top 5%.
    `min_pixels`: floor on the count so very sparse views still produce a
                  usable mask.
    """
    if depth is None or c2w is None or rgb is None or not target_query:
        return None
    rgb_gpu = rgb.to(perception.device)
    depth_gpu = depth.to(perception.device)

    # Per-pixel similarity to the target text. feats and text_embed are both
    # L2-normalized inside MaskCLIPSemantics, so the dot product is cosine.
    feats = perception.mask_clip.extract_dense_features(rgb_gpu)  # (H, W, 512)
    text_embed = perception.mask_clip.encode_text(target_query)   # (1, 512)
    sim_2d = (feats @ text_embed.T).squeeze(-1)                   # (H, W) in [-1, 1]

    depth_mask = (depth_gpu > cfg.min_sensor_dist) & (depth_gpu < cfg.max_sensor_dist)
    if not bool(depth_mask.any()):
        return None
    sim_in_valid = sim_2d[depth_mask]
    n_valid = int(sim_in_valid.numel())
    if n_valid == 0:
        return None
    k = max(min_pixels, int(top_frac * n_valid))
    k = min(k, n_valid)
    thresh = torch.topk(sim_in_valid, k).values.min()
    mask = (sim_2d >= thresh) & depth_mask
    if not bool(mask.any()):
        return None

    world_points = unprojection(
        depth_gpu, intrinsics, c2w.to(perception.device), perception.device, mask=mask,
    )
    if world_points.shape[0] == 0:
        return None

    sg = perception.similarity_grid
    z_idx = ((world_points[:, 2] - sg.min_z) / cfg.voxel_resolution).long().clamp(0, sg.num_z - 1)
    x_idx = ((world_points[:, 0] - sg.min_x) / cfg.voxel_resolution).long().clamp(0, sg.num_x - 1)
    cell_mask = torch.zeros((sg.num_z, sg.num_x), dtype=torch.bool, device=perception.device)
    cell_mask[z_idx, x_idx] = True
    return cell_mask.cpu().numpy()


def bev_cells_from_det_box(perception, cfg, intrinsics, det_box, depth, c2w):
    """BEV (z_idx, x_idx) cells that the pixels inside `det_box` project to.

    Returns a boolean numpy array of shape (num_z, num_x), or None if the box
    is missing / yields no valid depth pixels. Used to constrain goal
    selection in EXPLOIT mode: instead of the global argmax of bev_sim, we
    take the argmax only over cells that actually correspond to the detected
    object in the current frame.
    """
    if det_box is None or depth is None or c2w is None:
        return None
    xmin, ymin, xmax, ymax = det_box
    depth_gpu = depth.to(perception.device)
    H_d, W_d = depth_gpu.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(H_d, device=perception.device),
        torch.arange(W_d, device=perception.device),
        indexing='ij',
    )
    box_mask = (xx >= int(xmin)) & (xx <= int(xmax)) & (yy >= int(ymin)) & (yy <= int(ymax))
    depth_mask = (depth_gpu > cfg.min_sensor_dist) & (depth_gpu < cfg.max_sensor_dist)
    mask = box_mask & depth_mask
    if not bool(mask.any()):
        return None
    world_points = unprojection(depth_gpu, intrinsics, c2w.to(perception.device), perception.device, mask=mask)
    if world_points.shape[0] == 0:
        return None
    sg = perception.similarity_grid
    z_idx = ((world_points[:, 2] - sg.min_z) / cfg.voxel_resolution).long().clamp(0, sg.num_z - 1)
    x_idx = ((world_points[:, 0] - sg.min_x) / cfg.voxel_resolution).long().clamp(0, sg.num_x - 1)
    cell_mask = torch.zeros((sg.num_z, sg.num_x), dtype=torch.bool, device=perception.device)
    cell_mask[z_idx, x_idx] = True
    return cell_mask.cpu().numpy()


def plan_one_action(perception, sim_iface, mppi, cfg,
                    action_queue: deque, k_max: int = 5,
                    progress: float = 0.0, det_score: float = 0.0,
                    detected: bool = False, det_box=None, rgb=None, depth=None, c2w=None,
                    last_box_goal=None):
    """Run MPPI from current pose and return (action, ref_traj, opt_traj).

    action is [v_mps, w_rad_per_s] or None if idling.
    ref_traj / opt_traj are grid-coord lists [(z_idx, x_idx), ...]; None when
    falling back to the queue or idling (no new plan was computed).

    On a successful plan the full MPPI control sequence replaces the queue.
    If every A* goal is unreachable the queue is consumed so the agent keeps
    moving.

    `det_score` is the raw detector confidence for the current frame, used as
    `goal_confidence` in SEARCH mode (MPPI's threshold + hysteresis filter
    noisy detections). `detected=True` overrides it with goal_confidence=1.0,
    saturating the conf curve so `w_ig_conf = 0` and IG is hard-off.
    """
    bev_sim = _to_numpy(perception.similarity_grid.get_2d_map(min_y=0.1, max_y=1.5))
    bev_epi = _to_numpy(perception.ugrid.get_2d_map(type='epistemic'))
    bev_occ = _to_numpy(perception.occupancy_grid.get_2d_map())

    if bev_sim is None or bev_epi is None or bev_occ is None:
        print("  plan: missing map(s); idling")
        return (action_queue.popleft() if action_queue else None), None, None, None, None, None

    pos = sim_iface.agent_position
    heading = get_agent_heading(sim_iface.agent)
    start_grid = world_to_grid(pos[0], pos[2], perception.similarity_grid, cfg.voxel_resolution)

    # Drive straight at the highest-similarity cell, even if it's on an
    # obstacle (the target object itself). MPPI carves a small free disk
    # around the goal and forgives collisions once the rollout has arrived,
    # so the planner can commit instead of orbiting.
    sim_for_goal = bev_sim.copy().astype(np.float32)
    sim_for_goal[bev_occ == 0] = -np.inf  # exclude unseen
    # Goal selection priority:
    #   1. Detection box this frame → argmax of bev_sim restricted to BEV cells
    #      the box projects to.
    #   2. No box this frame, but a previous box-derived goal is cached →
    #      reuse that fixed world cell so the agent commits to the last
    #      sighting instead of chasing a far-away similarity peak.
    #   3. Otherwise → global argmax of observed bev_sim (exploration).
    goal = None
    box_goal = None  # None unless THIS frame's detection produced a fresh goal
    H, W = sim_for_goal.shape
    # Detector presence triggers the constrained search; the actual BEV
    # region comes from the top-K% most-target-similar pixels in the frame
    # (not the raw box, which can be loose at distance).
    if det_box is not None:
        det_cells = bev_cells_from_sim_pixels(
            perception, cfg, sim_iface.intrinsics,
            rgb, depth, c2w, perception.target_query,
        )
        if det_cells is not None and det_cells.any():
            sim_for_goal[~det_cells] = -np.inf
            if np.any(np.isfinite(sim_for_goal)):
                flat_idx = int(np.argmax(sim_for_goal))
                goal = (flat_idx // W, flat_idx % W)
                box_goal = goal
    if goal is None and last_box_goal is not None:
        gz, gx = int(last_box_goal[0]), int(last_box_goal[1])
        if 0 <= gz < H and 0 <= gx < W:
            goal = (gz, gx)
    if goal is None:
        if not np.any(np.isfinite(sim_for_goal)):
            print("  plan: no observed cells, can't pick a goal")
            return (action_queue.popleft() if action_queue else None), None, None, None, None, None
        flat_idx = int(np.argmax(sim_for_goal))
        goal = (flat_idx // W, flat_idx % W)
    goal_confidence = 1.0 if detected else float(det_score)

    opt_path, U_opt = mppi.optimize_trajectory(
        start_grid, goal, bev_epi, bev_occ,
        initial_heading=heading,
        intrinsics=sim_iface.intrinsics,
        sensor_height=cfg.sensor_height,
        progress=progress,
        ig_source=cfg.ig_source,
        goal_confidence=goal_confidence,
    )
    mppi_score = None
    if U_opt is None or U_opt.shape[0] == 0:
        # No safe MPPI plan this replan — spin in place to scan for new
        # options. Rotating doesn't translate the agent, so it's always safe
        # in a free start cell, and the heading change usually reveals new
        # geometry that makes the next replan more likely to find a safe plan.
        print("  plan: MPPI returned no safe control sequence, replanning")
        action_queue.clear()
        return [0.0, -np.pi/4], None, None, None, goal, box_goal

    # Successful plan: replace queue with full MPPI control sequence.
    # MPPI U units: v in [grid cells / mppi_step], w in [rad / mppi_step].
    # sim_iface.step expects [m/s, rad/s].
    action_queue.clear()
    for i in range(U_opt.shape[0]):
        v_mps = float(U_opt[i, 0]) * cfg.voxel_resolution / cfg.mppi_dt
        w_rps = cfg.mppi_w_sign * float(U_opt[i, 1]) / cfg.mppi_dt
        action_queue.append([v_mps, w_rps])
    return action_queue.popleft(), None, opt_path, mppi_score, goal, box_goal


def run(cfg: Config, save_enabled: bool = True,
        visualize: bool = True, viz_output: str = "./figs/nav_history.mp4",
        viz_fps: int = 5) -> None:
    # 1. Init simulator
    sim, agent = init_simulator(
        cfg.scene_path, resolution=cfg.img_width, fov_deg=cfg.fov, sensor_height=cfg.sensor_height,
    )
    spawn_agent_at_random_navpoint(sim, agent)
    # spawn_agent_at_pos(sim, agent, [0.0, 0.0, 0.0])
    scene_bounds = get_scene_bounds_from_pathfinder(sim)

    sim_iface = SimInterface(cfg, sim, agent)
    perception = PerceptionStack(cfg, scene_bounds)  # owns target_query; initialised from cfg

    mppi = MPPIPlanner(cfg, device=cfg.device)
    detector = make_detector(cfg.detector, device=cfg.device)
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

    # Two-phase planner state. `detected` latches to True (and never releases)
    # once `det_score >= cfg.detected_conf_threshold` for
    # `cfg.detected_persistence` consecutive replans. While latched, MPPI
    # silences IG and pulls hard toward the BEV similarity peak.
    detected = False
    detected_streak = 0

    # Best goal cell derived from a detection bounding box so far, with its
    # detector confidence. Persists across replans so a missing detection
    # this frame doesn't drop the agent back to global-argmax exploration.
    # Only overwritten when a NEW detection beats `last_box_conf` — a noisy
    # low-conf detection can't displace an earlier strong sighting.
    last_box_goal = None
    last_box_conf = -1.0

    # Tracks the last step actually executed by the loop. Used to cap the
    # post-run visualization so it doesn't pull in stale .npy snapshots left
    # over from a previous, longer run (traj_log.jsonl is truncated each run,
    # but the umaps/occ_maps/sim_maps directories are not).
    last_step = 0

    for step in range(1, cfg.iterations + 1):
        last_step = step
        # A. train every step
        loss = perception.train_step(super_pts, super_feats)

        # B. replan + 1 agent step
        if step % REPLAN_INTERVAL == 0:
            t_plan = time.time()
            pos = sim_iface.agent_position.copy()
            heading = get_agent_heading(sim_iface.agent)
            progress = step / max(1, cfg.iterations)

            rgb_cur, depth_cur, c2w_cur = perception.observe(sim_iface)
            det_score, det_box = detector.detect(rgb_cur, perception.target_query)

            # Latch into DETECTED once we get `detected_persistence` consecutive
            # detections above threshold. Never unlatches — see comment above.
            if not detected:
                if det_score >= cfg.detected_conf_threshold:
                    detected_streak += 1
                else:
                    detected_streak = 0
                if detected_streak >= cfg.detected_persistence:
                    detected = True
                    print(f"step {step}: DETECTED — entering exploit mode "
                          f"(det_score={det_score:.3f})")

            action, ref_traj, opt_traj, score, goal_cell, box_goal = plan_one_action(
                perception, sim_iface, mppi, cfg, action_queue,
                progress=progress, det_score=det_score, detected=detected,
                det_box=det_box, rgb=rgb_cur, depth=depth_cur, c2w=c2w_cur,
                last_box_goal=last_box_goal,
            )
            # Cache only when THIS frame's box yielded a valid goal AND its
            # confidence beats the best we've seen. Fallback paths (cached /
            # global argmax) return goal_cell but leave box_goal=None, so they
            # never touch the cache.
            if box_goal is not None and det_score > last_box_conf:
                last_box_goal = box_goal
                last_box_conf = float(det_score)
                print(f"  cached new goal {box_goal} (conf {det_score:.3f})")
            det_backend = detector.backend_for(perception.target_query)
            mode = 'EXPLOIT' if detected else 'SEARCH'
            w_conf = float(getattr(mppi, 'last_w_conf', 0.0))
            goal_str = f"({goal_cell[0]},{goal_cell[1]})" if goal_cell is not None else "—"
            if action is not None:
                sim_iface.step(action, dt=cfg.mppi_dt)
                print(f"step {step}: plan {time.time()-t_plan:.2f}s | "
                      f"action [v={action[0]:+.3f} m/s, w={action[1]:+.3f} rad/s] | "
                      f"position: {pos[0]:.2f}, {pos[2]:.2f} | "
                      f"det[{det_backend}]: {det_score:.3f} | "
                      f"goal: {goal_str} | w_conf: {w_conf:.2f} | "
                      f"mode: {mode}")
            else:
                print(f"step {step}: plan {time.time()-t_plan:.2f}s | no action | "
                      f"det[{det_backend}]: {det_score:.3f} | "
                      f"goal: {goal_str} | w_conf: {w_conf:.2f} | "
                      f"mode: {mode}")

            # Termination check: in DETECTED mode, compute distance from agent
            # to BEV similarity peak. Stop after the traj_log write below so the
            # final replan is captured for visualization.
            stop_now = False
            if detected:
                bev_sim = _to_numpy(perception.similarity_grid.get_2d_map(min_y=0.1, max_y=1.5))
                bev_occ = _to_numpy(perception.occupancy_grid.get_2d_map())
                if bev_sim is not None and bev_occ is not None:
                    sim_for_goal = bev_sim.astype(np.float32).copy()
                    sim_for_goal[bev_occ == 0] = -np.inf
                    if det_box is not None:
                        det_cells = bev_cells_from_sim_pixels(
                            perception, cfg, sim_iface.intrinsics,
                            rgb_cur, depth_cur, c2w_cur, perception.target_query,
                        )
                        if det_cells is not None and det_cells.any():
                            sim_for_goal[~det_cells] = -np.inf
                    if np.any(np.isfinite(sim_for_goal)):
                        H_g, W_g = sim_for_goal.shape
                        flat_idx = int(np.argmax(sim_for_goal))
                        gz, gx = flat_idx // W_g, flat_idx % W_g
                        sz, sx = world_to_grid(
                            pos[0], pos[2], perception.similarity_grid, cfg.voxel_resolution
                        )
                        dist_m = float(np.hypot(gz - sz, gx - sx)) * cfg.voxel_resolution
                        if dist_m <= cfg.stop_distance_m:
                            print(f"step {step}: TARGET REACHED "
                                  f"(dist={dist_m:.2f}m <= {cfg.stop_distance_m:.2f}m)")
                            stop_now = True

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
                    'det_conf': float(det_score),
                    'det_box': [float(v) for v in det_box] if det_box is not None else None,
                    'goal': [int(goal_cell[0]), int(goal_cell[1])] if goal_cell is not None else None,
                    'mode': mode,
                    'w_conf': w_conf,
                }) + '\n')

            if stop_now:
                sim_iface.step([0.0, 0.0], dt=cfg.mppi_dt)
                break

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

    if visualize:
        print(f"[viz] rendering navigation video up to step {last_step}...")
        try:
            render_navigation(cfg, viz_output, fps=viz_fps, max_step=last_step)
        except Exception as e:
            print(f"[viz] visualization failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None, help="Override target query")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device index")
    parser.add_argument("--no-save", action="store_true", default=False,
                        help="Skip saving BEV maps to disk during the live loop")
    parser.add_argument("--ig-source", type=str, choices=["unseen", "epistemic"], default="epistemic",
                        help="Information gain source for MPPI (unseen or epistemic)")
    parser.add_argument("--detector", type=str,
                        choices=["yolo", "coco_yolo", "hybrid", "grounding_dino"],
                        default="hybrid",
                        help="Detector backend: yolo (YOLO-Worldv2), coco_yolo "
                             "(closed-set YOLOv8), hybrid (COCO→YOLOv8 else "
                             "YOLO-Worldv2), or grounding_dino")
    parser.add_argument("--no-visualize", action="store_true", default=False,
                        help="Skip rendering nav_history video after the run")
    parser.add_argument("--viz-output", type=str, default="./figs/nav_history.mp4",
                        help="Output path for the post-run navigation video")
    parser.add_argument("--viz-fps", type=int, default=5,
                        help="FPS for the post-run navigation video")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    cfg = Config("config/config.yaml")
    if args.query is not None:
        cfg.target_query = args.query
    cfg.ig_source = args.ig_source
    cfg.detector = args.detector

    run(cfg, save_enabled=not args.no_save,
        visualize=not args.no_visualize,
        viz_output=args.viz_output,
        viz_fps=args.viz_fps)
