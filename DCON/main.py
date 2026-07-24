"""Live perception + planning loop — one navigation episode.

Three cadences in a single process:
    A. train_step              every step                  (fast)
    B. replan + 1 agent step   every REPLAN_INTERVAL       (planning is cheap)
    C. refresh buffer + maps   every cfg.hash_buffer_refresh_interval (bottleneck)

Startup: a single observation seeds perception and the first BEV maps are
built directly from the untrained feature field — no spin, no cold-train. The
maps fill in online as the loop trains and cadence C recomputes them.

Control is split by mode. SEARCH (pre-latch) runs MPPI over the BEV maps
(`src/planning/search.py`); once a detection latches, EXPLOIT hands locomotion
to the pretrained DD-PPO PointNav policy (`src/planning/exploit.py`). The
detector → field-verify → classify → latch pipeline lives in
`src/perception/detection.py`; evidence writing and scoring in `src/episode/`.
"""

import argparse
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
os.environ['GLOG_minloglevel'] = '2'
os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['HABITAT_SIM_LOG'] = 'quiet'

import gc
import random
import time
from collections import deque

import numpy as np
import torch

from src.config import Config
from src.episode.control import EarlyStop
from src.episode.recorder import EpisodeRecorder
from src.episode.scoring import score_episode
from src.habitat.habitat_utils import start_episode
from src.habitat.sim_interface import SimInterface
from src.perception.detection import DetectionGate, bev_cell_from_box_center, grid_to_world_xz
from src.perception.obj_detection import make_detector
from src.perception.perception_stack import PerceptionStack
from src.planning.ddppo_policy import DDPPO_FORWARD_M, DDPPO_TURN_DEG
from src.planning.exploit import ExploitController
from src.planning.mppi import MPPIPlanner
from src.planning.search import plan_search_action
from tools.visualize import render_navigation

REPLAN_INTERVAL = 100


def execute_action(sim_iface, cfg, action, mode: str) -> str:
    """Apply one action to the simulator; returns a log string describing it.

    EXPLOIT is always discrete-stepped: DD-PPO's action space is the fixed
    Habitat primitive set (STOP/MOVE_FORWARD/TURN_LEFT/TURN_RIGHT), independent
    of cfg.discrete_actions (which only governs SEARCH's continuous-vs-discrete
    MPPI tracking). Its magnitudes are the checkpoint's training convention
    (25 cm / 10°), NOT cfg.discrete_* (the ObjectNav-challenge 25 cm / 30° used
    by SEARCH's tracking controller).
    """
    if mode == 'EXPLOIT':
        sim_iface.step_discrete(action, forward_m=DDPPO_FORWARD_M,
                                turn_deg=DDPPO_TURN_DEG)
        return action
    if cfg.discrete_actions:
        sim_iface.step_discrete(action)
        return action
    sim_iface.step(action, dt=cfg.mppi_dt)
    return f"[v={action[0]:+.3f} m/s, w={action[1]:+.3f} rad/s]"


def run(cfg: Config, save_enabled: bool = True,
        save_video: bool = True, viz_output: str = "./figs/nav_history.mp4",
        viz_fps: int = 5,
        start_pos=None, start_rotation=None, goals=None,
        success_radius_m: float = 1.0) -> dict:
    """Run one navigation episode.

    `start_pos` is the spawn point (snapped to the navmesh); defaults to the
    historical [0, -3, 2.5]. `start_rotation`, if given, is the spawn
    orientation as an [x, y, z, w] quaternion (the habitat episode-dataset
    convention). `goals`, if given, is a list of ground-truth target
    world positions (xyz) used to score the episode — success requires the agent
    to self-stop within `success_radius_m` geodesic of the nearest goal, and SPL
    weights that success by start→nearest-goal geodesic over distance traveled.
    Returns a metrics dict (see `src/episode/scoring.py`).

    Evidence is written to `cfg.output_dir` — see `src/episode/recorder.py` for
    what each of `save_enabled` / `save_video` adds.
    """
    if start_pos is None:
        start_pos = [-2.0, -3.0, 6.0]

    # Seed every RNG this episode draws from, before anything is constructed:
    # MPPI's rollout noise, the feature field's init + minibatch sampling, and
    # DD-PPO's action sampling. Without this an episode is unrepeatable (the
    # field used to seed itself off wall-clock time), so an A/B difference
    # can't be attributed to the change under test rather than to drift — two
    # identical-config runs of tv__Collierville__ep5 latched objects 7 m apart.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # 1. Init simulator + the perception / planning / detection stacks.
    sim, agent, start_nav, scene_bounds = start_episode(cfg, start_pos, start_rotation)
    sim_iface = SimInterface(cfg, sim, agent)
    perception = PerceptionStack(cfg, scene_bounds)  # owns target_query; initialised from cfg
    mppi = MPPIPlanner(cfg, device=cfg.device)       # SEARCH
    exploit = ExploitController(cfg, device=cfg.device)  # EXPLOIT (DD-PPO)
    # `gate` owns the detector, the field-verify gate, and the latch: once
    # `gate.detected` flips (cfg.detected_persistence consecutive persistent
    # detections) it never releases, and control switches to EXPLOIT for good.
    gate = DetectionGate(cfg, make_detector(cfg), perception)
    action_queue: deque = deque()
    recorder = EpisodeRecorder(cfg, save_enabled=save_enabled, save_video=save_video)

    # Caches the goal cell of the most recent investigated detection so a missed
    # frame doesn't drop the agent back to global-argmax exploration; post-latch
    # it freezes on the object that triggered the latch.
    last_box_goal = None

    # 2. Startup: a single observation seeds the replay buffer + occupancy, then
    # the first BEV maps are built directly from the untrained feature field —
    # no spin, no cold-train. The maps start as field noise / mostly-unseen and
    # fill in online as the loop trains and cadence C recomputes them.
    print("[init] seeding maps from the untrained feature field (no bootstrap)...")
    rgb, depth, c2w = perception.observe(sim_iface)
    perception.update_replay_buffer(rgb, depth, c2w, sim_iface.intrinsics)
    perception.update_occupancy(depth, c2w, sim_iface.intrinsics)
    super_pts, super_feats = perception.make_super_batch()
    torch.cuda.empty_cache()

    # First maps (needed for the first plan) + the static evidence sidecars.
    perception.update_maps(step=0, save_enabled=save_video)
    recorder.snapshot(0, perception, rgb, depth, c2w, sim_iface.intrinsics)
    recorder.write_grid_extent(perception.similarity_grid)
    recorder.write_run_meta(perception.target_query, perception.semantics.distractors)

    # 3. Main loop
    print(f"[main] running for {cfg.iterations} iterations "
          f"(replan every {REPLAN_INTERVAL}, refresh every {cfg.hash_buffer_refresh_interval})")

    # Tracks the last step actually executed by the loop. Used to cap the
    # post-run visualization so it doesn't pull in stale .npy snapshots left
    # over from a previous, longer run (traj_log.jsonl is truncated each run,
    # but the umaps/occ_maps/sim_maps directories are not).
    last_step = 0

    # Episode metrics: accumulate the actual distance the agent travels (sum of
    # per-action displacements) for SPL, and remember whether the agent itself
    # decided to stop (vs. running out the step budget = timeout = failure).
    path_length = 0.0
    prev_pos = np.asarray(start_nav, dtype=np.float64).copy()
    agent_stopped = False
    # Discrete-primitive counter (MOVE_FORWARD/TURN), shown in the per-step log
    # for telemetry only — no budget is enforced here.
    agent_steps = 0

    with EarlyStop(cfg.output_dir) as early_stop:
        for step in range(1, cfg.iterations + 1):
            if early_stop.reason:
                print(f"[main] ending early at step {last_step} ({early_stop.reason})")
                break
            last_step = step
            # A. train every step
            perception.train_step(super_pts, super_feats)

            # B. replan + 1 agent step
            if step % REPLAN_INTERVAL == 0:
                t_plan = time.time()
                pos = sim_iface.agent_position.copy()
                heading = sim_iface.agent_heading
                rgb_cur, depth_cur, c2w_cur = perception.observe(sim_iface)

                # Detect (subject to the EXPLOIT throttle), field-verify,
                # classify into too-close / too-far / usable-band, and advance
                # the latch. `det.investigate` (= not too close) drives the goal
                # + confidence; `det.persistent` (usable band) also drives
                # latching, so the agent approaches a far sighting and commits
                # only once it has closed into the usable band. A throttled-off
                # replan reports no detection, so control falls through to the
                # cached goal cell below.
                det = gate.step(
                    sim_iface, rgb_cur, depth_cur, c2w_cur, pos,
                    run_detector=gate.should_run_detector(step // REPLAN_INTERVAL),
                    tag=f"step {step}")
                recorder.save_field_verify_frame(step, rgb_cur, det.box, det.field_score)

                # Refresh the cached goal from THIS frame's detection whenever
                # it's worth investigating. In SEARCH the planner does this
                # itself (and reports the cell back as `box_goal`); in EXPLOIT
                # the cache only moves on a redetect tick, and otherwise stays
                # frozen on the latched object.
                stop_now = False
                if gate.detected:
                    if det.box is not None and det.investigate:
                        cand = bev_cell_from_box_center(
                            perception, cfg, sim_iface.intrinsics, det.box,
                            depth_cur, c2w_cur)
                        if cand is not None:
                            last_box_goal = cand
                    # Keep occupancy fresh (observe() already ran this replan) —
                    # still consumed by SEARCH-mode IG/exploration even though
                    # EXPLOIT's own navigation (DD-PPO) no longer reads it.
                    perception.update_occupancy(depth_cur, c2w_cur, sim_iface.intrinsics)
                    goal_cell = last_box_goal
                    action, stop_now = exploit.step(
                        sim_iface, perception, goal_cell, pos, depth_cur)
                    opt_traj, mode, w_conf = None, 'EXPLOIT', 1.0
                else:
                    action, opt_traj, goal_cell, box_goal = plan_search_action(
                        perception, sim_iface, mppi, cfg, action_queue,
                        det_score=det.conf_score,
                        det_box=det.box, depth=depth_cur, c2w=c2w_cur,
                        last_box_goal=last_box_goal,
                        det_investigate=det.investigate,
                    )
                    # `box_goal` is non-None only for an investigated box,
                    # tracking the latest sighting worth steering toward —
                    # including too-far ones. Fallback paths (cached / global
                    # argmax) leave it None.
                    if box_goal is not None:
                        last_box_goal = box_goal
                        print(f"  cached new goal {box_goal} (det_score={det.score:.3f})")
                    mode, w_conf = 'SEARCH', float(mppi.last_w_conf)

                if goal_cell is not None:
                    gx_m, gz_m = grid_to_world_xz(
                        goal_cell, perception.similarity_grid, cfg.voxel_resolution)
                    goal_str = f"({gx_m:+.2f}, {gz_m:+.2f})m"
                else:
                    goal_str = "—"
                if action is not None:
                    action_str = execute_action(sim_iface, cfg, action, mode)
                    if mode == 'EXPLOIT' or cfg.discrete_actions:
                        agent_steps += 1
                        action_str = f"{action_str} ({agent_steps}/{cfg.max_agent_steps})"
                    cur_pos = np.asarray(sim_iface.agent_position, dtype=np.float64)
                    path_length += float(np.linalg.norm(cur_pos - prev_pos))
                    prev_pos = cur_pos
                else:
                    action_str = "no action"
                print(f"step {step}: plan {time.time()-t_plan:.2f}s | {action_str} | "
                      f"position: {pos[0]:.2f}, {pos[2]:.2f} | "
                      f"det: {det.score:.3f} | goal: {goal_str} | "
                      f"w_conf: {w_conf:.2f} | mode: {mode}")

                # `stop_now` came from EXPLOIT's stop_distance_m arrival check
                # (the sole stop signal; SEARCH never self-stops). The traj_log
                # write runs first so the final replan is captured for
                # visualization.
                fv = perception.last_field_verify if det.field_score is not None else None
                recorder.log_step({
                    'step': step,
                    'pos': [float(pos[0]), float(pos[2])],
                    'heading': float(heading),
                    'action': (action if (mode == 'EXPLOIT' or cfg.discrete_actions)
                               else ([float(action[0]), float(action[1])]
                                     if action is not None else [0.0, 0.0])),
                    'opt_traj': [[int(p[0]), int(p[1])] for p in opt_traj] if opt_traj else [],
                    'det_conf': float(det.score),
                    'field_score': (float(det.field_score)
                                    if det.field_score is not None else None),
                    # Pairwise-mode components (None otherwise): query-channel
                    # presence + per-term pooled scores behind the margin.
                    'field_presence': fv["presence"] if fv is not None else None,
                    'field_terms': fv["terms"] if fv is not None else None,
                    'det_box': [float(v) for v in det.box] if det.box is not None else None,
                    'goal': [int(goal_cell[0]), int(goal_cell[1])] if goal_cell is not None else None,
                    # EXPLOIT steers at the goal snapped onto navigable free
                    # space, which is NOT `goal` above (that is the raw box
                    # projection the arrival test uses). Logged so evidence
                    # shows what DD-PPO actually aimed at. None in SEARCH.
                    'nav_goal': ([int(exploit.nav_goal[0]), int(exploit.nav_goal[1])]
                                 if mode == 'EXPLOIT' and exploit.nav_goal is not None
                                 else None),
                    'mode': mode,
                    'w_conf': w_conf,
                })

                if stop_now:
                    agent_stopped = True
                    if cfg.discrete_actions:
                        agent_steps += 1  # the STOP primitive counts too
                    else:
                        sim_iface.step([0.0, 0.0], dt=cfg.mppi_dt)
                    break

            # C. refresh buffer + maps (slowest cadence — bottleneck)
            if step % cfg.hash_buffer_refresh_interval == 0:
                t_refresh = time.time()
                rgb, depth, c2w = perception.observe(sim_iface)
                perception.update_replay_buffer(rgb, depth, c2w, sim_iface.intrinsics)
                perception.update_occupancy(depth, c2w, sim_iface.intrinsics)
                recorder.snapshot(step, perception, rgb, depth, c2w, sim_iface.intrinsics)

                if super_pts is not None:
                    del super_pts, super_feats
                    torch.cuda.empty_cache()
                super_pts, super_feats = perception.make_super_batch()

                perception.update_maps(step=step, save_enabled=save_video)
                gc.collect()
                torch.cuda.empty_cache()
                print(f"step {step}: refresh+maps {time.time()-t_refresh:.2f}s | "
                      f"history {perception.buffer_size}/{perception.buffer_capacity} pts")

    recorder.save_final(last_step, perception, sim_iface)

    # Score while the pathfinder is still alive (before sim.close()).
    metrics = score_episode(
        cfg, sim.pathfinder, goals, start_nav, sim_iface.agent_position,
        path_length, agent_stopped, success_radius_m, last_step)
    sim.close()
    print("[main] done.")

    if save_video:
        print(f"[viz] rendering navigation video up to step {last_step}...")
        try:
            render_navigation(cfg, viz_output, fps=viz_fps, max_step=last_step)
        except Exception as e:
            print(f"[viz] visualization failed: {e}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None, help="Override target query")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device index")
    parser.add_argument("--no-save", action="store_true", default=False,
                        help="Save nothing but traj_log.jsonl + grid_extent.json "
                             "(skip the final occupancy map too)")
    parser.add_argument("--no-visualize", action="store_true", default=False,
                        help="Skip the full per-step map + RGB history and the "
                             "nav_history video render (keeps just the final "
                             "occupancy map)")
    parser.add_argument("--viz-output", type=str, default="./figs/nav_history.mp4",
                        help="Output path for the post-run navigation video")
    parser.add_argument("--viz-fps", type=int, default=5,
                        help="FPS for the post-run navigation video")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    cfg = Config("config/config.yaml")
    if args.query is not None:
        cfg.target_query = args.query

    run(cfg, save_enabled=not args.no_save,
        save_video=not args.no_visualize,
        viz_output=args.viz_output,
        viz_fps=args.viz_fps)
