"""Live perception + MPPI planning loop.

Three cadences in a single process:
    A. train_step              every step                  (fast)
    B. replan + 1 agent step   every REPLAN_INTERVAL       (MPPI is cheap)
    C. refresh buffer + maps   every cfg.hash_buffer_refresh_interval (bottleneck)

Startup: a single observation seeds perception and the first BEV maps are
built directly from the untrained feature field — no spin, no cold-train. The
maps fill in online as the loop trains and cadence C recomputes them.
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
import shutil
import signal
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
    spawn_agent_at_pos,
    geodesic_distance,
)
from src.habitat.sim_interface import SimInterface
from src.perception.obj_detection import make_detector
from src.perception.perception_stack import PerceptionStack
from src.perception.utils import unprojection
from src.planning.mppi import MPPIPlanner
from tools.visualize import render_navigation


REPLAN_INTERVAL = 100


def get_agent_heading(agent) -> float:
    """Agent yaw in MPPI's grid frame: atan2(forward_z_world, forward_x_world)."""
    rot = agent.get_state().rotation
    R = quaternion.as_rotation_matrix(rot)
    forward_world = R @ np.array([0.0, 0.0, -1.0])  # Habitat agent forward is local -Z
    return float(np.arctan2(forward_world[2], forward_world[0]))


def discrete_action_from_plan(opt_path, heading, cfg):
    """Convert an MPPI optimized path into one Habitat ObjectNav primitive.

    Pure-pursuit tracking controller (the continuous→discrete transformation):
    pick the first waypoint on `opt_path` at least `cfg.discrete_lookahead_m`
    ahead of the agent, take the bearing to it in the grid frame (same
    convention as get_agent_heading: atan2(Δz, Δx)), and emit the nearest
    primitive — TURN toward the bearing when the heading error exceeds half a
    turn, otherwise MOVE_FORWARD. Returns one of "move_forward" / "turn_left" /
    "turn_right", or None if the path is degenerate (idle this replan). The
    receding-horizon replan corrects any tracking drift each cycle.
    """
    if not opt_path or len(opt_path) < 2:
        return None
    sz, sx = float(opt_path[0][0]), float(opt_path[0][1])
    lookahead_cells = max(1.0, cfg.discrete_lookahead_m / cfg.voxel_resolution)
    target = None
    for cz, cx in opt_path[1:]:
        if np.hypot(cz - sz, cx - sx) >= lookahead_cells:
            target = (float(cz), float(cx))
            break
    if target is None:
        target = (float(opt_path[-1][0]), float(opt_path[-1][1]))
    tz, tx = target
    if tz == sz and tx == sx:
        return None
    desired = float(np.arctan2(tz - sz, tx - sx))
    dtheta = (desired - heading + np.pi) % (2 * np.pi) - np.pi
    turn_thresh = np.radians(cfg.discrete_turn_deg / 2.0)
    if abs(dtheta) > turn_thresh:
        # grid θ increases for "turn_left" (matches SimInterface.step_discrete).
        return "turn_left" if dtheta > 0 else "turn_right"
    return "move_forward"


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


def box_center_world_xz(perception, cfg, intrinsics, det_box, depth, c2w):
    """Median world (x, z) of a small patch around the center of `det_box`.

    Unprojects only a small window (not the single center pixel, so one depth
    hole at the exact center doesn't drop the result) and returns the median
    world point's BEV-plane coordinates (wx, wz), or None if no valid depth
    near the center. Shared by goal projection and the detection size/distance
    gate. Box pixel coords must be in `depth`'s pixel space.
    """
    if det_box is None or depth is None or c2w is None:
        return None
    xmin, ymin, xmax, ymax = det_box
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    depth_gpu = depth.to(perception.device)
    H_d, W_d = depth_gpu.shape[-2:]
    # Half-window: 5% of the smaller box side, clamped to >=1px.
    half = max(1, int(0.05 * min(xmax - xmin, ymax - ymin)))
    yy, xx = torch.meshgrid(
        torch.arange(H_d, device=perception.device),
        torch.arange(W_d, device=perception.device),
        indexing='ij',
    )
    win = ((xx >= int(cx) - half) & (xx <= int(cx) + half) &
           (yy >= int(cy) - half) & (yy <= int(cy) + half))
    depth_mask = (depth_gpu > cfg.min_sensor_dist) & (depth_gpu < cfg.max_sensor_dist)
    mask = win & depth_mask
    if not bool(mask.any()):
        return None
    world_points = unprojection(
        depth_gpu, intrinsics, c2w.to(perception.device), perception.device, mask=mask)
    if world_points.shape[0] == 0:
        return None
    return float(world_points[:, 0].median()), float(world_points[:, 2].median())


def bev_cell_from_box_center(perception, cfg, intrinsics, det_box, depth, c2w):
    """Single BEV (z_idx, x_idx) cell the *center* of `det_box` projects to.

    Projects only the box-center patch to one world point and returns its lone
    BEV cell, used as the goal verbatim — no similarity argmax, no snap-to-free.
    Relies on LLMDet emitting tight, accurate boxes. Returns (z_idx, x_idx)
    or None.
    """
    wxz = box_center_world_xz(perception, cfg, intrinsics, det_box, depth, c2w)
    if wxz is None:
        return None
    wx, wz = wxz
    sg = perception.similarity_grid
    z_idx = int(np.clip((wz - sg.min_z) / cfg.voxel_resolution, 0, sg.num_z - 1))
    x_idx = int(np.clip((wx - sg.min_x) / cfg.voxel_resolution, 0, sg.num_x - 1))
    return (z_idx, x_idx)


def classify_detection(perception, cfg, intrinsics, det_box, depth, c2w,
                       agent_pos):
    """Classify a detection by object distance + box size into how it may be used.

    Returns (is_persistent, contributes_confidence):
      - TOO CLOSE — object distance < cfg.detected_min_dist_m OR box covers more
        than cfg.detected_max_box_frac of the frame. The box fills the view and
        carries no usable localization → (False, False): ignored entirely
        (no goal, no confidence weight, no latch).
      - TOO FAR — object distance > cfg.detected_max_dist_m OR box smaller than
        cfg.detected_min_box_frac of the frame. A distant, uncertain sighting →
        (False, True): investigated (projected + cached as the goal, pulls the
        confidence weight) but not persistent — it never counts toward the latch.
      - USABLE BAND (anything else) → (True, True): a persistent detection that
        latches and is cached as the goal, and contributes the confidence weight.

    Each threshold disables at a non-positive value. A missing box or
    unrangeable depth (no valid depth at the box center) → (False, False).
    """
    if det_box is None:
        return (False, False)
    xmin, ymin, xmax, ymax = det_box
    H_img, W_img = depth.shape[-2:]
    box_frac = ((xmax - xmin) * (ymax - ymin)) / float(W_img * H_img)
    wxz = box_center_world_xz(perception, cfg, intrinsics, det_box, depth, c2w)
    if wxz is None:
        return (False, False)
    dist_m = float(np.hypot(wxz[0] - agent_pos[0], wxz[1] - agent_pos[2]))

    box_too_large = cfg.detected_max_box_frac > 0.0 and box_frac > cfg.detected_max_box_frac
    box_too_small = cfg.detected_min_box_frac > 0.0 and box_frac < cfg.detected_min_box_frac
    dist_too_small = cfg.detected_min_dist_m > 0.0 and dist_m < cfg.detected_min_dist_m
    dist_too_large = cfg.detected_max_dist_m > 0.0 and dist_m > cfg.detected_max_dist_m

    if dist_too_small or box_too_large:
        return (False, False)   # too close: ignore entirely
    if dist_too_large or box_too_small:
        return (False, True)    # too far: confidence only, not persistent
    return (True, True)         # usable band


def detect_classify_latch(detector, perception, sim_iface, cfg,
                          rgb, depth, c2w, pos, detected, detected_streak,
                          run_detector=True, tag=""):
    """Run the detector, classify the detection, and advance the latch state.

    Classifies the box by object distance + box size into
    three tiers (see `classify_detection`): *too close* → ignored; *too far* →
    investigate (steer + confidence) but not persistent; *usable band* →
    persistent (also counts toward the latch streak). Latches into DETECTED
    once `cfg.detected_persistence` consecutive persistent detections accrue
    (never unlatches).

    When `cfg.field_verify` is on, a detector box must additionally be
    confirmed by the learned relevance field before it counts: the box's
    valid-depth pixels are unprojected to 3D, the field is queried there, and
    the pooled score (`cfg.field_verify_pool`: top-`cfg.field_verify_top_frac`
    mean, or max) must clear `cfg.field_verify_threshold` (see
    PerceptionStack.field_score_in_box). A
    frame that fails the gate is treated as no detection at all — no goal, no
    confidence, no latch. `field_score` is the pooled score (None when the
    gate didn't run or the box couldn't be verified).

    Returns (det_score, det_box, det_persistent, det_investigate, conf_score,
    detected, detected_streak, field_score).
    """
    if run_detector:
        det_score, det_box = detector.detect(rgb, perception.target_query)
    else:
        det_score, det_box = 0.0, None

    field_score = None
    if cfg.field_verify and det_box is not None:
        field_score = perception.field_score_in_box(
            depth, c2w, sim_iface.intrinsics, det_box,
            top_frac=cfg.field_verify_top_frac,
            min_points=cfg.field_verify_min_points,
            pool=cfg.field_verify_pool)
        if field_score is None or field_score < cfg.field_verify_threshold:
            fs = "n/a" if field_score is None else f"{field_score:.3f}"
            print(f"{tag}: field-verify REJECTED detection "
                  f"(llmdet={det_score:.3f}, field={fs} < "
                  f"{cfg.field_verify_threshold:.2f})")
            det_score, det_box = 0.0, None
        else:
            print(f"{tag}: field-verify accepted "
                  f"(llmdet={det_score:.3f}, field={field_score:.3f})")

    det_persistent, det_investigate = classify_detection(
        perception, cfg, sim_iface.intrinsics, det_box, depth, c2w, pos)
    # No separate score gate here: the detector's own floor (llmdet_threshold)
    # already bounds the score of any surviving box, so every usable-band
    # detection counts toward the latch.
    conf_score = det_score if det_investigate else 0.0

    if not detected:
        if det_persistent:
            detected_streak += 1
        else:
            detected_streak = 0
        if detected_streak >= cfg.detected_persistence:
            detected = True
            print(f"{tag}: DETECTED — entering exploit mode "
                  f"(det_score={det_score:.3f})")
    return (det_score, det_box, det_persistent, det_investigate, conf_score,
            detected, detected_streak, field_score)


def plan_one_action(perception, sim_iface, mppi, cfg,
                    action_queue: deque,
                    det_score: float = 0.0,
                    detected: bool = False, det_box=None, depth=None, c2w=None,
                    last_box_goal=None,
                    det_investigate: bool = False):
    """Run MPPI from current pose and return (action, opt_traj, goal, box_goal).

    action is [v_mps, w_rad_per_s] (or a discrete primitive name) or None if
    idling. opt_traj is a grid-coord list [(z_idx, x_idx), ...]; None when
    falling back to the queue or idling (no new plan was computed).

    On a successful plan the full MPPI control sequence replaces the queue.

    `det_score` is the raw detector confidence for the current frame, used as
    `goal_confidence` in SEARCH mode (MPPI's threshold + hysteresis filter
    noisy detections). `detected=True` overrides it with goal_confidence=1.0,
    saturating the conf curve so `w_ig_conf = 0` and IG is hard-off.
    """
    bev_sim = _to_numpy(perception.similarity_grid.get_2d_map())
    bev_epi = _to_numpy(perception.ugrid.get_2d_map(type='epistemic'))
    bev_occ = _to_numpy(perception.occupancy_grid.get_2d_map())

    if bev_sim is None or bev_epi is None or bev_occ is None:
        print("  plan: missing map(s); idling")
        return (action_queue.popleft() if action_queue else None), None, None, None

    pos = sim_iface.agent_position
    heading = get_agent_heading(sim_iface.agent)
    start_grid = world_to_grid(pos[0], pos[2], perception.similarity_grid, cfg.voxel_resolution)

    # Drive straight at the highest-similarity cell, even if it's on an
    # obstacle (the target object itself). In EXPLOIT, MPPI freezes each
    # rollout at its first waypoint within the goal-arrival radius (with an
    # arrival bonus), so the planner commits to "stop next to the goal"
    # instead of orbiting; the frozen cell itself must be collision-free.
    sim_for_goal = bev_sim.copy().astype(np.float32)
    sim_for_goal[bev_occ == 0] = -np.inf  # exclude unseen
    # Goal selection priority:
    #   1. Detection box this frame → the BEV cell its center projects to,
    #      used verbatim.
    #   2. No box this frame, but a previous box-derived goal is cached →
    #      reuse that fixed world cell so the agent commits to the last
    #      sighting instead of chasing a far-away similarity peak.
    #   3. Otherwise → global argmax of observed bev_sim (exploration).
    goal = None
    box_goal = None  # None unless THIS frame's detection produced a fresh goal
    H, W = sim_for_goal.shape
    # Layer 1: project a fresh box-derived goal from THIS frame's detection
    # whenever it's worth investigating (`det_investigate` — anything but a
    # too-close box that fills the frame). This INCLUDES too-far detections, so
    # the agent steers toward and investigates a distant sighting even though it
    # won't latch on it yet (latching needs the usable band; see the caller).
    # Every such detection re-projects, so in SEARCH mode the caller's cache
    # tracks the most recent bounding box. Until a detection is worth
    # investigating the agent explores via the global similarity argmax
    # (Layer 3).
    if det_box is not None and det_investigate:
        # Project the box center straight to one BEV cell, used verbatim (a
        # cell on the target surface is fine: in EXPLOIT, MPPI freezes rollouts
        # at their first waypoint within the goal-arrival radius and drops any
        # frozen on an occupied cell, so winners stop on free cells nearby).
        cand = bev_cell_from_box_center(
            perception, cfg, sim_iface.intrinsics, det_box, depth, c2w)
        if cand is not None:
            goal = cand
            box_goal = goal
    if goal is None and last_box_goal is not None:
        gz, gx = int(last_box_goal[0]), int(last_box_goal[1])
        if 0 <= gz < H and 0 <= gx < W:
            goal = (gz, gx)
    if goal is None:
        if not np.any(np.isfinite(sim_for_goal)):
            print("  plan: no observed cells, can't pick a goal")
            return (action_queue.popleft() if action_queue else None), None, None, None
        flat_idx = int(np.argmax(sim_for_goal))
        goal = (flat_idx // W, flat_idx % W)
    # Goal weight (`w_conf` inside MPPI):
    #   EXPLOIT (detected latched True) → pin at 1.0 so the agent commits
    #     hard to the cached max-conf goal and never decays back to IG.
    #   SEARCH → raw per-frame det_score. MPPI's hysteresis
    #     (exploit_conf = max(incoming, prev * conf_decay)) ratchets up on
    #     strong sightings and decays after a stretch of misses, eventually
    #     falling below threshold and re-enabling IG-driven exploration.
    # The cached *goal cell* is independent of this — see Layer 1/2 above —
    # so the planner still aims at the strongest sighting in either mode.
    goal_confidence = 1.0 if detected else float(det_score)

    opt_path, U_opt = mppi.optimize_trajectory(
        start_grid, goal, bev_epi, bev_occ,
        initial_heading=heading,
        intrinsics=sim_iface.intrinsics,
        sensor_height=cfg.sensor_height,
        goal_confidence=goal_confidence,
    )
    if U_opt is None or U_opt.shape[0] == 0:
        # No safe MPPI plan this replan — every rollout collides (wedged in a
        # tight pocket). Instead of idling, rotate clockwise in place to sweep
        # the heading: the next replan starts from a new orientation, which
        # usually exposes a collision-free rollout out of the pocket. "turn_right"
        # (grid θ decreasing) is clockwise viewed top-down; the continuous form
        # produces the same physical Habitat yaw as the discrete primitive (both
        # go through mppi_w_sign, so they rotate the same way).
        print("  plan: MPPI returned no safe control sequence, rotating clockwise")
        action_queue.clear()
        if cfg.discrete_actions:
            return "turn_right", None, goal, box_goal
        w_rps = -cfg.mppi_w_sign * cfg.mppi_max_w_rps
        return [0.0, w_rps], None, goal, box_goal

    # Discrete mode: hand the optimized path to the tracking controller, which
    # emits a single Habitat ObjectNav primitive. The continuous control queue
    # is unused (the agent re-plans before every primitive).
    if cfg.discrete_actions:
        action_queue.clear()
        act = discrete_action_from_plan(opt_path, heading, cfg)
        return act, opt_path, goal, box_goal

    # Successful plan: replace queue with full MPPI control sequence.
    # MPPI U units: v in [grid cells / mppi_step], w in [rad / mppi_step].
    # sim_iface.step expects [m/s, rad/s].
    action_queue.clear()
    for i in range(U_opt.shape[0]):
        v_mps = float(U_opt[i, 0]) * cfg.voxel_resolution / cfg.mppi_dt
        w_rps = cfg.mppi_w_sign * float(U_opt[i, 1]) / cfg.mppi_dt
        action_queue.append([v_mps, w_rps])

    return action_queue.popleft(), opt_path, goal, box_goal


def nearest_goal_point(goal, p):
    """World xyz of the point of `goal` closest (in the x-z plane) to point `p`.

    A goal is either a point ([x, y, z]) — returned as-is — or an axis-aligned
    rectangular footprint, given as {"rect": [x_min, z_min, x_max, z_max],
    "y": <height>}. For a rect, `p`'s (x, z) is clamped into the rectangle, so a
    `p` outside maps to the nearest edge/corner and a `p` over the footprint maps
    to itself (distance 0). This is what lets the agent stop anywhere along a
    large table's perimeter and still be scored against the closest part of it.
    """
    if isinstance(goal, dict):
        x_min, z_min, x_max, z_max = goal["rect"]
        y = goal.get("y", float(p[1]))
        cx = min(max(float(p[0]), x_min), x_max)
        cz = min(max(float(p[2]), z_min), z_max)
        return [cx, y, cz]
    return [float(goal[0]), float(goal[1]), float(goal[2])]


def goal_geodesic(pathfinder, p, goal):
    """Geodesic distance from world point `p` to the nearest part of `goal`
    (point or rectangle). Both endpoints are navmesh-snapped inside
    `geodesic_distance`."""
    return geodesic_distance(pathfinder, p, nearest_goal_point(goal, p))


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
    Returns a metrics dict (see bottom of the function).

    Evidence saved to `cfg.output_dir` (all tiny except the video path):
      - always: `traj_log.jsonl` (the trajectory) + `grid_extent.json`.
      - `save_enabled` (default): additionally the FINAL BEV occupancy map
        (`occ_maps/bev_occupancy_<last_step>.npy`).
      - `save_video`: additionally the full per-step BEV map + RGB history
        (epistemic/occupancy/similarity `.npy` + `rgbs/*.png` at every refresh
        tick) and the rendered `nav_history.mp4`. This is the only heavy path.
    The feature-field checkpoint is never written (nothing consumes it)."""
    if start_pos is None:
        start_pos = [-2.0, -3.0, 6.0]
    # 1. Init simulator
    sim, agent = init_simulator(
        cfg.scene_path, width=cfg.img_width, height=cfg.img_height,
        fov_deg=cfg.fov, sensor_height=cfg.sensor_height,
        agent_radius=cfg.agent_radius,
        agent_height=cfg.agent_height,
    )
    start_nav = spawn_agent_at_pos(sim, agent, start_pos,
                                   rotation=start_rotation)  # snapped navmesh start
    snap_dist = float(np.linalg.norm(np.asarray(start_nav) - np.asarray(start_pos)))
    if snap_dist > cfg.max_spawn_snap_m:
        sim.close()
        raise RuntimeError(
            f"spawn point {list(start_pos)} snapped {snap_dist:.2f}m to "
            f"{list(np.asarray(start_nav))} (> max_spawn_snap_m="
            f"{cfg.max_spawn_snap_m}m) — requested spawn isn't connected to "
            "this scene's navmesh (disconnected floor / broken reconstruction "
            "/ bad annotation); not a fair episode to evaluate.")
    # `bev_height_min`/`bev_height_max` may have been set from the *nominal*
    # start_pos[1] by the caller (e.g. eval_scene.py's per-episode floor
    # band) before the sim existed to snap it. Shift the band by however much
    # snapping moved the agent vertically, so a small mismatch here doesn't
    # silently starve every BEV map of observed cells for the whole episode.
    dy = float(np.asarray(start_nav)[1] - start_pos[1])
    if dy:
        cfg.bev_height_min += dy
        cfg.bev_height_max += dy
    # grid_max_height is a hard absolute-Y cap applied once at grid
    # construction (VoxelGrid.initialize_from_bounds); grow it to cover the
    # (possibly just-shifted) band so upper-floor bands don't get silently
    # clipped to an empty Y-range.
    cfg.grid_max_height = max(cfg.grid_max_height, cfg.bev_height_max)
    scene_bounds = get_scene_bounds_from_pathfinder(sim)

    sim_iface = SimInterface(cfg, sim, agent)
    perception = PerceptionStack(cfg, scene_bounds)  # owns target_query; initialised from cfg

    mppi = MPPIPlanner(cfg, device=cfg.device)
    detector = make_detector(cfg)
    action_queue: deque = deque()

    # Planner state. `detected` latches True (and never releases) once
    # cfg.detected_persistence consecutive persistent detections accrue; while
    # latched MPPI silences IG and pulls hard toward the cached goal.
    # `last_box_goal` caches the goal cell of the most recent investigated
    # detection so a missed frame doesn't drop the agent back to global-argmax
    # exploration.
    detected = False
    detected_streak = 0
    last_box_goal = None

    # Clear per-step artifact dirs from any previous run sharing this
    # output_dir. They hold step-numbered .npy/.png/.npz files; unlike
    # traj_log.jsonl (truncated below) they were never cleared, so a previous,
    # longer run (often a different scene) left stale snapshots that visualize.py
    # then interleaved into the new navigation history. Wipe them so only the
    # current run's maps exist.
    if save_enabled or save_video:
        # `featurefield` is legacy — earlier runs wrote a checkpoint there that
        # nothing consumes; clear it too so re-runs reclaim the disk.
        for sub in ("umaps", "occ_maps", "sim_maps", "rgbs", "featurefield"):
            shutil.rmtree(os.path.join(cfg.output_dir, sub), ignore_errors=True)

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

    # First maps (needed for the first plan) + grid extent. The per-step map +
    # RGB history is only saved for the video; the minimal path saves just the
    # final occupancy map at the end.
    perception.update_maps(step=0, save_enabled=save_video)
    if save_video:
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
    # Discrete-mode primitive budget: every MOVE_FORWARD / TURN counts as one
    # ObjectNav step; exhausting `cfg.max_agent_steps` without self-stopping is a
    # timeout (failure), matching VLFM / SemExp / the Habitat challenge.
    agent_steps = 0

    # Graceful early stop. Two ways to request it; both let the loop break at
    # the next step and fall through to the final map snapshot + visualization
    # below (instead of losing them to a hard kill):
    #   1. Ctrl-C (SIGINT) → sets a flag. A second Ctrl-C hard-aborts in case
    #      something is wedged.
    #   2. Create the sentinel file `<output_dir>/STOP` (e.g. `touch` it, handy
    #      when running under `docker exec` without an attached TTY).
    # A stale STOP from a previous run is removed up front so it doesn't end the
    # new run immediately.
    stop_file = os.path.join(cfg.output_dir, "STOP")
    if os.path.exists(stop_file):
        os.remove(stop_file)
    stop_requested = {"flag": False}

    def _on_sigint(signum, frame):
        if stop_requested["flag"]:
            raise KeyboardInterrupt
        stop_requested["flag"] = True
        print("\n[main] stop requested (Ctrl-C) — finishing the current step, "
              "then saving maps + rendering the visualization. "
              "Press Ctrl-C again to abort immediately.")

    prev_sigint = signal.signal(signal.SIGINT, _on_sigint)

    for step in range(1, cfg.iterations + 1):
        if stop_requested["flag"] or os.path.exists(stop_file):
            reason = "Ctrl-C" if stop_requested["flag"] else f"{stop_file} sentinel"
            print(f"[main] ending early at step {last_step} ({reason})")
            break
        last_step = step
        # A. train every step
        loss = perception.train_step(super_pts, super_feats)

        # B. replan + 1 agent step
        if step % REPLAN_INTERVAL == 0:
            t_plan = time.time()
            pos = sim_iface.agent_position.copy()
            heading = get_agent_heading(sim_iface.agent)

            rgb_cur, depth_cur, c2w_cur = perception.observe(sim_iface)

            # Throttle the detector in EXPLOIT mode: the goal is already pinned
            # to the cached box cell, so skip the (expensive) detector on most
            # replans and reuse the cache. SEARCH mode always detects. A skipped
            # replan reports no detection (det_box=None), so plan_one_action
            # falls through to the cached box-goal (Layer 2). The latch logic is
            # guarded by `if not detected`, so a skipped 0.0 score is harmless.
            replan_idx = step // REPLAN_INTERVAL
            if detected and cfg.exploit_redetect_interval <= 0:
                run_detector = False
            elif detected:
                run_detector = (replan_idx % cfg.exploit_redetect_interval == 0)
            else:
                run_detector = True
            # Detect (subject to the throttle above), classify into too-close /
            # too-far / usable-band, and advance the latch. `det_investigate`
            # (= not too close) drives the goal + confidence; `det_persistent`
            # (usable band over threshold) also drives latching, so the agent
            # approaches a far sighting and commits only once it has closed into
            # the usable band.
            (det_score, det_box, det_persistent, det_investigate, conf_score,
             detected, detected_streak, field_score) = detect_classify_latch(
                detector, perception, sim_iface, cfg,
                rgb_cur, depth_cur, c2w_cur, pos, detected, detected_streak,
                run_detector=run_detector, tag=f"step {step}")

            # # SEARCH-mode goal disconfirmation: if the agent has reached its
            # # cached box-goal but nothing is detected there now, the earlier
            # # sighting was a false positive (a transient detector spike or a
            # # similarity blip). Drop the cache so the planner resumes exploration
            # # instead of dwelling next to the spike. EXPLOIT never reaches here
            # # (it has latched on a confirmed target), and a fresh investigated
            # # detection this frame leaves `det_investigate` True so we keep it.
            # if not detected and last_box_goal is not None and not det_investigate:
            #     agz, agx = world_to_grid(pos[0], pos[2], perception.similarity_grid, cfg.voxel_resolution)
            #     reach_cells = cfg.stop_distance_m / cfg.voxel_resolution
            #     if float(np.hypot(agz - last_box_goal[0], agx - last_box_goal[1])) <= reach_cells:
            #         print(f"  goal disconfirmed at {last_box_goal} "
            #               f"(reached, no detection) — clearing cache")
            #         last_box_goal = None

            action, opt_traj, goal_cell, box_goal = plan_one_action(
                perception, sim_iface, mppi, cfg, action_queue,
                det_score=conf_score, detected=detected,
                det_box=det_box, depth=depth_cur, c2w=c2w_cur,
                last_box_goal=last_box_goal,
                det_investigate=det_investigate,
            )
            # Cache the goal of the most recent investigated detection.
            # `box_goal` is non-None only for an investigated box (Layer 1 gates
            # on `det_investigate`), so this tracks the latest sighting worth
            # steering toward — including too-far ones — in SEARCH mode; in
            # EXPLOIT the detector is throttled off so box_goal stays None and
            # the cache freezes on the latched object. Fallback paths (cached /
            # global argmax) leave box_goal=None.
            if box_goal is not None:
                last_box_goal = box_goal
                print(f"  cached new goal {box_goal} (det_score={det_score:.3f})")
            mode = 'EXPLOIT' if detected else 'SEARCH'
            w_conf = float(getattr(mppi, 'last_w_conf', 0.0))
            if goal_cell is not None:
                sg = perception.similarity_grid
                goal_x_m = sg.min_x + goal_cell[1] * cfg.voxel_resolution
                goal_z_m = sg.min_z + goal_cell[0] * cfg.voxel_resolution
                goal_str = f"({goal_x_m:+.2f}, {goal_z_m:+.2f})m"
            else:
                goal_str = "—"
            if action is not None:
                if cfg.discrete_actions:
                    sim_iface.step_discrete(action)
                    agent_steps += 1
                    action_str = f"{action} ({agent_steps}/{cfg.max_agent_steps})"
                else:
                    sim_iface.step(action, dt=cfg.mppi_dt)
                    action_str = (f"[v={action[0]:+.3f} m/s, "
                                  f"w={action[1]:+.3f} rad/s]")
                cur_pos = np.asarray(sim_iface.agent_position, dtype=np.float64)
                path_length += float(np.linalg.norm(cur_pos - prev_pos))
                prev_pos = cur_pos
                print(f"step {step}: plan {time.time()-t_plan:.2f}s | "
                      f"action {action_str} | "
                      f"position: {pos[0]:.2f}, {pos[2]:.2f} | "
                      f"det: {det_score:.3f} | "
                      f"goal: {goal_str} | w_conf: {w_conf:.2f} | "
                      f"mode: {mode}")
            else:
                print(f"step {step}: plan {time.time()-t_plan:.2f}s | no action | "
                      f"det: {det_score:.3f} | "
                      f"goal: {goal_str} | w_conf: {w_conf:.2f} | "
                      f"mode: {mode}")

            # Termination check: measure distance to whatever goal the planner
            # just aimed at (the max-conf goal — current frame if it beat the
            # cache, else the cached cell). Stop after the traj_log write
            # below so the final replan is captured for visualization.
            stop_now = False
            if detected and goal_cell is not None:
                # Obstacle-aware distance from the planner's goal distance
                # field (matches the geodesic success metric: can't fire
                # through a wall). Straight-line fallback only if the planner
                # didn't run this replan.
                dist_m = mppi.last_goal_dist_m
                if dist_m is None:
                    gz, gx = int(goal_cell[0]), int(goal_cell[1])
                    sz, sx = world_to_grid(
                        pos[0], pos[2], perception.similarity_grid, cfg.voxel_resolution,
                    )
                    dist_m = float(np.hypot(gz - sz, gx - sx)) * cfg.voxel_resolution
                if dist_m <= cfg.stop_distance_m:
                    print(f"step {step}: TARGET REACHED "
                          f"(dist={dist_m:.2f}m <= {cfg.stop_distance_m:.2f}m)")
                    stop_now = True

            with open(traj_log_path, 'a') as _f:
                _f.write(json.dumps({
                    'step': step,
                    'pos': [float(pos[0]), float(pos[2])],
                    'heading': float(heading),
                    'action': (action if cfg.discrete_actions
                               else ([float(action[0]), float(action[1])]
                                     if action is not None else [0.0, 0.0])),
                    'opt_traj': [[int(p[0]), int(p[1])] for p in opt_traj] if opt_traj else [],
                    'det_conf': float(det_score),
                    'field_score': (float(field_score)
                                    if field_score is not None else None),
                    'det_box': [float(v) for v in det_box] if det_box is not None else None,
                    'goal': [int(goal_cell[0]), int(goal_cell[1])] if goal_cell is not None else None,
                    'mode': mode,
                    'w_conf': w_conf,
                }) + '\n')

            if stop_now:
                agent_stopped = True
                if cfg.discrete_actions:
                    agent_steps += 1  # the STOP primitive counts too
                else:
                    sim_iface.step([0.0, 0.0], dt=cfg.mppi_dt)
                break

            # Discrete-mode primitive budget: exhausting it without self-stopping
            # is a timeout (failure), so the agent never runs past the challenge's
            # action budget even though training continues for `cfg.iterations`.
            if cfg.discrete_actions and agent_steps >= cfg.max_agent_steps:
                print(f"step {step}: STEP BUDGET EXHAUSTED "
                      f"({agent_steps}/{cfg.max_agent_steps}) — timeout (failure)")
                break

        # C. refresh buffer + maps (slowest cadence — bottleneck)
        if step % cfg.hash_buffer_refresh_interval == 0:
            t_refresh = time.time()
            rgb, depth, c2w = perception.observe(sim_iface)
            perception.update_replay_buffer(rgb, depth, c2w, sim_iface.intrinsics)
            perception.update_occupancy(depth, c2w, sim_iface.intrinsics)

            if save_video:
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

            perception.update_maps(step=step, save_enabled=save_video)
            gc.collect()
            torch.cuda.empty_cache()
            print(f"step {step}: refresh+maps {time.time()-t_refresh:.2f}s | "
                  f"history {perception.buffer_size}/{perception.buffer_capacity} pts")

        # if step % 100 == 0:
        #     print(f"  step {step:05d} | loss {loss:.5f} | t {time.time()-start_time:.1f}s")

    # Restore the default Ctrl-C behavior so the (potentially long) finalization
    # + visualization below can be aborted normally.
    signal.signal(signal.SIGINT, prev_sigint)

    # Final evidence snapshot aligned to last_step. The C cadence only fires
    # every refresh interval, and early termination via `stop_now` or Ctrl-C can
    # leave it 100+ steps behind the actual last action, so refresh once here.
    if save_video and last_step % cfg.hash_buffer_refresh_interval != 0:
        # Video: the full map + RGB history so the visualizer's final frame is
        # up-to-date. Skipped when last_step already matches the most recent
        # refresh tick — the same files would just be overwritten.
        print(f"[main] saving final maps at step {last_step}...")
        rgb_f, depth_f, c2w_f = perception.observe(sim_iface)
        perception.update_replay_buffer(rgb_f, depth_f, c2w_f, sim_iface.intrinsics)
        perception.update_occupancy(depth_f, c2w_f, sim_iface.intrinsics)
        rgb_dir = os.path.join(cfg.output_dir, "rgbs")
        os.makedirs(rgb_dir, exist_ok=True)
        rgb_img = (rgb_f.numpy() * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(rgb_dir, f"rgb_{last_step:03d}.png"), rgb_img)
        perception.save_2d_similarity(last_step, depth_f, c2w_f, sim_iface.intrinsics)
        perception.update_maps(step=last_step, save_enabled=True)
    elif save_enabled and not save_video:
        # Minimal: just the final occupancy map (traj_log + grid_extent are
        # already on disk). Refresh occupancy from a final observation first so
        # it reflects the end pose, then save that one .npy — no feature field,
        # no per-step maps, no RGB.
        print(f"[main] saving final occupancy map at step {last_step}...")
        _, depth_f, c2w_f = perception.observe(sim_iface)
        perception.update_occupancy(depth_f, c2w_f, sim_iface.intrinsics)
        perception.occupancy_grid.save(last_step)

    # Episode scoring. Compute geodesics while the pathfinder is still alive
    # (before sim.close()). Without ground-truth `goals` we can only report the
    # self-reported subset (the agent's own stop decision + distance traveled).
    final_pos = np.asarray(sim_iface.agent_position, dtype=np.float64)
    # Recorded so the episode can be re-scored offline (e.g. against a
    # rectangular table footprint or a different radius) without re-running.
    final_pos_xyz = [float(v) for v in final_pos]
    start_nav_xyz = [float(v) for v in np.asarray(start_nav, dtype=np.float64)]
    if goals:
        # Goals may be points or rectangular footprints (see nearest_goal_point);
        # each is scored against its closest part to the query point.
        l_geo = min(goal_geodesic(sim.pathfinder, start_nav, g) for g in goals)
        d_final = min(goal_geodesic(sim.pathfinder, final_pos, g) for g in goals)
        success = bool(agent_stopped and d_final <= success_radius_m)
        if not success:
            spl = 0.0
        elif l_geo > 0.0 and np.isfinite(l_geo):
            spl = float(l_geo / max(path_length, l_geo))
        else:
            # Spawned already on the goal (l == 0) or goal unreachable on the
            # navmesh but the agent reported success — credit a perfect path.
            spl = 1.0
        print(f"[eval] success={success} spl={spl:.3f} | "
              f"l_geo={l_geo:.2f}m path={path_length:.2f}m "
              f"final_geo={d_final:.2f}m stopped={agent_stopped}")
        metrics = {
            'success': success, 'spl': spl,
            'l_geodesic': float(l_geo), 'final_geodesic': float(d_final),
            'path_length': float(path_length), 'agent_stopped': bool(agent_stopped),
            'final_pos': final_pos_xyz, 'start_nav': start_nav_xyz,
            'success_radius_m': float(success_radius_m),
            'steps': int(last_step), 'scene': cfg.scene_path, 'query': cfg.target_query,
        }
    else:
        metrics = {
            'success': None, 'spl': None,
            'l_geodesic': None, 'final_geodesic': None,
            'path_length': float(path_length), 'agent_stopped': bool(agent_stopped),
            'final_pos': final_pos_xyz, 'start_nav': start_nav_xyz,
            'success_radius_m': float(success_radius_m),
            'steps': int(last_step), 'scene': cfg.scene_path, 'query': cfg.target_query,
        }

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
