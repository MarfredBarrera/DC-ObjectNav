"""SEARCH-mode control: one MPPI replan → one executable action.

Pre-latch locomotion. Picks the goal cell (fresh detection → cached detection →
global similarity argmax), runs MPPI over the current BEV maps, and returns the
action to execute. Post-latch control lives in `exploit.py`.
"""

from collections import deque

import numpy as np

from src.perception.detection import bev_cell_from_box_center, world_to_grid
from src.perception.utils import to_numpy
from src.planning.tracking import discrete_action_from_plan


def plan_search_action(perception, sim_iface, mppi, cfg,
                       action_queue: deque,
                       det_score: float = 0.0,
                       det_box=None, depth=None, c2w=None,
                       last_box_goal=None,
                       det_investigate: bool = False):
    """Run MPPI from current pose and return (action, opt_traj, goal, box_goal).

    action is [v_mps, w_rad_per_s] (or a discrete primitive name) or None if
    idling. opt_traj is a grid-coord list [(z_idx, x_idx), ...]; None when
    falling back to the queue or idling (no new plan was computed).

    On a successful plan the full MPPI control sequence replaces the queue.

    `det_score` is the raw detector confidence for the current frame, passed to
    MPPI as `goal_confidence` (its threshold + hysteresis filter noisy
    detections).
    """
    bev_sim = to_numpy(perception.similarity_grid.get_2d_map())
    bev_epi = to_numpy(perception.ugrid.get_2d_map(type='epistemic'))
    bev_occ = to_numpy(perception.occupancy_grid.get_2d_map())

    if bev_sim is None or bev_epi is None or bev_occ is None:
        print("  plan: missing map(s); idling")
        return (action_queue.popleft() if action_queue else None), None, None, None

    pos = sim_iface.agent_position
    heading = sim_iface.agent_heading
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
    # Goal weight (`w_conf` inside MPPI): the raw per-frame det_score. MPPI's
    # hysteresis (exploit_conf = max(incoming, prev * conf_decay)) ratchets up
    # on strong sightings and decays after a stretch of misses, eventually
    # falling below threshold and re-enabling IG-driven exploration. The cached
    # *goal cell* is independent of this — see Layer 1/2 above — so the planner
    # still aims at the strongest sighting either way.
    opt_path, U_opt = mppi.optimize_trajectory(
        start_grid, goal, bev_epi, bev_occ,
        initial_heading=heading,
        intrinsics=sim_iface.intrinsics,
        sensor_height=cfg.sensor_height,
        goal_confidence=float(det_score),
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
    action_queue.clear()
    if cfg.discrete_actions:
        return discrete_action_from_plan(opt_path, heading, cfg), opt_path, goal, box_goal

    # Successful plan: replace queue with full MPPI control sequence.
    # MPPI U units: v in [grid cells / mppi_step], w in [rad / mppi_step].
    # sim_iface.step expects [m/s, rad/s].
    for i in range(U_opt.shape[0]):
        v_mps = float(U_opt[i, 0]) * cfg.voxel_resolution / cfg.mppi_dt
        w_rps = cfg.mppi_w_sign * float(U_opt[i, 1]) / cfg.mppi_dt
        action_queue.append([v_mps, w_rps])

    return action_queue.popleft(), opt_path, goal, box_goal
