"""Path tracking: turn an MPPI grid path into one executable action.

The planner works in continuous velocity space over BEV cells; the simulator
wants either a velocity command or a Habitat ObjectNav primitive. This module
is the pure-pursuit controller that bridges them (SEARCH mode only — EXPLOIT
locomotion comes from DD-PPO, see `exploit.py`).
"""

import numpy as np


def lookahead_heading_error(opt_path, heading, cfg):
    """Signed heading error (rad) toward the pure-pursuit lookahead waypoint.

    Walks along `opt_path` (grid coords [(z_idx, x_idx), ...]) to the first
    waypoint at least `cfg.discrete_lookahead_m` ahead of the agent — or the
    final waypoint if the path is shorter — and returns the wrapped difference
    between the bearing to it and `heading`, both in the grid frame
    (atan2(Δz, Δx), same convention as `SimInterface.agent_heading`). Returns
    None for a degenerate path (too short, or the lookahead lands on the agent),
    meaning "no command this replan".
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
    return (desired - heading + np.pi) % (2 * np.pi) - np.pi


def discrete_action_from_plan(opt_path, heading, cfg):
    """Convert an MPPI optimized path into one Habitat ObjectNav primitive.

    Emits the primitive nearest the pure-pursuit bearing — TURN toward it when
    the heading error exceeds half a turn, otherwise MOVE_FORWARD. Returns one
    of "move_forward" / "turn_left" / "turn_right", or None if the path is
    degenerate (idle this replan). The receding-horizon replan corrects any
    tracking drift each cycle.
    """
    dtheta = lookahead_heading_error(opt_path, heading, cfg)
    if dtheta is None:
        return None
    if abs(dtheta) > np.radians(cfg.discrete_turn_deg / 2.0):
        # dtheta > 0 means "increase grid θ". step_discrete takes NATIVE
        # Habitat turn names, so map the grid-θ turn direction onto Habitat
        # yaw here via mppi_w_sign (the same sign the continuous `step`
        # applies to ω).
        grid_left = dtheta > 0
        native_left = grid_left if cfg.mppi_w_sign >= 0 else not grid_left
        return "turn_left" if native_left else "turn_right"
    return "move_forward"
