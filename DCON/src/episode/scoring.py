"""Episode scoring: geodesic success + SPL against ground-truth goals.

Success is GEODESIC (navmesh shortest path), matching the Habitat ObjectNav
challenge — there is no straight-line success mode.
"""

import numpy as np

from src.habitat.habitat_utils import geodesic_distance


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


def score_episode(cfg, pathfinder, goals, start_nav, final_pos, path_length,
                  agent_stopped, success_radius_m, steps) -> dict:
    """Build the episode metrics dict.

    Success requires the agent to SELF-STOP within `success_radius_m` geodesic
    of the nearest goal; SPL weights that success by start→nearest-goal geodesic
    over distance travelled. Without ground-truth `goals` only the self-reported
    subset is available (stop decision + distance travelled) and the scored
    fields come back None. `final_pos`/`start_nav` are recorded so the episode
    can be re-scored offline (different radius, rect footprints) without
    re-running. Must be called while the pathfinder is alive (before sim.close()).
    """
    metrics = {
        'success': None, 'spl': None,
        'l_geodesic': None, 'final_geodesic': None,
        'path_length': float(path_length), 'agent_stopped': bool(agent_stopped),
        'final_pos': [float(v) for v in np.asarray(final_pos, dtype=np.float64)],
        'start_nav': [float(v) for v in np.asarray(start_nav, dtype=np.float64)],
        'success_radius_m': float(success_radius_m),
        'steps': int(steps), 'scene': cfg.scene_path, 'query': cfg.target_query,
    }
    if not goals:
        return metrics

    # Goals may be points or rectangular footprints (see nearest_goal_point);
    # each is scored against its closest part to the query point.
    l_geo = min(goal_geodesic(pathfinder, start_nav, g) for g in goals)
    d_final = min(goal_geodesic(pathfinder, final_pos, g) for g in goals)
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
    metrics.update(success=success, spl=spl,
                   l_geodesic=float(l_geo), final_geodesic=float(d_final))
    return metrics
