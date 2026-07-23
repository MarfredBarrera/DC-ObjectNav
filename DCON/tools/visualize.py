"""Post-hoc navigation visualization.

Loads saved maps (uncertainty, occupancy, similarity) and the trajectory log
written by main.py, then renders a video showing maps evolving over time with
the agent trail and MPPI optimised paths overlaid.

Usage:
    cd DCON
    python tools/visualize.py                         # defaults from config.yaml
    python tools/visualize.py --output figs/nav.mp4 --fps 4
"""

import argparse
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from src.config import Config
from src.visualization.visualizer import Visualizer


def _norm_box(box):
    """Return (x0, y0, x1, y1) with x0<=x1, y0<=y1, or None if degenerate.

    Defensive: PIL's draw.rectangle raises if x1<x0, so normalize corner order
    and drop empty boxes (older logs may contain out-of-order coordinates).
    """
    if box is None:
        return None
    xa, ya, xb, yb = box
    x0, x1 = (xa, xb) if xa <= xb else (xb, xa)
    y0, y1 = (ya, yb) if ya <= yb else (yb, ya)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def annotate_detection(rgb_uint8: np.ndarray, box, color=(255, 64, 64),
                       label=None) -> np.ndarray:
    """Draw a bounding box (and optional label) on a uint8 RGB image."""
    box = _norm_box(box)
    if box is None:
        return rgb_uint8
    img = Image.fromarray(rgb_uint8)
    draw = ImageDraw.Draw(img)
    xmin, ymin, xmax, ymax = box
    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
    if label is not None:
        draw.text((int(xmin) + 2, max(0, int(ymin) - 12)), label, fill=color)
    return np.asarray(img)



# ── Helpers ────────────────────────────────────────────────────────────────

def load_traj_log(path):
    """Load JSONL trajectory log; return list of dicts sorted by step."""
    if not os.path.exists(path):
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                # Skip the leading metadata record (e.g. the distractor
                # vocabulary) main.py stashes as a `meta` line with no `step`.
                if 'step' not in rec:
                    continue
                entries.append(rec)
    return sorted(entries, key=lambda e: e['step'])


def load_grid_extent(path):
    """Load grid extent saved by main.py. Returns dict or None."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def grid_to_world(z_idx, x_idx, min_x, min_z, res):
    return (min_x + x_idx * res, min_z + z_idx * res)


def load_map(directory, filename):
    path = os.path.join(directory, filename)
    return np.load(path) if os.path.exists(path) else None




# ── Main ───────────────────────────────────────────────────────────────────

def render_navigation(cfg, output_path: str, fps: int = 5,
                      snapshot_step: int = -1, max_step: Optional[int] = None) -> None:
    """Render the navigation video (or single-step snapshot) from disk artifacts.

    `max_step` caps which BEV map snapshots are included — pass the last step
    actually executed by the live loop to avoid pulling in stale .npy files
    left over from a previous, longer run (traj_log.jsonl is truncated each
    run, but the umaps/occ_maps/sim_maps directories are not).
    """
    viz = Visualizer(cfg)

    output_dir = cfg.output_dir

    # Load grid extent written by main.py
    extent_data = load_grid_extent(os.path.join(output_dir, "grid_extent.json"))
    if extent_data is None:
        print("ERROR: grid_extent.json not found. Run main.py first.")
        return
    min_x = extent_data['min_x']
    max_x = extent_data['max_x']
    min_z = extent_data['min_z']
    max_z = extent_data['max_z']
    res   = extent_data['voxel_resolution']
    extent = [min_x, max_x, min_z, max_z]

    def g2w(z_idx, x_idx):
        return grid_to_world(z_idx, x_idx, min_x, min_z, res)

    # Load trajectory log
    traj_log = load_traj_log(os.path.join(output_dir, "traj_log.jsonl"))
    if not traj_log:
        print("WARNING: traj_log.jsonl is empty or missing — trajectories won't be shown.")

    # Determine map steps from config, capped at the run's actual final step.
    # When max_step isn't given (standalone CLI), bound to the last step in the
    # freshly-truncated traj_log rather than cfg.iterations — otherwise we'd
    # scan all the way to cfg.iterations and risk pulling in stale .npy
    # snapshots left by a previous, longer run sharing this output_dir.
    if max_step is not None:
        cap = int(max_step)
    elif traj_log:
        cap = int(traj_log[-1]['step'])
    else:
        cap = cfg.iterations
    map_steps = [s for s in range(0, cap + 1, cfg.viz_interval)]

    # Always include the run's final step (main.py snapshots aligned maps at
    # last_step before exiting), even if it isn't a viz_interval tick.
    if cap not in map_steps:
        map_steps.append(cap)

    # Filter to only those that actually exist on disk
    map_steps = [s for s in map_steps if os.path.exists(os.path.join(output_dir, "umaps", f"bev_epistemic_uncertainty_{s}.npy"))]

    if snapshot_step >= 0:
        # User asked for a specific step — honor it whether or not it lined up
        # with viz_interval.
        map_steps = [snapshot_step]

    if not map_steps:
        print(f"ERROR: No saved BEV maps found in {output_dir} (cap step={cap}, "
              f"viz_interval={cfg.viz_interval}). Run main.py with save_enabled=True.")
        return
    print(f"Using {len(map_steps)} map snapshots (step 0 to {map_steps[-1]} every {cfg.viz_interval}) and {len(traj_log)} trajectory log entries.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    frames = []
    for map_step in map_steps:
        # Load BEV maps for this step
        epi = load_map(os.path.join(output_dir, "umaps"),
                       f"bev_epistemic_uncertainty_{map_step}.npy")
        occ = load_map(os.path.join(output_dir, "occ_maps"),
                       f"bev_occupancy_{map_step}.npy")
        sim = load_map(os.path.join(output_dir, "sim_maps"),
                       f"bev_similarity_{map_step}.npy")
        
        rgb_path = os.path.join(output_dir, "rgbs", f"rgb_{map_step:03d}.png")
        # if not os.path.exists(rgb_path):
        #     rgb_path = os.path.join(output_dir, "rgbs", f"bootstrap_{map_step:03d}.png") # fallback

        rgb = imageio.imread(rgb_path) if os.path.exists(rgb_path) else None

        if epi is None or occ is None or sim is None:
            print(f"  Skipping step {map_step}: one or more maps missing.")
            continue

        # Agent trail = all positions logged up to (and including) this map step
        trail_entries = [e for e in traj_log if e['step'] <= map_step]
        agent_trail = [(e['pos'][0], e['pos'][1]) for e in trail_entries]

        # Most recent plan at or before this step
        current_entry = trail_entries[-1] if trail_entries else None

        opt_traj_world = (
            [g2w(p[0], p[1]) for p in current_entry['opt_traj']]
            if current_entry and current_entry['opt_traj'] else []
        )
        current_pos     = tuple(current_entry['pos'])     if current_entry else None
        current_heading = float(current_entry['heading']) if current_entry else 0.0

        # Recorded detection from the most-recent plan step — draw the accepted
        # detector box on the RGB panel.
        det_conf = float(current_entry.get('det_conf', 0.0) or 0.0) if current_entry else 0.0
        det_box = current_entry.get('det_box') if current_entry else None
        if rgb is not None:
            rgb = annotate_detection(rgb, det_box)

        # Goal cell (z, x) chosen by the planner this step. Convert to world.
        goal_cell = current_entry.get('goal') if current_entry else None
        goal_world = g2w(goal_cell[0], goal_cell[1]) if goal_cell else None

        # Planner mode (SEARCH/EXPLOIT) and current exploit weight.
        mode = (current_entry.get('mode') if current_entry else None) or 'SEARCH'
        w_conf = float(current_entry.get('w_conf', 0.0) or 0.0) if current_entry else 0.0

        maps_dict = {'occ': occ, 'sim': sim, 'epi': epi, 'rgb': rgb}

        fig = viz.render_combined_grid(
            maps_dict, extent,
            agent_trail=agent_trail,
            opt_traj_world=opt_traj_world, current_pos=current_pos,
            current_heading=current_heading,
            step=map_step,
            det_conf=det_conf,
            goal_world=goal_world, goal_cell=goal_cell,
            mode=mode, w_conf=w_conf,
        )
        frames.append(viz.fig_to_numpy(fig))
        plt.close(fig)
        print(f"  Rendered step {map_step:6d} | trail pts: {len(agent_trail):4d} | "
              f"opt: {len(opt_traj_world):3d}")

    if not frames:
        print("No frames rendered — nothing to save.")
        return

    if snapshot_step >= 0:
        # If output still has a video extension, swap it for .png so imageio.imwrite doesn't fail
        out_path = output_path
        ext = os.path.splitext(out_path)[1].lower()
        if ext in ['.mp4', '.gif', '.avi', '.mov']:
            out_path = os.path.splitext(out_path)[0] + f"_step{snapshot_step}.png"

        imageio.imwrite(out_path, frames[0])
        print(f"\nSnapshot saved to: {out_path}")
        return

    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.gif':
        imageio.mimsave(output_path, frames, fps=fps)
    else:
        imageio.mimsave(output_path, frames, fps=fps, codec='libx264')

    print(f"\nVideo saved to: {output_path}  ({len(frames)} frames @ {fps} fps)")


def main():
    parser = argparse.ArgumentParser(description="Visualize DC-ObjectNav navigation history")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default="./figs/nav_history.mp4")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--snapshot_step", type=int, default=-1, help="If positive, generate a single PNG for this step instead of a video")
    parser.add_argument("--max_step", type=int, default=None, help="Cap visualized step (default: cfg.iterations)")
    args = parser.parse_args()

    cfg = Config(args.config)
    render_navigation(
        cfg, args.output, fps=args.fps,
        snapshot_step=args.snapshot_step, max_step=args.max_step,
    )


if __name__ == "__main__":
    main()
