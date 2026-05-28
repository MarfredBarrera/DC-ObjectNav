"""Post-hoc navigation visualization.

Loads saved maps (uncertainty, occupancy, similarity) and the trajectory log
written by main.py, then renders a video showing maps evolving over time with
agent trail, A* reference paths, and MPPI optimised paths overlaid.

Usage:
    cd DCON
    python visualize.py                         # defaults from config.yaml
    python visualize.py --output figs/nav.mp4 --fps 4
"""

import argparse
import json
import os
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from src.config import Config
from src.visualization.visualizer import Visualizer


def annotate_detection(rgb_uint8: np.ndarray, box) -> np.ndarray:
    """Draw the detector's bounding box on a uint8 RGB image. No text overlay."""
    if box is None:
        return rgb_uint8
    img = Image.fromarray(rgb_uint8)
    draw = ImageDraw.Draw(img)
    xmin, ymin, xmax, ymax = box
    draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 64, 64), width=3)
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
                entries.append(json.loads(line))
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

    # Determine map steps from config, capped at the run's actual final step
    # (defaults to cfg.iterations when called standalone via CLI).
    cap = cfg.iterations if max_step is None else int(max_step)
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
    
    # Calculate average MPPI cost
    costs = [e['mppi_cost'] for e in traj_log if 'mppi_cost' in e and e['mppi_cost'] is not None]
    costs = [c for c in costs if np.isfinite(c)]
    avg_cost = np.mean(costs) if costs else 0.0
    print(f"Average achieved MPPI cost: {avg_cost:.2f}")

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
        
        # Load extra diagnostic maps
        sim2d = load_map(os.path.join(output_dir, "2d_sim_maps"),
                         f"sim2d_{map_step:03d}.npy")
        
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

        ref_traj_world = (
            [g2w(p[0], p[1]) for p in current_entry['ref_traj']]
            if current_entry and current_entry['ref_traj'] else []
        )
        opt_traj_world = (
            [g2w(p[0], p[1]) for p in current_entry['opt_traj']]
            if current_entry and current_entry['opt_traj'] else []
        )
        current_pos     = tuple(current_entry['pos'])     if current_entry else None
        current_heading = float(current_entry['heading']) if current_entry else 0.0

        # Recorded detection from the most-recent plan step.
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

        maps_dict = {'occ': occ, 'sim': sim, 'epi': epi, 'rgb': rgb, 'sim2d': sim2d}
        
        # Optional: compute CombinedZ if we want to match analysis.py perfectly
        # For now, let's just pass the dict. Visualizer handles it.
        
        # Get cost for this specific step from traj_log
        current_cost = next((e['mppi_cost'] for e in trail_entries[::-1] if 'mppi_cost' in e and e['mppi_cost'] is not None and np.isfinite(e['mppi_cost'])), 0.0)

        fig = viz.render_combined_grid(
            maps_dict, extent,
            agent_trail=agent_trail, ref_traj_world=ref_traj_world,
            opt_traj_world=opt_traj_world, current_pos=current_pos,
            current_heading=current_heading,
            step=map_step,
            avg_cost=avg_cost, current_cost=current_cost,
            det_conf=det_conf,
            goal_world=goal_world, goal_cell=goal_cell,
            mode=mode, w_conf=w_conf,
        )
        frames.append(viz.fig_to_numpy(fig))
        plt.close(fig)
        print(f"  Rendered step {map_step:6d} | trail pts: {len(agent_trail):4d} | "
              f"ref: {len(ref_traj_world):3d} | opt: {len(opt_traj_world):3d}")

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
