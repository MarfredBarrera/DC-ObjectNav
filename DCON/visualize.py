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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np

from src.config import Config
from src.visualization.visualizer import Visualizer


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

def main():
    parser = argparse.ArgumentParser(description="Visualize DC-ObjectNav navigation history")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default="./figs/nav_history.mp4")
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()

    cfg = Config(args.config)
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

    # Determine map steps from config
    map_steps = [s for s in range(0, cfg.iterations + 1, cfg.viz_interval)]
    
    # Filter to only those that actually exist (optional but keeps it robust)
    map_steps = [s for s in map_steps if os.path.exists(os.path.join(output_dir, "umaps", f"bev_epistemic_uncertainty_{s}.npy"))]

    if not map_steps:
        print(f"ERROR: No saved BEV maps found in {output_dir} following viz_interval {cfg.viz_interval}. "
              "Run main.py with save_enabled=True.")
        return
    print(f"Using {len(map_steps)} map snapshots (step 0 to {map_steps[-1]} every {cfg.viz_interval}) and {len(traj_log)} trajectory log entries.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

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

        maps_dict = {'occ': occ, 'sim': sim, 'epi': epi, 'rgb': rgb, 'sim2d': sim2d}
        
        # Optional: compute CombinedZ if we want to match analysis.py perfectly
        # For now, let's just pass the dict. Visualizer handles it.
        
        fig = viz.render_combined_grid(
            maps_dict, extent,
            agent_trail=agent_trail, ref_traj_world=ref_traj_world, 
            opt_traj_world=opt_traj_world, current_pos=current_pos, 
            current_heading=current_heading,
            step=map_step,
        )
        frames.append(viz.fig_to_numpy(fig))
        plt.close(fig)
        print(f"  Rendered step {map_step:6d} | trail pts: {len(agent_trail):4d} | "
              f"ref: {len(ref_traj_world):3d} | opt: {len(opt_traj_world):3d}")

    if not frames:
        print("No frames rendered — nothing to save.")
        return

    ext = os.path.splitext(args.output)[1].lower()
    if ext == '.gif':
        imageio.mimsave(args.output, frames, fps=args.fps)
    else:
        imageio.mimsave(args.output, frames, fps=args.fps, codec='libx264')

    print(f"\nVideo saved to: {args.output}  ({len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
