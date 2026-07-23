"""Everything one episode writes to `cfg.output_dir`.

Three evidence levels, selected by the two flags the caller passes:
  - always: `traj_log.jsonl` (the trajectory), `grid_extent.json`,
    `run_meta.json` (query + distractor bank).
  - `save_enabled`: additionally the FINAL BEV occupancy map
    (`occ_maps/bev_occupancy_<last_step>.npy`).
  - `save_video`: additionally the full per-step BEV map + RGB history
    (epistemic/occupancy/similarity `.npy` + `rgbs/*.png` at every refresh
    tick). This is the only heavy path.
The feature-field checkpoint is never written (nothing consumes it).
"""

import json
import os
import shutil

import imageio
import numpy as np
from PIL import Image, ImageDraw

# Per-step artifact dirs, wiped at episode start. They hold step-numbered
# .npy/.png files; unlike traj_log.jsonl they were never cleared, so a
# previous, longer run (often a different scene) left stale snapshots that
# visualize.py then interleaved into the new navigation history. `featurefield`
# is legacy — earlier runs wrote a checkpoint there that nothing consumes;
# clear it too so re-runs reclaim the disk.
ARTIFACT_DIRS = ("umaps", "occ_maps", "sim_maps", "rgbs", "featurefield")


class EpisodeRecorder:
    """Writes an episode's evidence bundle; owns the output-dir layout."""

    def __init__(self, cfg, save_enabled: bool = True, save_video: bool = True):
        self.cfg = cfg
        self.save_enabled = save_enabled
        self.save_video = save_video
        self.out_dir = cfg.output_dir
        self.traj_log_path = os.path.join(self.out_dir, "traj_log.jsonl")
        os.makedirs(self.out_dir, exist_ok=True)
        if save_enabled or save_video:
            for sub in ARTIFACT_DIRS:
                shutil.rmtree(os.path.join(self.out_dir, sub), ignore_errors=True)
        open(self.traj_log_path, 'w').close()  # truncate / create fresh

    def write_grid_extent(self, grid) -> None:
        with open(os.path.join(self.out_dir, "grid_extent.json"), 'w') as f:
            json.dump({
                'min_x': grid.min_x, 'max_x': grid.max_x,
                'min_z': grid.min_z, 'max_z': grid.max_z,
                'voxel_resolution': self.cfg.voxel_resolution,
            }, f)

    def write_run_meta(self, query, distractors) -> None:
        """Auditability: the distractor vocabulary the pairwise field actually
        used, in a metadata sidecar (like grid_extent.json) so traj_log.jsonl
        stays homogeneous per-step records. Empty when the field isn't running
        distractors (plain sigmoid mode)."""
        distractors = list(distractors)
        print(f"[main] field distractors ({len(distractors)}): {distractors}")
        with open(os.path.join(self.out_dir, "run_meta.json"), 'w') as f:
            json.dump({'query': query, 'distractors': distractors}, f)

    def snapshot(self, step, perception, rgb, depth, c2w, intrinsics) -> None:
        """Per-step RGB frame + 2D similarity map (video evidence only)."""
        if not self.save_video:
            return
        rgb_dir = os.path.join(self.out_dir, "rgbs")
        os.makedirs(rgb_dir, exist_ok=True)
        rgb_img = (rgb.numpy() * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(rgb_dir, f"rgb_{step:03d}.png"), rgb_img)
        perception.save_2d_similarity(step, depth, c2w, intrinsics)

    def save_field_verify_frame(self, step, rgb, det_box, field_score) -> None:
        """Calibration aid: dump the exact frame + box `field_score` was computed
        on, so it can be visually verified later (see cfg.field_verify_save_frames)."""
        if not (self.cfg.field_verify_save_frames
                and field_score is not None and det_box is not None):
            return
        frames_dir = os.path.join(self.out_dir, "field_verify_frames")
        os.makedirs(frames_dir, exist_ok=True)
        frame_img = Image.fromarray((rgb.numpy() * 255).astype(np.uint8))
        ImageDraw.Draw(frame_img).rectangle(list(det_box), outline=(255, 0, 0), width=3)
        frame_img.save(os.path.join(frames_dir, f"step{step:06d}_fs{field_score:.3f}.png"))

    def log_step(self, row: dict) -> None:
        with open(self.traj_log_path, 'a') as f:
            f.write(json.dumps(row) + '\n')

    def save_final(self, step, perception, sim_iface) -> None:
        """Final evidence snapshot aligned to `step` (the last executed step).

        The refresh cadence only fires every `hash_buffer_refresh_interval`
        steps, and early termination (arrival or Ctrl-C) can leave it 100+ steps
        behind the actual last action, so refresh once here.
        """
        if self.save_video:
            if step % self.cfg.hash_buffer_refresh_interval == 0:
                return  # already up to date — the same files would be rewritten
            # Video: the full map + RGB history so the visualizer's final frame
            # is up-to-date.
            print(f"[main] saving final maps at step {step}...")
            rgb, depth, c2w = perception.observe(sim_iface)
            perception.update_replay_buffer(rgb, depth, c2w, sim_iface.intrinsics)
            perception.update_occupancy(depth, c2w, sim_iface.intrinsics)
            self.snapshot(step, perception, rgb, depth, c2w, sim_iface.intrinsics)
            perception.update_maps(step=step, save_enabled=True)
        elif self.save_enabled:
            # Minimal: just the final occupancy map (traj_log + grid_extent are
            # already on disk). Refresh occupancy from a final observation first
            # so it reflects the end pose, then save that one .npy — no feature
            # field, no per-step maps, no RGB.
            print(f"[main] saving final occupancy map at step {step}...")
            _, depth, c2w = perception.observe(sim_iface)
            perception.update_occupancy(depth, c2w, sim_iface.intrinsics)
            perception.occupancy_grid.save(step)
