import gc
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch

from src.config import Config
from src.perception.semantics import MaskCLIPSemantics
from src.perception.featurefield import FeatureField
from src.perception.grid import UncertaintyGrid, OccupancyGrid, SimilarityGrid
from src.perception.utils import unprojection


class PerceptionStack:
    """Perception pipeline: feature fields, uncertainty, occupancy, and similarity grids.

    Key methods and their intended cadence:
        observe()             — pull one RGB-D frame, extract world-space points + CLIP features
        update_replay_buffer()— add observation to CPU replay buffer (evicts oldest if full)
        update_occupancy()    — update occupancy grid from depth + pose
        make_super_batch()    — stage a random sample from the buffer onto GPU
        train_step()          — one gradient step over the ensemble  [HIGH FREQUENCY]
        update_maps()         — compute & save all BEV maps          [LOW FREQUENCY]
    """

    def __init__(self, cfg: Config, scene_bounds: list):
        self.cfg = cfg
        self.device = cfg.device
        self.scene_bounds = scene_bounds

        self.mask_clip = MaskCLIPSemantics(device=self.device)
        self.ensemble = [
            FeatureField(cfg, scene_bounds=scene_bounds, device=self.device)
            for _ in range(cfg.ensemble_num_models)
        ]

        self.ugrid = UncertaintyGrid(cfg, ensemble=self.ensemble, scene_bounds=scene_bounds)
        self.occupancy_grid = OccupancyGrid(cfg, scene_bounds=scene_bounds)
        self.similarity_grid = SimilarityGrid(
            cfg, ensemble=self.ensemble, semantics=self.mask_clip, scene_bounds=scene_bounds,
        )

        # Buffer stores already-unprojected, subsampled (pts, feats) tuples on
        # CPU. Unprojection runs once per frame (at insert), not per refresh,
        # which lets super-batches sample uniformly across history without
        # paying the unprojection cost for every buffered frame on every refresh.
        # Reservoir-sampled (Algorithm R), so old frames are retained with
        # probability k/n_promoted instead of FIFO-evicted — keeps the buffer
        # representative of the *entire* run history at bounded memory.
        self._replay_buffer = []          # reservoir of (pts_cpu, feats_cpu)
        self._latest_frame = None         # most-recent (pts_cpu, feats_cpu), held out
        self._n_promoted = 0              # frames promoted from latest into reservoir
        self.target_query = cfg.target_query
        
    def extract_and_unproject(self, rgb, depth, c2w, intrinsics, depth_near=None, depth_far=None) -> Tuple[torch.Tensor, torch.Tensor]:
        depth_gpu = depth.to(self.device)
        c2w_gpu = c2w.to(self.device)

        # MaskCLIP: pass the image tensor directly on GPU — no numpy conversion
        rgb_gpu = rgb.to(self.device)
        clip_features = self.mask_clip.extract_dense_features(rgb_gpu)

        if depth_near is None:
            depth_near = self.cfg.min_sensor_dist
        if depth_far is None:
            depth_far = self.cfg.max_sensor_dist

        mask = (depth_gpu > depth_near) & (depth_gpu < depth_far)
        world_points = unprojection(depth_gpu, intrinsics, c2w_gpu, self.device, mask=mask)
        gt_features = clip_features[mask]

        valid = gt_features.norm(dim=-1) > 1e-6
        world_points = world_points[valid]
        gt_features = gt_features[valid]

        return world_points, gt_features

    def observe(
        self,
        sim_iface,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (rgb, depth, c2w) — all on CPU."""
        rgb, depth, c2w = sim_iface.get_observations()
        return rgb.cpu(), depth.cpu(), c2w.cpu()

    def update_replay_buffer(self, rgb: torch.Tensor, depth: torch.Tensor, c2w: torch.Tensor, intrinsics: tuple) -> None:
        pts, feats = self.extract_and_unproject(rgb, depth, c2w, intrinsics)
        cache_size = self.cfg.hash_per_frame_cache_size
        if pts.shape[0] > cache_size:
            idx = torch.randperm(pts.shape[0], device=pts.device)[:cache_size]
            pts = pts[idx]
            feats = feats[idx]
        new_entry = (pts.cpu(), feats.cpu())

        # The previous latest frame is now eligible for the reservoir; the
        # newly arrived frame takes its place as latest.
        if self._latest_frame is not None:
            self._reservoir_insert(self._latest_frame)
        self._latest_frame = new_entry

    def _reservoir_insert(self, entry) -> None:
        """Algorithm R: keep `entry` with probability k/n where n = total
        frames promoted so far. Result: at any time, the reservoir is a
        uniform random sample of all promoted frames. Frames from early in
        the run keep a non-zero chance of surviving indefinitely."""
        k = self.cfg.hash_replay_buffer_size
        self._n_promoted += 1
        if len(self._replay_buffer) < k:
            self._replay_buffer.append(entry)
            return
        if torch.rand(1).item() < k / self._n_promoted:
            slot = int(torch.randint(0, k, (1,)).item())
            self._replay_buffer[slot] = entry

    def update_occupancy(self, depth: torch.Tensor, c2w: torch.Tensor, intrinsics: tuple) -> None:
        self.occupancy_grid.update_from_observation(
            depth.to(self.device), c2w.to(self.device), intrinsics,
        )

    def make_super_batch(
        self,
        recent_sample_portion: float = 0.2,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Sample a GPU super-batch from the replay buffer. Returns (None, None) if empty.

        Sampling is *point-wise* uniform across all frames in the history (not
        frame-wise then point-wise), so consecutive super-batches don't change
        in discrete chunks when the random frame subset shifts. The most-recent
        frame is still oversampled by `recent_sample_portion` to keep fresh
        observations weighted.
        """
        if self._latest_frame is None and not self._replay_buffer:
            print("Warning: replay buffer is empty — cannot build super-batch.")
            return None, None

        staging_size = self.cfg.hash_train_batch_size * self.cfg.hash_buffer_refresh_interval

        gpu_pts_chunks = []
        gpu_feat_chunks = []

        # Recent-frame oversample: pull n_recent points from the held-out
        # latest frame. Cap at the cache size — sampling more than the pool
        # size with replacement just duplicates points and blows GPU memory.
        if self._latest_frame is not None:
            recent_pts, recent_feats = self._latest_frame
            n_recent = min(int(staging_size * recent_sample_portion), recent_pts.shape[0])
            if n_recent > 0:
                idx = torch.randperm(recent_pts.shape[0])[:n_recent]
                gpu_pts_chunks.append(recent_pts[idx].to(self.device))
                gpu_feat_chunks.append(recent_feats[idx].to(self.device))
        else:
            n_recent = 0

        # History: draw uniformly across the reservoir pool, capped at the
        # pool size. Reservoir membership is a uniform random sample of all
        # frames ever observed, so old frames are represented in expectation.
        if self._replay_buffer:
            hist_pts = torch.cat([p for p, _ in self._replay_buffer], dim=0)
            hist_feats = torch.cat([f for _, f in self._replay_buffer], dim=0)
            n_history = min(staging_size - n_recent, hist_pts.shape[0])
            if n_history > 0:
                idx = torch.randperm(hist_pts.shape[0])[:n_history]
                gpu_pts_chunks.append(hist_pts[idx].to(self.device))
                gpu_feat_chunks.append(hist_feats[idx].to(self.device))

        if not gpu_pts_chunks:
            return None, None

        return torch.cat(gpu_pts_chunks, dim=0), torch.cat(gpu_feat_chunks, dim=0)

    def train_step(self, super_points: torch.Tensor, super_features: torch.Tensor) -> float:
        """Sample a mini-batch from the super-batch and run one gradient step. Returns avg loss."""
        mini_batch_size = self.cfg.hash_train_batch_size

        if super_points is None or super_points.shape[0] < mini_batch_size:
            return 0.0

        batch_idx = torch.randint(0, super_points.shape[0], (mini_batch_size,), device=self.device)
        batch_pts = super_points[batch_idx]
        batch_feats = super_features[batch_idx]

        total_loss = 0.0
        for model in self.ensemble:
            loss = model.train_step(batch_pts, batch_feats)
            if loss is not None:
                total_loss += loss

        return total_loss / len(self.ensemble)

    def update_maps(
        self,
        step: int,
        batch_size: int = 100_000,
        save_enabled: bool = True,
    ) -> None:
        """Compute all VOXEL maps (uncertainty, occupancy, similarity). Saves to disk only if save_enabled."""
        t0 = time.time()

        # Uncertainty Reduction
        self.ugrid.forward_pass(batch_size=batch_size)
        if save_enabled:
            self.ugrid.save(step)

        # Occupancy (already updated incrementally)
        if save_enabled:
            self.occupancy_grid.save(step)

        # Similarity
        self.similarity_grid.compute_similarity_map(self.target_query, occupancy_grid=self.occupancy_grid)
        if save_enabled:
            self.similarity_grid.save(step)

        verb = "saved" if save_enabled else "computed (in-memory)"
        print(f"Maps {verb} at step {step} ({time.time() - t0:.2f}s)")

    def save_models(self) -> None:
        for i, model in enumerate(self.ensemble):
            path = os.path.join(self.cfg.output_dir, f"ensemble/featurefield_ensemble_{i}.pt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            model.save(path)
            print(f"Saved ensemble model {i} -> {path}")

    def save_pretrained(self) -> None:
        """Save current models to ensemble/pretrained/ so a future run can load them."""
        for i, model in enumerate(self.ensemble):
            path = os.path.join(self.cfg.output_dir, f"ensemble/pretrained/pretrained_{i}.pt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            model.save(path)
            print(f"Saved pretrained model {i} -> {path}")

    def load_models(self, pretrained: bool = False) -> bool:
        """Load models from disk. Returns True if at least one model was loaded."""
        ensemble_dir = os.path.join(self.cfg.output_dir, "ensemble")
        if pretrained:
            ensemble_dir = os.path.join(ensemble_dir, "pretrained")

        if not os.path.exists(ensemble_dir):
            print(f"Warning: ensemble directory not found at {ensemble_dir}")
            return False

        loaded_any = False
        for i, model in enumerate(self.ensemble):
            # When loading pretrained, try pretrained_{i}.pt first then fall back
            # to featurefield_ensemble_{i}.pt (handles the case where the user
            # placed standard-save files in the pretrained directory).
            if pretrained:
                candidates = [f"pretrained_{i}.pt", f"featurefield_ensemble_{i}.pt"]
            else:
                candidates = [f"featurefield_ensemble_{i}.pt"]

            path = None
            for filename in candidates:
                candidate = os.path.join(ensemble_dir, filename)
                if os.path.exists(candidate):
                    path = candidate
                    break

            if path is not None:
                # Skip optimizer state for pretrained loads — fine-tuning on a new
                # data distribution with the converged Adam stats produces oversized
                # first steps that destroy the pretrained weights.
                model.load(path, load_optimizer=not pretrained)
                print(f"  -> Loaded {'pretrained ' if pretrained else ''}model {i} <- {path}")
                loaded_any = True
            else:
                tried = [os.path.join(ensemble_dir, f) for f in candidates]
                print(f"  -> Warning: model {i} not found (tried: {tried})")

        return loaded_any
        
    def save_2d_similarity(self, step: int, depth: torch.Tensor, c2w: torch.Tensor, intrinsics: tuple) -> None:
        """Query ensemble for 2D similarity from current camera view and save."""
        if not self.target_query:
            return

        # 1. Get Text Embedding
        with torch.no_grad():
            text_embed = self.mask_clip.encode_text(self.target_query)

        # 2. Project View to 3D
        fx, fy, cx, cy, H, W = intrinsics
        mask = (depth > self.cfg.min_sensor_dist) & (depth < self.cfg.max_sensor_dist)
        world_points = unprojection(depth.to(self.device), intrinsics, c2w.to(self.device), self.device, mask=mask)

        if world_points.shape[0] == 0:
            sim_2d = torch.zeros((H, W), device=self.device)
        else:
            # 3. Query Ensemble
            batch_size = 100_000
            num_batches = int(np.ceil(world_points.shape[0] / batch_size))
            all_sims = []
            
            with torch.no_grad():
                for i in range(num_batches):
                    start, end = i * batch_size, min((i + 1) * batch_size, world_points.shape[0])
                    batch_pts = world_points[start:end]
                    
                    batch_means = []
                    for model in self.ensemble:
                        m, _ = model.forward(batch_pts, normalize=True)
                        batch_means.append(m)
                    
                    ens_mean = torch.stack(batch_means, dim=0).mean(dim=0)
                    ens_mean = ens_mean / (ens_mean.norm(dim=-1, keepdim=True) + 1e-8)
                    
                    sim = (ens_mean @ text_embed.T).squeeze(-1)
                    sim = (sim + 1.0) / 2.0
                    all_sims.append(sim)
            
            sim_2d = torch.zeros((H, W), device=self.device)
            sim_2d[mask] = torch.cat(all_sims, dim=0)

        # 4. Save
        sim_dir = os.path.join(self.cfg.output_dir, "2d_sim_maps")
        os.makedirs(sim_dir, exist_ok=True)
        save_path = os.path.join(sim_dir, f"sim2d_{step:03d}.npy")
        np.save(save_path, sim_2d.cpu().numpy())

    @property
    def buffer_size(self) -> int:
        return len(self._replay_buffer)
