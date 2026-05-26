import gc
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch

from src.config import Config
from src.perception.semantics import MaskCLIPSemantics
from src.perception.featurefield import EvidentialFeatureField
from src.perception.grid import UncertaintyGrid, OccupancyGrid, SimilarityGrid
from src.perception.utils import unprojection


class _HistoryBuffer:
    """Flat history of observed points with bounded memory.

    Phase 1 (buffer not full): append in arrival order.
    Phase 2 (buffer full): every new point enters and overwrites a uniformly
    random existing slot. New data always wins; old data is continuously
    displaced. Sampling is one randint + one gather — no concat, no per-cell
    loop, no Algorithm R.
    """

    def __init__(self, capacity: int, feat_dim: int):
        self.cap = int(capacity)
        self.pts   = torch.empty((self.cap, 3),        dtype=torch.float32)
        self.feats = torch.empty((self.cap, feat_dim), dtype=torch.float32)
        self.fill = 0

    def insert(self, pts: torch.Tensor, feats: torch.Tensor) -> None:
        if pts.numel() == 0:
            return
        m = pts.shape[0]
        # Phase 1: fill empty slots.
        free = self.cap - self.fill
        take = min(free, m)
        if take > 0:
            s = self.fill
            self.pts[s:s + take]   = pts[:take]
            self.feats[s:s + take] = feats[:take]
            self.fill += take
        # Phase 2: random eviction for the rest, fully vectorized.
        rest = m - take
        if rest > 0:
            slots = torch.randint(0, self.cap, (rest,))
            self.pts[slots]   = pts[take:]
            self.feats[slots] = feats[take:]

    def sample(self, n: int):
        if self.fill == 0:
            return None, None
        n = min(n, self.fill)
        idx = torch.randint(0, self.fill, (n,))
        return self.pts[idx], self.feats[idx]


class PerceptionStack:
    """Perception pipeline: feature fields, uncertainty, occupancy, and similarity grids.

    Key methods and their intended cadence:
        observe()             — pull one RGB-D frame, extract world-space points + CLIP features
        update_replay_buffer()— insert observation points into the flat history buffer
        update_occupancy()    — update occupancy grid from depth + pose
        make_super_batch()    — stage a random sample from the buffer onto GPU
        train_step()          — one gradient step on the evidential field  [HIGH FREQUENCY]
        update_maps()         — compute & save all BEV maps                [LOW FREQUENCY]
    """

    def __init__(self, cfg: Config, scene_bounds: list):
        self.cfg = cfg
        self.device = cfg.device
        self.scene_bounds = scene_bounds

        self.mask_clip = MaskCLIPSemantics(device=self.device)
        # Single evidential field: aleatoric + epistemic uncertainty in one
        # forward pass via the Normal-Inverse-Gamma marginal (replaces the
        # ensemble whose only role was empirical epistemic from prediction var).
        self.feature_field = EvidentialFeatureField(
            cfg, scene_bounds=scene_bounds, device=self.device,
        )

        self.ugrid = UncertaintyGrid(cfg, feature_field=self.feature_field, scene_bounds=scene_bounds)
        self.occupancy_grid = OccupancyGrid(cfg, scene_bounds=scene_bounds)
        self.similarity_grid = SimilarityGrid(
            cfg, feature_field=self.feature_field, semantics=self.mask_clip, scene_bounds=scene_bounds,
        )

        # Flat history buffer: bounded-memory ring over all observed points
        # (minus the held-out latest frame). New points always enter and
        # displace a uniformly random existing slot once full, so recent data
        # is never silently dropped. Sampling is one randint + gather.
        self._history_buffer = _HistoryBuffer(
            capacity=cfg.history_buffer_capacity,
            feat_dim=cfg.hash_feature_dim,
        )
        self._latest_frame = None         # most-recent (pts_cpu, feats_cpu), held out
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

        # Previous latest frame is folded into the history buffer; the newly
        # arrived frame takes its place as the held-out latest for the
        # recent-oversample.
        if self._latest_frame is not None:
            prev_pts, prev_feats = self._latest_frame
            self._history_buffer.insert(prev_pts, prev_feats)
        self._latest_frame = new_entry

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
        if self._latest_frame is None and self._history_buffer.fill == 0:
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

        # History: one uniform draw over the flat buffer's current contents.
        n_history = staging_size - n_recent
        if n_history > 0:
            hist_pts, hist_feats = self._history_buffer.sample(n_history)
            if hist_pts is not None:
                gpu_pts_chunks.append(hist_pts.to(self.device))
                gpu_feat_chunks.append(hist_feats.to(self.device))

        if not gpu_pts_chunks:
            return None, None

        return torch.cat(gpu_pts_chunks, dim=0), torch.cat(gpu_feat_chunks, dim=0)

    def train_step(self, super_points: torch.Tensor, super_features: torch.Tensor) -> float:
        """Sample a mini-batch from the super-batch and run one gradient step. Returns loss."""
        mini_batch_size = self.cfg.hash_train_batch_size

        if super_points is None or super_points.shape[0] < mini_batch_size:
            return 0.0

        batch_idx = torch.randint(0, super_points.shape[0], (mini_batch_size,), device=self.device)
        batch_pts = super_points[batch_idx]
        batch_feats = super_features[batch_idx]

        loss = self.feature_field.train_step(batch_pts, batch_feats)
        return loss if loss is not None else 0.0

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
        path = os.path.join(self.cfg.output_dir, "featurefield/featurefield.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.feature_field.save(path)
        print(f"Saved feature field -> {path}")

    def save_pretrained(self) -> None:
        """Save current model to featurefield/pretrained/ so a future run can load it."""
        path = os.path.join(self.cfg.output_dir, "featurefield/pretrained/pretrained.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.feature_field.save(path)
        print(f"Saved pretrained feature field -> {path}")

    def load_models(self, pretrained: bool = False) -> bool:
        """Load feature field from disk. Returns True if loaded."""
        if pretrained:
            path = os.path.join(self.cfg.output_dir, "featurefield/pretrained/pretrained.pt")
        else:
            path = os.path.join(self.cfg.output_dir, "featurefield/featurefield.pt")

        if not os.path.exists(path):
            print(f"Warning: feature field checkpoint not found at {path}")
            return False

        # Skip optimizer state for pretrained loads — fine-tuning on a new
        # data distribution with the converged Adam stats produces oversized
        # first steps that destroy the pretrained weights.
        self.feature_field.load(path, load_optimizer=not pretrained)
        print(f"  -> Loaded {'pretrained ' if pretrained else ''}feature field <- {path}")
        return True
        
    def save_2d_similarity(self, step: int, depth: torch.Tensor, c2w: torch.Tensor, intrinsics: tuple) -> None:
        """Query feature field for 2D similarity from current camera view and save."""
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
            batch_size = 100_000
            num_batches = int(np.ceil(world_points.shape[0] / batch_size))
            all_sims = []

            with torch.no_grad():
                for i in range(num_batches):
                    start, end = i * batch_size, min((i + 1) * batch_size, world_points.shape[0])
                    batch_pts = world_points[start:end]

                    gamma, _, _, _ = self.feature_field.forward(batch_pts, normalize=True)
                    gamma = gamma / (gamma.norm(dim=-1, keepdim=True) + 1e-8)

                    sim = (gamma @ text_embed.T).squeeze(-1)
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
        return self._history_buffer.fill

    @property
    def buffer_capacity(self) -> int:
        return self._history_buffer.cap
