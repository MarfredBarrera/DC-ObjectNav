import gc
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch

from src.config import Config
from src.semantics import SAM_CLIP_Semantics
from src.featurefield import FeatureField
from src.grid import UncertaintyGrid, OccupancyGrid, SimilarityGrid
from src.utils import unprojection


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

        self.sam_clip = SAM_CLIP_Semantics(cfg, device=self.device)
        self.ensemble = [
            FeatureField(cfg, scene_bounds=scene_bounds, device=self.device)
            for _ in range(cfg.ensemble_num_models)
        ]

        self.ugrid = UncertaintyGrid(cfg, ensemble=self.ensemble, scene_bounds=scene_bounds)
        self.occupancy_grid = OccupancyGrid(cfg, scene_bounds=scene_bounds)
        self.similarity_grid = SimilarityGrid(
            cfg, ensemble=self.ensemble, sam_clip=self.sam_clip, scene_bounds=scene_bounds,
        )

        self._replay_buffer = []  # list of (points_cpu, features_cpu)

    def observe(
        self,
        sim_iface,
        depth_near: float = 0.1,
        depth_far: float = 10.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (world_points, gt_features, depth, c2w) — all on CPU."""
        rgb, depth, c2w = sim_iface.get_observations()

        depth_gpu = depth.to(self.device)
        c2w_gpu = c2w.to(self.device)

        rgb_np = (rgb.numpy() * 255).astype(np.uint8)
        clip_features = self.sam_clip.extract_dense_features(rgb_np)

        mask = (depth_gpu > depth_near) & (depth_gpu < depth_far)
        world_points = unprojection(depth_gpu, sim_iface.intrinsics, c2w_gpu, self.device, mask=mask)
        gt_features = clip_features[mask]

        valid = gt_features.norm(dim=-1) > 1e-6
        world_points = world_points[valid]
        gt_features = gt_features[valid]

        return world_points.cpu(), gt_features.cpu(), depth.cpu(), c2w.cpu()

    def update_replay_buffer(self, world_points: torch.Tensor, gt_features: torch.Tensor) -> None:
        if len(self._replay_buffer) >= self.cfg.hash_replay_buffer_size:
            old_pts, old_feats = self._replay_buffer.pop(0)
            del old_pts, old_feats
            gc.collect()
        self._replay_buffer.append((world_points, gt_features))

    def update_occupancy(self, depth: torch.Tensor, c2w: torch.Tensor, intrinsics: tuple) -> None:
        self.occupancy_grid.update_from_observation(
            depth.to(self.device), c2w.to(self.device), intrinsics,
        )

    def make_super_batch(
        self,
        recent_sample_portion: float = 0.2,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Sample a GPU super-batch from the replay buffer. Returns (None, None) if empty."""
        if not self._replay_buffer:
            print("Warning: replay buffer is empty — cannot build super-batch.")
            return None, None

        staging_size = self.cfg.hash_train_batch_size * self.cfg.hash_buffer_refresh_interval

        gpu_pts_chunks = []
        gpu_feat_chunks = []

        recent_pts, recent_feats = self._replay_buffer[-1]
        n_recent = int(staging_size * recent_sample_portion)
        if recent_pts.shape[0] > 0:
            n = min(n_recent, recent_pts.shape[0])
            idx = torch.randint(0, recent_pts.shape[0], (n,))
            gpu_pts_chunks.append(recent_pts[idx].to(self.device))
            gpu_feat_chunks.append(recent_feats[idx].to(self.device))

        n_history = staging_size - n_recent
        history = self._replay_buffer[:-1]
        if history and n_history > 0:
            per_frame = n_history // len(history)
            for pts, feats in history:
                if pts.shape[0] > 0:
                    n = min(per_frame, pts.shape[0])
                    idx = torch.randint(0, pts.shape[0], (n,))
                    gpu_pts_chunks.append(pts[idx].to(self.device))
                    gpu_feat_chunks.append(feats[idx].to(self.device))

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
        target_query: str = "a pillow",
        height_filter: tuple = (0.1, 2.0),
        height_samples: int = 10,
        batch_size: int = 100_000,
    ) -> None:
        """Compute and save all BEV maps (uncertainty, occupancy, similarity). Intended for low-frequency calls."""
        t0 = time.time()

        self.ugrid.forward_pass(height_filter=height_filter, height_samples=height_samples, batch_size=batch_size)
        self.ugrid.save(step)
        self.ugrid.clear_umaps()

        self.occupancy_grid.save(step)

        self.similarity_grid.compute_similarity_map(target_query, occupancy_grid=self.occupancy_grid)
        self.similarity_grid.save(step)

        print(f"Maps saved at step {step} ({time.time() - t0:.1f}s)")

    def save_models(self) -> None:
        for i, model in enumerate(self.ensemble):
            path = os.path.join(self.cfg.output_dir, f"ensemble/featurefield_ensemble_{i}.pt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            model.save(path)
            print(f"Saved ensemble model {i} -> {path}")

    def load_models(self) -> None:
        ensemble_dir = os.path.join(self.cfg.output_dir, "ensemble")
        if not os.path.exists(ensemble_dir):
            print(f"Error: ensemble directory not found at {ensemble_dir}")
            return
        for i, model in enumerate(self.ensemble):
            path = os.path.join(ensemble_dir, f"featurefield_ensemble_{i}.pt")
            if os.path.exists(path):
                model.load(path)
                print(f"Loaded ensemble model {i} <- {path}")
            else:
                print(f"Warning: model {i} not found at {path}")

    @property
    def buffer_size(self) -> int:
        return len(self._replay_buffer)
