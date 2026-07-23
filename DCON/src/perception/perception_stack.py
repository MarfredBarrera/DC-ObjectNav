import gc
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch

from src.config import Config
from src.perception.distractor_gen import build_distractor_vocabulary
from src.perception.semantics import CLIPSegSemantics
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

        # CLIPSeg is query-conditioned (a trained decoder scores one fixed
        # prompt), not a query-agnostic embedding like MaskCLIP -- accepted
        # tradeoff: target_query never changes mid-run, and CLIPSeg's per-
        # pixel signal is much cleaner, especially before the target has been
        # well-observed. See CLIPSegSemantics docstring.
        # Distractor vocabulary: the generic `background_terms` bank plus the
        # object confusers (a per-target LLM set when `llm_distractors` is on,
        # else the static `distractor_objects`). Assembled once here, before
        # the field/buffer claim GPU memory (the LLM path frees itself first).
        # `filter_distractors` inside CLIPSegSemantics still strips any phrase
        # sharing a content word with the query as a safety net.
        distractors = None
        if cfg.clipseg_contrastive or cfg.clipseg_pairwise:
            distractors = build_distractor_vocabulary(cfg.target_query, cfg)
        self.semantics = CLIPSegSemantics(
            query=cfg.target_query,
            device=self.device,
            model_name=cfg.clipseg_model_name,
            distractors=distractors,
            softmax_temp=cfg.clipseg_softmax_temp,
            pairwise=cfg.clipseg_pairwise,
        )
        # Pairwise mode: the field regresses one sigmoid channel per term
        # ([query] + the per-query filtered distractors), so the feature dim
        # is only known after filter_distractors ran. Override before the
        # field/buffer are sized. K can hit 0 if every distractor shares a
        # word with the query — then pairwise degenerates to the plain
        # sigmoid field (margin falls back to presence at verify time).
        if cfg.clipseg_pairwise:
            cfg.hash_feature_dim = 1 + len(self.semantics.distractors)
            print(f"[PerceptionStack] pairwise field: hash_feature_dim -> "
                  f"{cfg.hash_feature_dim} (query + {len(self.semantics.distractors)} "
                  f"distractor channels)")
        # Single evidential field: aleatoric + epistemic uncertainty in one
        # forward pass via the Normal-Inverse-Gamma marginal (replaces the
        # ensemble whose only role was empirical epistemic from prediction var).
        # Output is now a scalar CLIPSeg relevance score (hash_feature_dim=1),
        # not a 512-D embedding -- the field aggregates that score across
        # every observed viewpoint into one persistent, multi-view-consistent
        # relevance map instead of trusting any single noisy frame.
        self.feature_field = EvidentialFeatureField(
            cfg, scene_bounds=scene_bounds, device=self.device,
        )

        self.ugrid = UncertaintyGrid(cfg, feature_field=self.feature_field, scene_bounds=scene_bounds)
        self.occupancy_grid = OccupancyGrid(cfg, scene_bounds=scene_bounds)
        self.similarity_grid = SimilarityGrid(
            cfg, feature_field=self.feature_field, scene_bounds=scene_bounds,
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
        # Components of the most recent field_score_in_box call (pairwise
        # mode: presence / margin / per-term pooled scores) for logging.
        self.last_field_verify = None
        
    def extract_and_unproject(self, rgb, depth, c2w, intrinsics, depth_near=None, depth_far=None) -> Tuple[torch.Tensor, torch.Tensor]:
        depth_gpu = depth.to(self.device)
        c2w_gpu = c2w.to(self.device)

        # CLIPSeg: pass the image tensor directly on GPU — no numpy conversion
        rgb_gpu = rgb.to(self.device)
        clip_features = self.semantics.extract_dense_features(rgb_gpu)

        if depth_near is None:
            depth_near = self.cfg.min_sensor_dist
        if depth_far is None:
            depth_far = self.cfg.max_sensor_dist

        mask = (depth_gpu > depth_near) & (depth_gpu < depth_far)
        world_points = unprojection(depth_gpu, intrinsics, c2w_gpu, self.device, mask=mask)
        gt_features = clip_features[mask]

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

        # Uncertainty Reduction (free voxels masked to kill phantom traces —
        # the field never trains on air, so free-air uncertainty is noise).
        self.ugrid.forward_pass(
            batch_size=batch_size,
            occupancy_grid=self.occupancy_grid if self.cfg.mask_free_epistemic else None,
        )
        if save_enabled:
            self.ugrid.save(step)

        # Occupancy (already updated incrementally)
        if save_enabled:
            self.occupancy_grid.save(step)

        # Similarity
        self.similarity_grid.compute_similarity_map()
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

        # Project View to 3D. No text embedding needed anymore -- the field's
        # output IS the relevance score directly (trained to regress CLIPSeg's
        # scalar activation for the fixed target_query).
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

                    gamma, _, _, _ = self.feature_field.forward(batch_pts, normalize=False)
                    if gamma.shape[-1] > 1:
                        # Pairwise field: clamped worst-case margin (query
                        # channel minus hardest distractor channel).
                        sim = (gamma[..., 0] - gamma[..., 1:].max(dim=-1).values).clamp(0.0, 1.0)
                    else:
                        sim = gamma.squeeze(-1).clamp(0.0, 1.0)
                    all_sims.append(sim)

            sim_2d = torch.zeros((H, W), device=self.device)
            sim_2d[mask] = torch.cat(all_sims, dim=0)

        # 4. Save
        sim_dir = os.path.join(self.cfg.output_dir, "2d_sim_maps")
        os.makedirs(sim_dir, exist_ok=True)
        save_path = os.path.join(sim_dir, f"sim2d_{step:03d}.npy")
        np.save(save_path, sim_2d.cpu().numpy())

    @torch.no_grad()
    def field_score_in_box(
        self,
        depth: torch.Tensor,
        c2w: torch.Tensor,
        intrinsics: tuple,
        box,
        top_frac: float = 0.10,
        min_points: int = 20,
        pool: str = "topk",
    ) -> Optional[float]:
        """Pooled field relevance over the depth pixels inside a detector box.

        Unprojects every valid-depth pixel inside `box` (xmin, ymin, xmax,
        ymax, in `depth`'s pixel space) to a world point, forward-passes the
        trained CLIPSeg relevance field there, and pools the point scores:
        `pool="topk"` returns the mean of the top `top_frac` fraction (robust
        to background inside a loose box, unlike a whole-box mean; not a
        single-pixel statistic like max); `pool="max"` returns the single
        highest point score. Used to verify detections against the accumulated
        multi-view map: a look-alike that fools the detector on one frame
        reads low until the field itself has learned "target here".

        Pairwise mode (cfg.clipseg_pairwise): the field is multi-channel
        (channel 0 = query, 1..K = distractors). The top-frac cells are
        selected by the QUERY channel, every channel is pooled over those
        same cells (margin-of-means over the multi-view-converged field),
        and the returned score is the worst-case margin
        presence_q - max_i presence_i in [-1, 1]. The components (presence,
        margin, per-term pooled scores) land in `self.last_field_verify`.

        Returns None when the box is degenerate or has fewer than
        `min_points` valid-depth pixels (can't be verified).
        """
        self.last_field_verify = None
        if box is None:
            return None
        depth_gpu = depth.to(self.device)
        H, W = depth_gpu.shape[-2:]
        x0 = max(0, int(box[0]))
        y0 = max(0, int(box[1]))
        x1 = min(W, int(np.ceil(box[2])))
        y1 = min(H, int(np.ceil(box[3])))
        if x1 <= x0 or y1 <= y0:
            return None
        mask = torch.zeros((H, W), dtype=torch.bool, device=self.device)
        mask[y0:y1, x0:x1] = True
        mask &= (depth_gpu > self.cfg.min_sensor_dist) & (depth_gpu < self.cfg.max_sensor_dist)
        if int(mask.sum()) < max(1, int(min_points)):
            return None
        pts = unprojection(depth_gpu, intrinsics, c2w.to(self.device), self.device, mask=mask)

        batch_size = self.cfg.hash_inference_batch_size
        scores = []
        for i in range(0, pts.shape[0], batch_size):
            gamma, _, _, _ = self.feature_field.forward(pts[i:i + batch_size], normalize=False)
            scores.append(gamma.clamp(0.0, 1.0))
        flat = torch.cat(scores, dim=0)  # (N, C); C == 1 unless pairwise

        if self.cfg.clipseg_pairwise and flat.shape[-1] > 0:
            # Select cells by the query channel, pool every channel over the
            # SAME cells (top-k'ing channels independently would compare
            # different locations).
            q = flat[:, 0]
            if pool == "max":
                idx = q.argmax().unsqueeze(0)
            elif pool == "topk":
                k = max(1, int(round(q.numel() * top_frac)))
                idx = torch.topk(q, k).indices
            else:
                raise ValueError(f"Unknown field_verify pool {pool!r} (expected 'topk' or 'max')")
            pooled = flat[idx].mean(dim=0)          # (C,)
            presence = float(pooled[0])
            if pooled.numel() > 1:
                margin = presence - float(pooled[1:].max())
            else:
                margin = presence  # no distractor channels survived filtering
            self.last_field_verify = {
                "presence": presence,
                "margin": margin,
                "terms": [round(float(x), 4) for x in pooled],
            }
            return margin

        flat = flat.squeeze(-1)
        if pool == "max":
            score = float(flat.max())
        elif pool == "topk":
            k = max(1, int(round(flat.numel() * top_frac)))
            score = float(torch.topk(flat, k).values.mean())
        else:
            raise ValueError(f"Unknown field_verify pool {pool!r} (expected 'topk' or 'max')")
        self.last_field_verify = {"presence": score, "margin": None, "terms": None}
        return score

    @property
    def buffer_size(self) -> int:
        return self._history_buffer.fill

    @property
    def buffer_capacity(self) -> int:
        return self._history_buffer.cap
