# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DC-ObjectNav** is a robotics navigation system for direction-cognizant object navigation. It combines:
- **3D Scene Understanding**: Learned feature fields with uncertainty quantification
- **Semantic Grounding**: Vision-language models (MaskCLIP) for target object recognition
- **Motion Planning**: A* graph-based and MPPI stochastic planning (with DIAL-MPC-style annealing)
- **Simulation Environment**: Habitat-sim integration for indoor scene simulation

The system processes RGB-D observations from a robot, builds world-space 3D maps online, and plans trajectories to reach a text-queried target object — all in a single live loop.

## Directory Structure

```
DCON/
├── src/
│   ├── config.py              # Global configuration (YAML-backed)
│   ├── gaussians.py           # Gaussian splatting utilities
│   ├── perception/            # Core perception pipeline
│   │   ├── perception_stack.py # Main perception interface (observe, train, update maps)
│   │   ├── featurefield.py    # Learned 3D feature field model (hash-grid + MLP)
│   │   ├── grid.py            # Uncertainty, Occupancy, and Similarity grids
│   │   ├── semantics.py       # MaskCLIP semantic feature extraction wrapper
│   │   ├── obj_detection.py   # YOLO-World / COCO-YOLO / Grounding DINO + SamRefinedDetector
│   │   ├── segmentation.py    # MobileSAM auto-mask-gen wrapper (whole-image segmentation)
│   │   └── utils.py           # Unprojection and coordinate utilities
│   ├── planning/              # Motion planning
│   │   ├── astar.py           # A* graph-based pathfinder
│   │   ├── mppi.py            # MPPI stochastic trajectory optimization (DIAL annealing)
│   │   ├── pathfinder.py      # Pathfinding helpers
│   │   └── utils.py           # Planning utilities
│   ├── mask_clip/             # MaskCLIP semantic segmentation
│   │   ├── MaskCLIP.py        # Main MaskCLIP interface
│   │   ├── model.py           # Dense feature extraction from CLIP
│   │   ├── clip.py            # CLIP model loading
│   │   └── test_*.py          # Test scripts (require sample images in images/)
│   ├── habitat/               # Habitat-sim integration
│   │   ├── habitat_utils.py   # Scene initialization, pathfinding
│   │   └── sim_interface.py   # Robot control interface
│   └── visualization/         # Plotting helpers (visualizer.py)
│
├── main.py                    # Primary entry: live perception + planning + execution loop
├── perception.py              # Perception-only training loop (no planning)
├── planner.py                 # Offline planner / map analysis class (reads pre-built maps)
├── offline.py                 # Offline ensemble training from saved trajectories
├── analysis.py                # Map analysis & figure generation
├── visualize.py               # Renders nav_history.mp4 from traj_log.jsonl + saved maps
├── 3dgs_trainer.py            # 3D Gaussian Splatting training
├── exploration_env.py         # Exploration policy definitions
├── run.sh                     # Wrapper script (taskset + GPU selection)
│
├── config/config.yaml         # Active config (YAML overrides Python defaults)
├── output/                    # Results (scenes, maps, models, traj_log.jsonl)
├── figs/                      # Generated figures and videos
├── gibson_scenes/             # Scene assets (.glb)
├── gsplat_viewers/            # Splatting viewer scripts
├── SAM_models/                # MobileSAM weights (mobile_sam.pt, ~40 MB)
└── test_object_detection.py   # Standalone detector + SAM smoke-test viz
```

## Core Architecture & Data Flow

### `main.py` — the live loop

The primary entry point. Three cadences in one process:

| Cadence | Step trigger | Cost |
|--------|--------------|------|
| **A. Train** | every step | one gradient step on each ensemble member |
| **B. Replan + 1 agent action** | every `REPLAN_INTERVAL` (=100) steps | MPPI rollout + Habitat step |
| **C. Refresh buffer + maps** | every `cfg.hash_buffer_refresh_interval` (=200) steps | observation, buffer insert, super-batch stage, BEV map recompute (slowest) |

**Bootstrap** (before the main loop):
1. **Spin**: rotate the agent through 36 × 10° = 360°, observing each frame.
2. **Cold-train**: `BOOTSTRAP_TRAIN_STEPS=2000` gradient steps so the first BEV maps have signal.
3. **First maps**: compute & save uncertainty / occupancy / similarity grids at step 0.

After every replan, `traj_log.jsonl` gets a line with `step`, `pos`, `heading`, `action`, `ref_traj`, `opt_traj`, `mppi_cost` (= `-final_score`), `det_conf`, `det_box`, `sam_box`, `sam_score`, `goal`, `mode`, and `w_conf`. When MobileSAM produces a fresh mask, a bit-packed `.npz` is also written to `output/<scene>/sam_masks/sam_mask_<step:06d>.npz`. `visualize.py` consumes the log, saved BEV maps, and saved SAM masks to render `nav_history.mp4`.

### Perception Pipeline (`src/perception/perception_stack.py`)

**Purpose**: Online feature learning and map building from streaming RGB-D observations.

**Data Flow per observation**:
1. `observe(sim_iface)` — Pull RGB-D frame + camera pose from Habitat.
2. `update_replay_buffer()` — Unproject to world-space points, extract CLIP features via MaskCLIP, sub-sample to `hash_per_frame_cache_size`, then fold *the previous frame* into the flat `_HistoryBuffer` and hold the new frame out as `_latest_frame`.
3. `update_occupancy()` — Voxelize depth into the occupancy grid (independent of the feature-field buffer).
4. `make_super_batch()` — Stage one big GPU tensor for training: ~20% drawn from `_latest_frame`, ~80% sampled uniformly from `_HistoryBuffer` via one `randint` + gather (no concat over many chunks).
5. `train_step(super_pts, super_feats)` — Sample a mini-batch from the staged super-batch and run one gradient step over every ensemble member.
6. `update_maps()` — Forward-pass the trained ensemble through every BEV voxel; save epistemic uncertainty, aleatoric uncertainty, occupancy, and target-similarity maps.

**Key concepts**:
- **Feature Fields**: Hash-grid + MLP networks mapping 3D position → CLIP feature vector. Multiple are trained in parallel as an ensemble; ensemble disagreement = epistemic uncertainty.
- **`_HistoryBuffer`**: Flat pre-allocated CPU ring of `(history_buffer_capacity, 3)` points and `(history_buffer_capacity, hash_feature_dim)` features. New points always enter; once full, each new point overwrites a uniformly random existing slot. Bounded memory (~412 MB CPU at 200k × 512-dim), continuous fresh-data dominance, O(batch) sampling with no `torch.cat` over many small chunks.
- **`_latest_frame`**: The most-recent frame is held out of the history buffer so it contributes a guaranteed 20% of every super-batch (recent-frame oversample). Keeps fresh observations weighted regardless of buffer state.
- **Query-Specific Maps**: Similarity grids are computed w.r.t. a text query (e.g., `cfg.target_query = "a pillow"`).

### Planning (`src/planning/mppi.py`)

**MPPI with DIAL-style annealing**. Receding-horizon stochastic optimizer.

- Unicycle dynamics (turn-first integration), control vector `[v_cells_per_step, w_rad_per_step]`.
- Cost terms (combined as a score; lower cost = higher score):
  - **Goal-distance**: mean Euclidean from rollout waypoints to the single goal cell.
  - **Collision**: subsampled along-segment occupancy check (sub-sample rate scales with max velocity to prevent diagonal tunneling).
  - **Unseen-traversal**: penalty for stepping through cells that haven't been verified free.
  - **Information gain**: FOV raycast against epistemic-uncertainty map (only for collision-free rollouts).
- **DIAL action-level annealing** ([mppi.py:200-207](DCON/src/planning/mppi.py#L200-L207)): noise variance is *smaller* for early horizon indices (which will actually be committed) and grows toward the tail.
- **DIAL trajectory-level annealing** ([mppi.py:216-222](DCON/src/planning/mppi.py#L216-L222)): noise variance shrinks across MPPI iterations — iter 0 is wide exploration, iter N-1 is local refinement.
- **Exploration→exploitation schedule**: `scheduled_params(progress)` lerps `lambda_weight`, `w_ig`, `w_goal`, `cov_scale` from `*_start` to `*_end` based on `progress = step / iterations`.
- **Warm-start**: previously committed control sequence is shifted left by one step at the start of each replan.
- **Sample 0 pinned to zero noise** so the unmutated warm-start is always evaluated.

**A\* (`src/planning/astar.py`)**: graph search over occupancy grid with Euclidean heuristic. Used for sanity-check / fallback paths; the primary planner is MPPI.

**Goal selection** ([main.py:plan_one_action](DCON/main.py)): three-layer priority for picking the cell MPPI aims at.

1. **Fresh SAM-refined detection** — if the detector fires this replan AND its score beats the cached max, run [`bev_cells_from_sam`](DCON/main.py): MobileSAM auto-mask-gen on the WHOLE image (no box prompt), score every mask by mean MaskCLIP similarity to `cfg.target_query`, pick the best mask (if it clears `sam_min_clip_sim`), unproject its pixels to BEV cells, then `argmax(bev_sim)` restricted to those cells. The mask is cached on `segmenter.last_mask` and the goal cell is cached as `last_box_goal`/`last_box_conf`.
2. **Cached box-goal** — no fresh detection this frame but a previous box-derived goal exists → reuse that fixed world cell so the agent commits to the strongest historical sighting instead of chasing a transient peak.
3. **Global similarity argmax** — neither: pick the global argmax of observed BEV similarity (exploration default).

Rationale for layer 1: at distance the detector's box is often offset by ~one box-width, so anchoring SAM on the box propagates that error. Whole-image auto-gen lets SAM find object boundaries from scratch and CLIP picks the right one. The detector is only a trigger.

**Goal weight (MPPI `w_conf`)** is independent of which layer fired: `EXPLOIT` (latched once `det_score >= detected_conf_threshold` for `detected_persistence` consecutive replans) pins `w_conf=1.0` so the agent commits hard; `SEARCH` passes raw `det_score` and MPPI's hysteresis (`exploit_conf = max(incoming, prev * conf_decay)`) ratchets up on sightings and decays after misses.

### Semantics (`src/mask_clip/`)

**MaskCLIP**: Dense semantic feature extraction.
- Loads OpenAI CLIP (ViT-B/16 by default).
- **Text path**: query string → 512-D text embedding.
- **Image path**: image → dense per-patch CLIP features (spatial 512-D map).
- Per-pixel similarity: `patch_features @ text_features.T`, normalized to [0, 1].
- **Frozen**: weights never update. The feature fields learn *where* in 3D to render these embeddings.

### Object Detection + Goal Refinement (`src/perception/obj_detection.py`, `segmentation.py`)

**Detectors** (all share `detect(image, query) → (score, box)`; pick via `cfg.detector`):
- `yolo` — YOLO-Worldv2 open-vocabulary (~15 ms/frame, default).
- `coco_yolo` — closed-set YOLOv8 over the 80 COCO classes (fastest, returns 0 for non-COCO queries).
- `hybrid` — COCO-matching queries → `coco_yolo`, everything else → `yolo` (mirrors the VLFM paper's hybrid scheme but swaps Grounding DINO for YOLO-World on the open-vocab branch).
- `grounding_dino` — open-vocab Grounding DINO Tiny (~200 ms/frame, better on natural-language phrases).
- `sam_refined` — composite: base detector triggers + MobileSAM whole-image auto-gen + MaskCLIP best-mask scoring. Returns the SAM mask's bbox + CLIP score. Mainly for the standalone smoke test; the live loop reaches for the same pipeline directly via `bev_cells_from_sam`.

**MobileSAM** ([segmentation.py](DCON/src/perception/segmentation.py)): thin wrapper around MobileSAM's `SamAutomaticMaskGenerator`. Returns one mask dict per discovered object — class-agnostic. Semantic identity comes from MaskCLIP scoring downstream. Loaded once at startup from `cfg.sam_checkpoint` (default `SAM_models/mobile_sam.pt`). Failed load is non-fatal: EXPLOIT falls back to the older box-based path with a warning.

**State side channel**: when `bev_cells_from_sam` picks a winning mask, it stashes the 2D bool mask, bbox, and CLIP score on `segmenter.last_mask / last_box / last_score`. The main loop reads these immediately after the plan call to persist the mask to `output/<scene>/sam_masks/sam_mask_<step:06d>.npz` (bit-packed via `np.packbits` for ~8× compression) and writes `sam_box` / `sam_score` into the traj log. `visualize.py` looks these up and overlays a green-tinted mask + bbox on the RGB panel.

### Grids (`src/perception/grid.py`)

Three complementary BEV representations:

| Grid Type | Purpose | Input | Output (per voxel) |
|-----------|---------|-------|-------------------|
| **UncertaintyGrid** | Epistemic + aleatoric | Ensemble forward-pass disagreement | 2 scalars |
| **OccupancyGrid** | Free / occupied / unseen | Depth voxelization (incremental) | trinary value |
| **SimilarityGrid** | Target relevance | Mean CLIP feature ⋅ text-query embedding | 1 scalar |

**Coordinates**: World space with `scene_bounds = ((xmin,ymin,zmin),(xmax,ymax,zmax))`; BEV cells index `(z, x)`; `y` is ignored for planning.

## Running the Code

### Main Entry Points

**1. Live perception + planning loop (primary)**
```bash
cd DCON
python main.py --gpu 0 --query "a pillow"
```
Runs bootstrap (spin + cold-train), then the three-cadence main loop. Outputs to `output/current_scene/`: RGB frames, BEV maps (`.npy`), `grid_extent.json`, `traj_log.jsonl`, and ensemble checkpoints.

**2. Perception-only training**
```bash
python perception.py --gpu 0 --query "a pillow"
```
Or with the wrapper:
```bash
./run.sh -g 0 -c 112-127 -q "a pillow"
```
Same perception pipeline, but the agent follows a fixed policy (default: spin) instead of MPPI plans. Useful for collecting training data without entangling planner behavior.

**3. Offline ensemble training**
```bash
python offline.py
```
Replays a saved RGB-D trajectory and trains the ensemble from disk. Uses `hash_replay_buffer_size` for its own offline sampling logic (separate from the live `_HistoryBuffer`).

**4. Map / trajectory analysis**
```bash
python analysis.py
python visualize.py --config config/config.yaml --output ./figs/nav_history.mp4 --fps 5
```
`analysis.py` plots uncertainty/occupancy/similarity maps. `visualize.py` reads `traj_log.jsonl` + saved BEV maps and renders the navigation video.

### Configuration

All hyperparameters live in [src/config.py](DCON/src/config.py). YAML in [config/config.yaml](DCON/config/config.yaml) overrides Python defaults.

Key settings:
- **Perception**: `target_query`, `ensemble_num_models`, `iterations`, `device`.
- **Habitat**: `scene_path`, `img_width`, `img_height`, `fov`, `sensor_height`.
- **MaskCLIP**: `maskclip_model_name` ("ViT-B/16"), `maskclip_input_size` (448).
- **Hash-grid training**: `hash_train_batch_size`, `hash_buffer_refresh_interval`, `hash_per_frame_cache_size`, `history_buffer_capacity`, `hash_feature_dim`.
- **MPPI**: `mppi_dt`, `mppi_max_v_mps`, `mppi_max_w_rps`, `mppi_num_iters`, `mppi_anneal_beta_action`, `mppi_anneal_beta_traj`, `mppi_*_start`/`mppi_*_end` for the schedule.
- **Detection**: `detector` (`yolo` / `coco_yolo` / `hybrid` / `grounding_dino` / `sam_refined`), `detected_conf_threshold`, `detected_persistence`, `stop_distance_m`.
- **MobileSAM**: `use_mobile_sam` (toggle the SAM goal-refinement path), `sam_checkpoint` (default `SAM_models/mobile_sam.pt`), `sam_points_per_side` (auto-gen density; 16 → ~0.5 s/call, 32 → ~2 s/call), `sam_pred_iou_thresh`, `sam_stability_score_thresh`, `sam_min_mask_region_area`, `sam_min_mask_pixels` (post-gen mask size floor), `sam_min_clip_sim` (CLIP-score floor for accepting a mask), `sam_lookback_steps` (viz-only: how far back to grab the most recent saved SAM mask when overlaying on a viz frame).

## Testing

MaskCLIP smoke tests:
```bash
cd DCON/src/mask_clip
python test_masking.py     # Dense feature extraction
python test_mask_class.py  # MaskCLIP class API
```
Both require sample images in `images/`; they generate heatmap + overlay visualizations.

Detector + SAM standalone smoke test:
```bash
cd DCON
# Bare detector: red bbox on the input image
python -m src.perception.obj_detection yolo "a pillow" output/current_scene/rgbs/rgb_000.png
# Full SAM-refined pipeline: red box = base detector, green mask + box = SAM-refined.
python -m src.perception.obj_detection sam_refined "a pillow" output/current_scene/rgbs/rgb_000.png
```
Writes to `figs/det_<backend>_<query>.png`. The `sam_refined` overlay shows whether SAM correctly picked the right mask vs. the detector's (often offset) box.

## Common Development Tasks

### Adding a New Planner
1. Create `src/planning/my_planner.py` mirroring `MPPIPlanner`'s `plan` / `optimize_trajectory` signature.
2. Wire into [main.py:plan_one_action](DCON/main.py#L73) alongside the existing MPPI call.
3. Add a config flag or argparse switch to choose between planners at runtime.

### Tweaking Perception
- **Loss weights**: [src/config.py](DCON/src/config.py) (`ssim_weight`, `l1_weight`).
- **Ensemble size**: `cfg.ensemble_num_models`.
- **Buffer size / freshness**: `history_buffer_capacity` (larger = more history, slower drift; smaller = faster turnover, more recency bias).
- **Uncertainty computation**: `src/perception/grid.py` (`UncertaintyGrid` methods).

### Tweaking the planner
- **Cost weights**: `mppi_w_goal_*`, `mppi_w_ig_*` (schedule endpoints) and `w_occ`, `w_unseen` (constants).
- **Annealing aggressiveness**: `mppi_anneal_beta_action`, `mppi_anneal_beta_traj` — smaller β = sharper annealing.
- **Sampling envelope**: `mppi_cov_scale_start` / `_end`.

### Debugging Semantic Grounding
- Visualize MaskCLIP heatmaps: `src/mask_clip/test_mask_class.py`.
- Check similarity BEV map: `analysis.py` plots `bev_similarity_*.npy`.
- Tweak query: `cfg.target_query` (favors "a [object]" phrasing for better CLIP grounding).

### Adding Visualization
- `src/visualization/visualizer.py` (matplotlib-based).
- Extend `analysis.py` for new plot types.
- Saved maps are `.npy`; load with `np.load()` + `plt.imshow()`.

`render_combined_grid` layout (2×3) used by `nav_history.mp4`:
| | col 0 | col 1 | col 2 |
|---|---|---|---|
| **row 0** | Agent View (RGB w/ detector + SAM overlays) | _(empty)_ | BEV Similarity |
| **row 1** | Epistemic Uncertainty | Occupancy | Uncertainty + Occupancy overlay |

The RGB panel composites in order: detector bbox (red) → SAM mask tint + bbox (green) → score label. SAM overlays come from the most recent saved `sam_masks/sam_mask_<step>.npz` within `cfg.sam_lookback_steps` of the current viz frame.

## Key Invariants & Patterns

1. **Coordinate Systems**: World space (Habitat / simulator) vs. BEV voxel indices (in grids); unprojection maps camera→world via depth + camera pose.
2. **Ensemble as Uncertainty**: Don't mock disagreement — train separate models with independent inits.
3. **Frozen Semantics**: MaskCLIP weights are static; the feature fields learn *where* in 3D to render those embeddings.
4. **Streaming Data**: Perception is online. The `_HistoryBuffer` keeps a bounded-memory snapshot of past observations; the `_latest_frame` oversample guarantees fresh data participates in every gradient step.
5. **BEV Representation**: Plans operate in top-down (z, x) space; y is ignored for planning.
6. **Receding Horizon**: MPPI computes an H-step plan, but only the first action of `U_opt` is executed before the next replan. `last_U` carries the rest forward as a warm-start.

## Dependencies & Environment

- **PyTorch** (CUDA 11.8+)
- **Habitat-sim**: physics simulator, scene loading, agent control
- **CLIP** (OpenAI): frozen vision-language model
- **MobileSAM**: `pip install git+https://github.com/ChaoningZhang/MobileSAM.git`, weights at `SAM_models/mobile_sam.pt` (~40 MB)
- **Ultralytics** (`pip install ultralytics`): YOLO-World / YOLOv8 detectors
- **NumPy, Matplotlib, PIL, imageio, OpenCV**: data processing and visualization
- See `requirements.txt` for pinned versions

GPU memory: ~12 GB recommended for training (multi-model ensemble + super-batch staging).
CPU memory: ~500 MB for the `_HistoryBuffer` at default capacity (200k × 512-dim features).

## Debugging Tips

- **Out-of-memory (GPU)**: reduce `ensemble_num_models`, `hash_train_batch_size`, or `hash_buffer_refresh_interval`.
- **Out-of-memory (CPU)**: reduce `history_buffer_capacity` (linear in feature-dim × capacity).
- **NaN losses**: check input normalization (esp. CLIP embeddings); verify depth values are in `[min_sensor_dist, max_sensor_dist]`.
- **Planner picks bad goals**: inspect `bev_sim` peak via `analysis.py`; tweak `cfg.target_query` phrasing.
- **MPPI "NO SAFE ROLLOUTS"**: the agent is wedged. Check occupancy dilation, reduce `w_unseen`, or widen `mppi_cov_scale_*`.
- **Trajectory video looks wrong**: check `grid_extent.json` matches the scene bounds and that `traj_log.jsonl` was written cleanly (one JSON per line).
- **`[init] MobileSAM unavailable`** at startup: weights missing or `mobile_sam` package not installed. Install via `pip install git+https://github.com/ChaoningZhang/MobileSAM.git` and place weights at `cfg.sam_checkpoint`. The run continues without SAM (EXPLOIT falls back to the box-based path).
- **SAM never produces a goal** (no `sam_mask_*.npz` ever written): lower `sam_min_clip_sim` (default 0.18) — the CLIP scores on the chosen mask aren't clearing the floor. Use the `sam_refined` standalone test to inspect actual scores per frame.
- **SAM picks the wrong mask** (e.g. couch when target is pillow): raise `sam_min_clip_sim`, or bump `sam_points_per_side` from 16 → 24/32 so SAM proposes finer-grained candidates.
- **SAM is too slow**: `sam_points_per_side=16` is ~0.5 s/call; drop to 12 for a small cost in mask coverage. SAM only runs when the detector fires AND beats the cached score, so total per-run overhead is bounded.