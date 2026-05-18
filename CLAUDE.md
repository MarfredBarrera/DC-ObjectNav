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
│   │   ├── obj_detection.py   # OWLv2 object detection (used for goal grounding experiments)
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
└── SAM_models/                # Pretrained model weights
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

After every replan, `traj_log.jsonl` gets a line with `step`, `pos`, `heading`, `action`, `ref_traj`, `opt_traj`, and `mppi_cost` (= `-final_score`). `visualize.py` consumes this log + saved BEV maps to render `nav_history.mp4`.

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

**Goal selection** ([main.py:103](DCON/main.py#L103)): `mppi.get_goals_near_highest_sim` finds the highest-similarity peak and picks the closest *free* cell next to it (target objects are themselves marked occupied, so the peak is usually unreachable).

### Semantics (`src/mask_clip/`)

**MaskCLIP**: Dense semantic feature extraction.
- Loads OpenAI CLIP (ViT-B/16 by default).
- **Text path**: query string → 512-D text embedding.
- **Image path**: image → dense per-patch CLIP features (spatial 512-D map).
- Per-pixel similarity: `patch_features @ text_features.T`, normalized to [0, 1].
- **Frozen**: weights never update. The feature fields learn *where* in 3D to render these embeddings.

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

## Testing

MaskCLIP smoke tests:
```bash
cd DCON/src/mask_clip
python test_masking.py     # Dense feature extraction
python test_mask_class.py  # MaskCLIP class API
```
Both require sample images in `images/`; they generate heatmap + overlay visualizations.

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
- **NumPy, Matplotlib, PIL, imageio**: data processing and visualization
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