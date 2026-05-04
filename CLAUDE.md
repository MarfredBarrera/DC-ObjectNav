# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DC-ObjectNav** is a robotics navigation system for direction-cognizant object navigation. It combines:
- **3D Scene Understanding**: Learned feature fields with uncertainty quantification
- **Semantic Grounding**: Vision-language models (MaskCLIP) for target object recognition
- **Motion Planning**: A* graph-based and MPPI stochastic planning algorithms
- **Simulation Environment**: Habitat-sim integration for indoor scene simulation

The system processes RGB-D observations from a robot, builds world-space 3D maps, and plans trajectories to reach target objects.

## Directory Structure

```
DCON/
├── src/
│   ├── config.py              # Global configuration (YAML-backed)
│   ├── perception/            # Core perception pipeline
│   │   ├── perception_stack.py # Main perception interface (observe, train, update maps)
│   │   ├── featurefield.py    # Learned 3D feature field model
│   │   ├── grid.py            # Uncertainty, Occupancy, and Similarity grids
│   │   ├── semantics.py       # MaskCLIP semantic feature extraction
│   │   └── utils.py           # Unprojection and coordinate utilities
│   ├── planning/              # Motion planning
│   │   ├── astar.py           # A* graph-based pathfinder
│   │   ├── mppi.py            # MPPI stochastic trajectory optimization
│   │   └── utils.py           # Planning utilities
│   ├── mask_clip/             # MaskCLIP semantic segmentation
│   │   ├── MaskCLIP.py        # Main MaskCLIP interface
│   │   ├── model.py           # Dense feature extraction from CLIP
│   │   ├── clip.py            # CLIP model loading
│   │   └── test_*.py          # Test scripts
│   ├── habitat/               # Habitat-sim integration
│   │   ├── habitat_utils.py   # Scene initialization, pathfinding
│   │   └── sim_interface.py   # Robot control interface
│   ├── visualization/         # Plotting and debugging
│   └── gaussians.py           # Gaussian splatting utilities
│
├── perception.py              # Main perception loop (train & map building)
├── planner.py                 # Planning interface (reads pre-built maps)
├── offline.py                 # Offline processing script
├── analysis.py                # Analysis and visualization scripts
├── 3dgs_trainer.py            # 3D Gaussian Splatting training
├── exploration_env.py         # Exploration policy definitions
├── run.sh                      # Wrapper script (taskset + GPU selection)
│
├── config/                    # YAML config files
├── output/                    # Results (scenes, maps, models)
├── gibson_scenes/             # Scene assets
└── SAM_models/                # Pretrained model weights

```

## Core Architecture & Data Flow

### Perception Pipeline (`src/perception/perception_stack.py`)

**Purpose**: Online feature learning and map building from robot observations.

**Data Flow**:
1. `observe(sim_iface)` — Extract RGB-D frame, compute world-space points + CLIP text features
2. `update_replay_buffer()` — Cache observation on CPU (FIFO, configurable size)
3. `update_occupancy()` — Voxelization from depth + camera pose
4. `train_step()` — One gradient descent update over ensemble of feature fields
5. `update_maps()` — Compute and save BEV (bird's-eye-view) maps
   - Epistemic uncertainty (model disagreement across ensemble)
   - Aleatoric uncertainty (within-model variance)
   - Semantic similarity to target query

**Key Concepts**:
- **Ensemble**: Multiple independent feature field networks (default 4) trained on same data; disagreement signals uncertainty
- **Feature Fields**: Neural radiance-like models mapping 3D position + viewing direction → CLIP feature vector
- **Replay Buffer**: CPU-side sliding window of observations for stable training on streaming data
- **Query-Specific Maps**: Similarity grids computed w.r.t. a text query (e.g., "green plant")

### Planning Interface (`planner.py`)

**Purpose**: Read pre-built maps and compute safe trajectories to target.

**Key Planners**:
- **A\* (`src/planning/astar.py`)**: Graph-based shortest path over occupancy grid, uses Euclidean distance heuristic
- **MPPI (`src/planning/mppi.py`)**: Stochastic trajectory optimization; samples & weights trajectories based on cost (occupancy + semantic uncertainty)

**Usage Pattern**:
```python
planner = Planner(cfg, sim_iface, scene_bounds)
planner.load_umap()          # Load epistemic & aleatoric uncertainty maps
planner.load_sim_map()       # Load semantic similarity map
goal = planner.astar_planner.plan(start, goal_pos)  # or planner.mppi_planner.plan(...)
```

### Semantics (`src/mask_clip/`)

**MaskCLIP**: Dense semantic feature extraction.
- Loads OpenAI CLIP (ViT-B/16 by default) in two paths:
  1. **Text path**: Encode query string → text embedding (global, 512D)
  2. **Image path**: Encode image patch features → dense feature map (spatial, 512D)
- Computes per-pixel similarity: `patch_features @ text_features.T`
- Output: Heatmap (H × W) normalized to [0, 1]

**Note**: MaskCLIP is frozen (no training); feature fields learn *how to render CLIP embeddings* at 3D locations.

### Grids (`src/perception/grid.py`)

Three complementary BEV representations:

| Grid Type | Purpose | Input | Output (per voxel) |
|-----------|---------|-------|-------------------|
| **UncertaintyGrid** | Epistemic + aleatoric | Feature field ensemble disagreement | 2 scalars (epistemic, aleatoric) |
| **OccupancyGrid** | Free vs. occupied space | Depth-based voxelization | Binary occupancy |
| **SimilarityGrid** | Target relevance | Mean CLIP feature over ensemble | 1 scalar (cosine sim to query) |

**Coordinates**: World space with scene bounds; voxels correspond to BEV (top-down) cells.

## Running the Code

### Main Entry Points

**1. Perception + Training Loop**
```bash
cd DCON
python perception.py --gpu 0 --query "a pillow"
```
Or using the wrapper script:
```bash
./run.sh -g 0 -c 112-127 -q "a pillow"
```
- Runs agent in simulator, trains feature fields, saves BEV maps every N iterations
- Output: `output/current_scene/` (RGB frames, uncertainty maps, similarity maps)

**2. Planning (Read Pre-Built Maps)**
```bash
python planner.py
```
- Loads uncertainty & similarity maps from disk
- Executes A* or MPPI planning
- (Exact interface depends on main block; check file)

**3. Offline Analysis**
```bash
python offline.py
```
- Post-processes saved data (depth, RGB, camera poses)
- Reconstructs 3D maps offline for visualization/analysis

**4. Analysis & Visualization**
```bash
python analysis.py
```
- Plots uncertainty, occupancy, similarity maps
- Generates figures for papers/reports

### Configuration

All hyperparameters live in `src/config.py`:
```python
cfg = Config(yaml_path="path/to/config.yaml")
```

Key settings:
- **Perception**: `target_query`, `ensemble_num_models`, `iterations`, `device`
- **Habitat**: `scene_path`, `img_width`, `img_height`, `fov`, `sensor_height`
- **MaskCLIP**: `maskclip_model_name` (e.g., "ViT-B/16"), `maskclip_input_size` (448)
- **Training**: `ssim_weight`, `l1_weight`, learning rate schedule

YAML files in `config/` override defaults; Python defaults are fallback.

## Testing

Test scripts for MaskCLIP semantic extraction:
```bash
cd DCON/src/mask_clip
python test_masking.py          # Test dense feature extraction
python test_mask_class.py       # Test MaskCLIP class API
```
Both require sample images in `images/` folder; generate heatmap + overlay visualization.

## Common Development Tasks

### Adding a New Planner
1. Create `src/planning/my_planner.py` implementing planner interface
2. In `planner.py`, instantiate and wire into `Planner` class
3. Test with `python planner.py` or in analysis scripts

### Tweaking Perception
- **Loss weights**: `src/config.py` (e.g., `ssim_weight`, `l1_weight`)
- **Ensemble size**: `cfg.ensemble_num_models`
- **Uncertainty computation**: `src/perception/grid.py` (UncertaintyGrid methods)

### Debugging Semantic Grounding
- Visualize MaskCLIP heatmaps: `src/mask_clip/test_mask_class.py`
- Check similarity map: `analysis.py` plots `bev_similarity_*.npy`
- Tweak query: `cfg.target_query` in perception.py

### Adding Visualization
- Use `src/visualization/visualizer.py` (matplotlib-based)
- Extend analysis.py for new plot types
- Saved maps are `.npy` files; load and plot with `np.load()` + `plt.imshow()`

## Key Invariants & Patterns

1. **Coordinate Systems**: World space (from simulator) vs. voxel indices (in grids); unprojection maps camera→world via depth + pose
2. **Ensemble as Uncertainty**: Don't mock disagreement; train separate models
3. **Frozen Semantics**: MaskCLIP weights are static; we learn *where* in 3D to render those embeddings
4. **Streaming Data**: Perception is online; replay buffer prevents overfitting to latest frames
5. **BEV Representation**: Plans operate in top-down space; grid cell = (x, z) world coordinates, ignore y (height)

## Dependencies & Environment

- **PyTorch** (CUDA 11.8+)
- **Habitat-sim**: Physics simulator, scene loading, agent control
- **CLIP** (OpenAI): Frozen vision-language model
- **NumPy, Matplotlib, PIL**: Data processing and visualization
- See `requirements.txt` for pinned versions

GPU memory: ~12GB recommended for training (4-model ensemble + batch sampling).

## Debugging Tips

- **Out-of-memory during training**: Reduce `ensemble_num_models`, `data_queue_size`, or batch size
- **NaN losses**: Check input normalization (esp. CLIP embeddings); verify depth values are valid
- **Slow planning**: Reduce BEV grid resolution or enable path simplification in A*/MPPI
- **Poor semantic maps**: Verify `target_query` is descriptive; try "a [object]" format for better CLIP grounding
