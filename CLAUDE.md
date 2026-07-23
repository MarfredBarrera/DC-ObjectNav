# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DC-ObjectNav** is a robotics navigation system for direction-cognizant object navigation. It combines:
- **3D Scene Understanding**: Learned feature fields with uncertainty quantification
- **Semantic Grounding**: a pairwise CLIPSeg relevance field (query + distractor channels) + LLMDet (attention-sink-gated) for target detection, with field-verified box acceptance
- **Motion Planning**: MPPI stochastic planning (with DIAL-MPC-style annealing) for SEARCH; the pretrained DD-PPO PointNav policy for the EXPLOIT final approach
- **Simulation Environment**: Habitat-sim integration for indoor scene simulation

The system processes RGB-D observations from a robot, builds world-space 3D maps online, and plans trajectories to reach a text-queried target object — all in a single live loop.

## Directory Structure

```
DCON/
├── src/
│   ├── config.py              # Global configuration (YAML-backed)
│   ├── perception/            # Core perception pipeline
│   │   ├── perception_stack.py # Main perception interface (observe, train, update maps)
│   │   ├── featurefield.py    # EvidentialFeatureField: 3D feature field (hash-grid + MLP, NIG uncertainty)
│   │   ├── grid.py            # Uncertainty, Occupancy, and Similarity grids
│   │   ├── semantics.py       # CLIPSegSemantics: per-pixel relevance channels (query + distractors)
│   │   ├── distractor_gen.py  # Distractor vocabulary (background terms + object confusers)
│   │   ├── obj_detection.py   # LLMDet detector (MM-Grounding-DINO + attention sinks)
│   │   ├── detection.py       # DetectionGate: box→3D, field-verify, classify, EXPLOIT latch
│   │   └── utils.py           # Unprojection and coordinate utilities
│   ├── planning/              # Motion planning
│   │   ├── mppi.py            # MPPI stochastic trajectory optimization (DIAL annealing) — SEARCH
│   │   ├── search.py          # plan_search_action: goal selection + MPPI replan — SEARCH
│   │   ├── exploit.py         # ExploitController: DD-PPO approach + arrival check — EXPLOIT
│   │   ├── ddppo_policy.py    # Vendored DD-PPO PointNav policy + DDPPONavigator
│   │   ├── tracking.py        # Pure-pursuit path → one Habitat primitive (SEARCH discrete mode)
│   │   └── utils.py           # FOV-raycast information-gain, goal_distance_field, reachable_min
│   ├── episode/               # Per-episode plumbing for main.run()
│   │   ├── recorder.py        # EpisodeRecorder: everything written to cfg.output_dir
│   │   ├── scoring.py         # Geodesic success + SPL (nearest_goal_point, score_episode)
│   │   └── control.py         # EarlyStop: Ctrl-C / STOP-sentinel graceful termination
│   ├── habitat/               # Habitat-sim integration
│   │   ├── habitat_utils.py   # Scene init, spawn (start_episode), pathfinding, geodesic_distance
│   │   └── sim_interface.py   # Robot control interface (step, step_discrete, agent_heading)
│   └── visualization/         # Plotting helpers (visualizer.py)
│
├── main.py                    # Primary entry: the three-cadence live loop (run())
├── benchmarks/
│   ├── eval_scene.py          # Per-scene sweep CLI: run / review / report stages (thin over eval_core)
│   ├── eval_core.py           # Eval logic (torch-free): scenarios, verdicts, aggregation, BEV evidence
│   ├── make_eval_subset.py    # Materializes a standardized ObjectNav-val subset for cross-system eval
│   ├── gibson_scenes/         # Gibson scene assets (.glb) — git-tracked
│   ├── episodes/              # Gibson-val + HM3D-OVON episode datasets (gitignored)
│   └── scene_datasets/        # HM3D scene assets for OVON (gitignored; download separately)
├── tools/
│   ├── visualize.py           # Renders nav_history.mp4 from traj_log.jsonl + saved maps
│   ├── analyze_runs.py        # Eval analysis: progress, common-set SR/SPL comparison, failure attribution
│   └── exploration_env.py     # Exploration policy definitions
│
├── ddppo_weights/             # DD-PPO PointNav checkpoint (gitignored; cfg.ddppo_checkpoint_path)
├── config/config.yaml         # Active config (YAML overrides Python defaults; carries the canonical detection arm)
├── config/agent_configs/                       # Config overlays — see evaluation_configs/README.md
│   ├── agent_ovon_stretch.yaml                 # OVON Stretch embodiment profile (eval_scene.py --agent_config)
│   └── detector_pairwise_field_maxj.yaml       # THE canonical detection arm (τ0.47 LLMDet + pairwise field margin ≥ 0.0)
├── config/evaluation_configs/                  # Eval configs — see its README.md
│   ├── benchmarks.yaml                         # Named episode sources (eval_scene.py --benchmark: gibson, ovon, ... — dataset/scenarios + embodiment)
│   ├── experiments/*.yaml                      # Optional presets (eval_scene.py --experiment: non-detector overlays, action mode, caps, scoring, out)
│   └── scenarios_*.yaml                        # Per-scene sweep specs for benchmarks/eval_scene.py (targets × starts, rect goals)
├── output/                    # Results (scenes, maps, models, traj_log.jsonl; eval runs/, verdicts.yaml)
└── figs/                      # Generated figures and videos
```

## Core Architecture & Data Flow

### `main.py` — the live loop

The primary entry point, and only the loop driver: it owns the three cadences,
the mode switch, and the per-replan log line. Each stage it calls lives in its
own module — detection + latch in [`src/perception/detection.py`](DCON/src/perception/detection.py),
SEARCH in [`src/planning/search.py`](DCON/src/planning/search.py), EXPLOIT in
[`src/planning/exploit.py`](DCON/src/planning/exploit.py), evidence/scoring/early-stop
in [`src/episode/`](DCON/src/episode/), scene setup in `habitat_utils.start_episode`.

Three cadences in one process:

| Cadence | Step trigger | Cost |
|--------|--------------|------|
| **A. Train** | every step | one gradient step on the evidential feature field |
| **B. Replan + 1 agent action** | every `REPLAN_INTERVAL` (=100) steps | MPPI rollout + Habitat step |
| **C. Refresh buffer + maps** | every `cfg.hash_buffer_refresh_interval` (=200) steps | observation, buffer insert, super-batch stage, BEV map recompute (slowest) |

**Startup** (before the main loop): a single observation seeds the replay buffer + occupancy, then the first BEV maps are built directly from the (untrained) feature field — no spin, no cold-train. The maps start as field noise / mostly-unseen and fill in online as the loop trains and cadence C recomputes them.

After every replan, `traj_log.jsonl` gets a line with `step`, `pos`, `heading`, `action`, `opt_traj`, `det_conf`, `det_box`, `goal`, `mode`, and `w_conf`. `tools/visualize.py` consumes the log and saved BEV maps to render `nav_history.mp4`.

### Perception Pipeline (`src/perception/perception_stack.py`)

**Purpose**: Online feature learning and map building from streaming RGB-D observations.

**Data Flow per observation**:
1. `observe(sim_iface)` — Pull RGB-D frame + camera pose from Habitat.
2. `update_replay_buffer()` — Unproject to world-space points, extract per-pixel CLIPSeg relevance channels (query + distractors), sub-sample to `hash_per_frame_cache_size`, then fold *the previous frame* into the flat `_HistoryBuffer` and hold the new frame out as `_latest_frame`.
3. `update_occupancy()` — Voxelize depth into the occupancy grid (independent of the feature-field buffer).
4. `make_super_batch()` — Stage one big GPU tensor for training: ~20% drawn from `_latest_frame`, ~80% sampled uniformly from `_HistoryBuffer` via one `randint` + gather (no concat over many chunks).
5. `train_step(super_pts, super_feats)` — Sample a mini-batch from the staged super-batch and run one gradient step on the feature field.
6. `update_maps()` — Forward-pass the trained field through every BEV voxel; save epistemic uncertainty, aleatoric uncertainty, occupancy, and target-similarity maps.

**Key concepts**:
- **Feature Field** (`EvidentialFeatureField`): a single hash-grid + MLP mapping 3D position → the CLIPSeg relevance vector (one sigmoid channel per prompt: query + distractors; `hash_feature_dim` is overridden to 1+K at PerceptionStack init), trained with an evidential (Normal-Inverse-Gamma) head. The NIG marginal yields **both** aleatoric and epistemic uncertainty in one forward pass — no ensemble (the previous multi-model ensemble, whose only role was empirical epistemic from prediction variance, was replaced by this).
- **`_HistoryBuffer`**: Flat pre-allocated CPU ring of `(history_buffer_capacity, 3)` points and `(history_buffer_capacity, hash_feature_dim)` features. New points always enter; once full, each new point overwrites a uniformly random existing slot. Bounded memory (~412 MB CPU at 200k × 512-dim), continuous fresh-data dominance, O(batch) sampling with no `torch.cat` over many small chunks.
- **`_latest_frame`**: The most-recent frame is held out of the history buffer so it contributes a guaranteed 20% of every super-batch (recent-frame oversample). Keeps fresh observations weighted regardless of buffer state.
- **Query-Specific Maps**: Similarity grids are computed w.r.t. a text query (e.g., `cfg.target_query = "a pillow"`).

### Planning: SEARCH (`src/planning/mppi.py`) and EXPLOIT (`src/planning/ddppo_policy.py`)

**Control is split by mode.** SEARCH (pre-latch) runs the MPPI planner below. Once a detection **latches**, EXPLOIT hands locomotion to the pretrained **DD-PPO PointNav policy** (Wijmans et al. 2020; depth-only SE-ResNeXt101 + 2-layer LSTM, vendored dependency-free in `ddppo_policy.py` to bit-exactly match the `gibson-2plus-se-resneXt101-lstm1024` checkpoint): each replan, `DDPPONavigator.act(depth, pos, rotation, goal_world)` emits ONE native Habitat primitive, executed via `SimInterface.step_discrete` with the checkpoint's own training magnitudes (`DDPPO_FORWARD_M`=0.25 m / `DDPPO_TURN_DEG`=10° — deliberately NOT `cfg.discrete_*`). The goal is the raw cached detection-box projection (no free-space snapping — an obstacle-embedded goal is fine for a depth policy; a snapped goal drifting with the evolving occupancy map was a confirmed failure). The sole stop signal is the Euclidean `cfg.stop_distance_m` check against that goal; DD-PPO's own trained STOP is disregarded (trained against ~0.2 m exact PointGoals — never fires reliably on detection-projected goals). Action selection **samples** (never argmax — greedy locks into a turn_left/turn_right 2-cycle via the prev-action embedding). Depth input: resized 512→256 with `nearest-exact` (bilinear fabricates phantom edge depths), and sensor-miss `0.0` pixels are remapped to far/clear (a mesh-hole reading as a point-blank wall was the root cause of persistent turn-only stalls). The LSTM state resets once per latch (`ddppo_nav.reset()` when `detected` first flips).

**MPPI with DIAL-style annealing** (SEARCH). Receding-horizon stochastic optimizer. (Its EXPLOIT-specific cost behaviors below — arrival freeze, unseen-cell penalty — are retained in `mppi.py` but inactive in the live loop now that latching hands control to DD-PPO.)

- Unicycle dynamics (turn-first integration), control vector `[v_cells_per_step, w_rad_per_step]`.
- Cost terms (combined as a score; lower cost = higher score):
  - **Goal-distance**: mean squared *obstacle-aware* distance-to-go from rollout waypoints to the goal, scaled by the confidence-derived goal weight `w_conf`. The distance comes from a per-replan Dijkstra wavefront over the BEV seeded at the goal cell ([`goal_distance_field`](DCON/src/planning/utils.py)) — occupied cells are traversable at `mppi_occupied_cell_cost` per cell (so a goal projected onto the object surface still seeds a finite field, while crossing a wall costs ~thickness × that, losing to any indoor detour), and the field is anchored at the best cell in the agent's reachable component ([`reachable_min`](DCON/src/planning/utils.py): min over the **observed-FREE** cells of the start cell's 8-connected non-occupied component — NOT a global min, which an enclosed free pocket under/behind the goal furniture would hijack, and NOT over unseen members either, which for a goal on an outer wall (window) would put the anchor on the unseen exterior just past the wall; either hijack inflates every reachable cell past the arrival radius so the agent orbits forever and can never satisfy the stop check) so values ≈ free-space cells-to-go. In EXPLOIT, unseen cells additionally cost `mppi_unseen_cell_cost` (=3) each, so the committed approach prefers observed-free routes over optimistic shortcuts through unexplored space that may hide a wall and force an SPL-burning backtrack (SEARCH keeps unseen at cost 1 — exploration should enter unseen space). This kills the Euclidean corner local-minimum next to a wall with the goal on the far side: the gradient routes around geometry, the arrival test (`field ≤ arrival_radius`) can't leak across walls, and the caller's stop check reads the same field via `mppi.last_goal_dist_m` so TARGET REACHED can't fire through a wall either.
  - **Collision (hard exclusion)**: waypoint occupancy check plus `mppi_collision_substeps` interpolated checks per segment (prevents tunneling through thin walls). Only the start cell is forgiven. In EXPLOIT a rollout that arrives (within the goal-arrival radius) is *frozen* at its first arrival waypoint (position pinned, controls zeroed) with an arrival bonus — so "reach the target, then sit" wins and doubles as the stopping behavior; the freeze is applied *before* the collision check, so collisions are judged on the executed trajectory (no phantom hits from the never-executed post-arrival tail) and the frozen cell itself must be free — there is no arrival-region collision forgiveness. Colliding rollouts never win the argmax; when NO rollout is safe (wedged), the softmax update instead weights by survival time (steps before first collision, goal proximity as tiebreak) so the nominal steers out of the pocket. If no safe rollout is found, `best_U=None` and the caller idles this replan.
  - **Information gain**: FOV raycast (`compute_batch_fov_ig`) against the **epistemic-uncertainty** BEV map (masked field uncertainty); skipped when the IG weight is ~0 (EXPLOIT).
- **DIAL action-level annealing**: noise variance is *smaller* for early horizon indices (which will actually be committed) and grows toward the tail.
- **DIAL trajectory-level annealing**: noise variance shrinks across MPPI iterations — iter 0 is wide exploration, iter N-1 is local refinement.
- **Warm-start**: previously committed control sequence is shifted left by one step at the start of each replan.
- **Sample 0 pinned to zero noise** so the unmutated warm-start is always evaluated.

**Continuous→discrete action mode** (`cfg.discrete_actions`, default off; governs SEARCH only — EXPLOIT is always discrete-stepped by DD-PPO regardless): MPPI stays continuous, but a low-level **tracking controller** ([`discrete_action_from_plan`](DCON/src/planning/tracking.py)) converts each replan's optimized path into ONE Habitat ObjectNav primitive — MOVE_FORWARD (`discrete_forward_m`=0.25 m), TURN_LEFT/RIGHT (`discrete_turn_deg`=30°, the challenge convention), or STOP — executed via [`SimInterface.step_discrete`](DCON/src/habitat/sim_interface.py). Pure-pursuit: it takes the bearing to the first waypoint ≥ `discrete_lookahead_m` ahead and turns toward it when the heading error exceeds half a turn, else steps forward; receding-horizon replans correct drift. `step_discrete` takes NATIVE Habitat turn semantics unconditionally; the tracking controller maps its grid-θ turn direction onto Habitat yaw via `mppi_w_sign` at the source (applying a sign inside step_discrete inverted DD-PPO's native turns — a confirmed spin-in-place bug). Each primitive counts against `max_agent_steps` (=500) — exhausting it without self-stopping is a timeout/failure. This makes SR/SPL directly comparable to VLFM / Goal-Oriented Semantic Exploration in the discrete ObjectNav challenge. `traj_log.jsonl` logs the primitive name as `action` in this mode (nothing downstream consumes it; `tools/visualize.py` reads only `heading`).

**Goal selection** ([`plan_search_action`](DCON/src/planning/search.py)): each replan a detection is first **classified** by distance + size, then the goal cell is chosen by a three-layer priority.

**Detection classification** ([`classify_detection`](DCON/src/perception/detection.py)) sorts a detection into three bands using the agent→object distance (box-center depth) and the box's image-area fraction, returning two flags `(is_persistent, contributes_confidence)`:
- **too close** — `dist < detected_min_dist_m` OR `box_frac > detected_max_box_frac` → ignored entirely (no goal, no confidence weight, no latch); the box fills the frame and carries no usable localization.
- **too far** — `dist > detected_max_dist_m` OR `box_frac < detected_min_box_frac` → *investigate*: sets/caches the goal and pulls the confidence weight, but is NOT persistent (won't latch).
- **usable band** — anything else → *persistent*: investigate + contribute confidence AND count toward the latch streak.

So the agent steers toward a distant sighting and only commits to it once it has closed into the usable band. A missing box or unrangeable depth → ignored.

**Goal-cell priority:**
1. **Fresh detection goal** — if this replan's detection is worth investigating (`det_investigate`, i.e. not *too close*), project its box into a goal cell via [`bev_cell_from_box_center`](DCON/src/perception/detection.py): unproject a small patch around the box-center pixel to one world point and use that BEV cell **verbatim** (no argmax, no snapping); relies on LLMDet's tight boxes. A goal cell on the object surface is fine: in EXPLOIT, MPPI freezes each rollout at its first waypoint within the goal-arrival radius and excludes it if that cell is occupied, so winning rollouts stop on free cells near the surface rather than entering it. In SEARCH mode every investigated detection re-projects, so the cache (`last_box_goal`) tracks the **most recent** bounding box.
2. **Cached box-goal** — no fresh investigated detection this frame but a cached box-goal exists → reuse that fixed world cell. In EXPLOIT the detector is throttled off, so the cache freezes on the object that triggered the latch and the agent commits to it.
3. **Global similarity argmax** — neither: pick the global argmax of observed BEV similarity (exploration default).

**Detector throttle in EXPLOIT** ([`DetectionGate.should_run_detector`](DCON/src/perception/detection.py)): once latched the goal is pinned to the cached cell, so the detector runs only every `exploit_redetect_interval` replans (`<=0` → never re-detect after latching, reuse the cache for the rest of the run). SEARCH always detects every replan.

**Latching / goal weight (MPPI `w_conf`)**: `detected` latches once `detected_persistence` consecutive *persistent* (usable-band) detections accrue — there is no separate latch score threshold; the detector's own floor (`llmdet_threshold`) bounds every surviving box's score — a non-persistent frame resets the streak; never unlatches. `EXPLOIT` pins `w_conf=1.0` so the agent commits hard; `SEARCH` passes the per-frame detection score (zeroed for *too-close* detections so they don't pull the goal weight) and MPPI's hysteresis (`exploit_conf = max(incoming, prev * conf_decay)`) ratchets up on sightings and decays after misses.

### Semantics (`src/perception/semantics.py` + `distractor_gen.py`)

**CLIPSegSemantics**: dense per-pixel relevance for a FIXED text query (frozen CIDAS/clipseg-rd64-refined; the query is fixed at construction — `cfg.target_query` never changes mid-run). In **pairwise mode** (`cfg.clipseg_pairwise`, the canonical setup) it emits one sigmoid channel per prompt — the query plus the distractor bank from `build_distractor_vocabulary(cfg)` (`cfg.background_terms` generic-background prompts + `cfg.distractor_objects` structurally-similar/distinct-material object confusers, deduped; phrases sharing a content word with the query are stripped by `filter_distractors`). All K logit maps come from one batched forward pass; text-conditional embeddings are cached at construction. The feature field regresses the full channel vector; the query-vs-distractor **contrast happens at verify time** on the multi-view-converged field, not per frame.

**Field-verified detection** (`cfg.field_verify`, [`DetectionGate.step`](DCON/src/perception/detection.py)): a detector box must also pass the field — its valid-depth pixels are unprojected, the field is read there, the top-`field_verify_top_frac` in-box cells (selected by the query channel) are pooled, and the box counts only if the worst-case margin `presence(query) − max_i presence(distractor_i)` clears `field_verify_threshold` (canonically **0.0**, the tuning-free max-Youden-J point — calibrated against the static distractor bank; a materially different bank warrants a re-sweep). This is what suppresses the geometric-look-alike FPs that survive LLMDet's sinks.

### Object Detection (`src/perception/obj_detection.py`)

**`LLMDetDetector`** (`make_detector(cfg)`; `detect(image, query) → (score, box)`) — LLMDet (MM-Grounding-DINO + LLM-supervised backbone) via HuggingFace, with the **training-free attention sinks** of Ruis et al. (ICLR 2026) for background false-positive mitigation. MUST load `iSEE-Laboratory/llmdet_{tiny,base,large}` (`model_type="mm-grounding-dino"`, native in transformers ≥4.52); the `fushh7/*_hf` weights declare plain `grounding-dino` and load with a broken, non-discriminative contrastive head — do NOT use them. Sinks reuse `[unused*]` vocab slots (init proved **inert** — the BERT text encoder recontextualizes them); a box survives only if its query phrase out-scores every sink. Config: `llmdet_model_name`, `llmdet_threshold` (= τ), `llmdet_use_sinks`, `llmdet_num_sinks` (48), `llmdet_sink_init`.

The YOLO-Worldv2 / COCO-YOLOv8 / VLFM-hybrid backends, the LocateAnything→LLMDet verification cascade, the CLIPSegDetector verifier candidate, the contrastive (sigmoid×softmax-share) CLIPSeg target, and the per-target LLM distractor generator were all removed — recover from git history if ever needed. Look-alike FP suppression lives in the pairwise field-verify gate above.

**LLMDet threshold (τ) calibration** — `llmdet_threshold` is the per-box query-score floor; raising it trades true-positive recall for background-FP rejection. From a 60-frame sweep (87 COCO-YOLO-oracle TPs, 600 out-of-domain FPs) on the **large** model. TP-ret = true-positive retention at the τ that achieves the target background FP-rejection:

| target FP-rejection | τ (sinks48) | TP-ret (sinks48) | τ (no-sinks) | TP-ret (no-sinks) |
|---|---|---|---|---|
| 90% | 0.38 | 0.94 | 0.49 | 0.89 |
| 95% | 0.41 | 0.87 | 0.56 | 0.74 |
| 99% | 0.47 | 0.71 | 0.68 | 0.54 |

`large` beats `tiny`/`base` (no-sink separation 0.32 vs 0.24); sinks help most at the aggressive end (99% FP-rej: TP-ret 0.54→0.71). Default `llmdet_threshold=0.42` ≈ the 95% FP-rejection point; use ~0.47 for ~99% (toward the paper's near-elimination). Numbers are from rendered frames in these scenes — re-validate τ on real data.

### Grids (`src/perception/grid.py`)

Three complementary BEV representations:

| Grid Type | Purpose | Input | Output (per voxel) |
|-----------|---------|-------|-------------------|
| **UncertaintyGrid** | Epistemic + aleatoric | Evidential (NIG) field forward-pass | 2 scalars; observed-FREE voxels zeroed when `mask_free_epistemic` (field never trains on air → free-air uncertainty is phantom noise; BEV reduction switches to max-over-Y) |
| **OccupancyGrid** | Free / occupied / unseen | Depth voxelization (incremental) | trinary value |
| **SimilarityGrid** | Target relevance | Field forward-pass; pairwise: clamped worst-case margin (query − max distractor channel) | 1 scalar |

**Coordinates**: World space with `scene_bounds = ((xmin,ymin,zmin),(xmax,ymax,zmax))`; BEV cells index `(z, x)`; `y` is ignored for planning.

## Running the Code

### Main Entry Points

**1. Live perception + planning loop (primary)**
```bash
cd DCON
python main.py --gpu 0 --query "a pillow"
```
Seeds the maps from one observation (no spin/cold-train), then runs the three-cadence main loop. Outputs to `output/current_scene/`: always `traj_log.jsonl` + `grid_extent.json`; `save_enabled` (default) adds the final occupancy BEV `.npy`; `save_video` (the `--no-visualize` inverse) adds the full per-step BEV map + RGB history and renders `nav_history.mp4`. No feature-field checkpoint is written. `main.run(cfg, start_pos=..., start_rotation=..., goals=..., success_radius_m=..., save_enabled=..., save_video=...)` is also importable and returns a per-episode metrics dict (`success`, `spl`, `l_geodesic`, `final_geodesic`, `path_length`, `agent_stopped`, `final_pos`, `start_nav`, `success_radius_m`, `steps`; used by `benchmarks/eval_scene.py`).

**Goals** (for the evaluator) may be **points** (`[x, y, z]`) or **axis-aligned rectangles** (`{rect: [x_min, z_min, x_max, z_max], y: <height>}`, e.g. a table footprint). Success/geodesic score against the **nearest point** of the goal ([`nearest_goal_point`](DCON/src/episode/scoring.py) clamps the agent's x,z into the rectangle). Success is **geodesic** distance (navmesh shortest path via `geodesic_distance`) — matching the Habitat ObjectNav challenge; there is no straight-line success mode. `final_pos` is recorded so runs can be re-inspected/re-scored offline.

**2. Sweep with human adjudication (`benchmarks/eval_scene.py`)**

A minimal eval needs two named configs (see `config/evaluation_configs/README.md`); the CLI carries only the per-invocation ones:
- **`--benchmark <name>`** — WHICH EPISODES: a named entry in `config/evaluation_configs/benchmarks.yaml` (`gibson`, `ovon`, `ovon_unseen`, `ovon_synonyms`, `goffs`): the episode source (`dataset:` split or `scenarios:` sweep), `scenes_root`, and any protocol embodiment overlay (OVON's Stretch camera) + scoring defaults.
- **`--agent_config <path>`** — THE DETECTOR ARM: a `Config` overlay yaml under `config/agent_configs/` (repeatable; stacks after the benchmark's own embodiment overlay; a bare name is found anywhere under `config/`). Also names the default `--out` dir.
- **`--experiment <name>`** (optional) — EVERYTHING ELSE, only needed when bundling non-detector settings into a reusable preset: `config/evaluation_configs/experiments/<name>.yaml` — action mode (`discrete:`), caps (`max_per_combo:`, `categories:`, `scenes:`), scoring (`radius:`, `viewpoints:`), `out:`, base `config:`, even a default `benchmark:`/`agent_config:`. Keys are validated; a typo fails before anything runs.
- **Session flags** — `--gpu`, `--video`/`--no-evidence`, `--rerun`, `--only`, `--verdicts`, `--results`, `--refresh-bev`, `--out` (override).

Precedence per key: CLI > experiment > benchmark > default; overlays stack (benchmark embodiment → experiment → CLI `--agent_config`) so the most specific wins per config key. `--out` defaults to `output/<benchmark>_<agent_config>` (or `output/<benchmark>_<experiment>` when `--experiment` is given).
```bash
python benchmarks/eval_scene.py run    --benchmark gibson --agent_config config/agent_configs/detector_pairwise_field_maxj.yaml --gpu 3   # execute + save evidence
python benchmarks/eval_scene.py review --benchmark gibson --agent_config config/agent_configs/detector_pairwise_field_maxj.yaml           # (re)build verdicts.yaml + BEV evidence
# ...inspect the out dir's ep/<id>/ (nav_history.mp4, bev_final.png), edit verdicts.yaml...
python benchmarks/eval_scene.py report --benchmark gibson --agent_config config/agent_configs/detector_pairwise_field_maxj.yaml           # aggregate SR/SPL
python benchmarks/eval_scene.py report --out output/<any_existing_dir>                                                                    # re-score records on disk, no configs
```
`config/agent_configs/` holds `detector_pairwise_field_maxj.yaml` — THE canonical detection arm (τ0.47 sink-gated LLMDet + pairwise-field margin gate at 0.0; also baked into base `config.yaml`) — and the OVON embodiment profile. Superseded sweep arms (fieldverify thresholds, contrastive, LLM distractors) were deleted; git history has them. `--experiment` presets are for bundling caps/scoring, e.g. a `*_smoke` variant capped at 2 eps per category×scene — freeze one under `experiments/` once it's worth re-running by name.
A benchmark's episode source is either `scenarios:` (a hand-written per-scene sweep of **targets × start positions**; `runs_per_combo` repeats each; run id `{target}__{start}[__r{k}]`) **or** `dataset:` (standard benchmark episodes; below). Either way the pipeline separates **evidence** from **judgment**:
- **`run`** executes each combo (skips completed unless `--rerun`), saving a raw record to `runs/<id>.json` + an evidence bundle in `ep/<id>/`. Evidence is **minimal by default** — `traj_log.jsonl`, `grid_extent.json`, the FINAL occupancy BEV `.npy`, and `bev_final.png` (goals/start/final marked) — a few tens of KB per run. `--video` adds the full per-step BEV map + RGB history and renders `nav_history.mp4` (megabytes/run). `--no-evidence` keeps only `traj_log` + `grid_extent`. The feature-field checkpoint is never saved (nothing consumes it). Bakes in no verdict — auto-success is only a suggestion.
- **`review`** (re)generates a single **`verdicts.yaml`** (one entry per run, `status: auto|success|fail|exclude`, pre-filled with the auto-suggestion + evidence inline as comments). Preserves your edits on re-run. This is the authoritative human judgment (replaces the older exclude/override files).
- **`report`** aggregates SR/SPL from records + verdicts → `results.json` (auto = use computed success; success/fail = override with SPL recomputed; exclude = drop). Per-combo rollup included.

Key behaviors: `run` **requires an episode source** — `--benchmark`, or an `--experiment` whose yaml sets `benchmark:`/`dataset:`/`scenarios:`. For `review`/`report`, `--out` alone aggregates exactly the records on disk in `<out>/runs/` (immune to a since-changed source; passing a source warns about records it doesn't cover). `--verdicts <name|path>` chooses which verdicts file to score (bare name resolves under `<out>`; results file is named to match so curations don't clobber). `--only <id...>` restricts the set. Scenario targets accept an optional per-target `success_radius_m`. `benchmarks/eval_core.py` holds the torch-free logic (importable without CUDA); only `run` imports `main.run`.

**Standard ObjectNav benchmark episodes (`dataset:` benchmarks)** — evaluate on the same episode datasets VLFM / Goal-Oriented Semantic Exploration report on (Gibson/SemExp, HM3D, MP3D, HM3D-OVON), while the custom `scenarios:` path stays available. [`eval_core.load_objectnav_dataset`](DCON/benchmarks/eval_core.py) parses the habitat-lab ObjectNav-v1 schema (`content/<scene>.json.gz`; goal instances inline or in `goals_by_category`) into the same run specs the scenarios path produces — `review`/`report`/`verdicts.yaml` work unchanged. Per episode it carries: the local scene `.glb` (found by a recursive basename-matched walk of the benchmark's `scenes_root`, so both flat Gibson layouts and HM3D's nested `00xxx-X/X.basis.glb` resolve; episodes with no local scene are skipped with a warning), the episode's `start_rotation` (honored at spawn), the query `"a {category}"`, all goal-instance positions (success = geodesic to the nearest ≤ `radius`, default 1.0 m — the "stop within 1 m of the object" protocol), and a per-floor **BEV height band** derived from the start Y (`BEV_BAND_ABOVE_FLOOR` = floor+0.2..floor+1.5 — benchmark scenes are multi-floor, so the static config band is overridden per run). Run id `{category}__{scene}__ep{episode_id}`; the per-combo rollup then groups by (category × scene). Episode subsetting (`categories:`, `scenes:`, `max_per_combo:`) and the action space (`discrete: true` for challenge-comparable 25 cm / 30° primitives, 500-step budget) are experiment-yaml keys. Note the datasets + scene assets themselves must be downloaded separately; only Gibson `.glb`s are in the repo (the Gibson-val benchmark scenes are Collierville, Corozal, Darden, Markleeville, Wiconisco — the first two are present).

**HM3D-OVON** (open-vocabulary ObjectNav; episodes at HF `nyokoyama/hm3d_ovon`, splits `val_seen` / `val_seen_synonyms` / `val_unseen`; scenes = HM3D val v0.2, Matterport ToS) loads through the same `dataset:` path — same ObjectNav-v1 schema, `goals_by_category` keyed `<scene>.basis.glb_<category>`. The `ovon`/`ovon_unseen`/`ovon_synonyms` benchmarks bake in the protocol's **Stretch embodiment** via the `agent_ovon_stretch` overlay (360×640 portrait, 42° HFOV, camera at 1.31 m, agent 1.41 m × r0.17 m; applied through `Config.apply_yaml` — only the keys present override, and per-run fields (scene, query, BEV band) are applied after so the overlay can't clobber them). The narrow portrait camera meaningfully shrinks per-frame coverage for the detector / CLIPSeg / IG raycast (all consume the same `cfg.fov`/`img_*` via `SimInterface.intrinsics`, so they stay consistent automatically). OVON ships ~3k episodes per split — subsample with the experiment's `max_per_combo:`. **Scoring**: the **primary reported SR/SPL number is the non-viewpoints reading** — goals are the object positions, radius 1.0 (the SemExp/VLFM "within 1 m of the object" convention). This is the number to headline; it's what's comparable to VLFM/SemExp and doesn't penalize a system (ours) that has no explicit visibility/standoff-seeking behavior. Set `viewpoints: true` in the experiment for the official protocol as a secondary, stricter cross-check — goals become every goal instance's `view_points` agent positions (habitat's `DistanceToGoal` VIEW_POINTS measure) and the default radius switches to 0.1 (the official `success_distance`; verified to match the episodes' precomputed `info.geodesic_distance` to <1 mm). Our agent's self-stop (`cfg.stop_distance_m`, also MPPI's arrival radius — see `mppi.py`) checks proximity to the detected object's surface with no visibility requirement, so it systematically under-performs the 0.1 m viewpoint radius even on runs that reach the correct object (e.g. a 2026-07 OVON smoke episode stopped 0.12 m from the nearest viewpoint — 2 cm over). Episodes lacking `view_points` fall back to positions with a warning.

**3. Trajectory video**
```bash
python tools/visualize.py --config config/config.yaml --output ./figs/nav_history.mp4 --fps 5
```
`tools/visualize.py` reads `traj_log.jsonl` + saved BEV maps and renders the navigation video.

**4. Run analysis (`tools/analyze_runs.py`, torch-free)**
```bash
python tools/analyze_runs.py output/gibson_pairwise_maxj                                              # mid-run progress + SR/SPL + failure attribution
python tools/analyze_runs.py maxj=output/gibson_pairwise_maxj jul4=output/saved_data/objectnav_val_total_jul4   # compare arms on the COMMON episode set
```
Reads only what `eval_scene.py run` leaves on disk (`runs/*.json`, `ep/<id>/traj_log.jsonl` + `grid_extent.json`, `verdicts.yaml`). With ≥2 dirs: common-set SR/SPL plus per-category and per-scene tables. Failure attribution per arm: **FP** (latched onto the wrong object — final EXPLOIT goal > `--near-thresh` (1.0 m) from every true goal), **NEVER** (never latched — recall bucket), **NEAR** (latched near the target but failed — navigation bucket, tail-classified as frozen / orbiting / approach / jitter). Verdicts respected (success/fail override, exclude drops).

### Configuration

All hyperparameters live in [src/config.py](DCON/src/config.py). YAML in [config/config.yaml](DCON/config/config.yaml) overrides Python defaults.

Key settings:
- **Perception**: `target_query`, `iterations`, `device`.
- **Habitat**: `scene_path`, `img_width`, `img_height`, `fov`, `sensor_height` (1.25 — DD-PPO's training camera height).
- **Hash-grid training**: `hash_train_batch_size`, `hash_buffer_refresh_interval`, `hash_per_frame_cache_size`, `history_buffer_capacity`, `hash_feature_dim` (overridden to 1+K channels in pairwise mode).
- **MPPI** (`planning:` YAML): `mppi_dt`, `mppi_max_v_mps`, `mppi_max_w_rps`, `mppi_horizon`, `mppi_num_iters`, `mppi_anneal_beta_action`, `mppi_anneal_beta_traj`, `mppi_w_goal`, `mppi_w_ig`, `mppi_lambda`, `mppi_collision_substeps`, `mppi_occupied_cell_cost` (goal-distance-field wall-crossing penalty), `mppi_unseen_cell_cost` (EXPLOIT-only unseen-cell penalty in the same field), and the confidence-hysteresis knobs (`mppi_conf_*`).
- **Discrete actions** (`planning:` YAML; SEARCH only): `discrete_actions` (off = continuous velocity; on = Habitat ObjectNav primitives via the tracking controller), `discrete_forward_m` (0.25), `discrete_turn_deg` (30, challenge convention — DD-PPO EXPLOIT uses its own `DDPPO_FORWARD_M`/`DDPPO_TURN_DEG` = 25 cm/10° constants in ddppo_policy.py), `discrete_lookahead_m` (0.5), `max_agent_steps` (500 primitive budget). `ddppo_checkpoint_path` points at the pretrained PointNav weights.
- **Detection**: `detected_persistence`, `stop_distance_m` (also EXPLOIT's arrival check), `exploit_redetect_interval` (replans between detector runs once latched; `<=0` = never re-detect after latching). Detection-classification gates (each disables at `<=0`): `detected_min_box_frac` / `detected_max_box_frac` (box area as a fraction of the image) and `detected_min_dist_m` / `detected_max_dist_m` (agent→object distance, m) — *too close* (near OR fills the frame) is ignored, *too far* (distant OR tiny) investigates but doesn't latch, *usable band* latches.
- **LLMDet**: `llmdet_model_name`, `llmdet_threshold`, `llmdet_use_sinks`, `llmdet_num_sinks`, `llmdet_sink_init` (see the LLMDet section above).
- **Pairwise field verification**: `clipseg_pairwise`, `clipseg_model_name`, `background_terms` + `distractor_objects` (the competing-class bank), `field_verify`, `field_verify_threshold` (margin gate, canonically 0.0), `field_verify_top_frac` / `field_verify_min_points` / `field_verify_pool`, `field_verify_presence_floor` (separate "is anything here" conjunct).
- **Uncertainty masking** (`grid:` YAML section): `mask_free_epistemic` (zero epistemic/aleatoric at observed-FREE voxels — the field never trains on air, so free-air uncertainty is phantom noise over traversed rooms; also switches the uncertainty BEV reduction from bottom-k-mean to max-over-Y).

## Common Development Tasks

### Adding a New Planner
1. Create `src/planning/my_planner.py` mirroring `MPPIPlanner`'s `plan` / `optimize_trajectory` signature.
2. Wire into [`plan_search_action`](DCON/src/planning/search.py) alongside the existing MPPI call.
3. Add a config flag or argparse switch to choose between planners at runtime.

### Tweaking Perception
- **Buffer size / freshness**: `history_buffer_capacity` (larger = more history, slower drift; smaller = faster turnover, more recency bias).
- **Uncertainty computation**: `src/perception/grid.py` (`UncertaintyGrid` methods) + the evidential head in `src/perception/featurefield.py` (`EvidentialFeatureField`).

### Tweaking the planner
- **Cost weights**: `mppi_w_goal`, `mppi_w_ig`; the confidence curve (`mppi_conf_weight_a`, `mppi_conf_weight_scale`, `mppi_conf_decay`, `mppi_conf_threshold`) shapes how detection confidence trades goal pull against IG.
- **Annealing aggressiveness**: `mppi_anneal_beta_action`, `mppi_anneal_beta_traj` — smaller β = sharper annealing.
- **Collision robustness**: `mppi_collision_substeps` (thin-wall tunneling checks).

### Debugging Semantic Grounding
- Check similarity BEV map: load saved `bev_similarity_*.npy` with `np.load()` + `plt.imshow()`.
- Tweak query: `cfg.target_query` (favors "a [object]" phrasing for better CLIP grounding).
- Inspect the distractor bank a run actually used: `<out>/run_meta.json` (query + deduped/filtered phrases).

### Adding Visualization
- `src/visualization/visualizer.py` (matplotlib-based).
- Saved maps are `.npy`; load with `np.load()` + `plt.imshow()`.

`render_combined_grid` layout (2×3) used by `nav_history.mp4`:
| | col 0 | col 1 | col 2 |
|---|---|---|---|
| **row 0** | Agent View (RGB w/ detector bbox) | _(empty)_ | BEV Similarity |
| **row 1** | Epistemic Uncertainty | Occupancy | Uncertainty + Occupancy overlay |

The RGB panel draws the accepted detector bbox (red) from the most-recent plan step's `det_box`.

## Key Invariants & Patterns

1. **Coordinate Systems**: World space (Habitat / simulator) vs. BEV voxel indices (in grids); unprojection maps camera→world via depth + camera pose.
2. **Evidential Uncertainty**: a single `EvidentialFeatureField` yields aleatoric + epistemic uncertainty from its Normal-Inverse-Gamma head in one forward pass — no ensemble.
3. **Frozen Semantics**: CLIPSeg weights are static; the feature field learns *where* in 3D to render its per-prompt relevance channels.
4. **Streaming Data**: Perception is online. The `_HistoryBuffer` keeps a bounded-memory snapshot of past observations; the `_latest_frame` oversample guarantees fresh data participates in every gradient step.
5. **BEV Representation**: Plans operate in top-down (z, x) space; y is ignored for planning.
6. **Receding Horizon**: MPPI computes an H-step plan, but only the first action of `U_opt` is executed before the next replan. `last_U` carries the rest forward as a warm-start.

## Dependencies & Environment

- **PyTorch** (CUDA 11.8+)
- **Habitat-sim**: physics simulator, scene loading, agent control
- **Transformers** (HuggingFace): downloads `iSEE-Laboratory/llmdet_{tiny,base,large}` (MM-Grounding-DINO, native in transformers ≥4.52) and `CIDAS/clipseg-rd64-refined` on first use.
- **DD-PPO checkpoint**: `ddppo_weights/gibson-2plus-se-resneXt101-lstm1024.pth` (gitignored; `cfg.ddppo_checkpoint_path`) — loaded once per process (memoized).
- **NumPy, Matplotlib, PIL, imageio, OpenCV**: data processing and visualization
- See `requirements.txt` for pinned versions

GPU memory: ~12 GB recommended for training (feature field + super-batch staging).
CPU memory: ~500 MB for the `_HistoryBuffer` at default capacity (200k × 512-dim features).

## Debugging Tips

- **Out-of-memory (GPU)**: reduce `hash_train_batch_size` or `hash_buffer_refresh_interval`.
- **Out-of-memory (CPU)**: reduce `history_buffer_capacity` (linear in feature-dim × capacity).
- **NaN losses**: check input normalization (esp. CLIP embeddings); verify depth values are in `[min_sensor_dist, max_sensor_dist]`.
- **Planner picks bad goals**: inspect the saved `bev_similarity_*.npy` peak; tweak `cfg.target_query` phrasing.
- **Agent wedged / dwelling in corners** (SEARCH): every sampled rollout collides so `best_U=None` and the loop idles that replan (the survival-time weighting steers subsequent iterations out of the pocket). Check `agent_radius` (the navmesh insets walls by it, so gaps the BEV considers open can be sealed in the navmesh).
- **EXPLOIT turn-only stall** (DD-PPO spins without advancing): historically caused by (a) the turn-sign inversion in step_discrete (fixed — native semantics now), (b) 30° turns vs the 10° training convention (fixed — DDPPO_TURN_DEG), or (c) depth sensor-miss `0.0` blocks read as point-blank walls through Gibson mesh holes (fixed — misses remapped to far). If it recurs, dump the depth frame first.
- **Trajectory video looks wrong**: check `grid_extent.json` matches the scene bounds and that `traj_log.jsonl` was written cleanly (one JSON per line).
- **Never latches into EXPLOIT** (stuck exploring): the detection never reaches the *usable band*. Check the classification gates — widen `detected_max_dist_m` (object farther than the threshold only investigates, never latches), lower `detected_min_box_frac`, or raise `detected_max_box_frac`. Latching also needs `detected_persistence` consecutive usable-band frames (the score floor is the detector's own `llmdet_threshold`).
- **Latches too eagerly / on the wrong thing**: tighten the band — lower `detected_max_dist_m` so it must get closer, raise `detected_min_box_frac` so tiny far blobs only investigate, or raise `detected_persistence`.
- **Agent ignores an obvious nearby target** (box fills the frame): the *too-close* gate is dropping it — lower `detected_min_dist_m` or raise `detected_max_box_frac`.