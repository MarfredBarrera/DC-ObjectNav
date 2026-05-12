# DCON Handoff Notes

Snapshot of the current state of the MPPI planning stack and how the main loop
flows. Captures the changes the user has applied directly (some without
discussion), so future-me reads this instead of guessing.

---

## High-level algorithm

The agent runs a perception–planning loop with three cadences:

1. **Every step** — one gradient step on the feature-field ensemble.
2. **Every `REPLAN_INTERVAL = 200` steps** — call MPPI for a new plan, execute
   exactly **one** action from the resulting control sequence, then discard the
   rest of the queue next replan.
3. **Every `cfg.hash_buffer_refresh_interval = 2000` steps** — pull a fresh
   RGB-D frame, update the replay buffer + occupancy, rebuild the BEV maps.

The replan path is now A\*-free and top-k-free:

```
plan_one_action():
    bev_sim, bev_epi, bev_occ = perception.* .get_2d_map(...)
    approach_points = mppi.get_goals_near_highest_sim(bev_sim, bev_occ, max_candidates=1)
    goal = approach_points[0]                          # closest free cell to highest-sim peak
    opt_path, U_opt, _ = mppi.optimize_trajectory(
        start_grid, goal, bev_epi, bev_occ,
        initial_heading=heading, progress=progress, ...
    )
    # Fill action queue with U_opt, pop one, return.
    # On failure (U_opt = None): try queue, else idle.
```

`astar` is still imported and passed into `plan_one_action` for parameter
compatibility, but it is **not called anywhere on the MPPI path**.

---

## MPPI in detail (`src/planning/mppi.py`)

### Signature

```python
optimize_trajectory(
    start, goal, epi_map, occ_map,
    num_samples=100, num_iters=3, horizon=100,
    lambda_weight=None, w_goal=None, w_ig=None,
    w_occ=1e6, w_unseen=0,
    intrinsics=None, sensor_height=1.5,
    initial_heading=None, progress=0.0, cov_scale=None,
)
```

Takes a `(z, x)` start cell, a `(z, x)` goal cell, BEV maps. No reference
trajectory.

### Goal selection — `get_goals_near_highest_sim`

BFS outward from the single highest-similarity cell through *any* cell type
(unseen, free, obstacle). Returns free cells in BFS-distance order. The peak
similarity often lands on an obstacle (e.g. a bed is marked occupied), so the
caller can't step *on* it; the closest free cell is the approach point. With
`max_candidates=1` in main.py, only the single closest free cell is returned.

### Nominal control — warm-started receding-horizon

`U_nom` is **not** zeros. It's the previous successful plan, shifted left by
one (the action just executed is dropped):

```python
U_nom = torch.zeros((horizon, 2), device=...)
if self.last_U is not None:
    shifted_len = min(self.last_U.shape[0] - 1, horizon)
    U_nom[:shifted_len] = self.last_U[1:1 + shifted_len]
```

`self.last_U` is only updated at the end of `optimize_trajectory` if MPPI found
a safe rollout this call. On first call (or after a manual reset), `last_U`
is `None` and `U_nom = 0`.

### Sample noise

```python
cov_matrix = cov_scale * tensor([[0.5, 0], [0, π/4]])
noise = randn(K, H, 2) @ cov_matrix
noise[0] = 0.0                  # row 0 evaluates the warm-started plan as-is
U_samples = U_nom + noise
U_samples[..., 0] = clamp(..., min_v_cells, max_v_cells)
U_samples[..., 1] = clamp(..., -max_w_rad, max_w_rad)
```

The first row pins zero noise so the unmutated warm start is always among the
candidates — protects against the case where all noisy variants happen to be
worse than the carried-over plan.

### Rollout (turn-first integration)

```python
Theta[:, 1:] = θ_start + cumsum(U[:, :-1, 1])
Z[:, 1:] = z_start + cumsum(U[:, :-1, 0] * sin(Theta[:, 1:]))   # post-rotation heading
X[:, 1:] = x_start + cumsum(U[:, :-1, 0] * cos(Theta[:, 1:]))
```

Translation each step uses the heading **after** that step's rotation. This
matters for in-place spins: without it, `w_0` couldn't steer step 1, so an
agent pinned facing a wall would always commit the first translation into the
wall.

### Cost terms

1. **Goal distance** — `dist_cost = mean(||(Z, X) - goal||)` per rollout.
   Weight `w_goal` from `scheduled_params`.
2. **Collision** — subsampled along each segment (alpha-interpolated
   between waypoints, `n_sub = max(2, ceil(max_v_cells * 2.5))` per segment,
   plus the final waypoint). Cells at `(start_z_idx, start_x_idx)` are
   *forgiven* (treated as free) so an agent technically inside an obstacle
   isn't penalized for being where it is — only for moving into *new* obstacle
   cells. Weight `w_occ = 1e6` (catastrophic).
3. **Unseen traversal** — count of cells with value 0 (unseen). Currently
   `w_unseen = 0` — unseen cells are not penalized at all. The exploration
   gradient comes entirely from IG.
4. **Information gain** — discounted FOV raycast IG from each rollout
   waypoint, but **only computed for collision-free rollouts** (`safe_mask`).
   Weight `w_ig` from schedule (currently 30).

### Hard-exclude on collision

```python
safe_mask = (collision_cost == 0)
if safe_mask.any():
    masked_scores = where(safe_mask, scores, -inf)
    iter_best = argmax(masked_scores)
    # ... update best_U if scores improve
    weights = exp((masked_scores - beta) / λ)        # softmax over safe only
else:
    print("NO SAFE ROLLOUTS")
    weights = exp((scores - beta) / λ)               # fall back to soft penalty
weights /= weights.sum()
U_nom += sum(weights * noise)
```

When any rollout is safe, MPPI never picks a colliding plan — colliders get
zero softmax weight and are excluded from `argmax`. When all rollouts collide
in some iter, the soft `-w_occ * collision_cost` is still in `scores`, so the
softmax-weighted update at least nudges `U_nom` toward less-bad samples.

### Final fallback

After all iters, if `best_U` is still `None` (no safe rollout was ever found
in any iter), MPPI reconstructs a trajectory from the shifted previous plan
and returns *that* rather than `(., None, None)`:

```python
if best_U is None and self.last_U is not None:
    print("MPPI: No safe rollout found, falling back to shifted previous plan")
    # rebuild U_fallback from self.last_U shifted, then roll out kinematics
    best_U = U_fallback.cpu().numpy()
    best_traj = [reconstructed_(z, x) per step]
```

This keeps the agent moving along its last-known-safe trajectory when the
current state confuses MPPI. The action queue in main.py then executes one
step of this fallback plan.

### Escape rollouts — **currently disabled**

The escape-rollout block (stop / CCW spin / CW spin injected as deterministic
samples) is **commented out** in the current file. Search for
`# Inject deterministic escape rollouts` to find it. The user removed these
when they added warm-starting; the warm-start preserves multi-step plans
across replans, which addresses the IG-seesaw failure mode that the escapes
were originally meant to fix. If the agent gets stuck oscillating in tight
spots again, re-enable the escape block.

---

## Map semantics

`occ_map` values: `0 = unseen`, `1 = free`, `2 = obstacle`.

- `OccupancyGrid.update_from_observation` now uses
  `thickness = voxel_resolution * 1.2` (was hardcoded `0.05`). At
  `voxel_resolution = 0.10`, that's a 0.12 m occupied shell around the
  observed depth surface — enough to register a wall as ≥2 contiguous voxel
  layers rather than a sub-voxel stripe with unseen gaps.
- `OccupancyGrid.get_2d_map_dilated(radius)` exists but isn't currently used.
  It dilates obstacles **only into free cells**, never into unseen — preserves
  exploration.

`SimilarityGrid.compute_similarity_map`:
- Masks query points to `voxels == occupied_val` only (so similarity is
  evaluated only at obstacle cells).
- Returns `(sim + 1) / 2` of `ensemble_mean · text_embedding`.
- `get_2d_map` does **top-5%-mean** along Y (rather than mean or max), then
  returns the BEV.

---

## Config state (`src/config.py` + `config/config.yaml`)

Key values (YAML overrides Python defaults):

| Knob | YAML value | Notes |
|---|---|---|
| `voxel_resolution` | `0.10` | |
| `iterations` | `60000` | |
| `sensor_height` | `1.0` | |
| `mppi_dt` | `0.1` | Very short. Each MPPI step is 0.1 s. |
| `mppi_max_v_mps` | `1.0` (default) | → `max_v_cells = 1.0 * 0.1 / 0.10 = 1` cell/step |
| `mppi_max_w_rps` | `2.0` (default) | → `max_w_rad = 0.2` rad/step (~11.5° per step) |
| `mppi_w_sign` | `-1.0` (default) | Habitat ω is opposite sign from MPPI heading convention |
| `hash_buffer_refresh_interval` | `2000` | |
| `target_query` | `"a pillow"` | |
| `scene_path` | `gibson_scenes/Annawan.glb` | |

### Schedule values (all `*_start == *_end` currently — no active scheduling)

| Knob | start | end |
|---|---|---|
| `mppi_lambda` | 4.0 | 4.0 |
| `mppi_w_ig` | 30.0 | 30.0 |
| `mppi_w_goal` | 0.0 | 0.0 |
| `mppi_cov_scale` | 4.0 | 4.0 |

`w_goal = 0` everywhere means MPPI is currently driven purely by IG — there's
no goal-distance pull at all. The agent moves toward whatever direction has
the most uncertainty-rich cells visible by raycast. The "goal" computed by
`get_goals_near_highest_sim` is therefore informational only right now; it
doesn't shape the rollout cost.

`scheduled_params` still has the delayed-ramp logic (`p_goal = max(0, p -
0.75)`) but with `*_start == *_end` it has no effect.

`cov_scale = 4.0` means noise std for v ≈ 2.0 cells/step, w ≈ π·1 rad/step
clamped to `max_w_rad = 0.2`. The w-clamp will fire on most samples → most
rollouts saturate at full-CW or full-CCW per-step turn.

---

## Perception (`src/perception/perception_stack.py`)

The replay buffer interface changed:

- `observe(sim_iface)` now returns `(rgb, depth, c2w)` — **raw** frame, no
  unprojection.
- `update_replay_buffer(rgb, depth, c2w, intrinsics)` stores the raw tuple.
- `make_super_batch()` is the place where unprojection happens, on a randomly
  sampled subset of buffer frames. This pushes the per-frame CLIP-feature
  extraction cost from observation time to batch-build time, and lets the
  super-batch reuse frames across many training steps.
- `extract_and_unproject(rgb, depth, c2w, intrinsics)` is the helper.

`save_2d_similarity(step, depth, c2w, intrinsics)` is new — renders a 2D
similarity map from the current camera view by querying the ensemble at the
unprojected 3D points of the current frame, then resampling into image space.

---

## Failure modes & current best-known knobs to turn

If the agent stays stuck or plans through walls again, the things to check, in
order:

1. **Is `last_U` getting persistently bad?** Add a manual reset
   (`mppi.last_U = None`) whenever a replan triggers `"NO SAFE ROLLOUTS"`
   or the final fallback. The shifted-warm-start can lock in a stale direction.
2. **Re-enable the escape rollouts** at the commented block in `optimize_trajectory`.
   Stop / CCW spin / CW spin give MPPI guaranteed-safe candidates regardless of
   warm-start state.
3. **`w_unseen = 0`** — if you see rollouts slipping through partially-observed
   walls, give unseen cells a small penalty (e.g. 10–100). Anything ≥1e3 tends
   to suppress exploration entirely.
4. **`w_goal = 0`** — no goal-distance signal at all. If exploration is fine
   but the agent never *commits* to approaching the target, bump
   `mppi_w_goal_end` (and consider undoing the delayed ramp).
5. **`mppi_dt = 0.1`** with `horizon = 100` → 10 s of look-ahead. Short
   per-step distance (1 cell at v_max) means rollouts don't spread far. Either
   shorten the horizon or raise `mppi_dt`.

---

## Quick reference: key call paths

| Where | What |
|---|---|
| [src/planning/mppi.py:7-14](src/planning/mppi.py#L7-L14) | `MPPIPlanner.__init__` — sets `self.last_U = None` for warm start |
| [src/planning/mppi.py:40-81](src/planning/mppi.py#L40-L81) | `get_goals_near_highest_sim` — BFS from similarity peak |
| [src/planning/mppi.py:117-139](src/planning/mppi.py#L117-L139) | `scheduled_params` — exploration→exploitation interpolation |
| [src/planning/mppi.py:141-399](src/planning/mppi.py#L141-L399) | `optimize_trajectory` — main MPPI body |
| [src/planning/mppi.py:185-198](src/planning/mppi.py#L185-L198) | Warm-start construction |
| [src/planning/mppi.py:209-219](src/planning/mppi.py#L209-L219) | Sample + clamp |
| [src/planning/mppi.py:221-243](src/planning/mppi.py#L221-L243) | Escape rollouts (commented out) |
| [src/planning/mppi.py:255-262](src/planning/mppi.py#L255-L262) | Turn-first kinematic rollout |
| [src/planning/mppi.py:272-312](src/planning/mppi.py#L272-L312) | Collision/unseen subsampling + cost |
| [src/planning/mppi.py:333-367](src/planning/mppi.py#L333-L367) | Hard-exclude collision + softmax update |
| [src/planning/mppi.py:369-393](src/planning/mppi.py#L369-L393) | Final fallback (shifted previous plan) |
| [src/planning/mppi.py:395-399](src/planning/mppi.py#L395-L399) | Save `last_U` for next call |
| [src/perception/grid.py:198-252](src/perception/grid.py#L198-L252) | `OccupancyGrid.update_from_observation` (depth-thickness shell) |
| [src/perception/grid.py:259-281](src/perception/grid.py#L259-L281) | `OccupancyGrid.get_2d_map_dilated` (unused — available if needed) |
| [main.py:73-141](main.py#L73-L141) | `plan_one_action` — single goal, no A\*, no top-k |
