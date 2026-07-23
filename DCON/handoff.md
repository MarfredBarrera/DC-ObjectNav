# Handoff: main.py decomposition (2026-07-23, done)

`main.py` went 1077 → 337 lines; `run()` 554 → ~200. Pure structural
refactor — no algorithm change, verified by differential tests on the pure
helpers and three real smoke episodes (SEARCH, forced-latch EXPLOIT/DD-PPO,
field-verify gate). New homes:

| was in main.py | now |
|---|---|
| `box_center_world_xz`, `bev_cell_from_box_center`, `classify_detection`, `detect_classify_latch`, `world_to_grid` | `src/perception/detection.py` — `DetectionGate` class owns the detector, field-verify gate, latch state (`detected`/`streak`) and the redetect throttle; `step()` returns a `Detection` dataclass instead of the old 8-tuple |
| `plan_one_action` | `src/planning/search.py::plan_search_action` |
| the inline DD-PPO EXPLOIT branch | `src/planning/exploit.py::ExploitController` (owns the per-latch LSTM reset + arrival check) |
| `discrete_action_from_plan` | `src/planning/tracking.py` (pure-pursuit shared with `lookahead_heading_error`) |
| `get_agent_heading` | `SimInterface.agent_heading` property |
| sim init + spawn + BEV-band reconciliation | `habitat_utils.start_episode` |
| all output_dir writes | `src/episode/recorder.py::EpisodeRecorder` |
| `nearest_goal_point`, `goal_geodesic`, the two metrics dicts | `src/episode/scoring.py::score_episode` |
| SIGINT + STOP-sentinel handling | `src/episode/control.py::EarlyStop` (context manager) |

Deleted as dead: `continuous_action_from_plan` (defined, never called — the
MPPI-idle path builds its `[0, w]` command inline), the commented-out
SEARCH-mode goal-disconfirmation block, and `plan_search_action`'s `detected`
parameter (EXPLOIT never calls it, so `goal_confidence` is always `det_score`).
`main.run()`'s signature and return dict are unchanged, so `eval_scene.py` is
unaffected. Only cosmetic behavior change: the per-replan `goal:` log string
now prints the goal cell's CENTER (matching what EXPLOIT navigates to) instead
of its corner — 5 cm, stdout only, nothing downstream parses it.

---

# Handoff: codebase cleanup sweep (2026-07-23, done)

Committed in two commits: a **checkpoint commit** (`Checkpoint before cleanup
sweep`) snapshotting the full pre-cleanup working state — everything deleted
below is recoverable from it — followed by the cleanup itself. Canonical
system going forward: **pairwise CLIPSeg field, maxj operating point
(τ_margin = 0.0) + τ0.47 sink-gated LLMDet + DD-PPO EXPLOIT**; the canonical
detection arm is now also baked into base `config/config.yaml` (which had a
leftover `detector_cascade: true`).

Deleted (per user sign-off): the A* EXPLOIT cluster (`exploit_astar_action`,
`nearest_free_cell`, `astar_free`, `get_2d_map_dilated`); the
LocateAnything→LLMDet cascade + `CLIPSegDetector` + their config fields;
the whole `src/mask_clip/` package + `MaskCLIPSemantics` + `maskclip_*`
config; the contrastive (sigmoid×softmax-share) CLIPSeg mode +
`clipseg_softmax_temp`; the per-target LLM distractor generator
(`distractor_gen.py` is now just background + static confusers, no Qwen, no
fallback); `benchmarks/evaluate.py`; `tools/rgb_probe_calibration.py` +
`tools/calibrate_field_verify.py`; all superseded sweep agent-configs (only
`detector_pairwise_field_maxj` + `agent_ovon_stretch` remain); 74 tracked
`.pyc` files (now gitignored). ~50 untracked scratch analysis scripts/logs
were archived to `~/dcon_scratch_2026-07-23.tar.gz` and removed; their
capabilities were consolidated into **`tools/analyze_runs.py`** (progress,
common-set SR/SPL comparison, per-category/scene tables, FP / never-latched /
near-target failure attribution with tail classification) — verified against
the maxj-vs-jul4 250-episode data.

Refactors in the same sweep: `step_discrete` is now always NATIVE Habitat
semantics with per-call magnitudes (the MPPI tracking controller applies
`mppi_w_sign` at the source; DD-PPO passes `DDPPO_FORWARD_M`/`DDPPO_TURN_DEG`
= 25 cm/10° from `ddppo_policy.py`); `discrete_turn_deg` is back to the 30°
challenge convention (the 10°-vs-30° latent conflict below is resolved);
the DD-PPO checkpoint load is memoized per process; the distractor-vocab
metadata moved from a `meta` line in `traj_log.jsonl` to a `run_meta.json`
sidecar (visualize.py's skip-guard reverted); dead `deterministic`/argmax
path removed from the policy (sampling is mandatory — argmax was the
confirmed 2-cycle lock). CLAUDE.md rewritten to match. NOTE: the DD-PPO
handoff below predates this sweep — its next-step #2 (debug scaffolding
removal) was already done, and file references to deleted code are
historical.

---

# Handoff: DD-PPO EXPLOIT navigation (2026-07-22/23, in progress)

**Uncommitted.** This work package replaces the previous handoff's newest
item (#4, eval CLI rework) as the front of the stack — it's a different
subsystem (EXPLOIT locomotion, not detection) and doesn't touch the
CLIPSeg/field-verify work below at all. That work is unaffected and its
section (further down this file) is historical reference only now.

## What this is

EXPLOIT (the final-approach control mode once a detection has latched) used
to run a deterministic A* + waypoint controller over the BEV occupancy grid.
Per user direction, it was replaced entirely with **DD-PPO** (Wijmans et al.
2020) — the actual pretrained Habitat PointNav RL policy
(`ddppo_weights/gibson-2plus-se-resneXt101-lstm1024.pth`: depth-only,
SE-ResNeXt101 backbone, 2-layer LSTM, discrete 4-action space), since point
navigation is exactly the sub-problem it was built to solve and it sidesteps
the BEV-vs-navmesh collision mismatches A* kept running into on Gibson.

- **New file**: `src/planning/ddppo_policy.py` — a from-scratch,
  dependency-free reimplementation of
  `habitat_baselines.rl.ddppo.policy.resnet_policy.PointNavResNetPolicy`,
  vendored to match the checkpoint's exact `state_dict` (module names,
  including the upstream `tgt_embeding` typo). `DDPPONavigator` is the
  stateful wrapper (`act()` per replan; owns LSTM hidden state / prev-action
  / not-done mask across the whole latch).
- **main.py**: EXPLOIT branch now calls `ddppo_nav.act(depth, ..., pos,
  rotation, goal_world)` instead of A*; `exploit_astar_action` is still
  defined (dead code, kept for reference/recovery) but no longer called.
  EXPLOIT is always discrete-stepped via `sim_iface.step_discrete`
  regardless of the global `cfg.discrete_actions` flag (that flag now only
  governs SEARCH).
- **Verified bit-exact against upstream** (fetched actual habitat-lab v0.1.7
  source, not reverse-engineered from shapes): the PointGoal polar transform
  (`compute_pointgoal_polar` — quaternion rotation, `cartesian_to_polar`
  sign/argument order), the `[rho, cos(-phi), sin(-phi)]` goal embedding,
  the visual encoder's exact op order (permute → avg_pool2d(2) → running
  norm (identity here) → backbone → compression), and the
  `prev_action_embedding` start-token/+1 offset convention. None of these
  were the bug.

## Bugs found and fixed this session (roughly in the order hit)

1. **Goal-precision mismatch** — DD-PPO's own STOP is trained against a
   strict ~0.2m threshold on an exact ground-truth PointGoal; our goal is a
   single detection-box projection, rarely that precise. Fixed: DD-PPO
   drives locomotion only, our own `cfg.stop_distance_m` Euclidean check
   (bumped to 0.9m earlier) is the sole arrival trigger. (DD-PPO's own stop
   is still checked as a secondary path — harmless, rarely fires.)
2. **Deterministic-action 2-cycle lock** — greedy argmax action selection
   let the previous-action embedding feed back into the LSTM and lock into
   an inescapable turn_left/turn_right alternation forever. Fixed:
   `deterministic=False` (sample, don't argmax) in `DDPPONavigator.act`,
   matching Habitat's own reference eval wrapper
   (`habitat_baselines/agents/ppo_agents.py`).
3. **`discrete_turn_deg` was 30°, DD-PPO trained at Habitat PointNav's
   default 10°** — a 3x turn-magnitude mismatch vs. what the recurrent
   policy's implicit heading-correction assumes per action. Fixed: `10.0`
   in both `config.py` and `config.yaml`. (This field is currently shared
   with SEARCH's optional discrete-action tracking controller, which wants
   the ObjectNav-challenge 30° convention instead — a latent conflict, not
   yet an active bug since `discrete_actions` defaults off for SEARCH. Split
   into two config fields if SEARCH discrete mode is ever turned on
   alongside EXPLOIT DD-PPO.)
4. **Free-space goal snapping — tried, then fully REVERTED.** Reasoning at
   the time: DD-PPO trains on always-reachable PointGoals, so a detection
   goal embedded in an obstacle (on the object's surface) seemed
   out-of-distribution; snapped it to the nearest observed-free cell via
   `nearest_free_cell` (recomputed each replan, then cached-by-cell-change
   to reduce churn). **This was wrong and actively harmful**: the snap
   target depends on the *evolving occupancy map*, which keeps growing as
   the agent turns to look around — so even "cached by agent cell" still
   let the fed goal silently drift mid-chase, violating PointNav's core
   fixed-target-per-episode assumption. Confirmed by reproduction: caused a
   NaN-gradient feature-field regression in Corozal and reintroduced a
   permanent turn-lock in Collierville that hadn't been there before.
   **Current state: DD-PPO is fed the raw, fixed `goal_world` directly, same
   as the arrival check uses** — an obstacle-embedded goal is not actually a
   problem for a depth-based policy; it just approaches until physically
   blocked, same as a real PointGoal behind furniture.
5. **`sensor_height` — reverted to 1.0, then restored to 1.25 per explicit
   user direction** (Habitat's PointNav default camera POSITION is
   `[0, 1.25, 0]`, matching DD-PPO's training height; ours was 1.0). An
   earlier attempt at 1.25 combined with the (buggy) goal-snapping above
   produced the NaN-gradient regression; with the goal-snap removed, 1.25
   has not reproduced that regression in subsequent testing (still
   confirming — see Current Status).
6. **Depth resize: bilinear → nearest-exact.** Our sim renders at
   512x512 (for MaskCLIP/detector needs) but DD-PPO trained on native
   256x256 — this resize step has no upstream equivalent at all. Bilinear
   blends near/far depth across real object edges (doorframes, furniture
   silhouettes) into fabricated intermediate depths, which looks like a
   phantom slanted obstacle to a depth-only collision-avoidance policy.
   Switched `F.interpolate(..., mode="nearest-exact")` in
   `DDPPONavigator.act`.
7. **Depth-sensor-miss normalization — the likely primary root cause of the
   persistent turn-only stalls.** Added debug instrumentation (pointgoal
   rho/phi logging + depth dump on a 15-replan same-position stall — see
   `ddppo_debug_last_pos`/`ddppo_debug_stall_count` in `main.py`, currently
   left in as active debug scaffolding, **remove once this is confirmed
   fixed**) and found every observed stall showed a solid contiguous block
   of *exactly* `0.0` depth pixels (hundreds to thousands of pixels,
   centered in frame) — a sensor miss (no ray intersection), almost
   certainly a hole in the scanned Gibson mesh (matches this session's
   earlier, separate observation that "the Gibson dataset has lots of holes
   in it"). Our normalization treated `0.0` as "closest possible obstacle"
   (correct reading would be "unknown/no return", i.e. far/clear) — exactly
   backwards, and exactly the kind of input that would make a depth-trained
   policy perceive a solid wall filling a third of the frame where there's
   actually nothing, and never resolve a confident forward path. Fixed in
   `ddppo_policy.py`: depth `<= min_depth_m` (i.e. `0.0`) is remapped to
   `max_depth_m` (far/clear) before normalizing, matching how the rest of
   the codebase already excludes `depth <= min_sensor_dist` as invalid
   rather than "near".

## Current status (as of this handoff)

Re-testing the single previously-broken episode
(`toilet__Collierville__ep0`, the exact episode + position where the
stall/depth-hole was diagnosed) with fix #7 applied, `--video` on so the
stall can be visually cross-checked against `nav_history.mp4`, and
`--agent_config config/agent_configs/detector_pairwise_field_maxj.yaml` —
**use this agent_config on every future DD-PPO run**: `config.yaml`'s base
`detector_cascade: true` (a leftover from a different, unrelated detector
sweep — see the CLIPSeg work further down this file) is NOT what this
project's canonical detector setup is; the maxj pairwise-CLIPSeg-field arm
is. Earlier DD-PPO debug runs in this session accidentally used the cascade
detector by omission, which is both slower (~2x latency/replan) and not
representative of how the full validation run should score.

Not yet confirmed: whether fix #7 actually resolves the stalls end-to-end,
or whether it's a partial fix. Once confirmed on this one episode:

## Next steps (in order)

1. Confirm fix #7 (and cumulatively #1–7) resolves `toilet__Collierville__ep0`
   without a multi-hundred-replan turn lock; inspect `nav_history.mp4` if
   still ambiguous from logs alone.
2. **Remove the debug instrumentation** (`ddppo_debug_last_pos`,
   `ddppo_debug_stall_count`, the `[ddppo-debug]` prints and depth-dump
   block in `main.py`'s EXPLOIT branch) once confirmed — it's diagnostic
   scaffolding, not intended to ship.
3. Re-run the original 8-episode mini validation batch (`output/
   exploit_validation_ddppo`, ids: `chair__Collierville__ep12`,
   `chair__Collierville__ep3`, `chair__Corozal__ep15`,
   `couch__Corozal__ep11`, `toilet__Collierville__ep0`,
   `toilet__Corozal__ep3`, `tv__Collierville__ep11`,
   `tv__Collierville__ep25`) with all fixes in place and the correct
   `--agent_config`, `--rerun`. Compare SR/SPL against the pre-DD-PPO A*
   baseline and the earlier (broken) 0.50 SR / 0.128 SPL DD-PPO attempt.
4. If the 8-episode batch looks healthy: launch the full 250-episode Gibson
   run per the original instruction ("if it is effective, launch a full 250
   episode run").
5. If stalls persist even after fix #7: the video + debug prints are now in
   place to keep diagnosing — check whether the remaining stalls still
   correlate with a depth-miss block (a different mesh hole, or a
   genuinely-open area DD-PPO still refuses to enter), and whether it's
   scene-specific (Corozal was already established as harder than
   Collierville across every arm tested this session, independent of
   DD-PPO).
6. Decide whether DD-PPO nets out ahead of the A* controller it replaced —
   this hasn't been re-validated end-to-end since the original A* work
   (see the mermaid-flowchart artifact from earlier this session:
   https://claude.ai/code/artifact/f0e2409d-1821-4825-8323-1c8d66183d4c,
   predates the DD-PPO pivot).

## Repo reorg (separate, smaller item, same session)

Moved all Gibson + HM3D-OVON asset directories under `benchmarks/` for
cleanliness: `gibson_scenes/` → `benchmarks/gibson_scenes/` (git-tracked,
done via `git mv`, history preserved), `episodes/` →
`benchmarks/episodes/`, `scene_datasets/` → `benchmarks/scene_datasets/`
(both gitignored, plain `mv`). All references updated
(`config/evaluation_configs/benchmarks.yaml`, `scenarios_Goffs.yaml`,
`tools/exploration_env.py`, `eval_scene.py`/`eval_core.py`/`evaluate.py`
defaults, `CLAUDE.md`). Verified: `.gitignore`'s unanchored patterns already
cover the new paths (no edit needed); dataset loads + scene resolution
confirmed end-to-end post-move. This is done, not part of the DD-PPO
open items above.

---

# Handoff: CLIPSeg feature field + field-verified detection

Three stacked, **uncommitted** work packages (working tree dirty — `git status` from
the repo root); the newest is #3:

## Work package 4: eval CLI rework — benchmark × experiment (2026-07-13)

`benchmarks/eval_scene.py`'s CLI was rebuilt (per user spec: only gpu,
benchmark choice, video, and agent-config on the command line; everything
else in a pointed-to yaml). An earlier `--preset` design from the same day
was replaced by this. BREAKING: the old `--dataset/--scenarios/--scenes-root/
--categories/--scenes/--max-per-combo/--viewpoints/--radius/--discrete/
--config` flags are GONE from the CLI:

- `--benchmark <name>` → entry in `config/evaluation_configs/benchmarks.yaml`
  (gibson, ovon, ovon_unseen, ovon_synonyms, goffs): episode source +
  scenes_root + protocol embodiment overlay (ovon* bake in
  `agent_ovon_stretch`).
- `--experiment <name>` → `config/evaluation_configs/experiments/<name>.yaml`:
  everything else (agent_config overlays, discrete, max_per_combo,
  categories/scenes, radius/viewpoints, out, base config). Keys validated.
  Current: `fieldverify_thr050`, `contrastive_field`, + `*_smoke` (cap 2).
- Session flags: `--gpu --video --no-evidence --agent-config (repeatable,
  bare names resolve anywhere under evaluation_configs) --rerun --only
  --verdicts --results --refresh-bev --out`.
- Precedence per key CLI > experiment > benchmark > default; overlays STACK
  (benchmark → experiment → CLI). `--out` defaults to
  `output/<benchmark>_<experiment>`; `report --out <dir>` alone still
  re-scores any existing directory.

Typical: `python benchmarks/eval_scene.py run --benchmark gibson
--experiment contrastive_field --gpu 3`. See
`config/evaluation_configs/README.md`. Existing overlay yamls were
deliberately NOT moved: the live gibson_val_CLIPSEG_jul12 run re-reads
`detector_fieldverify_thr050.yaml` by path every episode.

## Work package 3: look-alike FP suppression (2026-07-12)

Targets the 13 FP-stops that SURVIVE the field gate (CLIPSeg corroborates
geometric look-alikes, e.g. bed__Corozal 0.83).

**Contrastive CLIPSeg field target** (`clipseg_contrastive`,
`clipseg_softmax_temp`, default OFF) — the field's per-pixel training target
becomes sigmoid(target) × softmax over [target] + `cfg.distractor_objects`
(18 canonical phrases incl. wall/floor as background classes; one batched
CLIPSeg pass, K conditional embeddings cached at construction; the query's
own category is word-overlap-filtered out per query —
`semantics.filter_distractors`). The map answers "more couch than bed?"
instead of "couch-like?". Field/gate/BEV machinery unchanged (still scalar
[0,1]); `field_verify_threshold` 0.50 was swept on the plain sigmoid, so
re-sweep if timeouts rise.

Smoke-tested on the gate-resistant FP frame itself
(bed__Corozal__ep0/rgbs/rgb_12500.png, the couch the run latched on):
bed-on-couch topk10 0.730 → 0.210 while couch-on-couch reads 0.571;
distractors-off path byte-identical to plain sigmoid. Test:
`output/scratch_distractor/smoke_distractor.py` (run with
`-e PYTHONPATH=/workspace/DCON`).

Eval overlay (stacked on the fieldverify thr050 winner):
`config/evaluation_configs/detector_distractors_field.yaml`. Suggested A/B:
vs `fieldverify_sweep_thr050` on the same 50-ep subset.

**LLMDet distractor-phrase gate — implemented, then REMOVED per user decision
(same day).** Confusable phrases appended to the detection prompt as competing
classes; killed the bed__Corozal FP cleanly. Findings worth keeping (also in
LLMDetDetector's docstring) for anyone resurrecting it (git has no trace —
it never got committed):
- Sinks and distractor phrases CANNOT coexist in one prompt: the 48-sink
  suffix is ~240 wordpieces (each `[unusedN]` tokenizes to ~5 pieces — which
  also explains the "inert sink init" mystery: the re-initialized embeddings
  never appear in the tokenized prompt) and it crushes every non-lead
  phrase's per-box score (couch box: "a couch"=0.607/"a bed"=0.036 sink-free
  in either order, vs 0.061/0.536 with sinks). The gate must REPLACE sinks.
- Sink-free multi-phrase prompts discriminate cleanly and order-invariantly,
  but shift the TP score distribution (couch TP 0.495 joint vs 0.54
  single-phrase), so τ0.47 is marginal and would need re-validation.
LLMDet itself is now byte-identical to the pre-WP3 sink-gated original.

---

# Previous handoff (work packages 1–2)

Two stacked, **uncommitted** work packages (working tree dirty — `git status` from
the repo root):

1. **CLIPSeg feature-field swap** (earlier session, verified) — the field now
   regresses a scalar CLIPSeg relevance score instead of a 512-D MaskCLIP
   embedding. Background section below.
2. **Field-verified detection** (this session, 2026-07-09/10) — LLMDet detections
   are gated by querying that trained field inside the detection box. Sweep in
   progress; live status below.

## Operational constraints (user instructions, this campaign)

- All python runs inside the container: `docker exec -w /workspace/DCON objectnav_container python ...`
  (`/workspace/DCON` = bind mount of `DCON/`).
- **GPU 3 only, one job at a time** — shared server; check `nvtop`/`nvidia-smi`
  for other users' processes before every launch. No multi-GPU parallelism.
- If mp4 assembly OOM-kills the container (known issue, see bottom): prioritize
  retrying for the video; only fall back to a last-frame PNG
  (`tools.visualize.render_navigation(cfg, out, snapshot_step=<last step>)`)
  if retries keep failing.

## Field-verified detection — what was built

The "next step #1" of the previous handoff, per user spec: τ=0.47-gated LLMDet
(no LocateAnything cascade); for every frame where LLMDet fires, unproject the
valid-depth pixels inside the box to 3D, query the trained CLIPSeg relevance
field (`gamma`), pool, and count the frame as a detection only if the pooled
score clears a threshold. A rejected frame is a full non-detection (no goal, no
confidence, no latch-streak).

Changes (all uncommitted):
- `src/perception/perception_stack.py` — `PerceptionStack.field_score_in_box(depth,
  c2w, intrinsics, box, top_frac, min_points, pool)`: pooled field relevance over
  the box's valid-depth pixels. `pool="topk"` = mean of the top `top_frac`
  fraction of point scores; `pool="max"` = single best point.
- `main.py` — the gate lives in `detect_classify_latch` (after `detector.detect`,
  before `classify_detection`); prints accept/reject per frame; the pooled score
  is logged as `field_score` in every `traj_log.jsonl` replan line (also when
  the gate is off-threshold — it logs whatever was computed).
- `src/config.py` — `detection:` knobs, all defaulting to OFF/neutral:
  `field_verify` (False), `field_verify_threshold` (0.30), `field_verify_pool`
  ("topk"), `field_verify_top_frac` (0.10), `field_verify_min_points` (20 —
  fewer valid-depth pixels ⇒ unverifiable ⇒ rejected).
- `config/evaluation_configs/detector_fieldverify_*.yaml` — overlays for
  `eval_scene.py run --agent-config` (only the keys present override):
  `calib` (τ0.47, cascade off, gate log-only at threshold 0.0 — doubles as the
  no-gate reference), `thr010/030/050/070` (topk), `max050/max080` (max-pool).

Verified by `output/scratch_fieldverify/smoke_test.py`: cold/untrained field
reads ~0.018 in-box (the `COLD_START_BIAS` sigmoid shift), degenerate boxes →
None, a briefly-trained region → 1.0; real episode logs `field_score` on every
fired frame.

## Calibration findings (11 full episodes, gate log-only → `output/fieldverify_calib/`)

SR 0.545 / SPL 0.400; **all 5 failures were false-positive self-stops** — the
failure mode the gate targets. Latch-level analysis
(`output/scratch_fieldverify/analyze_latch.py`):
- FP latches often score HIGH on the field (bed__Corozal 0.83, tv__Corozal 0.77):
  CLIPSeg itself corroborates geometric look-alikes — no threshold fixes those.
- Some correct latches happen COLD (potted plant__Corozal 0.02, toilet__Collierville
  0.05): first sighting precedes field training, so a static per-frame ROC is
  flat (`analyze_calib.py`). The gate's value is **dynamic**: rejecting a cold
  latch keeps the agent searching while multi-view CLIPSeg evidence accumulates
  (couch__Collierville re-latched later at 0.88).

## Sweep — COMPLETE (50-episode subset: `--max-per-combo 2`, all 5 Gibson-val scenes)

| sweep point | out dir | SR | SPL | FP-stops | timeouts |
|---|---|---|---|---|---|
| no gate (thr 0.0) | `output/fieldverify_sweep_thr000` | 0.500 | 0.335 | 22 | 3 |
| topk ≥ 0.10 | `output/fieldverify_sweep_thr010` | 0.500 | 0.257 | 21 | 4 |
| topk ≥ 0.30 | `output/fieldverify_sweep_thr030` | 0.560 | 0.281 | 19 | 3 |
| **topk ≥ 0.50 (WINNER)** | `output/fieldverify_sweep_thr050` | **0.620** | **0.264** | 13 | 6 |
| topk ≥ 0.70 | `output/fieldverify_sweep_thr070` | 0.620 | 0.232 | — | — |
| max ≥ 0.50 | `output/fieldverify_sweep_max050` | 0.600 | 0.272 | — | — |
| max ≥ 0.80 | `output/fieldverify_sweep_max080` | 0.620 | 0.255 | — | — |

Read: the gate monotonically converts FP-stops into successes (22→13 at topk
0.50) at only +3 timeouts; SR plateaus at 0.620 (thr070/max080 match it with
worse SPL), so **topk ≥ 0.50 wins** (SR-primary, SPL tiebreak). SPL falls vs
no-gate because successful paths lengthen (mean 6.0 m → 13.4 m) — latching
waits for map corroboration. Max pooling ties on SR, never beats topk.
Per-episode flips thr000→thr050: 11 fixed (9 FP-stops + 2 timeouts → success),
5 broken (3 → timeout — two of them reach the goal but never pass the gate —
and 2 → new FP-stops). 13 FP-stops survive the gate (CLIPSeg corroborates the
look-alike, e.g. bed__Corozal). Sweep runners:
`output/scratch_fieldverify/run_sweep*.sh`.

## Final results (chain completed 2026-07-11 16:58)

**Inspection videos** (all rendered first-attempt, no OOM) →
`output/fieldverify_videos/ep/<id>/nav_history.mp4`:
tv__Collierville__ep4 + toilet__Wiconisco__ep14 (gate fixed these FP-stops),
toilet__Collierville__ep0 (user-requested), couch__Collierville__ep6 (gate
broke it: reaches the couch, never latches, timeout), bed__Corozal__ep0
(gate-resistant FP).

**Full 250-episode run** (`output/fieldverify_full_thr050`, seeded with the 50
identical-config sweep records):

| system (250 eps, gibson v1.1_sub10) | SR | SPL | FP-stops | timeouts |
|---|---|---|---|---|
| jul3 baseline (MaskCLIP field + cascade τ0.45) | 0.604 | 0.336 | — | — |
| jul4 baseline (same) | 0.616 | 0.348 | 86 | 10 |
| **CLIPSeg field + τ0.47 LLMDet + gate topk≥0.50** | **0.572** | **0.262** | 79 | 28 |

Read carefully — two opposing effects:
- **Within the new stack the gate clearly helps** (subset: 0.500 → 0.620 SR),
  and the full run's FP-stops are lower than the baseline's (79 vs 86).
- **But the new stack overall still trails the old cascade+MaskCLIP baseline**
  (−4 SR points, −0.086 SPL): timeouts nearly tripled (10 → 28) and successful
  paths are longer (delayed latching), which is where both the SR gap and the
  SPL gap live. The confound (field swap + detector swap + gate in one diff)
  means the gap can't be attributed to the gate — the subset A/B says the gate
  component is strongly positive within the CLIPSeg stack.

Obvious next experiments (NOT run): (a) cascade detector + field gate combined
(the gate replaced the cascade this campaign; they're complementary — the
cascade kills uncorrelated-FP frames, the gate kills map-uncorroborated ones);
(b) relax `stop_distance_m`/arrival to cut the new timeouts; (c) re-tune
`detected_persistence` under the gate (gated frames are already filtered, so
persistence 2 may be redundant and is costing time-to-latch.

## OVON failure-mode analysis (side quest, 2026-07-10)

Per user request, every failed run of `output/ovon_val_seen_full` (79 eps, SR
0.114) was classified; all 48 self-stopped failures were **visually
adjudicated** by re-rendering each run's final pose as a 4-view panorama
(`output/scratch_fieldverify/finalviews/`, generator `render_final_views.py`).
Result (artifact: https://claude.ai/code/artifact/10835fdc-3ca3-44da-b0db-02ea62e89847
— flowchart generator `make_flowchart.py`): of 70 failures, 19 (27%) actually
found the right object (7 near-miss 1–2 m, 3 right-object-stopped-too-far,
9 correct-but-not-in-OVON-goals annotation gaps) → effective SR ≈ 35%, not
11.4%. True detector FPs: 22 (31%, not 59% as distance-only suggests). Vague
categories: 7. Timeouts: 4 stuck (<2 m travel), 8 latched-never-arrived,
1 sightings-no-latch, 9 never-detected (recall). Implications: stop-distance
calibration is the cheapest win; the FP gate addresses the largest bucket;
recall needs a different lever.

All eval runs use: `python benchmarks/eval_scene.py run --dataset
benchmarks/episodes/gibson/v1.1_sub10/val --scenes-root benchmarks/gibson_scenes --gpu 3
--agent-config config/evaluation_configs/detector_fieldverify_<point>.yaml
[--max-per-combo 2] --out output/<name>` — continuous actions (no `--discrete`),
matching the jul3/jul4 baselines. Success = self-stop ≤ 1.0 m geodesic
(SemExp rect goals from `val_info.pbz2`).

Analysis helpers (in `output/scratch_fieldverify/`, gitignored): `analyze_calib.py`
(per-frame near/off-goal score distributions + TP-ret/FP-rej table),
`analyze_latch.py` (field scores of the latch-causing detections). Both take an
eval `--out` dir.

---

# Background: CLIPSeg feature-field swap (previous session, verified)

The feature field used to learn a 512-D MaskCLIP embedding at every 3D point,
dotted against a text embedding later. CLIPSeg gives a much cleaner per-pixel
relevance signal at the cost of being query-conditioned — accepted because
`target_query` is fixed per run. The field's job became: aggregate CLIPSeg's
per-pixel score across every observed viewpoint into one persistent,
multi-view-consistent scalar relevance map. This map is exactly what the
field-verify gate above queries.

What changed: `src/perception/semantics.py` (new `CLIPSegSemantics`; MaskCLIP
kept but unwired), `perception_stack.py` (wiring; both similarity call sites
use the scalar directly), `grid.py` (`SimilarityGrid` needs no text embedding),
`config.py` + `config/config.yaml` (`hash_feature_dim` 512→1 in BOTH places —
the YAML silently overrides), and `featurefield.py` (the substance):

1. `safe_normalize()` no-ops at `feature_dim==1` (a scalar isn't a direction).
2. Fixed a latent bug: `torch.nan_to_num(loss, ...)` result was discarded.
3. `v`/`beta` epsilon floors raised 1e-6 → 1e-2 (near-uniform scalar batches
   drove `omega` → 0 and blew up the NLL's log).
4. `alpha` capped at 100 (unbounded evidence with omega at its floor explodes
   the `-alpha/omega` gradient).
5. Root cause of the NaNs: unbounded `gamma` drift → `gamma = sigmoid(raw)` at
   `feature_dim==1` (CLIPSeg targets are provably in [0,1]).
6. `COLD_START_BIAS = 4.0`: `gamma = sigmoid(raw - 4.0)` so untrained regions
   read ~0.018, not ~1.0 ("everything is the target"). Holds even for regions
   that stay untrained after convergence elsewhere.

Verified: standalone CLIPSeg peak inside GT box; synthetic scalar-field
convergence (MAE 0.02–0.07); 1000–2000-step near-constant-batch stress test
with zero NaN; end-to-end `main.run()` on Goffs + a real OVON episode, real
spatial structure in `bev_similarity_*.npy`, cold regions uniformly low.

## Known issues / open items (carried over)

- **Video rendering (mp4 assembly) OOM-crashed the container twice** (exit
  137/SIGKILL) during the full multi-frame `render_navigation` at the end of
  `main.run(..., save_video=True)` — infra contention on this shared host, not
  a code bug. Workaround: monkeypatch `main.render_navigation = lambda *a, **kw:
  None` and render separately afterward (single-frame PNG has no observed OOM
  risk). For this campaign: retry for the real mp4 first (user instruction).
- **The OVON cabinet episode wedging** (`cabinet__5cdEh9F2hJL__ep3979`: "NO SAFE
  ROLLOUTS" near spawn, no confirmed detection in 6000 steps under the CLIPSeg
  field, unlike the earlier MaskCLIP run that latched ~step 2800) is still not
  root-caused. Candidates: collision/navmesh (field-independent) vs. changed
  IG/exploration near the start under the CLIPSeg field.
- `output/scratch_fieldverify/` holds this campaign's scratch scripts/logs
  (gitignored via `output/`); clean up when the campaign concludes. Don't
  `git add -A` blindly — this repo has been bitten by scratch files before.

