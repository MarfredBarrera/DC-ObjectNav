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
episodes/gibson/v1.1_sub10/val --scenes-root gibson_scenes --gpu 3
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
