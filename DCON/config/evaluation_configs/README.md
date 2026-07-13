# Evaluation configs

An eval invocation for `benchmarks/eval_scene.py` is the product of three
orthogonal choices. The CLI carries only the per-invocation ones; everything
that defines the experiment lives in a yaml here:

```bash
# inside the container, from /workspace/DCON
python benchmarks/eval_scene.py run    --benchmark gibson --experiment fieldverify_thr050 --gpu 3 [--video]
python benchmarks/eval_scene.py review --benchmark gibson --experiment fieldverify_thr050
python benchmarks/eval_scene.py report --benchmark gibson --experiment fieldverify_thr050
# re-score any existing output dir with no configs at all:
python benchmarks/eval_scene.py report --out output/gibson_val_CLIPSEG_jul12
```

| choice | where | what it holds |
|---|---|---|
| **Which episodes** | `--benchmark <name>` → an entry in [`benchmarks.yaml`](benchmarks.yaml) | Episode source (`dataset:` split dir / `scenarios:` sweep), `scenes_root`, protocol embodiment overlay (e.g. OVON's Stretch camera), protocol scoring defaults. Entries: `gibson`, `ovon`, `ovon_unseen`, `ovon_synonyms`, `goffs`. |
| **Everything else** | `--experiment <name>` → `experiments/<name>.yaml` | The experiment's identity: `agent_config:` overlay(s) (detector arm), `discrete:` action mode, `max_per_combo:`/`categories:`/`scenes:` caps, `radius:`/`viewpoints:` scoring, `out:`, `config:` (base config), or even a default `benchmark:`. Allowed keys are validated — a typo fails before anything runs. |
| **Session flags** | CLI only | `--gpu`, `--video` / `--no-evidence`, `--agent-config` (extra overlay tweaks, repeatable), `--rerun`, `--only`, `--verdicts`, `--results`, `--refresh-bev`, `--out` (override). These never define an experiment. |

Precedence per key: **CLI > experiment > benchmark > built-in default**.
Overlays instead *stack* (benchmark embodiment first, then the experiment's,
then CLI `--agent-config`), so the most specific layer wins per config key.
`--out` defaults to `output/<benchmark>_<experiment>`.

## File kinds in this directory

- **`benchmarks.yaml`** — the named episode sources (above).
- **`experiments/*.yaml`** — one file per experiment. Current:
  `fieldverify_thr050` (τ0.47 LLMDet + field gate 0.50, the sweep winner),
  `contrastive_field` (adds the contrastive CLIPSeg field target), and
  `*_smoke` variants (capped at 2 eps per category×scene for fast A/Bs).
  When a configuration proves itself, freeze it here.
- **`agent_*.yaml` / `detector_*.yaml`** — partial `Config` overlays (same
  schema as `config/config.yaml`; only the keys present override). `agent_*`
  = sensor/body profiles; `detector_*` = detection-stack settings. Referenced
  by bare name from benchmarks/experiments/`--agent-config`; found anywhere
  under this directory.
- **`scenarios_<Scene>.yaml`** — hand-written per-scene sweeps (targets ×
  starts with ground-truth rect goals), referenced by a benchmark's
  `scenarios:` key.
- **`episodes.yaml`** — hand-annotated flat episode list for the older
  `benchmarks/evaluate.py`.

Benchmark episode datasets themselves live under `episodes/` (and HM3D scenes
under `scene_datasets/`), referenced by `benchmarks.yaml`.
