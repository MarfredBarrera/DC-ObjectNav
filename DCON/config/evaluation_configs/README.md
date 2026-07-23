# Evaluation configs

A minimal `benchmarks/eval_scene.py` invocation needs two named configs plus
`--gpu`:

```bash
# inside the container, from /workspace/DCON
python benchmarks/eval_scene.py run    --benchmark gibson --agent_config config/agent_configs/detector_distractors_field.yaml --gpu 3 [--video]
python benchmarks/eval_scene.py review --benchmark gibson --agent_config config/agent_configs/detector_distractors_field.yaml
python benchmarks/eval_scene.py report --benchmark gibson --agent_config config/agent_configs/detector_distractors_field.yaml
# re-score any existing output dir with no configs at all:
python benchmarks/eval_scene.py report --out output/gibson_val_CLIPSEG_jul12
```

| choice | where | what it holds |
|---|---|---|
| **Which episodes** | `--benchmark <name>` → an entry in [`benchmarks.yaml`](benchmarks.yaml) | Episode source (`dataset:` split dir / `scenarios:` sweep), `scenes_root`, protocol embodiment overlay (e.g. OVON's Stretch camera), protocol scoring defaults. Entries: `gibson`, `ovon`, `ovon_unseen`, `ovon_synonyms`, `goffs`. |
| **The detector arm** | `--agent_config <path>` (repeatable) → yaml under [`../agent_configs/`](../agent_configs/) | A partial `Config` overlay (same schema as `config/config.yaml`; only the keys present override). Names the default `--out` dir. |
| **Everything else (optional)** | `--experiment <name>` → `experiments/<name>.yaml` | Only needed when you want to bundle non-detector settings into a reusable preset: `discrete:` action mode, `max_per_combo:`/`categories:`/`scenes:` caps, `radius:`/`viewpoints:` scoring, `out:`, `config:` (base config), or even a default `benchmark:`/`agent_config:`. Allowed keys are validated — a typo fails before anything runs. |
| **Session flags** | CLI only | `--gpu`, `--video` / `--no-evidence`, `--rerun`, `--only`, `--verdicts`, `--results`, `--refresh-bev`, `--out` (override). These never define an experiment. |

Precedence per key: **CLI > experiment > benchmark > built-in default**.
Overlays instead *stack* (benchmark embodiment first, then the experiment's,
then CLI `--agent_config`), so the most specific layer wins per config key.
`--out` defaults to `output/<benchmark>_<agent_config>` (or
`output/<benchmark>_<experiment>` when `--experiment` is given — it names the
run instead).

## File kinds

- **`benchmarks.yaml`** (this directory) — the named episode sources (above).
- **`../agent_configs/*.yaml`** — partial `Config` overlays (same schema as
  `config/config.yaml`; only the keys present override). `agent_*` = sensor/
  body profiles (e.g. `agent_ovon_stretch`); `detector_*` = detection-stack
  settings. Referenced by full path or bare name from `--agent_config` /
  a benchmark's `agent_config:`; bare names are found anywhere under
  `config/`.
- **`experiments/*.yaml`** (optional) — reusable presets for the non-detector
  settings (discrete mode, caps, scoring, out dir). Most one-off runs don't
  need one; freeze a configuration here once it's worth re-running by name.
- **`scenarios_<Scene>.yaml`** — hand-written per-scene sweeps (targets ×
  starts with ground-truth rect goals), referenced by a benchmark's
  `scenarios:` key.
- **`episodes.yaml`** — hand-annotated flat episode list for the older
  `benchmarks/evaluate.py`.

Benchmark episode datasets themselves live under `benchmarks/episodes/` (and
HM3D scenes under `benchmarks/scene_datasets/`), referenced by `benchmarks.yaml`.
