"""ObjectNav evaluation — three transparent stages.

A minimal run needs just three flags:
    python benchmarks/eval_scene.py run --benchmark gibson \\
        --agent_config config/agent_configs/detector_distractors_field.yaml --gpu 3

  --benchmark    WHICH EPISODES — a named entry in
                 config/evaluation_configs/benchmarks.yaml (gibson, ovon, ...):
                 the episode source (standard Habitat ObjectNav dataset or a
                 hand-written scenarios sweep) plus any protocol embodiment
                 overlay (e.g. the OVON Stretch camera).
  --agent_config THE DETECTOR ARM — a Config overlay yaml under
                 config/agent_configs/ (or a bare name found anywhere under
                 config/; same schema as config/config.yaml, only the keys
                 present override). Repeatable; stacks after the benchmark's
                 own embodiment overlay.
  --gpu          device index (default 0).

  --experiment   OPTIONAL — everything else (discrete action mode, subset
                 caps, scoring, output dir), bundled into one reusable yaml
                 under config/evaluation_configs/experiments/. Rarely needed
                 for a one-off run.
  --video / --rerun / --only / --verdicts / --out / --refresh-bev
                 other per-invocation choices that never define an experiment.

The pipeline separates *evidence* from *judgment*:

    run     execute episodes; save a raw record + evidence bundle per run
            (final pose, geodesics, BEV maps, trajectory video). Bakes in NO
            verdict — the metric's computed success is only a suggestion.

    review  (re)generate verdicts.yaml: one entry per run, pre-filled with the
            auto-suggestion and the evidence inline as comments (final dist,
            steps, path, links to ep/<id>/nav_history.mp4 + bev_final.png). You
            open one file, inspect each run, and set its status.

    report  aggregate SR / SPL from the records + your verdicts.

verdicts.yaml is the single, authoritative human-judgment file (status per run:
auto / success / fail / exclude). It replaces the old excluded.txt + overrides.txt.

Typical loop (inside the docker container; --out defaults to
output/<benchmark>_<agent_config>):
    python benchmarks/eval_scene.py run    --benchmark gibson --agent_config config/agent_configs/detector_distractors_field.yaml --gpu 3
    python benchmarks/eval_scene.py review --benchmark gibson --agent_config config/agent_configs/detector_distractors_field.yaml
    # ...edit the out dir's verdicts.yaml after watching the evidence...
    python benchmarks/eval_scene.py report --benchmark gibson --agent_config config/agent_configs/detector_distractors_field.yaml
Re-scoring an existing directory needs no configs at all:
    python benchmarks/eval_scene.py report --out output/gibson_val_CLIPSEG_jul12

Layout under <out>/:
    runs/<id>.json     raw per-run record (immutable evidence index)
    ep/<id>/           per-run evidence (BEV maps, rgbs, traj_log, video, bev_final.png)
    verdicts.yaml      human adjudication (single source of judgment)
    results.json       report output
"""

import argparse
import gc
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import benchmarks.eval_core as core


# ── path resolution ───────────────────────────────────────────────────────────

def resolve_verdicts_path(args):
    """Verdicts file to read/write. Default <out>/verdicts.yaml. `--verdicts`
    overrides: a bare filename resolves under <out> (so you can keep several
    curations like verdicts_jul1.yaml side by side); a path with a directory is
    used as given (point at another run's verdicts)."""
    if getattr(args, "verdicts", None) is None:
        return os.path.join(args.out, "verdicts.yaml")
    v = args.verdicts
    return v if os.path.dirname(v) else os.path.join(args.out, v)


def resolve_results_path(args, verdicts_path):
    """Where `report` writes results.json. Default keeps parity with the chosen
    verdicts file so different curations don't clobber each other:
      verdicts.yaml            -> results.json
      verdicts_jul1.yaml       -> results_jul1.json
    `--results` (bare name under <out>, or a full path) overrides."""
    r = getattr(args, "results", None)
    if r is not None:
        return r if os.path.dirname(r) else os.path.join(args.out, r)
    stem = os.path.splitext(os.path.basename(verdicts_path))[0]
    suffix = stem[len("verdicts"):].lstrip("_-") if stem.startswith("verdicts") else stem
    name = "results.json" if not suffix else f"results_{suffix}.json"
    return os.path.join(args.out, name)


# ── benchmark + experiment resolution ────────────────────────────────────────
#
# A minimal eval needs two named configs; --experiment is optional:
#   --benchmark    WHICH EPISODES: a named entry in benchmarks.yaml (episode
#                  source + protocol embodiment overlay, e.g. gibson / ovon).
#   --agent_config THE DETECTOR ARM: a Config overlay yaml under
#                  config/agent_configs/ (repeatable).
#   --experiment   optional: EVERYTHING ELSE, in one yaml under experiments/ —
#                  action mode, subset caps, scoring options, output dir.
#   --gpu / --video / --rerun / --only / --verdicts / --out
#                  session choices (hardware, evidence volume, re-scoring)
#                  that never define an experiment.

CONFIG_ROOT = "config"
EVAL_CONFIG_DIR = os.path.join(CONFIG_ROOT, "evaluation_configs")
BENCHMARKS_FILE = os.path.join(EVAL_CONFIG_DIR, "benchmarks.yaml")
EXPERIMENT_DIR = os.path.join(EVAL_CONFIG_DIR, "experiments")

# Keys an experiment yaml may set. A benchmark entry may set the same minus
# `benchmark`/`out` (it IS the episode source; it doesn't own the campaign).
EXPERIMENT_KEYS = {
    "benchmark", "scenarios", "dataset", "scenes_root", "categories",
    "scenes", "max_per_combo", "viewpoints", "radius", "out", "discrete",
    "config", "agent_config",
}
BENCHMARK_KEYS = EXPERIMENT_KEYS - {"benchmark", "out"}


def _load_yaml_mapping(path: str, what: str) -> dict:
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{what} {path} must be a YAML mapping")
    return data


def _check_keys(data: dict, allowed: set, what: str):
    unknown = {k for k in data if k.replace("-", "_") not in allowed}
    if unknown:
        raise SystemExit(f"{what}: unknown key(s) {sorted(unknown)} "
                         f"(allowed: {', '.join(sorted(allowed))})")


def load_benchmark(name: str) -> dict:
    """Named episode source from benchmarks.yaml (dataset/scenarios paths,
    scenes root, protocol embodiment overlay, protocol scoring defaults)."""
    benchmarks = _load_yaml_mapping(BENCHMARKS_FILE, "benchmarks file")
    if name not in benchmarks:
        raise SystemExit(f"unknown benchmark {name!r} "
                         f"(available: {', '.join(sorted(benchmarks))} — "
                         f"defined in {BENCHMARKS_FILE})")
    entry = benchmarks[name] or {}
    _check_keys(entry, BENCHMARK_KEYS, f"benchmark {name!r}")
    return entry


def resolve_experiment_path(name: str) -> str:
    """A real path is used as-is; a bare name resolves to
    config/evaluation_configs/experiments/<name>.yaml."""
    if os.path.exists(name):
        return name
    fname = name if name.endswith((".yaml", ".yml")) else name + ".yaml"
    path = os.path.join(EXPERIMENT_DIR, fname)
    if os.path.exists(path):
        return path
    avail = sorted(os.path.splitext(f)[0] for f in os.listdir(EXPERIMENT_DIR)
                   if f.endswith((".yaml", ".yml"))) if os.path.isdir(EXPERIMENT_DIR) else []
    raise SystemExit(f"experiment not found: {name!r} "
                     f"(available: {', '.join(avail) or 'none'})")


def resolve_overlay_path(name: str) -> str:
    """agent_config value: a real path is used as-is; a bare name is searched
    recursively under config/ (config/agent_configs/ holds the detector/sensor
    overlays; must match exactly one file so overlays can be referenced
    without their directory)."""
    if os.path.exists(name):
        return name
    fname = name if name.endswith((".yaml", ".yml")) else name + ".yaml"
    matches = sorted(glob.glob(os.path.join(CONFIG_ROOT, "**", fname),
                               recursive=True))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(f"agent-config not found: {name!r} "
                         f"(no {fname} under {CONFIG_ROOT}/)")
    raise SystemExit(f"agent-config name {name!r} is ambiguous: {matches}")


def _as_list(v) -> list:
    if v is None:
        return []
    return [v] if isinstance(v, str) else list(v)


def resolve_settings(args):
    """Merge experiment yaml + benchmark entry + CLI into the flat attrs the
    stages consume. Precedence per key: CLI > experiment > benchmark >
    built-in default. Overlays stack instead (benchmark embodiment first,
    then the experiment's, then CLI --agent_config) so the most specific
    layer wins per config key."""
    exp, exp_name = {}, None
    if args.experiment:
        path = resolve_experiment_path(args.experiment)
        exp = _load_yaml_mapping(path, "experiment")
        exp = {k.replace("-", "_"): v for k, v in exp.items()}
        _check_keys(exp, EXPERIMENT_KEYS, f"experiment {path}")
        exp_name = os.path.splitext(os.path.basename(path))[0]
        print(f"[experiment] {path}")

    bench_name = args.benchmark or exp.get("benchmark")
    bench = load_benchmark(bench_name) if bench_name else {}
    bench = {k.replace("-", "_"): v for k, v in bench.items()}
    if bench_name:
        print(f"[benchmark] {bench_name}: "
              f"{bench.get('dataset') or bench.get('scenarios')}")

    def pick(key, default=None):
        return exp.get(key, bench.get(key, default))

    args.scenarios = pick("scenarios")
    args.dataset = pick("dataset")
    args.scenes_root = pick("scenes_root", "gibson_scenes")
    args.categories = pick("categories")
    args.scenes = pick("scenes")
    args.max_per_combo = pick("max_per_combo")
    args.viewpoints = bool(pick("viewpoints", False))
    args.radius = pick("radius")           # None → protocol default below
    args.discrete = bool(pick("discrete", False))

    # CLI --agent_config names, kept raw (pre-resolution) so a run without
    # --experiment still gets a distinguishing out-dir name below — otherwise
    # e.g. two different detector arms against the same --benchmark would
    # both default to output/<benchmark> and clobber each other.
    cli_agent_config = _as_list(getattr(args, "agent_config", None))

    if args.out is None:
        args.out = exp.get("out")
    if args.out is None:
        name_hint = exp_name or "_".join(
            os.path.splitext(os.path.basename(p))[0] for p in cli_agent_config)
        parts = [p for p in (bench_name, name_hint) if p]
        if parts:
            args.out = os.path.join("output", "_".join(parts))
    # review/report on a plain directory: --out alone is a valid spec.
    if args.out is None:
        raise SystemExit("no output dir: pass --benchmark (out defaults to "
                         "output/<benchmark>[_<agent_config>][_<experiment>]) "
                         "or an explicit --out")

    if hasattr(args, "gpu"):               # run-only settings
        args.gpu = "0" if args.gpu is None else str(args.gpu)
        args.config = exp.get("config", "config/config.yaml")
        args.video = bool(args.video)
        args.no_evidence = bool(args.no_evidence)
        overlays = (_as_list(bench.get("agent_config"))
                    + _as_list(exp.get("agent_config"))
                    + cli_agent_config)
        args.agent_config = [resolve_overlay_path(p) for p in overlays]


# ── run ──────────────────────────────────────────────────────────────────────

def cmd_run(args, scn, runs):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from main import run as run_episode  # imports torch/habitat; after CUDA env
    from src.config import Config

    evidence = "off" if args.no_evidence else ("minimal + video" if args.video else "minimal")
    overlays = ", ".join(args.agent_config) if args.agent_config else "none"
    print(f"[run] scene={os.path.basename(scn['scene'])} | {len(runs)} run(s) "
          f"({scn['runs_per_combo']}x per combo) | discrete={args.discrete} | "
          f"evidence={evidence} | overlays={overlays}")

    for i, r in enumerate(runs):
        rid = r["id"]
        rp = core.run_record_path(args.out, rid)
        if os.path.exists(rp) and not args.rerun:
            print(f"[run] ({i+1}/{len(runs)}) {rid}: result exists, skipping "
                  f"(use --rerun to redo)")
            continue

        print(f"\n[run] === ({i+1}/{len(runs)}) {rid} | "
              f"'{r['query']}' @ {r['start_name']} ===")

        cfg = Config(args.config)
        # Agent/sensor/detector profiles overlaid on the base config (e.g. the
        # HM3D-OVON Stretch camera), in order, applied before the per-run
        # overrides below so they can never clobber the episode's
        # scene/query/band. Already resolved to real paths in main().
        for overlay in args.agent_config:
            cfg.apply_yaml(overlay)
        # Dataset runs carry their own scene (episodes span scenes) and the BEV
        # height band for their floor; scenarios runs use the file-level scene
        # and the config's band.
        cfg.scene_path = r.get("scene") or scn["scene"]
        cfg.target_query = r["query"]
        if r.get("bev_band"):
            # main.run() grows cfg.grid_max_height to cover this band (and
            # re-shifts the band itself to the agent's actual post-snap
            # spawn height) once the sim exists to compute it.
            cfg.bev_height_min, cfg.bev_height_max = r["bev_band"]
        if args.discrete:
            cfg.discrete_actions = True

        # Evidence into ep/<id>/. Default is minimal (traj_log + final occupancy
        # map + the review bev_final.png); `--video` adds the full per-step map
        # history + the nav_history.mp4 (large). `--no-evidence` saves nothing
        # but traj_log + grid_extent.
        save = not args.no_evidence
        video = save and args.video
        run_dir = core.ep_dir(args.out, rid)
        if save:
            cfg.output_dir = run_dir
        viz_out = os.path.join(run_dir, "nav_history.mp4")

        record = {
            "id": rid, "combo": r["combo"], "repeat": r["repeat"],
            "target": r["target"], "start_name": r["start_name"],
            "start": r["start"], "query": r["query"],
            "goals": r["goals"], "requested_radius_m": r["success_radius_m"],
        }
        if r.get("episode_id") is not None:
            record["episode_id"] = r["episode_id"]
        try:
            metrics = run_episode(
                cfg, save_enabled=save, save_video=video, viz_output=viz_out,
                start_pos=r["start"], start_rotation=r.get("start_rotation"),
                goals=r["goals"],
                success_radius_m=r["success_radius_m"],
            )
            record.update(metrics)
            record["status"] = "ok"
            if save:
                record["evidence_dir"] = os.path.relpath(run_dir, args.out)
        except Exception as e:
            import traceback
            traceback.print_exc()
            record["status"] = "error"
            record["error"] = f"{type(e).__name__}: {e}"
            print(f"[run] {rid} FAILED ({record['error']}) — recorded, continuing")

        os.makedirs(os.path.dirname(rp), exist_ok=True)
        with open(rp, "w") as f:
            json.dump(record, f, indent=2)

        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Refresh verdicts skeleton + print a report so `run` leaves a usable state.
    cmd_review(args, scn, runs, render=not args.no_evidence)
    cmd_report(args, scn, runs)


# ── review ────────────────────────────────────────────────────────────────────

def cmd_review(args, scn, runs, render=True):
    records = core.load_records(args.out, runs)

    # Render the static final-BEV evidence for each completed run (best-effort).
    if render:
        for r in runs:
            rec = records.get(r["id"])
            if not rec or rec.get("status") != "ok":
                continue
            run_dir = core.ep_dir(args.out, r["id"])
            out_png = os.path.join(run_dir, "bev_final.png")
            if os.path.exists(out_png) and not args.refresh_bev:
                continue
            try:
                if core.render_final_bev(run_dir, rec, r["goals"], out_png):
                    print(f"[review] rendered {os.path.relpath(out_png, args.out)}")
            except Exception as e:
                print(f"[review] bev render failed for {r['id']}: {e}")

    verdicts_path = resolve_verdicts_path(args)
    existing = core.load_verdicts(verdicts_path)
    core.write_verdicts(verdicts_path, runs, records, existing)

    n_done = sum(1 for r in runs if r["id"] in records)
    n_set = sum(1 for v in existing.values() if v.get("status", "auto") != "auto")
    print(f"[review] {verdicts_path} | {n_done}/{len(runs)} run(s) completed | "
          f"{n_set} non-auto verdict(s) preserved")
    print(f"[review] open {os.path.basename(verdicts_path)}, inspect ep/<id>/ evidence, "
          f"set each status, then: benchmarks/eval_scene.py report ...")


# ── report ────────────────────────────────────────────────────────────────────

def cmd_report(args, scn, runs):
    verdicts_path = resolve_verdicts_path(args)
    out_path = resolve_results_path(args, verdicts_path)
    if not os.path.exists(verdicts_path):
        raise SystemExit(f"[report] verdicts file not found: {verdicts_path}\n"
                         f"         run `benchmarks/eval_scene.py review ...` first, or pass "
                         f"--verdicts <name|path>.")
    print(f"[report] verdicts: {verdicts_path}")
    records = core.load_records(args.out, runs)
    verdicts = core.load_verdicts(verdicts_path)
    summary = core.aggregate(runs, records, verdicts)

    summary = {
        "scene": scn["scene"], "discrete_actions": args.discrete,
        "runs_per_combo": scn["runs_per_combo"],
        "verdicts_file": verdicts_path, **summary,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Console table.
    w = max(26, *(len(r["id"]) for r in runs)) if runs else 26
    by_id = {row["id"]: row for row in summary["included"] + summary["excluded"]}
    print("\n" + "=" * (w + 40))
    print(f"{'run id':<{w}} {'verdict':>8} {'count':>6} {'succ':>5} {'spl':>6} {'fgeo':>6}")
    print("-" * (w + 40))
    for r in runs:
        row = by_id.get(r["id"])
        if row is None:
            print(f"{r['id']:<{w}} {'--':>8} {'MISSING':>6}")
            continue
        counted = "no" if row in summary["excluded"] else "yes"
        if counted == "no" and row["kind"] == "error":
            print(f"{r['id']:<{w}} {row['status']:>8} {'no':>6} {'ERR':>5}  "
                  f"{(row.get('error') or '')[:24]}")
            continue
        succ = "-" if row["success"] is None else str(row["success"])
        spl = "-" if row["spl"] is None else f"{row['spl']:.3f}"
        fgeo = row.get("final_geodesic")
        fgeo = f"{fgeo:.2f}" if isinstance(fgeo, (int, float)) else "n/a"
        print(f"{r['id']:<{w}} {row['status']:>8} {counted:>6} {succ:>5} {spl:>6} {fgeo:>6}")
    print("-" * (w + 40))

    if scn["runs_per_combo"] > 1 and summary["per_combo"]:
        print(f"per combo ({scn['runs_per_combo']} run(s) each):")
        for c in sorted(summary["per_combo"]):
            d = summary["per_combo"][c]
            print(f"  {c:<{w}} SR {d['successes']}/{d['n']}={d['sr']:.2f}  SPL {d['spl']:.3f}")
        print("-" * (w + 40))

    s = summary
    print(f"included {s['num_included']}  |  excluded {s['num_excluded']} "
          f"(errors {s['num_errored']})  |  missing {s['num_missing']}")
    print(f"SR  = {s['success_rate']:.3f}   ({s['successes']}/{s['num_included']} counted)")
    print(f"SPL = {s['spl']:.3f}")
    print(f"[report] wrote {out_path}")
    if s["num_missing"]:
        print(f"[report] not yet run: {', '.join(s['missing'])}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--benchmark", default=None,
                       help="Which episodes: a named entry in "
                            "config/evaluation_configs/benchmarks.yaml "
                            "(gibson, ovon, ...). Provides the episode source "
                            "+ any protocol embodiment overlay.")
        p.add_argument("--experiment", default=None,
                       help="Optional: bundle non-detector settings (discrete "
                            "action mode, max_per_combo/categories/scenes caps, "
                            "radius/viewpoints scoring, out dir) into one "
                            "reusable yaml: config/evaluation_configs/"
                            "experiments/<name>.yaml (or a path). Pick the "
                            "detector arm itself with --agent_config, not "
                            "here. CLI flags override its values.")
        p.add_argument("--agent_config", "--agent-config", dest="agent_config",
                       action="append", default=None,
                       help="Agent/sensor/detector profile YAML (same schema "
                            "as config/config.yaml, only the keys present "
                            "override) — the primary way to pick a detector "
                            "arm, e.g. --agent_config "
                            "config/agent_configs/detector_distractors_field.yaml. "
                            "Stacks after any benchmark/experiment overlay, and "
                            "names the default --out dir. Repeatable — overlays "
                            "apply in order (run only; review/report use it "
                            "only to reconstruct the default --out). A bare "
                            "name (e.g. detector_fieldverify_thr050) is found "
                            "anywhere under config/.")
        p.add_argument("--out", default=None,
                       help="Output directory (default output/<benchmark>_"
                            "<agent_config>[_<experiment>]). Alone it is a valid spec for "
                            "review/report: aggregate exactly the records in "
                            "<out>/runs/.")
        p.add_argument("--only", nargs="*", default=None, help="Restrict to these run ids")
        p.add_argument("--verdicts", default=None,
                       help="Verdicts file to use (default <out>/verdicts.yaml). "
                            "A bare name resolves under <out> (keep several curations "
                            "side by side); a path is used as-is (point at another run).")

    p_run = sub.add_parser("run", help="Execute episodes + save evidence")
    common(p_run)
    p_run.add_argument("--gpu", default=None, help="GPU device index (default 0)")
    p_run.add_argument("--video", action="store_true", default=False,
                       help="Also save the full per-step BEV map + RGB history and "
                            "render nav_history.mp4 (large; default keeps only the "
                            "final occupancy map + bev_final.png)")
    p_run.add_argument("--no-evidence", action="store_true", default=False,
                       help="Save nothing but traj_log + grid_extent (no final "
                            "occupancy map, no inspection bundle)")
    p_run.add_argument("--rerun", action="store_true", default=False,
                       help="Re-simulate runs even if a saved result exists")
    p_run.add_argument("--refresh-bev", action="store_true", default=False,
                       help="Re-render bev_final.png even if it exists")

    p_rev = sub.add_parser("review", help="(Re)generate verdicts.yaml + BEV evidence")
    common(p_rev)
    p_rev.add_argument("--refresh-bev", action="store_true", default=False,
                       help="Re-render bev_final.png even if it exists")

    p_rep = sub.add_parser("report", help="Aggregate SR/SPL from verdicts")
    common(p_rep)
    p_rep.add_argument("--results", default=None,
                       help="Results JSON to write (default results.json, or "
                            "results_<suffix>.json to match a non-default --verdicts). "
                            "A bare name resolves under <out>.")

    args = parser.parse_args()

    # Fold the experiment yaml + benchmark entry into the flat settings the
    # stages consume (CLI > experiment > benchmark > default per key).
    resolve_settings(args)

    # `run` must know what to execute: a benchmark (or an experiment carrying
    # a dataset/scenarios source). `review` and `report` operate on data
    # already on disk: with a source they use that run set (and warn about
    # records it doesn't cover); with only --out they aggregate exactly the
    # records present in <out>/runs/ — immune to the source having since
    # changed (renamed combos, restricted targets).
    if args.scenarios and args.dataset:
        raise SystemExit("config error: both scenarios and dataset ended up "
                         "set (benchmark + experiment overlap?) — they are "
                         "mutually exclusive")
    if args.command == "run" and not (args.scenarios or args.dataset):
        parser.error("run needs an episode source: --benchmark <name> "
                     "(see config/evaluation_configs/benchmarks.yaml) or an "
                     "--experiment whose yaml sets benchmark:/dataset:/scenarios:")
    if args.radius is None:
        args.radius = 0.1 if args.viewpoints else 1.0

    if args.scenarios:
        scn = core.load_scenarios(args.scenarios)
        scn["scene"] = core.resolve_scene(scn["scene"])
        runs = core.build_runs(scn)
    elif args.dataset:
        scn, runs = core.load_objectnav_dataset(
            args.dataset, scenes_root=args.scenes_root,
            categories=args.categories, scenes=args.scenes,
            max_per_combo=args.max_per_combo, success_radius_m=args.radius,
            use_viewpoints=args.viewpoints)
        print(f"[dataset] {len(runs)} runnable episode(s) from {args.dataset}")
        if not runs:
            raise SystemExit("no runnable episodes (scene/category filters, or "
                             "scene .glb files missing under --scenes-root).")
    else:
        scn, runs = core.from_records(args.out)
        print(f"[info] no --scenarios/--dataset: using {len(runs)} record(s) "
              f"found in {args.out}/runs/")
        if not runs:
            raise SystemExit(f"no run records in {args.out}/runs/ — nothing to do.")

    if (args.scenarios or args.dataset) and args.command != "run":
        orphans = core.orphan_record_ids(args.out, runs)
        if orphans:
            shown = ", ".join(orphans[:8]) + (" ..." if len(orphans) > 8 else "")
            print(f"[warn] {len(orphans)} record(s) in {args.out}/runs/ are NOT "
                  f"covered by the given source and will be IGNORED: {shown}")
            print("[warn] omit --scenarios/--dataset to report everything on disk.")

    if args.only:
        keep = set(args.only)
        unknown = keep - {r["id"] for r in runs}
        if unknown:
            raise SystemExit(f"--only references unknown run ids: {sorted(unknown)}")
        runs = [r for r in runs if r["id"] in keep]

    os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)

    if args.command == "run":
        cmd_run(args, scn, runs)
    elif args.command == "review":
        cmd_review(args, scn, runs)
    elif args.command == "report":
        cmd_report(args, scn, runs)


if __name__ == "__main__":
    main()
