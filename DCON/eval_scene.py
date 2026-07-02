"""Per-scene ObjectNav evaluation — three transparent stages.

A single scene is swept over a cross product of target objects x agent start
positions (optionally repeated). The pipeline separates *evidence* from
*judgment*:

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

Typical loop (inside the docker container):
    python eval_scene.py run    --scenarios eval/scenarios_Goffs.yaml --out output/goffs
    python eval_scene.py review --scenarios eval/scenarios_Goffs.yaml --out output/goffs
    # ...edit output/goffs/verdicts.yaml after watching the evidence...
    python eval_scene.py report --scenarios eval/scenarios_Goffs.yaml --out output/goffs

Layout under <out>/:
    runs/<id>.json     raw per-run record (immutable evidence index)
    ep/<id>/           per-run evidence (BEV maps, rgbs, traj_log, video, bev_final.png)
    verdicts.yaml      human adjudication (single source of judgment)
    results.json       report output
"""

import argparse
import gc
import json
import os

import eval_core as core


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


# ── run ──────────────────────────────────────────────────────────────────────

def cmd_run(args, scn, runs):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from main import run as run_episode  # imports torch/habitat; after CUDA env
    from src.config import Config

    detector = args.detector or scn["detector"]
    print(f"[run] scene={os.path.basename(scn['scene'])} | {len(runs)} run(s) "
          f"({scn['runs_per_combo']}x per combo) | ig={args.ig_source} | "
          f"detector={detector or 'config default'} | discrete={args.discrete} | "
          f"evidence={'off' if args.no_evidence else 'on'}")

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
        cfg.scene_path = scn["scene"]
        cfg.target_query = r["query"]
        cfg.ig_source = args.ig_source
        if detector is not None:
            cfg.detector = detector
        if args.discrete:
            cfg.discrete_actions = True

        # Evidence: save BEV maps + render the trajectory video into ep/<id>/.
        save = not args.no_evidence
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
        try:
            metrics = run_episode(
                cfg, save_enabled=save, visualize=save, viz_output=viz_out,
                start_pos=r["start"], goals=r["goals"],
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
          f"set each status, then: eval_scene.py report ...")


# ── report ────────────────────────────────────────────────────────────────────

def cmd_report(args, scn, runs):
    verdicts_path = resolve_verdicts_path(args)
    out_path = resolve_results_path(args, verdicts_path)
    if not os.path.exists(verdicts_path):
        raise SystemExit(f"[report] verdicts file not found: {verdicts_path}\n"
                         f"         run `eval_scene.py review ...` first, or pass "
                         f"--verdicts <name|path>.")
    print(f"[report] verdicts: {verdicts_path}")
    records = core.load_records(args.out, runs)
    verdicts = core.load_verdicts(verdicts_path)
    summary = core.aggregate(runs, records, verdicts)

    detector = args.detector or scn["detector"]
    summary = {
        "scene": scn["scene"], "detector": detector,
        "ig_source": args.ig_source, "discrete_actions": args.discrete,
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
        p.add_argument("--scenarios", default=None,
                       help="Per-scene scenarios YAML/JSON. Required for `run`. "
                            "For `review`/`report`, omit it to aggregate exactly the "
                            "records in <out>/runs/ (independent of any scenarios file).")
        p.add_argument("--out", default="output/scene_eval", help="Output directory")
        p.add_argument("--ig-source", choices=["unseen", "epistemic"], default="epistemic",
                       help="Information-gain source for MPPI (recorded in results)")
        p.add_argument("--detector", default=None,
                       choices=["yolo", "coco_yolo", "hybrid", "grounding_dino",
                                "locate_anything", "llmdet"],
                       help="Override the detector (else scenarios `detector:` / config)")
        p.add_argument("--discrete", action="store_true", default=False,
                       help="Discrete Habitat ObjectNav action mode")
        p.add_argument("--only", nargs="*", default=None, help="Restrict to these run ids")
        p.add_argument("--verdicts", default=None,
                       help="Verdicts file to use (default <out>/verdicts.yaml). "
                            "A bare name resolves under <out> (keep several curations "
                            "side by side); a path is used as-is (point at another run).")

    p_run = sub.add_parser("run", help="Execute episodes + save evidence")
    common(p_run)
    p_run.add_argument("--gpu", default="0", help="GPU device index")
    p_run.add_argument("--config", default="config/config.yaml", help="Base config YAML")
    p_run.add_argument("--rerun", action="store_true", default=False,
                       help="Re-simulate runs even if a saved result exists")
    p_run.add_argument("--no-evidence", action="store_true", default=False,
                       help="Skip BEV-map saving + video render (faster, no inspection bundle)")
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

    # `run` must know what to execute, so it needs a scenarios file. `review`
    # and `report` operate on data already on disk: with --scenarios they use
    # that run set (and warn about records it doesn't cover); without it they
    # aggregate exactly the records present in <out>/runs/ — immune to the
    # scenarios file having since changed (renamed combos, restricted targets).
    if args.command == "run" and not args.scenarios:
        parser.error("run requires --scenarios")

    if args.scenarios:
        scn = core.load_scenarios(args.scenarios)
        scn["scene"] = core.resolve_scene(scn["scene"])
        runs = core.build_runs(scn)
        if args.command != "run":
            orphans = core.orphan_record_ids(args.out, runs)
            if orphans:
                shown = ", ".join(orphans[:8]) + (" ..." if len(orphans) > 8 else "")
                print(f"[warn] {len(orphans)} record(s) in {args.out}/runs/ are NOT "
                      f"covered by --scenarios and will be IGNORED: {shown}")
                print("[warn] omit --scenarios to report everything on disk instead.")
    else:
        scn, runs = core.from_records(args.out)
        print(f"[info] no --scenarios: using {len(runs)} record(s) found in "
              f"{args.out}/runs/")
        if not runs:
            raise SystemExit(f"no run records in {args.out}/runs/ — nothing to do.")

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
