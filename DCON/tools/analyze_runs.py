"""Consolidated eval-run analysis: progress, cross-run comparison, failure attribution.

Replaces the one-off scratch scripts (common_compare / three_way / fp_rate_compare /
latch_analysis / failure_attribution / stuck_vs_orbit / frozen_diagnosis — archived,
see git history / the 2026-07-23 scratch tarball) with one torch-free CLI that reads
only what `benchmarks/eval_scene.py run` leaves on disk:

    <outdir>/runs/<id>.json          — per-episode record (status, success, spl, goals)
    <outdir>/ep/<id>/traj_log.jsonl  — per-replan trajectory (pos, mode, goal)
    <outdir>/ep/<id>/grid_extent.json— BEV extent (goal-cell -> world conversion)
    <outdir>/verdicts.yaml           — optional human adjudication (auto|success|fail|exclude)

Usage:
    # Mid-run progress + SR/SPL of one (possibly still-running) eval dir:
    python tools/analyze_runs.py output/gibson_pairwise_maxj

    # Compare arms on their COMMON episode set (any number of dirs; first = reference):
    python tools/analyze_runs.py output/gibson_pairwise_maxj output/saved_data/objectnav_val_total_jul4

    # Name the arms explicitly (label=dir), full failure attribution included:
    python tools/analyze_runs.py maxj=output/gibson_pairwise_maxj jul4=output/saved_data/objectnav_val_total_jul4

Per run dir it reports: episodes on disk / ok / excluded, SR, SPL. With >= 2 dirs it
additionally intersects the ok-episode ids and reports, ON THE COMMON SET: SR/SPL per
arm (the apples-to-apples comparison), per-category and per-scene tables, and a
failure attribution per arm:

    FP    — latched onto the wrong object (final EXPLOIT goal > --near-thresh from
            every goal): the detection false-positive bucket. Also reported as a
            rate over all common episodes and over failures only.
    NEVER — never latched into EXPLOIT (or no evidence bundle): the search/recall
            bucket.
    NEAR  — latched near the true target but still failed: the navigation bucket,
            sub-classified from the trajectory tail (last --tail-frac of positions):
              frozen    — tail extent < 5 cm (premature stop / wedged)
              orbiting  — lots of path, little net displacement (circling the goal)
              approach  — still closing on the goal at timeout
              jitter    — none of the above cleanly

Run ids follow `{category}__{scene}__ep{k}` (dataset benchmarks) or
`{target}__{start}` (scenario sweeps); category/scene tables use the first two
`__` fields. Verdicts are applied when present: success/fail override the record,
exclude drops the episode (from that arm AND from the common set).
"""

import argparse
import collections
import glob
import json
import math
import os

try:
    import yaml
except ImportError:  # verdicts.yaml support degrades gracefully
    yaml = None


# ── loading ──────────────────────────────────────────────────────────────────

def load_arm(outdir):
    """Load one eval dir -> {id: record}, applying verdicts.yaml if present.

    Record fields used: success (bool, post-verdict), spl (float), goals (list),
    status. Episodes with status != 'ok' are counted but not returned; excluded
    episodes are dropped.
    """
    verdicts = {}
    vpath = os.path.join(outdir, "verdicts.yaml")
    if yaml is not None and os.path.exists(vpath):
        verdicts = yaml.safe_load(open(vpath)) or {}

    ok, bad, excluded = {}, 0, 0
    for f in sorted(glob.glob(os.path.join(outdir, "runs", "*.json"))):
        r = json.load(open(f))
        rid = r.get("id") or os.path.splitext(os.path.basename(f))[0]
        if r.get("status") != "ok":
            bad += 1
            continue
        v = (verdicts.get(rid) or {}).get("status", "auto")
        if v == "exclude":
            excluded += 1
            continue
        if v == "success":
            r["success"] = True
        elif v == "fail":
            r["success"] = False
            r["spl"] = 0.0
        ok[rid] = r
    return ok, bad, excluded


# ── latch / failure attribution ──────────────────────────────────────────────

def nearest_goal_dist(pt, goals):
    """x-z distance from `pt` to the nearest goal (point [x,y,z] or rect dict)."""
    x, z = pt
    best = float("inf")
    for g in goals or []:
        if isinstance(g, dict) and "rect" in g:
            xmin, zmin, xmax, zmax = g["rect"]
            d = math.hypot(x - min(max(x, xmin), xmax), z - min(max(z, zmin), zmax))
        else:
            p = g["point"] if isinstance(g, dict) else g
            d = math.hypot(x - p[0], z - p[2] if len(p) > 2 else z - p[1])
        best = min(best, d)
    return best


def load_traj(outdir, rid):
    path = os.path.join(outdir, "ep", rid, "traj_log.jsonl")
    if not os.path.exists(path):
        return None
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def latch_class(outdir, rid, record, near_thresh_m):
    """'latched_near_target' | 'latched_wrong_object' | 'never_latched' | 'no_evidence'.

    Judged from the FINAL EXPLOIT replan's goal cell, converted to world via
    grid_extent.json and compared against the record's goals.
    """
    ge_path = os.path.join(outdir, "ep", rid, "grid_extent.json")
    traj = load_traj(outdir, rid)
    if traj is None or not os.path.exists(ge_path):
        return "no_evidence"
    exploit = [e for e in traj if e.get("mode") == "EXPLOIT" and e.get("goal")]
    if not exploit:
        return "never_latched"
    ge = json.load(open(ge_path))
    z_idx, x_idx = exploit[-1]["goal"]
    pt = (ge["min_x"] + x_idx * ge["voxel_resolution"],
          ge["min_z"] + z_idx * ge["voxel_resolution"])
    d = nearest_goal_dist(pt, record.get("goals"))
    return "latched_near_target" if d <= near_thresh_m else "latched_wrong_object"


def tail_class(outdir, rid, tail_frac, min_window=10):
    """Sub-classify a NEAR failure from the trajectory tail geometry."""
    traj = load_traj(outdir, rid)
    if not traj:
        return "no_evidence"
    positions = [e["pos"] for e in traj if "pos" in e]
    n = len(positions)
    if n < 2:
        return "no_evidence"
    w = min(n, max(min_window, int(n * tail_frac)))
    tail = positions[-w:]
    seg = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
    path_len = sum(seg(tail[i], tail[i + 1]) for i in range(len(tail) - 1))
    net_disp = seg(tail[0], tail[-1])
    xs = [p[0] for p in tail]
    zs = [p[1] for p in tail]
    extent = math.hypot(max(xs) - min(xs), max(zs) - min(zs))
    if extent < 0.05:
        return "frozen"
    if net_disp < 0.5 * extent and path_len > 2.0 * extent:
        return "orbiting"
    if net_disp > 0.6 * extent:
        return "approach"
    return "jitter"


BUCKET = {
    "latched_wrong_object": "FP",
    "never_latched": "NEVER",
    "no_evidence": "NEVER",
    "latched_near_target": "NEAR",
}


# ── aggregation / printing ───────────────────────────────────────────────────

def sr_spl(records, ids):
    ids = [i for i in ids if i in records]
    if not ids:
        return 0, 0.0, 0.0
    succ = sum(1 for i in ids if records[i]["success"])
    spl = sum(records[i].get("spl", 0.0) for i in ids) / len(ids)
    return len(ids), succ / len(ids), spl


def print_group_table(title, key_fn, common, arms):
    groups = collections.defaultdict(list)
    for rid in common:
        groups[key_fn(rid)].append(rid)
    names = list(arms)
    header = f"{title:<16}{'n':>4}   " + "   ".join(f"{n[:14]:<16}" for n in names)
    print(header)
    for g in sorted(groups):
        ids = groups[g]
        parts = []
        for name in names:
            _, sr, spl = sr_spl(arms[name], ids)
            parts.append(f"{sr * 100:5.1f}% / {spl:.3f}")
        print(f"{g:<16}{len(ids):>4}   " + "   ".join(f"{p:<16}" for p in parts))
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="+",
                    help="eval out-dirs, optionally labeled as name=path")
    ap.add_argument("--near-thresh", type=float, default=1.0,
                    help="latched goal within this (m) of a true goal = NEAR (default 1.0)")
    ap.add_argument("--tail-frac", type=float, default=0.4,
                    help="trailing fraction of the trajectory used for NEAR sub-classification")
    ap.add_argument("--full", action="store_true",
                    help="attribute failures on each arm's FULL episode set too, not just the common set")
    args = ap.parse_args()

    arms, dirs, meta = {}, {}, {}
    for spec in args.dirs:
        name, _, path = spec.rpartition("=")
        name = name or os.path.basename(os.path.normpath(path))
        records, bad, excluded = load_arm(path)
        arms[name], dirs[name], meta[name] = records, path, (bad, excluded)

    # 1. Per-arm progress / full-set numbers (valid mid-run: reflects disk state).
    print(f"{'arm':<20}{'ok':>5}{'err':>5}{'excl':>6}{'SR':>9}{'SPL':>8}")
    for name, records in arms.items():
        bad, excluded = meta[name]
        n, sr, spl = sr_spl(records, records.keys())
        print(f"{name:<20}{n:>5}{bad:>5}{excluded:>6}{sr * 100:>8.1f}%{spl:>8.3f}")
    print()

    single = len(arms) == 1
    common = set.intersection(*(set(r) for r in arms.values()))
    if not single:
        print(f"common episodes: {len(common)}\n")
        if not common:
            return
        print(f"{'arm':<20}{'n':>5}{'SR':>9}{'SPL':>8}   (common set)")
        for name, records in arms.items():
            n, sr, spl = sr_spl(records, common)
            print(f"{name:<20}{n:>5}{sr * 100:>8.1f}%{spl:>8.3f}")
        print()
        print_group_table("category", lambda r: r.split("__")[0], common, arms)
        if all("__" in r for r in common):
            print_group_table("scene", lambda r: r.split("__")[1], common, arms)

    # 2. Failure attribution (per arm, on the common set unless --full/single).
    for name, records in arms.items():
        ids = records.keys() if (single or args.full) else common
        fails = [i for i in ids if i in records and not records[i]["success"]]
        counts = collections.Counter()
        near_tail = collections.Counter()
        fp_by_cat = collections.defaultdict(lambda: [0, 0])  # cat -> [fp, n]
        for rid in ids:
            fp_by_cat[rid.split("__")[0]][1] += 1
        for rid in fails:
            lc = latch_class(dirs[name], rid, records[rid], args.near_thresh)
            b = BUCKET[lc]
            counts[b] += 1
            if b == "FP":
                fp_by_cat[rid.split("__")[0]][0] += 1
            elif b == "NEAR":
                near_tail[tail_class(dirs[name], rid, args.tail_frac)] += 1
        n = len([i for i in ids if i in records])
        print(f"=== {name}: {len(fails)} failures / {n} episodes ===")
        if n:
            print(f"  FP (wrong-object latch): {counts['FP']:>3}"
                  f"  ({counts['FP'] / n * 100:.1f}% of all"
                  + (f", {counts['FP'] / len(fails) * 100:.1f}% of fails" if fails else "")
                  + ")")
            print(f"  NEVER (no latch):        {counts['NEVER']:>3}")
            print(f"  NEAR (nav failure):      {counts['NEAR']:>3}"
                  + (f"   tails: {dict(near_tail)}" if near_tail else ""))
            fp_cats = {c: f"{fp}/{tot}" for c, (fp, tot) in sorted(fp_by_cat.items()) if fp}
            if fp_cats:
                print(f"  FP by category: {fp_cats}")
        print()


if __name__ == "__main__":
    main()
