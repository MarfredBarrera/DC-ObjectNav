"""Sweep candidate `field_verify_threshold` values against real field_score
samples collected by a log-only calibration run (field_verify_threshold: 0.0,
see config/agent_configs/detector_distractors_field_calib.yaml).

field_score is logged in traj_log.jsonl on every replan the detector fires,
even when a gate would reject it (main.py:detect_classify_latch computes it
before the threshold check). There is no per-box ground truth, so each sample
is labeled TP/FP by a distance-to-goal proxy: the planner's `goal` cell that
same replan is the box's own projected world location whenever the detection
was investigated (Layer 1, bev_cell_from_box_center) — convert it to world
(x, z) via the run's grid_extent.json and compare to the episode's true goal.
Samples where the detection wasn't investigated (`det_conf == 0`, e.g. the
"too close" gate) are dropped: their `goal` cell may be a stale cache/argmax
unrelated to this box, so it's not a valid proxy.

Usage:
    python tools/calibrate_field_verify.py --out output/gibson_fieldverify_contrastive_calib
"""

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.eval_core import nearest_goal_xz


def collect_samples(out_dir: str, tp_radius: float):
    """[(field_score, label, dist_to_goal_m), ...] across every completed run
    under `out_dir`, plus the number of runs that contributed a sample."""
    samples = []
    n_runs = 0
    for rp in sorted(glob.glob(os.path.join(out_dir, "runs", "*.json"))):
        with open(rp) as f:
            rec = json.load(f)
        if rec.get("status") != "ok" or not rec.get("goals"):
            continue
        rid = rec.get("id") or os.path.splitext(os.path.basename(rp))[0]
        run_dir = os.path.join(out_dir, "ep", rid)
        traj_path = os.path.join(run_dir, "traj_log.jsonl")
        extent_path = os.path.join(run_dir, "grid_extent.json")
        if not (os.path.exists(traj_path) and os.path.exists(extent_path)):
            continue
        with open(extent_path) as f:
            extent = json.load(f)
        min_x, min_z, res = extent["min_x"], extent["min_z"], extent["voxel_resolution"]

        run_had_sample = False
        with open(traj_path) as f:
            for line in f:
                d = json.loads(line)
                fs = d.get("field_score")
                goal_cell = d.get("goal")
                if fs is None or goal_cell is None:
                    continue
                if not d.get("det_conf"):
                    # Detection wasn't investigated this step (e.g. "too
                    # close") — goal_cell may be an unrelated stale cache/
                    # argmax cell, not this box's location. Drop.
                    continue
                x = min_x + goal_cell[1] * res
                z = min_z + goal_cell[0] * res
                dist = min(math.hypot(x - gx, z - gz)
                           for gx, gz in (nearest_goal_xz(g, (x, 0.0, z))
                                          for g in rec["goals"]))
                label = "TP" if dist <= tp_radius else "FP"
                samples.append((float(fs), label, dist))
                run_had_sample = True
        n_runs += run_had_sample
    return samples, n_runs


def sweep(samples, thresholds):
    tp = [s for s, l, _ in samples if l == "TP"]
    fp = [s for s, l, _ in samples if l == "FP"]
    rows = []
    for t in thresholds:
        tp_ret = (sum(1 for s in tp if s >= t) / len(tp)) if tp else float("nan")
        fp_rej = (sum(1 for s in fp if s < t) / len(fp)) if fp else float("nan")
        rows.append((t, tp_ret, fp_rej))
    return rows, len(tp), len(fp)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="Calibration run's output dir")
    ap.add_argument("--tp-radius", type=float, default=1.5,
                    help="Distance-to-goal (m) below which a sample is "
                         "labeled TP (default 1.5, matches cfg.stop_distance_m)")
    ap.add_argument("--thresholds", type=float, nargs="*",
                    default=[round(0.05 * i, 2) for i in range(20)],
                    help="Candidate field_verify_threshold values to sweep "
                         "(default 0.00..0.95 step 0.05)")
    args = ap.parse_args()

    samples, n_runs = collect_samples(args.out, args.tp_radius)
    if not samples:
        raise SystemExit(f"no usable field_score samples found under {args.out} "
                         "(check the run used field_verify_threshold: 0.0 and "
                         "completed successfully)")
    n_tp = sum(1 for _, l, _ in samples if l == "TP")
    n_fp = len(samples) - n_tp
    print(f"[calibrate] {n_runs} run(s), {len(samples)} field_score sample(s) "
          f"({n_tp} TP-labeled, {n_fp} FP-labeled, tp_radius={args.tp_radius}m)")

    rows, n_tp2, n_fp2 = sweep(samples, sorted(args.thresholds))
    print(f"\n{'threshold':>9} {'TP-ret':>8} {'FP-rej':>8}")
    print("-" * 28)
    for t, tp_ret, fp_rej in rows:
        tr = f"{tp_ret:>8.2%}" if not math.isnan(tp_ret) else f"{'n/a':>8}"
        fr = f"{fp_rej:>8.2%}" if not math.isnan(fp_rej) else f"{'n/a':>8}"
        print(f"{t:>9.2f} {tr} {fr}")

    print()
    for target in (0.90, 0.95, 0.99):
        candidates = [r for r in rows if not math.isnan(r[2])]
        if not candidates:
            continue
        best = min(candidates, key=lambda r: abs(r[2] - target))
        print(f"~{target:.0%} FP-rejection -> threshold {best[0]:.2f} "
              f"(TP-ret {best[1]:.2%}, FP-rej {best[2]:.2%})")


if __name__ == "__main__":
    main()
