"""Aggregate + diagnose `exploit_probe.py` output, and A/B two probe runs.

Classifies every approach by WHY it ended where it did, which is the thing the
raw arrival rate hides. The controller has two targets — the raw detection and
the snapped cell it steers at — so an approach can fail for opposite reasons:

  arrived        stopped itself (either arrival condition fired)
  parked         reached the snapped cell (<= nav_eps) but never stopped: the
                 policy did its job and the stop rule could not see it
  orbiting       never reached the snapped cell, closest approach stalled out:
                 the pointgoal is physically non-standable, so DD-PPO circles
  no_progress    ended no closer than it started (a genuine nav failure)

    python benchmarks/probe_report.py output/probe_v1 [output/probe_v2]
"""

import argparse
import json
import os
import sys

import numpy as np


def classify(r, nav_eps, stop_m):
    """Bucket one approach. `parked` is detected by a proxy, deliberately.

    The exact test is "distance to the snapped cell <= nav_eps", but the world
    position of that cell is only recorded from probe v2 on. The proxy needs
    nothing extra: an agent standing on the snapped cell is, by definition of
    the snap, exactly `snap_m` from the raw goal — so a stalled approach whose
    closest raw-goal approach equals its snap distance parked on its pointgoal.
    """
    if r["stopped"]:
        return "arrived"
    snap = r["snap_m"]
    if snap is not None and abs(r["dmin"] - snap) <= max(nav_eps, 0.25):
        return "parked"
    if r["dfinal"] >= r["d0"]:
        return "no_progress"
    return "orbiting"


def rect_dist(rect, x, z):
    """x-z distance from (x, z) to an axis-aligned rect, 0 inside it.

    Mirrors episode/scoring.nearest_goal_point: Gibson-val success is measured
    to the nearest point of the target's rectangular FOOTPRINT, not to its
    centroid, so anything scored against the centroid is pessimistic by up to
    half the object's diagonal — enough to flip a verdict on a bed or a couch.
    """
    x_min, z_min, x_max, z_max = rect
    return float(np.hypot(x - min(max(x, x_min), x_max),
                          z - min(max(z, z_min), z_max)))


def recover_rects(dataset, scenes_root, scenes, goals_per_combo):
    """case-id prefix -> goal rect, rebuilt from the dataset.

    The probe records only the centroid it drove to, so the footprint has to
    come back from the source. Goal selection there is deterministic (first N
    distinct goals in episode order), so replaying it recovers the exact
    mapping without re-running anything.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from benchmarks.eval_core import load_objectnav_dataset
    from benchmarks.exploit_probe import goal_point

    _, runs = load_objectnav_dataset(dataset, scenes_root=scenes_root,
                                     scenes=scenes)
    by_combo = {}
    for r in runs:
        by_combo.setdefault(r["combo"], []).append(r)
    out = {}
    for combo, eps in by_combo.items():
        seen, gi = set(), 0
        for ep in eps:
            for g in ep["goals"]:
                key = tuple(np.round(goal_point(g), 2))
                if key in seen:
                    continue
                seen.add(key)
                if isinstance(g, dict) and "rect" in g:
                    out[f"{combo}__g{gi}"] = g["rect"]
                gi += 1
                if gi >= goals_per_combo:
                    break
            if gi >= goals_per_combo:
                break
    return out


def load(path, nav_eps, stop_m, rects=None):
    blob = json.load(open(os.path.join(path, "probe.json")))
    out = []
    for r in blob["results"]:
        rr = dict(r)
        rr["klass"] = classify(rr, nav_eps, stop_m)
        # Final pose: the last logged step is pre-action, so for a run that
        # self-stopped (no action executed) it IS the final pose exactly; for a
        # capped run it is within one 0.25 m primitive.
        fx, fz = r["trace"][-1]["pos"]
        key = r["case"].rsplit("__s", 1)[0]
        rect = (rects or {}).get(key)
        rr["d_rect"] = rect_dist(rect, fx, fz) if rect else None
        out.append(rr)
    return blob, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--nav-eps", type=float, default=0.3)
    ap.add_argument("--stop-m", type=float, default=0.75)
    ap.add_argument("--success-m", type=float, default=1.0,
                    help="episode success radius: final distance judged against this")
    ap.add_argument("--dataset", default="benchmarks/episodes/gibson/v1.1_sub10/val")
    ap.add_argument("--scenes-root", default="benchmarks/gibson_scenes")
    ap.add_argument("--scenes", nargs="*", default=["Collierville", "Corozal"])
    ap.add_argument("--goals-per-combo", type=int, default=2)
    ap.add_argument("--no-rects", action="store_true", default=False,
                    help="skip footprint recovery (report centroid distance only)")
    args = ap.parse_args()

    rects = None
    if not args.no_rects:
        rects = recover_rects(args.dataset, args.scenes_root, args.scenes,
                              args.goals_per_combo)
        print(f"[rects] recovered {len(rects)} goal footprints")

    tables = {}
    for d in args.dirs:
        blob, rows = load(d, args.nav_eps, args.stop_m, rects)
        tables[d] = rows
        n = len(rows)
        ks = {}
        for r in rows:
            ks[r["klass"]] = ks.get(r["klass"], 0) + 1
        steps = np.array([r["steps"] for r in rows])
        dfin = np.array([r["dfinal"] for r in rows])
        snap = np.array([r["snap_m"] if r["snap_m"] is not None else 0.0 for r in rows])
        have_rect = [r for r in rows if r["d_rect"] is not None]
        would = np.array([r["d_rect"] <= args.success_m for r in have_rect]) \
            if have_rect else np.array([r["final_euclid"] <= args.success_m for r in rows])

        print(f"\n{'='*88}\n{d}   ({n} approaches)\n{'='*88}")
        for k in ["arrived", "parked", "orbiting", "no_progress"]:
            c = ks.get(k, 0)
            print(f"  {k:12s} {c:4d}  ({100*c/n:5.1f}%)")
        # SR is the conjunction: an episode scores only if the agent STOPS
        # itself and is in position. Reported apart from its two factors
        # because they fail for unrelated reasons — being in position is
        # DD-PPO's job, stopping there is the controller's.
        sr = np.mean([r["stopped"] and r["d_rect"] is not None
                      and r["d_rect"] <= args.success_m for r in rows])
        stranded = sum(1 for r in rows if not r["stopped"]
                       and r["d_rect"] is not None and r["d_rect"] <= args.success_m)
        print(f"\n  self-stop rate      {100*np.mean([r['stopped'] for r in rows]):5.1f}%")
        print(f"  in position <= {args.success_m}m {100*would.mean():5.1f}%   "
              f"(final pose to the goal FOOTPRINT — the Gibson-val rule)")
        print(f"  SR = stop AND pos   {100*sr:5.1f}%   "
              f"<- {stranded} in position but never stopped (score 0)")
        if have_rect:
            dr = np.array([r["d_rect"] for r in have_rect])
            dc = np.array([r["final_euclid"] for r in have_rect])
            print(f"  median d(footprint) {np.median(dr):5.2f} m   "
                  f"vs d(centroid) {np.median(dc):5.2f} m")
        print(f"  median steps        {np.median(steps):5.0f}   "
              f"(arrived only: {np.median([r['steps'] for r in rows if r['stopped']]) if any(r['stopped'] for r in rows) else float('nan'):.0f})")
        print(f"  median dfinal       {np.median(dfin):5.2f} m")
        print(f"  snap > stop_m       {int((snap > args.stop_m).sum()):4d}  "
              f"<- raw-goal arrival is unsatisfiable for these")

        # The core correlation: does a large snap predict a non-arrival?
        big = snap > args.stop_m
        if big.any() and (~big).any():
            sr_big = np.mean([r["stopped"] for r, b in zip(rows, big) if b])
            sr_small = np.mean([r["stopped"] for r, b in zip(rows, big) if not b])
            print(f"  self-stop | snap>{args.stop_m}m : {100*sr_big:5.1f}%")
            print(f"  self-stop | snap<={args.stop_m}m: {100*sr_small:5.1f}%")

    if len(args.dirs) == 2:
        a, b = args.dirs
        ra = {r["case"]: r for r in tables[a]}
        rb = {r["case"]: r for r in tables[b]}
        common = sorted(set(ra) & set(rb))
        print(f"\n{'='*88}\nA/B on {len(common)} common cases:  A={a}  B={b}\n{'='*88}")
        sa = sum(ra[c]["stopped"] for c in common)
        sb = sum(rb[c]["stopped"] for c in common)
        fa = np.median([ra[c]["dfinal"] for c in common])
        fb = np.median([rb[c]["dfinal"] for c in common])
        ta = np.median([ra[c]["steps"] for c in common])
        tb = np.median([rb[c]["steps"] for c in common])
        def succ(t, c):
            d = t[c]["d_rect"]
            return (d if d is not None else t[c]["final_euclid"]) <= args.success_m
        wa = np.mean([succ(ra, c) for c in common])
        wb = np.mean([succ(rb, c) for c in common])
        sra = np.mean([ra[c]["stopped"] and succ(ra, c) for c in common])
        srb = np.mean([rb[c]["stopped"] and succ(rb, c) for c in common])
        print(f"  self-stop        {sa:3d} -> {sb:3d}   ({100*sa/len(common):.0f}% -> {100*sb/len(common):.0f}%)")
        print(f"  in position      {100*wa:.0f}% -> {100*wb:.0f}%   (footprint)")
        print(f"  SR               {100*sra:.0f}% -> {100*srb:.0f}%")
        print(f"  median dfinal    {fa:.2f} -> {fb:.2f} m")
        print(f"  median steps     {ta:.0f} -> {tb:.0f}")
        fixed = [c for c in common if not ra[c]["stopped"] and rb[c]["stopped"]]
        broke = [c for c in common if ra[c]["stopped"] and not rb[c]["stopped"]]
        print(f"\n  newly arriving ({len(fixed)}): {fixed[:12]}")
        print(f"  regressed ({len(broke)}): {broke[:12]}")


if __name__ == "__main__":
    main()
