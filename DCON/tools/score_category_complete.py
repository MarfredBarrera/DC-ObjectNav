"""Category-complete footprint scoring for OVON runs.

Re-scores completed records against the footprint of EVERY HM3D instance whose
label maps to the query category (config/ovon_category_hm3d_labels.yaml), not
just the OVON goal subset. Success = geodesic <= radius to the nearest such
footprint on the agent's floor. Reports SR/SPL vs the OVON-official (centroid)
number, per category, with instance-density flagged (ubiquitous categories are
trivially satisfiable and marked).

Run with CUDA_VISIBLE_DEVICES pinned off the training GPU.
"""
import os, glob, json, argparse
import numpy as np
import habitat_sim
from collections import defaultdict
import yaml

from src.habitat.habitat_utils import init_simulator
from tools.hm3d_instance_footprints import build_scene_cache

MP = yaml.safe_load(open("config/ovon_category_hm3d_labels.yaml"))
FLOOR_BAND = 1.5   # m; instance counts only if its AABB y-range is within this of the agent


def scene_stem(rec):
    return os.path.splitext(os.path.basename(rec["scene"]))[0].replace(".basis", "")


def geo(pf, a, b):
    sp = habitat_sim.ShortestPath()
    sp.requested_start = pf.snap_point(np.array(a, np.float32))
    sp.requested_end = pf.snap_point(np.array(b, np.float32))
    if pf.find_path(sp) and np.isfinite(sp.geodesic_distance):
        return float(sp.geodesic_distance)
    return float("inf")


def clamp_xz(p, mn, mx, y):
    return [min(max(p[0], mn[0]), mx[0]), y, min(max(p[2], mn[2]), mx[2])]


def score_scene(stem, recs, radius, gpu):
    cache = build_scene_cache(stem, verbose=True)
    # label -> list of (mn, mx, centroid)
    by_label = defaultdict(list)
    for v in cache.values():
        by_label[v["label"]].append((np.array(v["aabb_min"]), np.array(v["aabb_max"]),
                                      np.array(v["centroid"])))
    sd = glob.glob(f"benchmarks/scene_datasets/hm3d/val/*-{stem}")[0]
    sim, _ = init_simulator(os.path.join(sd, f"{stem}.basis.glb"), agent_radius=0.0)
    pf = sim.pathfinder
    out = []
    for r in recs:
        cat = r["target"]
        labels = MP.get(cat, [cat])
        cands = [ins for lb in labels for ins in by_label.get(lb, [])]
        fp = np.array(r["final_pos"]); start = np.array(r.get("start_nav") or r["start"])
        # same-floor candidates (AABB y overlaps agent band)
        lo, hi = fp[1] - FLOOR_BAND, fp[1] + FLOOR_BAND
        cands = [(mn, mx, c) for (mn, mx, c) in cands if mx[1] >= lo and mn[1] <= hi]
        dgeo, deuc, lmin = float("inf"), float("inf"), float("inf")
        for mn, mx, c in cands:
            pt = clamp_xz(fp, mn, mx, fp[1])
            dgeo = min(dgeo, geo(pf, fp, pt))
            deuc = min(deuc, float(np.hypot(fp[0] - pt[0], fp[2] - pt[2])))
            lmin = min(lmin, geo(pf, start, clamp_xz(start, mn, mx, start[1])))
        succ_geo = dgeo <= radius
        succ_euc = deuc <= radius
        spl = 0.0
        if succ_geo and np.isfinite(lmin):
            pl = r.get("path_length") or 0.0
            spl = lmin / max(lmin, pl) if max(lmin, pl) > 0 else 1.0
        out.append(dict(id=r["id"], cat=cat, official=bool(r.get("success")),
                        cc_geo=succ_geo, cc_euc=succ_euc, cc_spl=spl,
                        dgeo=dgeo, deuc=deuc, n_inst=len(cands)))
    sim.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/ovon_detector_pairwise_field_maxj")
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--gpu", default="1")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu

    recs = []
    for f in glob.glob(f"{a.out}/runs/*.json"):
        r = json.load(open(f))
        if not r.get("error"):
            recs.append(r)
    by_scene = defaultdict(list)
    for r in recs:
        by_scene[scene_stem(r)].append(r)

    rows = []
    for stem, rs in by_scene.items():
        rows += score_scene(stem, rs, a.radius, a.gpu)
    json.dump({r["id"]: {"official": r["official"], "cc_geo": r["cc_geo"],
                         "cc_euc": r["cc_euc"], "n_inst": r["n_inst"]} for r in rows},
              open(f"{a.out}/_cc_scores.json", "w"))

    # adjudication-based residuals: visually-confirmed right object that HM3D
    # doesn't label under any synonym (agent self-stopped within radius of it).
    RESIDUALS = {"blanket__4ok3usBNeis__ep5238", "blanket__4ok3usBNeis__ep5298",
                 "rack__4ok3usBNeis__ep1045", "sink cabinet__4ok3usBNeis__ep6290"}

    n = len(rows)
    off = sum(r["official"] for r in rows)
    geo_sr = sum(r["cc_geo"] for r in rows)
    euc_sr = sum(r["cc_euc"] for r in rows)
    euc_res = sum(r["cc_euc"] or r["id"] in RESIDUALS for r in rows)
    cc_spl = sum(r["cc_spl"] for r in rows)
    print(f"\n=== rescoring  ({n} records, radius {a.radius} m) ===")
    print(f"OVON-official (centroid geodesic):     SR {off}/{n} = {off/n:.3f}")
    print(f"category-complete, GEODESIC footprint: SR {geo_sr}/{n} = {geo_sr/n:.3f}   SPL {cc_spl/n:.3f}")
    print(f"category-complete, EUCLIDEAN footprint:SR {euc_sr}/{n} = {euc_sr/n:.3f}")
    print(f"  + adjudicated residuals ({len(RESIDUALS)}):     SR {euc_res}/{n} = {euc_res/n:.3f}")
    print(f"  euclidean recovers over geodesic:    {sum(r['cc_euc'] and not r['cc_geo'] for r in rows)} "
          f"(wall/closet goals)")

    print("\nper category  (off / geo / eucl SR;  ~inst, DENSE>=6 = trivially satisfiable):")
    bycat = defaultdict(list)
    for r in rows:
        bycat[r["cat"]].append(r)
    print(f"  {'category':16s} {'n':>3s} {'off':>5s} {'geo':>5s} {'euc':>5s} {'~inst':>6s}")
    for cat in sorted(bycat):
        g = bycat[cat]
        m = np.median([x["n_inst"] for x in g])
        flag = "  DENSE" if m >= 6 else ""
        print(f"  {cat:16s} {len(g):3d} {sum(x['official'] for x in g)/len(g):5.2f} "
              f"{sum(x['cc_geo'] for x in g)/len(g):5.2f} {sum(x['cc_euc'] for x in g)/len(g):5.2f} "
              f"{m:6.0f}{flag}")


if __name__ == "__main__":
    main()
