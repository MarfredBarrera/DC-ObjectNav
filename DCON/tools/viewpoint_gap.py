"""Quantify the viewpoint metric's visibility-coupling cost.

For each completed OVON episode, compute geodesic distance from the agent's
stopping pose (final_pos) to the nearest VIEWPOINT (visibility-gated pose) of
its target category, and compare to:
  - current success under the centroid-1.0m headline (from the record)
  - our right-object adjudication (did we actually reach the correct object?)

The episodes that reached the correct object but sit far from every viewpoint
are the 'at the object, not at a viewpoint' failures the viewpoint metric would
wrongly score as misses.
"""
import os, glob, json, gzip
import numpy as np
import habitat_sim
from src.habitat.habitat_utils import init_simulator

BASE = "output/ovon_detector_pairwise_field_maxj"
EPI = "benchmarks/episodes/hm3d_ovon/hm3d/val_seen/content"

# right-object failures from the visual adjudication (25)
RIGHT_OBJ = {
 # near-miss (15)
 "window__4ok3usBNeis__ep1779","heater__4ok3usBNeis__ep3928","washbasin__4ok3usBNeis__ep2427",
 "washbasin__4ok3usBNeis__ep2459","window__4ok3usBNeis__ep1637","couch__4ok3usBNeis__ep4096",
 "bench__4ok3usBNeis__ep318","chair__4ok3usBNeis__ep779","chair__4ok3usBNeis__ep940",
 "bed__4ok3usBNeis__ep4510","bed__4ok3usBNeis__ep4807","washbasin__4ok3usBNeis__ep2083",
 "bed__4ok3usBNeis__ep4600","sink cabinet__4ok3usBNeis__ep6078","chair__4ok3usBNeis__ep565",
 # too-far (7)
 "blanket__4ok3usBNeis__ep5326","window__4ok3usBNeis__ep1686","countertop__4ok3usBNeis__ep3378",
 "countertop__4ok3usBNeis__ep3468","countertop__4ok3usBNeis__ep3426","countertop__4ok3usBNeis__ep3019",
 "sink cabinet__4ok3usBNeis__ep6290",
 # correct-not-in-goals (3)
 "rack__4ok3usBNeis__ep1045","blanket__4ok3usBNeis__ep5298","blanket__4ok3usBNeis__ep5238",
}

# cache raw episodes per scene: {scene_stem: {episode_id: episode}}, plus goals_by_category
_raw = {}
def load_scene_raw(stem):
    if stem in _raw: return _raw[stem]
    d = json.load(gzip.open(f"{EPI}/{stem}.json.gz"))
    by_id = {str(e["episode_id"]): e for e in d["episodes"]}
    _raw[stem] = (by_id, d.get("goals_by_category") or {})
    return _raw[stem]

def category_viewpoints(stem, category):
    by_id, gbc = load_scene_raw(stem)
    # goals_by_category keyed "<scene>.basis.glb_<category>"
    pts = []
    for k, goals in gbc.items():
        if k.endswith("_" + category):
            for g in goals:
                for vp in (g.get("view_points") or []):
                    pts.append(vp["agent_state"]["position"])
    return np.array(pts, dtype=np.float32) if pts else None

recs = []
for f in glob.glob(BASE + "/runs/*.json"):
    r = json.load(open(f))
    if r.get("error"): continue
    recs.append(r)
scene = os.path.abspath(recs[0]["scene"])
sim, agent = init_simulator(scene, agent_radius=0.0)
pf = sim.pathfinder

def geo_to_nearest_vp(fp, vps):
    start = pf.snap_point(np.array(fp, dtype=np.float32))
    best = float("inf")
    for v in vps:
        sp = habitat_sim.ShortestPath()
        sp.requested_start = start
        sp.requested_end = pf.snap_point(v)
        if pf.find_path(sp) and np.isfinite(sp.geodesic_distance):
            best = min(best, sp.geodesic_distance)
    return best

rows = []
for r in recs:
    stem = "4ok3usBNeis"
    if "4ok3usBNeis" not in r["id"]:  # skip scene-2 (pending anyway)
        continue
    vps = category_viewpoints(stem, r["target"])
    if vps is None:
        continue
    gvp = geo_to_nearest_vp(r["final_pos"], vps)
    rows.append(dict(id=r["id"], target=r["target"], succ=bool(r.get("success")),
                     right=r["id"] in RIGHT_OBJ, vp=gvp))
sim.close()

def rate(sub, thr):
    return sum(x["vp"] <= thr for x in sub)

reached = [x for x in rows if x["succ"] or x["right"]]
print(f"completed (scene 4ok3usBNeis): {len(rows)}   reached-correct-object: {len(reached)} "
      f"(={sum(x['succ'] for x in rows)} success + {sum(x['right'] and not x['succ'] for x in rows)} right-object-fail)")
print()
print(f"{'group':28s} {'n':>3s} " + " ".join(f'vp<={t}' for t in (0.1,0.25,0.5,1.0,2.0)))
for name, sub in [("ALL completed", rows), ("REACHED correct object", reached),
                  ("current successes", [x for x in rows if x['succ']]),
                  ("right-object FAILURES", [x for x in rows if x['right'] and not x['succ']])]:
    n=len(sub)
    cells=" ".join(f'{rate(sub,t):>5d}' for t in (0.1,0.25,0.5,1.0,2.0))
    print(f"{name:28s} {n:>3d} {cells}")

print()
print("REACHED-correct-object episodes, geodesic to nearest viewpoint (sorted):")
for x in sorted(reached, key=lambda z:z['vp']):
    flag = "SUCCESS" if x['succ'] else "right-obj-fail"
    passes = "PASS@0.1" if x['vp']<=0.1 else ("pass@1.0" if x['vp']<=1.0 else "FAIL(vp)")
    print(f"  {x['id'].replace('__4ok3usBNeis',''):32s} vp={x['vp']:6.2f}m  {passes:9s} [{flag}]")
