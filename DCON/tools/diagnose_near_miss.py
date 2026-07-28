"""Diagnose why right-object near-miss failures aren't within the success radius.

Decompose each into:
  d_agent_box   agent final_pos -> its own latched box-goal   (locomotion)
  d_box_fp      box-goal -> nearest true object footprint      (goal projection)
  d_agent_fp    agent -> nearest object footprint  (= the cc distance that failed)

If d_box_fp is large, the detection was right but the goal CELL was projected off
the object (projection error). If d_box_fp is small but d_agent_fp large, it's the
stop radius / locomotion. FP is ruled out separately (these are adjudicated right-object).
"""
import glob, json
import numpy as np
import yaml

BASE = "output/ovon_detector_pairwise_field_maxj"
MP = yaml.safe_load(open("config/ovon_category_hm3d_labels.yaml"))

RIGHT = {"window__ep1779","heater__ep3928","washbasin__ep2427","washbasin__ep2459","window__ep1637",
 "couch__ep4096","bench__ep318","chair__ep779","chair__ep940","bed__ep4510","bed__ep4807",
 "washbasin__ep2083","bed__ep4600","sink cabinet__ep6078","chair__ep565","blanket__ep5326",
 "window__ep1686","countertop__ep3378","countertop__ep3468","countertop__ep3426",
 "countertop__ep3019","sink cabinet__ep6290","rack__ep1045","blanket__ep5298","blanket__ep5238"}

cache = {"4ok3usBNeis": json.load(open("output/_instance_cache/4ok3usBNeis.json"))}

def fp_dist(xz, cat, cval, y):
    labels = set(MP.get(cat, [cat]))
    best = float("inf")
    for v in cval.values():
        if v["label"] not in labels: continue
        mn, mx = v["aabb_min"], v["aabb_max"]
        if mx[1] < y-1.5 or mn[1] > y+1.5: continue   # same floor
        cx = min(max(xz[0], mn[0]), mx[0]); cz = min(max(xz[1], mn[2]), mx[2])
        best = min(best, float(np.hypot(xz[0]-cx, xz[1]-cz)))
    return best

def box_goal_world(rid):
    ep = f"{BASE}/ep/{rid}"
    ge = json.load(open(f"{ep}/grid_extent.json")); res=ge["voxel_resolution"]
    g=None
    for line in open(f"{ep}/traj_log.jsonl"):
        d=json.loads(line)
        if d.get("mode")=="EXPLOIT" and d.get("goal") is not None: g=d["goal"]
    if g is None: return None
    zc,xc=g
    return np.array([ge["min_x"]+xc*res, ge["min_z"]+zc*res])  # world x,z

print(f"{'episode':24s} {'cat':14s} {'d_agent_box':>11s} {'d_box_fp':>9s} {'d_agent_fp':>10s}  verdict")
print("-"*95)
buckets={"projection":0,"stop-radius":0,"unlabeled":0,"loco":0}
for f in sorted(glob.glob(BASE+"/runs/*.json")):
    r=json.load(open(f)); short=r["id"].replace("__4ok3usBNeis","")
    if short not in RIGHT: continue
    cval=cache["4ok3usBNeis"]; y=r["final_pos"][1]
    axz=np.array([r["final_pos"][0], r["final_pos"][2]])
    bg=box_goal_world(r["id"])
    d_agent_fp=fp_dist(axz,r["target"],cval,y)
    if bg is None:
        continue
    d_agent_box=float(np.hypot(*(axz-bg)))
    d_box_fp=fp_dist(bg,r["target"],cval,y)
    if not np.isfinite(d_agent_fp):
        v="UNLABELED (annotation)"; buckets["unlabeled"]+=1
    elif d_agent_fp<=1.0:
        v="cc-credited"
    elif d_box_fp>1.0:
        v="PROJECTION (goal off object)"; buckets["projection"]+=1
    elif d_agent_box>1.0:
        v="LOCOMOTION (stopped short of goal)"; buckets["loco"]+=1
    else:
        v="stop-radius marginal"; buckets["stop-radius"]+=1
    print(f"{short:24s} {r['target']:14s} {d_agent_box:11.2f} {d_box_fp:9.2f} {d_agent_fp:10.2f}  {v}")
print("\nfailure attribution among cc-failing right-object near-misses:", buckets)
