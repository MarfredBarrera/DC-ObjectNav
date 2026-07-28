"""Validate the OVON->HM3D label map using decoded instance footprints
(output/_instance_cache/<scene>.json from hm3d_instance_footprints.py).

For each adjudicated stop, find the nearest instance whose HM3D label is in the
query category's mapped set, and report the footprint (x,z AABB) distance.
Right-object stops should have a mapped instance within ~1 m; FP stops should
not (or only the same-category instance the detector confused it with).
"""
import glob, json
import numpy as np
import yaml

BASE = "output/ovon_detector_pairwise_field_maxj"
CACHE = json.load(open("output/_instance_cache/4ok3usBNeis.json"))
MP = yaml.safe_load(open("config/ovon_category_hm3d_labels.yaml"))

RIGHT = {"window__ep1779","heater__ep3928","washbasin__ep2427","washbasin__ep2459","window__ep1637",
 "couch__ep4096","bench__ep318","chair__ep779","chair__ep940","bed__ep4510","bed__ep4807",
 "washbasin__ep2083","bed__ep4600","sink cabinet__ep6078","chair__ep565","blanket__ep5326",
 "window__ep1686","countertop__ep3378","countertop__ep3468","countertop__ep3426",
 "countertop__ep3019","sink cabinet__ep6290","rack__ep1045","blanket__ep5298","blanket__ep5238"}
FP = {"sink cabinet__ep6207","sink cabinet__ep6469","chair__ep627","chair__ep750","couch__ep4187",
 "couch__ep4180","tv__ep5691","tv__ep5923","washing machine__ep2824","blanket__ep5357",
 "washing machine__ep2983","washing machine__ep2632","bed__ep4530","bed__ep4791"}

INST = [(v["label"], np.array(v["aabb_min"]), np.array(v["aabb_max"]), np.array(v["centroid"]))
        for v in CACHE.values()]

def nearest(fp, labels):
    ax, az, ay = fp[0], fp[2], fp[1]
    best = (float("inf"), None, None, None)
    for lab, mn, mx, c in INST:
        if lab not in labels:
            continue
        cx = min(max(ax, mn[0]), mx[0]); cz = min(max(az, mn[2]), mx[2])
        d = float(np.hypot(ax - cx, az - cz))
        if d < best[0]:
            best = (d, lab, abs(ay - c[1]), abs(ay - c[1]) > 1.2)  # dy, other-floor flag
    return best

def anylabel(fp):
    """nearest instance of ANY label -- what HM3D thinks the agent is standing at."""
    ax, az = fp[0], fp[2]
    best = (float("inf"), None)
    for lab, mn, mx, c in INST:
        cx = min(max(ax, mn[0]), mx[0]); cz = min(max(az, mn[2]), mx[2])
        d = float(np.hypot(ax - cx, az - cz))
        if d < best[0]:
            best = (d, lab)
    return best

def run(group, ids, want):
    print(f"\n=== {group}  (want: {want}) ===")
    print(f"{'episode':24s} {'cat':15s} {'nearest mapped':17s} {'fp':>5s} {'floor':>6s}   {'HM3D@stop(any label)':20s}")
    hit = 0
    for f in sorted(glob.glob(BASE + "/runs/*.json")):
        r = json.load(open(f))
        if r.get("error"): continue
        short = r["id"].replace("__4ok3usBNeis", "")
        if short not in ids: continue
        labels = set(MP.get(r["target"], [r["target"]]))
        d, lab, dy, offfloor = nearest(r["final_pos"], labels)
        ad, alab = anylabel(r["final_pos"])
        ok = d <= 1.0 and not offfloor
        hit += ok
        fl = "OTHER" if offfloor else "same"
        print(f"{short:24s} {r['target']:15s} {(lab or '-'):17s} {d:5.2f} {fl:>6s}   {alab}@{ad:.2f}m")
    print(f"  -> mapped instance within 1.0 m (same floor): {hit}/{len(ids)}")

run("RIGHT-OBJECT failures", RIGHT, "hit")
run("FALSE POSITIVES (control)", FP, "miss / confused same-category")
