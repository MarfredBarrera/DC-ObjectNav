"""Decode HM3D per-instance footprints from the .semantic.glb texture into the
habitat world frame (calibrated from OVON goal positions), since this habitat
build's OBB/semantic-sensor APIs are broken for HM3D.

`build_scene_cache(stem)` returns/caches {instance_id: {label, centroid, aabb}}
in world coords. Reused by the label-map validation and the category-complete
scorer. Run as a script to (re)build one or more scenes.
"""
import os, re, gzip, json, glob, sys
import numpy as np
import trimesh

SCENES_ROOT = "benchmarks/scene_datasets/hm3d/val"
CACHE_DIR = "output/_instance_cache"


def _scene_dir(stem):
    hits = glob.glob(f"{SCENES_ROOT}/*-{stem}")
    if not hits:
        raise FileNotFoundError(f"no scene dir for stem {stem} under {SCENES_ROOT}")
    return hits[0]


def _episode_gz(stem):
    for split in ("val_seen", "val_seen_synonyms", "val_unseen"):
        p = f"benchmarks/episodes/hm3d_ovon/hm3d/{split}/content/{stem}.json.gz"
        if os.path.exists(p):
            return p
    return None


def _umeyama(A, B):
    mA, mB = A.mean(0), B.mean(0)
    H = (A - mA).T @ (B - mB) / len(A)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, mB - R @ mA


def build_scene_cache(stem, force=False, verbose=True):
    cpath = f"{CACHE_DIR}/{stem}.json"
    if os.path.exists(cpath) and not force:
        return json.load(open(cpath))
    sd = _scene_dir(stem)

    # palette: semantic.txt  id -> (label, rgb)
    pal_rgb, pal_id, id_label = [], [], {}
    line = re.compile(r'^\s*(\d+)\s*,\s*([0-9A-Fa-f]{6})\s*,\s*"(.*?)"')
    for ln in open(f"{sd}/{stem}.semantic.txt"):
        m = line.match(ln)
        if not m:
            continue
        iid, hexc, lab = int(m.group(1)), m.group(2), m.group(3).strip().lower()
        pal_rgb.append([int(hexc[i:i+2], 16) for i in (0, 2, 4)])
        pal_id.append(iid); id_label[iid] = lab
    pal_rgb = np.array(pal_rgb, np.int32); pal_id = np.array(pal_id)

    # decode vertices -> instance id via texture
    scene = trimesh.load(f"{sd}/{stem}.semantic.glb", process=False)
    inst_pts = {}
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        g = scene.geometry[gname]
        V = trimesh.transformations.transform_points(g.vertices, T)
        uv = np.asarray(g.visual.uv)
        img = np.asarray(g.visual.material.baseColorTexture)
        H, W = img.shape[:2]
        px = np.clip((uv[:, 0] * (W - 1)).astype(int), 0, W - 1)
        py = np.clip(((1 - uv[:, 1]) * (H - 1)).astype(int), 0, H - 1)
        texel = img[py, px, :3].astype(np.int32)
        d = np.abs(texel[:, None, :] - pal_rgb[None, :, :]).sum(2)
        j = d.argmin(1); dmin = d[np.arange(len(j)), j]
        for k in range(len(V)):
            if dmin[k] <= 12:
                inst_pts.setdefault(int(pal_id[j[k]]), []).append(V[k])

    inst = {i: (np.array(p).mean(0), np.array(p).min(0), np.array(p).max(0))
            for i, p in inst_pts.items()}

    # calibrate root->world from OVON goal positions
    epi = _episode_gz(stem)
    ovon = {}
    if epi:
        d = json.load(gzip.open(epi))
        for goals in (d.get("goals_by_category") or {}).values():
            for g in goals:
                oid = g.get("object_id")
                if oid and oid.rsplit("_", 1)[-1].isdigit():
                    ovon[int(oid.rsplit("_", 1)[-1])] = np.array(g["position"], float)
    ids = [i for i in ovon if i in inst]
    if len(ids) < 3:
        raise RuntimeError(f"{stem}: only {len(ids)} OVON correspondences, cannot calibrate")
    src = np.array([inst[i][0] for i in ids]); dst = np.array([ovon[i] for i in ids])
    R, t = _umeyama(src, dst)
    res = np.linalg.norm(src @ R.T + t - dst, axis=1)

    out = {}
    for iid, (c, mn, mx) in inst.items():
        corners = np.array([[x, y, z] for x in (mn[0], mx[0])
                            for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])
        wc = corners @ R.T + t
        out[str(iid)] = {"label": id_label.get(iid, "?"),
                         "centroid": (c @ R.T + t).tolist(),
                         "aabb_min": wc.min(0).tolist(), "aabb_max": wc.max(0).tolist()}
    os.makedirs(CACHE_DIR, exist_ok=True)
    json.dump(out, open(cpath, "w"))
    if verbose:
        print(f"[cache] {stem}: {len(out)} instances, {len(ids)} corr, "
              f"residual mean={res.mean()*100:.0f}cm max={res.max()*100:.0f}cm")
    return out


if __name__ == "__main__":
    stems = sys.argv[1:] or [os.path.basename(p).split(".")[0]
                             for p in glob.glob(f"{SCENES_ROOT}/*/*.semantic.txt")]
    for s in stems:
        try:
            build_scene_cache(s, force="--force" in sys.argv)
        except Exception as e:
            print(f"[cache] {s}: FAILED {e}")
