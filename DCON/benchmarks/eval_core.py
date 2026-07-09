"""Core logic for the per-scene ObjectNav evaluation (no simulator / torch).

This module is the shared, side-effect-light backbone behind `eval_scene.py`'s
three stages:

    run     -> execute episodes, save a raw record + evidence bundle per run
    review  -> (re)generate verdicts.yaml with the auto-suggestion + evidence
    report  -> aggregate SR/SPL from the records + the human verdicts

The design separates *evidence* (what happened: final pose, geodesics, maps,
video — immutable, in runs/<id>.json and ep/<id>/) from *judgment* (whether a
run counts: verdicts.yaml, human-editable). The metric's computed success is
only a SUGGESTION; the verdict is authoritative. This keeps scoring transparent
and re-runnable without touching the simulator.

Importable without CUDA/habitat (only the `run` stage imports `main.run`).
"""

import json
import math
import os


# ── Scenario loading + run expansion ─────────────────────────────────────────

def load_scenarios(path):
    """Parse the per-scene scenarios file (YAML or JSON). Returns a dict with
    scene, success_radius_m, runs_per_combo, starts (name->xyz),
    targets (name->{query, goals, success_radius_m}), combos (or None)."""
    with open(path, "r") as f:
        text = f.read()
    if os.path.splitext(path)[1].lower() in (".yaml", ".yml"):
        import yaml
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    for key in ("scene", "targets", "starts"):
        if key not in data:
            raise ValueError(f"{path}: missing top-level '{key}'")

    starts_raw = data["starts"]
    if isinstance(starts_raw, dict):
        starts = {str(k): list(v) for k, v in starts_raw.items()}
    else:
        starts = {f"start{i}": list(v) for i, v in enumerate(starts_raw)}

    targets = {}
    for name, spec in data["targets"].items():
        if "query" not in spec or "goals" not in spec:
            raise ValueError(f"{path}: target '{name}' needs 'query' and 'goals'")
        # Goals are points ([x,y,z]) or rectangles
        # ({rect: [x_min,z_min,x_max,z_max], y: <height>}); success_radius_m
        # (optional) widens the tolerance for this target (e.g. big furniture).
        targets[str(name)] = {
            "query": spec["query"],
            "goals": spec["goals"],
            "success_radius_m": spec.get("success_radius_m"),
        }

    runs_per_combo = int(data.get("runs_per_combo", 1))
    if runs_per_combo < 1:
        raise ValueError(f"{path}: runs_per_combo must be >= 1")

    return {
        "scene": data["scene"],
        "success_radius_m": float(data.get("success_radius_m", 1.0)),
        "runs_per_combo": runs_per_combo,
        "starts": starts,
        "targets": targets,
        "combos": data.get("combos"),
    }


def build_runs(scn):
    """Expand scenarios into ordered run specs: one per (target x start) combo,
    repeated runs_per_combo times. id = '{target}__{start}' (+ '__r{k}' when
    repeated). Each spec carries everything a run needs and everything review/
    report need to interpret it (query, goals, radius)."""
    starts, targets, combos, n_rep = (
        scn["starts"], scn["targets"], scn["combos"], scn["runs_per_combo"])
    if combos is None:
        pairs = [(t, s) for t in targets for s in starts]
    else:
        pairs = []
        for c in combos:
            t, s = c["target"], c["start"]
            if t not in targets:
                raise ValueError(f"combo references unknown target '{t}'")
            if s not in starts:
                raise ValueError(f"combo references unknown start '{s}'")
            pairs.append((t, s))

    runs = []
    for t, s in pairs:
        combo_id = f"{t}__{s}"
        for k in range(n_rep):
            rid = combo_id if n_rep == 1 else f"{combo_id}__r{k}"
            runs.append({
                "id": rid, "combo": combo_id, "repeat": k,
                "target": t, "start_name": s, "start": starts[s],
                "query": targets[t]["query"], "goals": targets[t]["goals"],
                "success_radius_m": (targets[t]["success_radius_m"]
                                     if targets[t]["success_radius_m"] is not None
                                     else scn["success_radius_m"]),
            })
    return runs


def resolve_scene(scene):
    return scene if os.path.isabs(scene) else os.path.abspath(scene)


# ── Habitat ObjectNav episode datasets (Gibson/SemExp, HM3D, MP3D) ───────────

# BEV height band offsets above the episode's floor (= the start position's Y,
# which in Habitat is the agent base on the floor). Mirrors the convention
# documented in config.py: min ~0.2 m above the floor surface (the occupancy
# grid marks the floor plane occupied and bleeds up ~one voxel), max at
# standing-obstacle height. Benchmark scenes are multi-floor, so the band must
# follow each episode's floor instead of the static config value.
BEV_BAND_ABOVE_FLOOR = (0.2, 1.5)


def _read_json_maybe_gz(path):
    import gzip
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    with open(path, "r") as f:
        return json.load(f)


def _dataset_files(path):
    """Episode files for a dataset path: a single .json/.json.gz file, or a
    directory in the standard habitat-lab layout (`<split>/content/<scene>.json.gz`;
    the top-level `<split>.json.gz` holds no episodes for ObjectNav and is
    ignored when a content/ dir exists)."""
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"dataset path not found: {path}")
    content = os.path.join(path, "content")
    scan_dir = content if os.path.isdir(content) else path
    files = [os.path.join(scan_dir, fn) for fn in sorted(os.listdir(scan_dir))
             if fn.endswith((".json", ".json.gz"))]
    if not files:
        raise FileNotFoundError(f"no .json/.json.gz episode files under {scan_dir}")
    return files


def _scene_glb_index(scenes_root):
    """Map .glb basename -> absolute path for every scene under `scenes_root`
    (recursive walk, built once per dataset load). Handles both the flat
    Gibson layout (gibson_scenes/Collierville.glb) and HM3D's nested one
    (hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb — the layout the
    HM3D-OVON episodes reference). Duplicate basenames keep the first hit in
    walk order (deterministic: os.walk with sorted children)."""
    index = {}
    for dirpath, dirnames, filenames in os.walk(scenes_root):
        dirnames.sort()
        for fn in sorted(filenames):
            if fn.endswith(".glb"):
                index.setdefault(fn, os.path.abspath(os.path.join(dirpath, fn)))
    return index


def _rotation_xyzw(rot):
    """Normalize a start-rotation quaternion to [x, y, z, w].

    habitat-lab episode files store [x, y, z, w]; the SemExp Gibson episodes
    store [w, x, y, z]. Both datasets only use yaw (rotation about +Y), whose
    quaternion has exactly two non-zero components — (y, w) — so the ordering
    is identified by which pair of slots is zero."""
    a, b, c, d = (float(v) for v in rot)
    if abs(b) < 1e-5 and abs(d) < 1e-5 and (abs(a) > 1e-5 or abs(c) > 1e-5):
        return [b, c, d, a]  # was [w, x, y, z]
    return [a, b, c, d]


# SemExp's category list (Gibson ObjectNav v1.1). `sem_map` channel 0 is the
# navigable area; channel i+1 is the mask of category SEMEXP_CATEGORIES[i].
# Only the first six are used as goals in the benchmark.
SEMEXP_CATEGORIES = ["chair", "couch", "potted plant", "bed", "toilet", "tv",
                     "dining-table", "oven", "sink", "refrigerator", "book",
                     "clock", "vase", "cup", "bottle"]
SEMEXP_MAP_RES = 0.05  # m per sem_map cell


def _semexp_goal_lookup(split_dir):
    """Ground-truth goals for the SemExp Gibson ObjectNav v1.1 variant, whose
    `<scene>_episodes.json.gz` files carry no goal positions — the ground truth
    is the rasterized per-floor semantic map in `<split>_info.pbz2`.

    Map→world (from SemExp's sim_continuous_to_sim_map, verified against the
    scenes' navmesh bounds): habitat x = row * 0.05 + origin[1]/100,
    habitat z = col * 0.05 + origin[0]/100 (origin is in cm).

    Returns lookup(scene_stem, floor_id, category) -> list of rect goals
    (each connected component of the category mask becomes an axis-aligned
    {rect, y} footprint, scored against its nearest point — equivalent to
    SemExp's "geodesic to the nearest category cell" up to bbox concavity),
    or None when no *_info.pbz2 exists next to the dataset."""
    import glob
    cands = sorted(glob.glob(os.path.join(split_dir, "*_info.pbz2")))
    if not cands:
        return None
    import bz2
    import pickle
    with bz2.BZ2File(cands[0], "rb") as f:
        info = pickle.load(f)
    from scipy import ndimage
    cache = {}

    def lookup(stem, floor_id, category):
        key = (stem, floor_id, category)
        if key in cache:
            return cache[key]
        goals = []
        floor = (info.get(stem) or {}).get(floor_id)
        if floor is not None and category in SEMEXP_CATEGORIES:
            sem, origin = floor["sem_map"], floor["origin"]
            y = float(floor["floor_height"])
            mask = sem[SEMEXP_CATEGORIES.index(category) + 1] > 0
            labels, n = ndimage.label(mask)
            for sl in ndimage.find_objects(labels):
                r, c = sl[0], sl[1]
                x_min = r.start * SEMEXP_MAP_RES + origin[1] / 100.0
                x_max = r.stop * SEMEXP_MAP_RES + origin[1] / 100.0
                z_min = c.start * SEMEXP_MAP_RES + origin[0] / 100.0
                z_max = c.stop * SEMEXP_MAP_RES + origin[0] / 100.0
                goals.append({"rect": [x_min, z_min, x_max, z_max], "y": y})
        cache[key] = goals
        return goals

    return lookup


def load_objectnav_dataset(path, scenes_root="gibson_scenes",
                           categories=None, scenes=None, max_per_combo=None,
                           success_radius_m=1.0, query_template="a {category}",
                           use_viewpoints=False):
    """Load a standard Habitat ObjectNav episode dataset (the benchmarks VLFM /
    SemExp evaluate on) into the same (scn, runs) shape the scenarios path
    produces, so `run`/`review`/`report` work unchanged.

    Handles the habitat-lab ObjectNav-v1 JSON schema: per-scene `episodes`
    (scene_id, start_position, start_rotation, object_category) with goal
    instances either inline (`episode["goals"]`), in the file-level
    `goals_by_category` (keyed `<scene_basename>_<category>`), or — for the
    SemExp Gibson v1.1 variant, whose episode files carry no goal positions —
    extracted from the `<split>_info.pbz2` GT semantic maps as rectangular
    footprints (see `_semexp_goal_lookup`; episodes without a category, like
    SemExp's `<scene>.glb.json.gz` stubs, are skipped). Each episode
    becomes one run:
      id           {category}__{scene_stem}__ep{episode_id}
      combo        {category}__{scene_stem}  (per-combo rollup = per scene+category)
      query        query_template.format(category=...)  (underscores → spaces)
      goals        every goal instance's object position (scored geodesically
                   against the nearest, success_radius_m default 1.0 m — the
                   SemExp/VLFM "stop within 1 m of the object" protocol). With
                   `use_viewpoints=True`, instead every goal instance's
                   `view_points` agent positions (the navigable poses around
                   the object habitat's DistanceToGoal VIEW_POINTS measure
                   scores against; official success_distance is 0.1 m — pass
                   that as success_radius_m). Episodes whose goals carry no
                   view_points fall back to object positions with a warning.
    plus per-run extras the scenarios path doesn't carry: `scene` (local .glb
    found by a recursive walk of <scenes_root>, matched by basename — covers
    both flat Gibson layouts and HM3D's nested `00xxx-X/X.basis.glb`, so the
    HM3D-OVON splits load unchanged), `start_rotation` ([x,y,z,w] quaternion),
    `episode_id`, and `bev_band` (the BEV height band for the episode's floor,
    derived from the start Y).

    `categories` / `scenes` filter by object category / scene stem;
    `max_per_combo` caps episodes per (category × scene) for balanced subsets.
    Episodes whose scene .glb is missing locally are skipped with a warning.
    """
    cat_filter = ({c.lower().replace("_", " ") for c in categories}
                  if categories else None)
    scene_filter = {s.lower() for s in scenes} if scenes else None

    # SemExp-variant ground truth (goals live in <split>_info.pbz2, not in the
    # episode files). Search next to the given file, or in the split dir.
    semexp_lookup = None
    for d in ({os.path.dirname(path), os.path.dirname(os.path.dirname(path))}
              if os.path.isfile(path) else {path}):
        if d:
            semexp_lookup = _semexp_goal_lookup(d)
            if semexp_lookup is not None:
                break

    glb_index = _scene_glb_index(scenes_root)
    runs, missing_scenes, combo_counts = [], set(), {}
    vp_fallbacks = 0
    for fp in _dataset_files(path):
        data = _read_json_maybe_gz(fp)
        goals_by_cat = data.get("goals_by_category") or {}
        for ep in data.get("episodes", []):
            scene_id = str(ep["scene_id"])
            scene_base = os.path.basename(scene_id)
            stem = os.path.splitext(scene_base)[0]
            # Strip stacked suffixes: "X.glb" (scene ids like "gibson/X.glb")
            # and "X.basis" (HM3D's basis-compressed "X.basis.glb").
            for suffix in (".glb", ".basis"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
            if scene_filter and stem.lower() not in scene_filter:
                continue
            category = ep.get("object_category")
            if category is None:
                continue  # e.g. SemExp's <scene>.glb.json.gz stubs (no category)
            category = str(category)
            cat_human = category.replace("_", " ")
            if cat_filter and cat_human.lower() not in cat_filter:
                continue
            combo = f"{category}__{stem}"
            if max_per_combo is not None and combo_counts.get(combo, 0) >= max_per_combo:
                continue

            # Exact basename first (matches HM3D's "X.basis.glb" as-is), then
            # the stripped stem (flat Gibson "X.glb", or a local plain-glb copy
            # of a basis-named scene).
            glb = glb_index.get(scene_base) or glb_index.get(stem + ".glb")
            if glb is None:
                missing_scenes.add(stem)
                continue

            goals = (ep.get("goals")
                     or goals_by_cat.get(f"{scene_base}_{category}")
                     or [])
            goal_pts = []
            if use_viewpoints:
                goal_pts = [[float(v) for v in vp["agent_state"]["position"]]
                            for g in goals for vp in (g.get("view_points") or [])]
                if not goal_pts and goals:
                    vp_fallbacks += 1
            if not goal_pts:
                goal_pts = [[float(v) for v in g["position"]]
                            for g in goals if g.get("position")]
            if not goal_pts and semexp_lookup is not None:
                # SemExp variant: rect goals from the GT semantic map.
                goal_pts = semexp_lookup(stem, ep.get("floor_id", 0), category)
            if not goal_pts:
                continue

            start = [float(v) for v in ep["start_position"]]
            rot = ep.get("start_rotation")
            eid = str(ep.get("episode_id", len(runs)))
            combo_counts[combo] = combo_counts.get(combo, 0) + 1
            runs.append({
                "id": f"{combo}__ep{eid}",
                "combo": combo, "repeat": 0,
                "target": category, "start_name": f"ep{eid}",
                "start": start,
                "start_rotation": (_rotation_xyzw(rot) if rot else None),
                "query": query_template.format(category=cat_human),
                "goals": goal_pts,
                "success_radius_m": float(success_radius_m),
                "scene": os.path.abspath(glb),
                "episode_id": eid,
                "bev_band": [start[1] + BEV_BAND_ABOVE_FLOOR[0],
                             start[1] + BEV_BAND_ABOVE_FLOOR[1]],
            })

    if missing_scenes:
        print(f"[dataset] skipped {len(missing_scenes)} scene(s) with no local .glb "
              f"under {scenes_root}: {', '.join(sorted(missing_scenes))}")
    if vp_fallbacks:
        print(f"[dataset] WARNING: --viewpoints requested but {vp_fallbacks} "
              f"episode(s) carry no view_points — scored against object "
              f"positions instead (a 0.1 m radius is unreachable for those; "
              f"this dataset variant likely wants position scoring).")
    scn = {
        "scene": f"dataset:{path}",
        "success_radius_m": float(success_radius_m),
        # Episodes per (category × scene); drives the per-combo rollup display.
        "runs_per_combo": max(combo_counts.values()) if combo_counts else 1,
    }
    return scn, runs


# ── Goal geometry (offline, euclidean — for evidence display) ────────────────

def nearest_goal_xz(goal, p):
    """(x, z) of the point of `goal` closest to point `p` in the floor plane.
    Mirrors main.nearest_goal_point but euclidean/offline (no pathfinder)."""
    if isinstance(goal, dict):
        x_min, z_min, x_max, z_max = goal["rect"]
        cx = min(max(float(p[0]), x_min), x_max)
        cz = min(max(float(p[2]), z_min), z_max)
        return cx, cz
    return float(goal[0]), float(goal[2])


def goal_label(goal):
    if isinstance(goal, dict):
        return "rect" + str(goal["rect"])
    return f"[{float(goal[0]):.1f},{float(goal[2]):.1f}]"


def per_goal_distances(goals, final_pos):
    """List of (label, euclidean_dist_m) from `final_pos` to each goal, nearest
    first. Returns [] if final_pos is missing. Euclidean (floor plane) — a quick
    sanity signal for review, not the scored geodesic."""
    if not final_pos or not goals:
        return []
    out = []
    for g in goals:
        gx, gz = nearest_goal_xz(g, final_pos)
        d = math.hypot(final_pos[0] - gx, final_pos[2] - gz)
        out.append((goal_label(g), d))
    return sorted(out, key=lambda t: t[1])


# ── Run records ──────────────────────────────────────────────────────────────

def run_record_path(out_dir, rid):
    return os.path.join(out_dir, "runs", f"{rid}.json")


def ep_dir(out_dir, rid):
    return os.path.join(out_dir, "ep", rid)


def load_records(out_dir, runs):
    """Load every saved run record (in `runs` order). Returns dict id->record."""
    records = {}
    for r in runs:
        p = run_record_path(out_dir, r["id"])
        if os.path.exists(p):
            with open(p, "r") as f:
                records[r["id"]] = json.load(f)
    return records


def _record_files(out_dir):
    runs_dir = os.path.join(out_dir, "runs")
    if not os.path.isdir(runs_dir):
        return []
    return [os.path.join(runs_dir, fn)
            for fn in sorted(os.listdir(runs_dir)) if fn.endswith(".json")]


def from_records(out_dir):
    """Reconstruct (scn, runs) from the records actually present in
    <out>/runs/, so report/review aggregate exactly what's on disk regardless
    of the current scenarios file. `scn` carries only the metadata report needs
    (scene, runs_per_combo); `runs` are self-describing (each record
    stores its id/combo/goals). Runs are ordered by id."""
    runs, scene = [], None
    for p in _record_files(out_dir):
        with open(p, "r") as f:
            rec = json.load(f)
        rid = rec.get("id") or os.path.splitext(os.path.basename(p))[0]
        scene = scene or rec.get("scene")
        runs.append({
            "id": rid,
            "combo": rec.get("combo", rid),
            "repeat": rec.get("repeat", 0),
            "target": rec.get("target"),
            "start_name": rec.get("start_name"),
            "start": rec.get("start"),
            "query": rec.get("query"),
            "goals": rec.get("goals", []),
            "success_radius_m": (rec.get("success_radius_m")
                                 or rec.get("requested_radius_m")),
        })
    combo_counts = {}
    for r in runs:
        combo_counts[r["combo"]] = combo_counts.get(r["combo"], 0) + 1
    scn = {
        "scene": scene or "(from records)",
        "runs_per_combo": max(combo_counts.values()) if combo_counts else 1,
    }
    return scn, runs


def orphan_record_ids(out_dir, runs):
    """Ids of records on disk in <out>/runs/ that are NOT in `runs` (i.e. present
    but not enumerated by the supplied scenarios). Used to warn that a mismatched
    scenarios file would silently drop them from the report."""
    have = {r["id"] for r in runs}
    return [os.path.splitext(os.path.basename(p))[0]
            for p in _record_files(out_dir)
            if os.path.splitext(os.path.basename(p))[0] not in have]


# ── Verdicts (single human-judgment file) ────────────────────────────────────

VALID_STATUS = ("auto", "success", "fail", "exclude")


def load_verdicts(path):
    """Read verdicts.yaml -> {id: {'status': ..., 'note': ...}}. Comments are
    ignored (they are regenerated by `write_verdicts`); only status/note are
    authoritative. Missing file => {}."""
    if not os.path.exists(path):
        return {}
    import yaml
    data = yaml.safe_load(open(path, "r")) or {}
    verdicts = {}
    for rid, v in data.items():
        if v is None:
            verdicts[rid] = {"status": "auto", "note": ""}
            continue
        status = str(v.get("status", "auto")).lower()
        if status not in VALID_STATUS:
            raise ValueError(
                f"{path}: run '{rid}' has bad status {status!r} "
                f"(expected one of {VALID_STATUS})")
        verdicts[rid] = {"status": status, "note": v.get("note", "") or ""}
    return verdicts


def _yaml_str(s):
    """Quote a note string safely for the hand-emitted YAML."""
    return json.dumps(str(s))  # JSON strings are valid YAML double-quoted scalars


def write_verdicts(path, runs, records, existing):
    """(Re)write verdicts.yaml. Preserves each run's existing status/note;
    new runs default to status 'auto'. Evidence is (re)emitted as comments so
    the human edits status while seeing the justification inline."""
    lines = [
        "# Per-run verdicts — the authoritative adjudication for SR / SPL.",
        "#",
        "# status:",
        "#   auto    -> trust the metric's computed success (shown as auto=... below)",
        "#   success -> count as success (SPL recomputed from the recorded geodesic)",
        "#   fail    -> count as failure (SPL 0)",
        "#   exclude -> drop from metrics (sim artifact / spoiled run)",
        "#",
        "# Edit `status` (and optional `note`) after inspecting the evidence under",
        "# ep/<id>/ (nav_history.mp4, bev_final.png). Re-running `review` refreshes",
        "# the evidence comments but keeps your status/note.",
        "",
    ]
    out_dir = os.path.dirname(path)
    for r in runs:
        rid = r["id"]
        rec = records.get(rid)
        prev = existing.get(rid, {})
        status = prev.get("status", "auto")
        note = prev.get("note", "")

        lines.append(f"{rid}:")
        lines.append(f"  status: {status}")
        lines.append(f"  note: {_yaml_str(note)}")

        # Evidence comments (derived; regenerated each review).
        if rec is None:
            lines.append("  # (not yet run)")
        elif rec.get("status") != "ok":
            lines.append(f"  # RUN ERROR: {rec.get('error', '')}")
        else:
            auto = "success" if rec.get("success") else "fail"
            fg = rec.get("final_geodesic")
            fg_s = f"{fg:.2f}m" if isinstance(fg, (int, float)) else "n/a"
            lines.append(
                f"  # auto={auto} | final_geo={fg_s} | stopped={rec.get('agent_stopped')}"
                f" | steps={rec.get('steps')} | path={rec.get('path_length', 0):.2f}m"
                f" | radius={rec.get('success_radius_m', '?')}m")
            dists = per_goal_distances(r["goals"], rec.get("final_pos"))
            if dists:
                shown = "  ".join(f"{lbl}={d:.2f}m" for lbl, d in dists[:4])
                lines.append(f"  # goals(euclid from final): {shown}")
            ep = ep_dir(out_dir, rid)
            vid = os.path.join(ep, "nav_history.mp4")
            bev = os.path.join(ep, "bev_final.png")
            if os.path.exists(vid):
                lines.append(f"  # video: {os.path.relpath(vid, out_dir)}")
            if os.path.exists(bev):
                lines.append(f"  # bev:   {os.path.relpath(bev, out_dir)}")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ── Aggregation (records + verdicts -> SR / SPL) ─────────────────────────────

def _spl_for_success(rec):
    """SPL of a run counted as success: l / max(path, l), matching main.run.
    Falls back to 1.0 when l is 0/non-finite (spawned on goal / unreachable)."""
    l_geo, path = rec.get("l_geodesic"), rec.get("path_length")
    if l_geo is None or path is None:
        return 1.0
    if l_geo > 0.0 and math.isfinite(l_geo):
        return float(l_geo / max(path, l_geo))
    return 1.0


def resolve_verdict(rec, status):
    """Resolve one run to ('include'|'exclude', success, spl, note_kind).

    status: the human verdict (auto/success/fail/exclude). 'auto' defers to the
    record's computed success (and counts a run that errored as excluded)."""
    if status == "exclude":
        return "exclude", None, None, "manual"
    if status == "success":
        return "include", True, _spl_for_success(rec), "forced-success"
    if status == "fail":
        return "include", False, 0.0, "forced-fail"
    # auto
    if rec.get("status") != "ok":
        return "exclude", None, None, "error"
    return "include", bool(rec.get("success")), float(rec.get("spl") or 0.0), "auto"


def aggregate(runs, records, verdicts):
    """Combine run records + verdicts into a summary dict. Pure function."""
    included, excluded, missing, errored = [], [], [], []
    per_combo = {}

    for r in runs:
        rid = r["id"]
        rec = records.get(rid)
        status = verdicts.get(rid, {}).get("status", "auto")
        note = verdicts.get(rid, {}).get("note", "")

        if rec is None:
            missing.append(rid)
            continue

        disp, success, spl, kind = resolve_verdict(rec, status)
        row = {
            "id": rid, "combo": r["combo"], "status": status, "kind": kind,
            "note": note, "success": success, "spl": spl,
            "auto_success": rec.get("success"),
            "final_geodesic": rec.get("final_geodesic"),
            "l_geodesic": rec.get("l_geodesic"),
            "path_length": rec.get("path_length"),
            "run_status": rec.get("status"),
            "error": rec.get("error"),
        }
        if disp == "exclude":
            excluded.append(row)
            if kind == "error":
                errored.append(rid)
            continue
        included.append(row)
        c = r["combo"]
        d = per_combo.setdefault(c, {"n": 0, "succ": 0, "spl_sum": 0.0})
        d["n"] += 1
        d["succ"] += int(bool(success))
        d["spl_sum"] += spl

    n = len(included)
    successes = sum(1 for r in included if r["success"])
    sr = successes / n if n else 0.0
    spl = sum(r["spl"] for r in included) / n if n else 0.0
    per_combo = {
        c: {"n": d["n"], "successes": d["succ"],
            "sr": d["succ"] / d["n"], "spl": d["spl_sum"] / d["n"]}
        for c, d in per_combo.items()
    }

    return {
        "num_total": len(runs), "num_included": n,
        "num_excluded": len(excluded), "num_errored": len(errored),
        "num_missing": len(missing),
        "successes": successes, "success_rate": sr, "spl": spl,
        "per_combo": per_combo,
        "included": included, "excluded": excluded, "missing": missing,
    }


# ── Final-BEV evidence render (matplotlib, reads saved maps) ──────────────────

def _latest_map_step(run_dir):
    """Highest step index for which an occupancy map exists. Occupancy is the
    one BEV map saved in every evidence mode (the minimal path saves only it),
    so it is the common key; similarity/uncertainty exist only with --video."""
    occ_dir = os.path.join(run_dir, "occ_maps")
    if not os.path.isdir(occ_dir):
        return None
    steps = []
    for fn in os.listdir(occ_dir):
        if fn.startswith("bev_occupancy_") and fn.endswith(".npy"):
            try:
                steps.append(int(fn[len("bev_occupancy_"):-len(".npy")]))
            except ValueError:
                pass
    return max(steps) if steps else None


def render_final_bev(run_dir, record, goals, out_png):
    """Render a static final-BEV PNG with the agent's final pose and the goal
    geometry marked. Uses the occupancy map (always saved) plus similarity when
    present (--video only): two panels with --video, one occupancy panel in the
    default minimal-evidence mode. Returns the path on success, None if the
    occupancy map isn't on disk. Used as review evidence."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    extent_path = os.path.join(run_dir, "grid_extent.json")
    step = _latest_map_step(run_dir)
    if step is None or not os.path.exists(extent_path):
        return None
    ext = json.load(open(extent_path))
    min_x, max_x = ext["min_x"], ext["max_x"]
    min_z, max_z = ext["min_z"], ext["max_z"]
    extent = [min_x, max_x, min_z, max_z]

    occ = os.path.join(run_dir, "occ_maps", f"bev_occupancy_{step}.npy")
    if not os.path.exists(occ):
        return None
    panels = [(np.load(occ), "Occupancy", "gray")]
    sim = os.path.join(run_dir, "sim_maps", f"bev_similarity_{step}.npy")
    if os.path.exists(sim):
        panels.insert(0, (np.load(sim), "Similarity", "viridis"))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5),
                             squeeze=False)
    for ax, (data, title, cmap) in zip(axes[0], panels):
        ax.imshow(data, origin="lower", extent=extent, cmap=cmap, aspect="equal")
        ax.set_title(f"{title} (step {step})")
        ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
        # Goals.
        for g in goals or []:
            if isinstance(g, dict):
                x0, z0, x1, z1 = g["rect"]
                ax.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0,
                                       fill=False, edgecolor="lime", lw=2))
            else:
                ax.scatter([g[0]], [g[2]], c="lime", marker="*", s=180,
                           edgecolors="black", zorder=5)
        # Start + final agent pose.
        start = record.get("start_nav") or record.get("start")
        if start:
            ax.scatter([start[0]], [start[2]], c="deepskyblue", marker="o",
                       s=70, edgecolors="black", zorder=6, label="start")
        fp = record.get("final_pos")
        if fp:
            ax.scatter([fp[0]], [fp[2]], c="red", marker="X", s=140,
                       edgecolors="black", zorder=7, label="final")
        ax.legend(loc="upper right", fontsize=7)

    auto = "success" if record.get("success") else "fail"
    fig.suptitle(f"{record.get('id', '')}  |  auto={auto}  "
                 f"final_geo={record.get('final_geodesic', float('nan')):.2f}m  "
                 f"steps={record.get('steps')}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png
