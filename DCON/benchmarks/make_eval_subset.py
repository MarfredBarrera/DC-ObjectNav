#!/usr/bin/env python3
"""Materialize a standardized ObjectNav-val subset for cross-system eval.

Produces a filtered copy of the Gibson v1.1 val split containing exactly
`--per-combo` episodes per (scene x object category), selected as the lowest
episode_ids (deterministic, trivially reproducible). Because the output is a
standard habitat/SemExp split (same schema, same `<split>_info.pbz2` goal
source, same `.glb.json.gz` stubs), BOTH VLFM (habitat-lab) and DCON
(`eval_scene.py --dataset`) can point their data path straight at it and run
the identical episode set.

    python benchmarks/make_eval_subset.py \
        --src datasets/gibson/v1.1/val \
        --out datasets/gibson/v1.1_sub5/val \
        --per-combo 5

Then:
    DCON:  python benchmarks/eval_scene.py run --dataset <out> --out output/sub5 --discrete ...
    VLFM:  set the habitat dataset data_path to <out>/  (or <out>/val.json.gz)

A `manifest.csv` (scene, category, episode_id) is also written next to the
output as a portable source of truth, in case a system needs to be constrained
by explicit episode_id instead of by the filtered files.
"""
import argparse, csv, gzip, json, os, shutil
from collections import defaultdict


def _read(fp):
    op = gzip.open if fp.endswith(".gz") else open
    with op(fp, "rt") as f:
        return json.load(f)


def _write_gz(obj, fp):
    with gzip.open(fp, "wt") as f:
        json.dump(obj, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/gibson/v1.1/val",
                    help="source split dir (must contain content/ and <split>_info.pbz2)")
    ap.add_argument("--out", default="datasets/gibson/v1.1_sub5/val",
                    help="destination split dir to create")
    ap.add_argument("--per-combo", type=int, default=5,
                    help="episodes per (scene x category) to keep")
    args = ap.parse_args()

    src_content = os.path.join(args.src, "content")
    out_content = os.path.join(args.out, "content")
    os.makedirs(out_content, exist_ok=True)

    manifest = []  # (scene_stem, category, episode_id)
    total_kept = 0

    for fn in sorted(os.listdir(src_content)):
        src_fp = os.path.join(src_content, fn)
        if not fn.endswith("_episodes.json.gz"):
            # .glb.json.gz stubs and anything else: copy verbatim (harmless; the
            # stubs carry no category and both loaders skip them).
            shutil.copy2(src_fp, os.path.join(out_content, fn))
            continue

        scene_stem = fn.replace("_episodes.json.gz", "")
        data = _read(src_fp)
        by_cat = defaultdict(list)
        for ep in data.get("episodes", []):
            by_cat[ep["object_category"]].append(ep)

        kept = []
        for cat in sorted(by_cat):
            chosen = sorted(by_cat[cat], key=lambda e: int(e["episode_id"]))[:args.per_combo]
            kept.extend(chosen)
            for e in chosen:
                manifest.append((scene_stem, cat, str(e["episode_id"])))

        # Preserve original ordering (by episode_id) in the written file.
        kept.sort(key=lambda e: int(e["episode_id"]))
        out = dict(data)
        out["episodes"] = kept
        _write_gz(out, os.path.join(out_content, fn))
        total_kept += len(kept)
        print(f"  {scene_stem:14s} kept {len(kept):3d} "
              f"({len(by_cat)} categories x <= {args.per_combo})")

    # Copy the top-level split files (val.json.gz stub + goal-source pbz2).
    for fn in os.listdir(args.src):
        p = os.path.join(args.src, fn)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(args.out, fn))

    man_fp = os.path.join(os.path.dirname(args.out.rstrip("/")), "manifest.csv")
    with open(man_fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "category", "episode_id"])
        w.writerows(sorted(manifest))

    print(f"\nWrote {total_kept} episodes to {args.out}")
    print(f"Manifest ({len(manifest)} rows): {man_fp}")


if __name__ == "__main__":
    main()
