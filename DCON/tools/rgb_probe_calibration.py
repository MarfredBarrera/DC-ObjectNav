"""Collect real detection-candidate frames from saved episode RGB histories,
score each candidate box with plain-sigmoid and contrastive-softmax CLIPSeg
directly on the frame (CLIPSegSemantics, no trained 3D field involved), and
dump crops for visual (human) TP/FP labeling.

This sidesteps both confounds of the field_score-based calibration: the
trained-field cold-start bias, and the distance-to-goal TP/FP proxy.

Usage (inside the container):
    python tools/rgb_probe_calibration.py --episodes-root output/fieldverify_videos \
        --out /tmp/rgb_probe
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.perception.obj_detection import LLMDetDetector
from src.perception.semantics import CLIPSegSemantics


def pool_topk_in_box(score_map: torch.Tensor, box, top_frac: float = 0.10) -> float:
    """Mean of the top `top_frac` fraction of pixel scores inside `box`
    (xmin, ymin, xmax, ymax), mirroring PerceptionStack.field_score_in_box's
    pooling but directly in image space (no unprojection)."""
    H, W = score_map.shape[:2]
    x0 = max(0, int(box[0])); y0 = max(0, int(box[1]))
    x1 = min(W, int(np.ceil(box[2]))); y1 = min(H, int(np.ceil(box[3])))
    if x1 <= x0 or y1 <= y0:
        return float("nan")
    region = score_map[y0:y1, x0:x1, 0].reshape(-1)
    k = max(1, int(round(region.numel() * top_frac)))
    return float(torch.topk(region, k).values.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes-root", required=True,
                    help="Dir containing runs/<id>.json + ep/<id>/rgbs/*.png")
    ap.add_argument("--out", required=True, help="Where to dump crops + manifest.json")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--max-boxes-per-frame", type=int, default=1)
    ap.add_argument("--episode-glob", default="*",
                    help="fnmatch pattern over run ids, e.g. '*__r0' to take "
                         "only the first repeat of each combo")
    ap.add_argument("--append", action="store_true",
                    help="Append to an existing manifest.json in --out instead "
                         "of overwriting it")
    args = ap.parse_args()

    cfg = Config(args.config)
    os.makedirs(args.out, exist_ok=True)

    detector = LLMDetDetector.from_config(cfg)

    import fnmatch
    manifest = []
    if args.append:
        existing_path = os.path.join(args.out, "manifest.json")
        if os.path.exists(existing_path):
            with open(existing_path) as f:
                manifest = json.load(f)
    run_paths = sorted(glob.glob(os.path.join(args.episodes_root, "runs", "*.json")))
    run_paths = [rp for rp in run_paths
                if fnmatch.fnmatch(os.path.splitext(os.path.basename(rp))[0], args.episode_glob)]
    for rp in run_paths:
        with open(rp) as f:
            rec = json.load(f)
        rid = rec.get("id") or os.path.splitext(os.path.basename(rp))[0]
        query = rec["query"]
        rgb_dir = os.path.join(args.episodes_root, "ep", rid, "rgbs")
        if not os.path.isdir(rgb_dir):
            continue

        plain_sem = CLIPSegSemantics(
            query=query, device=cfg.device, model_name=cfg.clipseg_model_name,
            distractors=None, softmax_temp=cfg.clipseg_softmax_temp)
        contrastive_sem = CLIPSegSemantics(
            query=query, device=cfg.device, model_name=cfg.clipseg_model_name,
            distractors=cfg.distractor_objects, softmax_temp=cfg.clipseg_softmax_temp)

        frame_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
        print(f"[{rid}] query={query!r}, {len(frame_paths)} frame(s)")
        for fp in frame_paths:
            step = os.path.splitext(os.path.basename(fp))[0].replace("rgb_", "")
            arr = np.asarray(Image.open(fp).convert("RGB"))
            boxes = detector.detect_all(arr, query)
            if not boxes:
                continue

            rgb_t = torch.from_numpy(arr).to(cfg.device).float() / 255.0
            plain_map = plain_sem.extract_dense_features(rgb_t)
            contrastive_map = contrastive_sem.extract_dense_features(rgb_t)

            for score, box in boxes[:args.max_boxes_per_frame]:
                plain_s = pool_topk_in_box(plain_map, box, args.top_frac)
                contrastive_s = pool_topk_in_box(contrastive_map, box, args.top_frac)

                root_tag = os.path.basename(os.path.normpath(args.episodes_root))
                crop_name = f"{root_tag}__{rid}_step{step}.png"
                annotated = Image.fromarray(arr).convert("RGB")
                d = ImageDraw.Draw(annotated)
                d.rectangle(list(box), outline=(255, 0, 0), width=3)
                annotated.save(os.path.join(args.out, crop_name))

                item = {
                    "episode": rid, "query": query, "step": step,
                    "box": [float(v) for v in box],
                    "llmdet_score": float(score),
                    "plain_score": plain_s, "contrastive_score": contrastive_s,
                    "image": crop_name,
                }
                manifest.append(item)
                print(f"  step {step}: llmdet={score:.3f} plain={plain_s:.3f} "
                      f"contrastive={contrastive_s:.3f} -> {crop_name}")

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n{len(manifest)} candidate(s) written to {args.out}/manifest.json "
          f"(+ annotated crops)")


if __name__ == "__main__":
    main()
