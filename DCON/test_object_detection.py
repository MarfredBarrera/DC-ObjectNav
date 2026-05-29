"""Open-vocabulary object detectors.

All backends share the same interface — `detect(image, query) -> (score, box)` —
so the rest of the system can swap between them by config without further
plumbing. Pick one via `make_detector(name, ...)`.

- `ObjectDetector`: YOLO-Worldv2 (fast, ~15 ms/frame, class-name-style queries).
- `CocoYoloDetector`: closed-set COCO YOLOv8 (fast, high precision on the 80
  COCO classes, returns score=0 for queries that don't map to any COCO class).
- `GroundingDinoDetector`: Grounding DINO (slow, ~200 ms/frame, much better
  on natural-language phrases like "the green plant by the window").
- `HybridDetector`: routes COCO-matching queries to `CocoYoloDetector` and
  everything else to `ObjectDetector` (YOLO-World). Mirrors the VLFM paper's
  split, swapping Grounding DINO for YOLO-World on the open-vocab branch
  to stay in the ~15 ms regime.
"""

from typing import Optional, Tuple, Union
import time

import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO
from src.perception.obj_detection import ObjectDetector, CocoYoloDetector, HybridDetector, GroundingDinoDetector, SamRefinedDetector

def _to_float_rgb_tensor(image: Union[torch.Tensor, np.ndarray, Image.Image], device: str) -> torch.Tensor:
    """(H, W, 3) float in [0, 1] on `device` — MaskCLIPSemantics input format."""
    arr = ObjectDetector._to_uint8(image)
    t = torch.from_numpy(arr).to(device).float() / 255.0
    return t


def make_detector(name: str = "yolo", device: str = "cuda", **kwargs):
    """Factory: pick a detector backend by short name.

    - "yolo": fast YOLO-Worldv2 (open-vocab).
    - "coco_yolo": closed-set COCO YOLOv8.
    - "hybrid": COCO classes → YOLOv8, everything else → YOLO-Worldv2.
    - "grounding_dino": Grounding DINO Tiny via HuggingFace.
    """
    key = name.lower().strip()
    if key in ("yolo", "yolo_world", "yoloworld"):
        return ObjectDetector(device=device, **kwargs)
    if key in ("coco_yolo", "yolov8", "coco"):
        return CocoYoloDetector(device=device, **kwargs)
    if key in ("hybrid",):
        return HybridDetector(device=device, **kwargs)
    if key in ("grounding_dino", "gdino", "groundingdino"):
        return GroundingDinoDetector(device=device, **kwargs)
    if key in ("sam_refined", "sam"):
        # Composite: base detector triggers, MobileSAM proposes whole-image
        # masks, MaskCLIP scores them, best mask wins. Heavy deps imported
        # lazily so the other branches don't pay the cost.
        from src.perception.segmentation import MobileSAMSegmenter
        from src.perception.semantics import MaskCLIPSemantics
        base_name = kwargs.pop("base", "yolo")
        sam_checkpoint = kwargs.pop("sam_checkpoint", "SAM_models/mobile_sam.pt")
        min_clip_sim = float(kwargs.pop("min_clip_sim", 0.18))
        min_mask_pixels = int(kwargs.pop("min_mask_pixels", 200))
        maskclip_input_size = int(kwargs.pop("maskclip_input_size", 448))
        maskclip_model_name = kwargs.pop("maskclip_model_name", "ViT-B/16")
        base = make_detector(base_name, device=device, **kwargs)
        segmenter = MobileSAMSegmenter(checkpoint=sam_checkpoint, device=device)
        mask_clip = MaskCLIPSemantics(
            device=device, model_name=maskclip_model_name, input_size=maskclip_input_size,
        )
        return SamRefinedDetector(
            base, segmenter, mask_clip, device=device,
            min_clip_sim=min_clip_sim, min_mask_pixels=min_mask_pixels,
        )
    raise ValueError(f"Unknown detector backend: {name!r}")


if __name__ == "__main__":
    # Usage:
    #   python -m src.perception.obj_detection [backend] [query] [image_path]
    # Examples:
    #   python -m src.perception.obj_detection yolo "shower"
    #   python -m src.perception.obj_detection sam_refined "a pillow" output/current_scene/rgbs/rgb_60000.png
    import os
    import sys

    backend = sys.argv[1] if len(sys.argv) > 1 else "yolo"
    query = sys.argv[2] if len(sys.argv) > 2 else "shower"
    image_path = sys.argv[3] if len(sys.argv) > 3 else "./output/current_scene/rgbs/rgb_15900.png"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = make_detector(backend, device=device)
    image = Image.open(image_path).convert("RGB")

    # Warm up: first call pays for kernel compile + text embed (+ SAM, if used).
    detector.detect(image, query)

    # SAM auto-gen is ~1s/call so timing 10 is overkill; do 3 for sam_refined.
    N = 3 if isinstance(detector, SamRefinedDetector) else 10
    t0 = time.time()
    for _ in range(N):
        score, box = detector.detect(image, query)
    print(f"[{backend}] per-call (warmed): {(time.time() - t0) / N * 1000:.1f} ms  (N={N})")

    if box is None:
        print("No detection.")
        sys.exit(0)

    rounded = tuple(round(v, 2) for v in box)
    print(f"Detected with confidence {score:.3f} at {rounded}")

    import cv2
    bgr = cv2.imread(image_path)

    if isinstance(detector, SamRefinedDetector):
        # Compare base-detector box vs. SAM-refined mask side-by-side.
        base_score, base_box = detector.base.detect(image, query)
        _, sam_box, mask = detector.detect_with_mask(image, query)
        if base_box is not None:
            x0, y0, x1, y1 = (int(v) for v in base_box)
            cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 0, 255), 2)  # red: base detector
            cv2.putText(bgr, f"base {base_score:.2f}", (x0, max(0, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        if sam_box is not None:
            x0, y0, x1, y1 = (int(v) for v in sam_box)
            cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)  # green: SAM mask bbox
            cv2.putText(bgr, f"sam {score:.2f}", (x0, min(bgr.shape[0] - 4, y1 + 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if mask is not None:
            # Tint the mask region green at 40% so the contour is visible.
            overlay = bgr.copy()
            overlay[mask] = (0, 255, 0)
            bgr = cv2.addWeighted(overlay, 0.4, bgr, 0.6, 0)
        out_path = f"./figs/det_{backend}_{query.replace(' ', '_')}.png"
    else:
        x0, y0, x1, y1 = (int(v) for v in box)
        cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
        out_path = f"./figs/det_{backend}_{query.replace(' ', '_')}.png"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, bgr)
    print(f"Wrote {out_path}")
