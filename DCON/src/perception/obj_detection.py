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


class ObjectDetector:
    """Open-vocabulary detector using YOLO-World (via Ultralytics)."""

    name = "yolo_world"

    def backend_for(self, query: str) -> str:
        return self.name


    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "yolo/yolov8s-worldv2.pt",
        threshold: float = 0.1,
        imgsz: int = 640,
        half: bool = True,
    ):
        self.device = device
        self.threshold = float(threshold)
        self.imgsz = int(imgsz)
        # fp16 only on CUDA; Ultralytics silently no-ops it on CPU.
        self.half = bool(half) and "cuda" in str(device)
        self.model = YOLO(model_name)
        self.model.to(device)
        # set_classes() re-embeds the query through CLIP, which is non-trivial.
        # Cache the current query so back-to-back calls with the same text
        # skip the embedding step.
        self._current_query: Optional[str] = None

    @torch.no_grad()
    def detect(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        query: str,
    ) -> Tuple[float, Optional[Tuple[float, float, float, float]]]:
        """Run detection on a single image with a single text query.

        Args:
            image: RGB image. Accepts PIL.Image, numpy uint8 (H, W, 3),
                or torch tensor (H, W, 3) in [0, 1] or [0, 255].
            query: Free-form text query, e.g. "a green plant".

        Returns:
            (score, box) where `score` ∈ [0, 1] and `box` is
            (xmin, ymin, xmax, ymax) in pixel coordinates of the input image.
            If no detection clears `self.threshold`, returns (0.0, None).
        """
        if query != self._current_query:
            self.model.set_classes([query])
            self._current_query = query

        arr = self._to_uint8(image)

        results = self.model.predict(
            arr,
            imgsz=self.imgsz,
            half=self.half,
            conf=self.threshold,
            verbose=False,
            device=self.device,
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return 0.0, None

        confs = r.boxes.conf
        best = int(torch.argmax(confs))
        score = float(confs[best])
        xyxy = r.boxes.xyxy[best].tolist()
        box = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
        return score, box

    @staticmethod
    def _to_uint8(image: Union[torch.Tensor, np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(image, Image.Image):
            return np.asarray(image.convert("RGB"))
        if isinstance(image, torch.Tensor):
            arr = image.detach().cpu().numpy()
        else:
            arr = np.asarray(image)
        # Float arrays expected in [0, 1]; uint8 in [0, 255].
        if arr.dtype != np.uint8:
            if float(arr.max()) <= 1.0 + 1e-3:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) RGB image, got shape {arr.shape}")
        return arr


class CocoYoloDetector:
    """Closed-set COCO detector using YOLOv8 (via Ultralytics).

    Restricts predictions to a single class derived from the query string.
    `query_to_class_id` normalizes the query (lower-case, strip leading
    article, strip trailing period) and looks it up in `self.model.names`,
    with a small synonym table for common ObjectNav targets that don't
    exactly match a COCO label. Returns (0.0, None) when the query can't
    be mapped to a COCO class — the hybrid wrapper uses that to fall back.
    """

    name = "coco_yolo"

    def backend_for(self, query: str) -> str:
        return self.name


    # Query strings that don't exactly match a COCO label but should still
    # route through the closed-set detector.
    _SYNONYMS = {
        "plant": "potted plant",
        "sofa": "couch",
        "television": "tv",
    }

    _ARTICLES = ("a ", "an ", "the ")

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "yolo/yolov8s.pt",
        threshold: float = 0.25,
        imgsz: int = 640,
        half: bool = True,
    ):
        self.device = device
        self.threshold = float(threshold)
        self.imgsz = int(imgsz)
        self.half = bool(half) and "cuda" in str(device)
        self.model = YOLO(model_name)
        self.model.to(device)
        # Inverse of model.names ({id: name}) so we can convert a class name
        # straight to the integer id Ultralytics wants in `classes=`.
        self.name_to_id = {name: idx for idx, name in self.model.names.items()}
        self._current_query: Optional[str] = None
        self._current_class_id: Optional[int] = None

    def query_to_class_id(self, query: str) -> Optional[int]:
        """Map a free-form query to a COCO class id, or None if no match."""
        q = query.strip().lower().rstrip(".")
        for art in self._ARTICLES:
            if q.startswith(art):
                q = q[len(art):]
                break
        q = self._SYNONYMS.get(q, q)
        return self.name_to_id.get(q)

    @torch.no_grad()
    def detect(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        query: str,
    ) -> Tuple[float, Optional[Tuple[float, float, float, float]]]:
        if query != self._current_query:
            self._current_query = query
            self._current_class_id = self.query_to_class_id(query)
        if self._current_class_id is None:
            return 0.0, None

        arr = ObjectDetector._to_uint8(image)
        results = self.model.predict(
            arr,
            imgsz=self.imgsz,
            half=self.half,
            conf=self.threshold,
            classes=[self._current_class_id],
            verbose=False,
            device=self.device,
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return 0.0, None
        confs = r.boxes.conf
        best = int(torch.argmax(confs))
        score = float(confs[best])
        xyxy = r.boxes.xyxy[best].tolist()
        box = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
        return score, box


class HybridDetector:
    """COCO-matching queries → closed-set YOLOv8; everything else → YOLO-World.

    Mirrors the VLFM paper's hybrid scheme. Routing decision is cached per
    query string, so back-to-back calls with the same query skip the
    normalization + lookup.
    """

    name = "hybrid"

    def backend_for(self, query: str) -> str:
        backend = self._route_cache.get(query)
        if backend is None:
            backend = self.coco if self.coco.query_to_class_id(query) is not None else self.world
            self._route_cache[query] = backend
        return backend.name


    def __init__(
        self,
        device: str = "cuda",
        coco_kwargs: Optional[dict] = None,
        world_kwargs: Optional[dict] = None,
    ):
        self.coco = CocoYoloDetector(device=device, **(coco_kwargs or {}))
        self.world = ObjectDetector(device=device, **(world_kwargs or {}))
        self._route_cache: dict = {}

    def detect(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        query: str,
    ) -> Tuple[float, Optional[Tuple[float, float, float, float]]]:
        backend = self._route_cache.get(query)
        if backend is None:
            backend = self.coco if self.coco.query_to_class_id(query) is not None else self.world
            self._route_cache[query] = backend
        return backend.detect(image, query)


class GroundingDinoDetector:
    """Open-vocabulary detector using Grounding DINO via HuggingFace transformers.

    Slower than YOLO-World but considerably better at natural-language queries
    ("the small red mug on the table"). Score scale differs from YOLO-World —
      strong detections typically sit at 0.35–0.55 rather than 0.5–0.8, so any
    downstream threshold tuned for YOLO-World likely needs to be retuned.
    """

    name = "gdino"

    def backend_for(self, query: str) -> str:
        return self.name


    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "IDEA-Research/grounding-dino-tiny",
        threshold: float = 0.25,
        text_threshold: float = 0.25,
    ):
        # Imported here so the YOLO-only path doesn't pay the transformers
        # import cost or require it to be installed.
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.device = device
        self.threshold = float(threshold)
        self.text_threshold = float(text_threshold)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device)
        self.model.eval()
        # Cache the last query text — re-tokenizing is cheap but the
        # post-processor expects the same string used at inference time.
        self._current_query: Optional[str] = None
        self._query_text: Optional[str] = None

    @torch.no_grad()
    def detect(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        query: str,
    ) -> Tuple[float, Optional[Tuple[float, float, float, float]]]:
        # GD expects the prompt to end with a period (separator between phrases).
        if query != self._current_query:
            self._current_query = query
            self._query_text = query.strip().rstrip(".") + "."

        arr = ObjectDetector._to_uint8(image)
        pil = Image.fromarray(arr)

        inputs = self.processor(images=pil, text=self._query_text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # post_process expects (H, W) target sizes for box rescaling.
        h, w = arr.shape[:2]
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            # box_threshold=self.threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(h, w)],
        )[0]

        scores = results.get("scores")
        boxes = results.get("boxes")
        if scores is None or len(scores) == 0:
            return 0.0, None

        best = int(torch.argmax(scores))
        score = float(scores[best])
        xyxy = boxes[best].tolist()
        box = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
        return score, box


class SamRefinedDetector:
    """Detector → MobileSAM (whole-image) → CLIP-scored best mask.

    Pipeline (mirrors what `bev_cells_from_sam` does in main.py, minus the
    BEV projection so it can be tested standalone on a single RGB):

      1. Run a base detector to decide whether the query target is present.
         If no box clears its threshold, return (0.0, None).
      2. Run MobileSAM's automatic mask generator on the WHOLE image
         (no box prompt — at distance the box is often a full box-width
         off, so anchoring SAM on it propagates the error).
      3. Score every mask by mean MaskCLIP cosine similarity to ``query``.
      4. Pick the highest-scoring mask. If it clears ``min_clip_sim``,
         return that mask's bbox + the CLIP score. Otherwise (0.0, None).

    `detect(image, query)` matches the base interface and returns
    (score, box) from the refined mask. `detect_with_mask(image, query)`
    additionally returns the (H, W) bool mask itself — used by the
    __main__ visualization to overlay the contour.
    """

    name = "sam_refined"

    def backend_for(self, query: str) -> str:
        return self.name

    def __init__(
        self,
        base_detector,
        segmenter,
        mask_clip,
        device: str = "cuda",
        min_clip_sim: float = 0.18,
        min_mask_pixels: int = 200,
    ):
        self.base = base_detector
        self.segmenter = segmenter
        self.mask_clip = mask_clip
        self.device = device
        self.min_clip_sim = float(min_clip_sim)
        self.min_mask_pixels = int(min_mask_pixels)

    @torch.no_grad()
    def detect_with_mask(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        query: str,
    ) -> Tuple[float, Optional[Tuple[float, float, float, float]], Optional[np.ndarray]]:
        det_score, det_box = self.base.detect(image, query)
        if det_box is None or det_score <= 0.0:
            return 0.0, None, None

        masks = self.segmenter.segment_all(image)
        if not masks:
            return 0.0, None, None

        rgb_gpu = _to_float_rgb_tensor(image, self.device)
        feats = self.mask_clip.extract_dense_features(rgb_gpu)  # (H, W, 512)
        text_embed = self.mask_clip.encode_text(query)          # (1, 512)
        sim_2d = (feats @ text_embed.T).squeeze(-1)             # (H, W) in [-1, 1]
        H, W = sim_2d.shape

        best_score = -float("inf")
        best_seg = None
        for m in masks:
            seg = m["segmentation"]
            if seg.sum() < self.min_mask_pixels:
                continue
            seg_t = torch.from_numpy(seg).to(self.device)
            if seg_t.shape != (H, W):
                seg_t = torch.nn.functional.interpolate(
                    seg_t[None, None].float(), size=(H, W), mode="nearest",
                )[0, 0].bool()
            if not bool(seg_t.any()):
                continue
            score = float(sim_2d[seg_t].mean())
            if score > best_score:
                best_score = score
                best_seg = seg

        if best_seg is None or best_score < self.min_clip_sim:
            return 0.0, None, None

        ys, xs = np.where(best_seg)
        box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        return best_score, box, best_seg

    @torch.no_grad()
    def detect(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        query: str,
    ) -> Tuple[float, Optional[Tuple[float, float, float, float]]]:
        score, box, _ = self.detect_with_mask(image, query)
        return score, box


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