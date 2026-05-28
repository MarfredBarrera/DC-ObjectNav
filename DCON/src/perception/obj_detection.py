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
    raise ValueError(f"Unknown detector backend: {name!r}")


if __name__ == "__main__":
    import sys
    backend = sys.argv[1] if len(sys.argv) > 1 else "yolo"
    detector = make_detector(backend, device="cuda" if torch.cuda.is_available() else "cpu")
    image_path = "./output/current_scene/rgbs/rgb_13000.png"
    image = Image.open(image_path)

    # Warm up: first call pays for kernel compile + text embed.
    detector.detect(image, "a small green potted plant")

    N = 10
    t0 = time.time()
    for _ in range(N):
        score, box = detector.detect(image, "a small green potted plant")
    print(f"[{backend}] per-call (warmed): {(time.time() - t0) / N * 1000:.1f} ms")

    if box is None:
        print("No detection.")
    else:
        rounded = tuple(round(v, 2) for v in box)
        print(f"Detected with confidence {score:.3f} at {rounded}")

    # Display image with box
    import cv2
    image = cv2.imread(image_path)
    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    cv2.imwrite("./figs/tv_gdino.png", image)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
