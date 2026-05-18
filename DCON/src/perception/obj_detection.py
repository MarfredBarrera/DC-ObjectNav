"""YOLO-World open-vocabulary object detector.

Given an RGB frame and a free-form text query, returns the highest-confidence
detection's score and bounding box. Used to gate exploration→approach behavior
on visible-target confidence. Swapped from OWLv2 for ~20× lower inference time
at the cost of some accuracy on unusual queries.
"""

from typing import Optional, Tuple, Union
import time

import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO


class ObjectDetector:
    """Open-vocabulary detector using YOLO-World (via Ultralytics)."""

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


if __name__ == "__main__":
    detector = ObjectDetector(device="cuda" if torch.cuda.is_available() else "cpu")
    image_path = "./output/current_scene/rgbs/rgb_18000.png"
    image = Image.open(image_path)

    # Warm up: first call pays for kernel compile + CLIP text embed.
    detector.detect(image, "a photo of a bed")

    N = 10
    t0 = time.time()
    for _ in range(N):
        score, box = detector.detect(image, "a photo of a bed")
    print(f"per-call (warmed): {(time.time() - t0) / N * 1000:.1f} ms")

    if box is None:
        print("No detection.")
    else:
        rounded = tuple(round(v, 2) for v in box)
        print(f"Detected with confidence {score:.3f} at {rounded}")
