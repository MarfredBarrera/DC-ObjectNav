"""OWLv2-based open-vocabulary object detector.

Given an RGB frame and a free-form text query, returns the highest-confidence
detection's score and bounding box. Used to gate exploration→approach behavior
on visible-target confidence.
"""

from typing import Optional, Tuple, Union
import time
import numpy as np
import torch
from PIL import Image

from transformers import Owlv2ForObjectDetection, Owlv2Processor


class ObjectDetector:
    """Wraps OWLv2 for single-image, single-query open-vocabulary detection."""

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "google/owlv2-base-patch16",
        threshold: float = 0.1,
    ):
        self.device = device
        self.threshold = float(threshold)
        self.processor = Owlv2Processor.from_pretrained(model_name, use_fast=True)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(device).eval()

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
        pil = self._to_pil(image)
        text_labels = [[query]]

        inputs = self.processor(text=text_labels, images=pil, return_tensors="pt").to(self.device)
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([(pil.height, pil.width)], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.threshold,
            text_labels=text_labels,
        )[0]

        scores = results["scores"]
        boxes = results["boxes"]
        if scores.numel() == 0:
            return 0.0, None

        best = int(torch.argmax(scores))
        score = float(scores[best])
        box = tuple(float(v) for v in boxes[best].tolist())
        return score, box

    @staticmethod
    def _to_pil(image: Union[torch.Tensor, np.ndarray, Image.Image]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, torch.Tensor):
            arr = image.detach().cpu().numpy()
        else:
            arr = np.asarray(image)
        # Float arrays are expected in [0, 1]; uint8 in [0, 255].
        if arr.dtype != np.uint8:
            if float(arr.max()) <= 1.0 + 1e-3:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) RGB image, got shape {arr.shape}")
        return Image.fromarray(arr)


if __name__ == "__main__":
    # Smoke test against the canonical COCO val image.
    detector = ObjectDetector(device="cuda" if torch.cuda.is_available() else "cpu")
    image_path = "./output/current_scene/rgbs/rgb_14000.png"
    image = Image.open(image_path)
    N = 5
    t0 = time.time()
    for _ in range(N):
        score, box = detector.detect(image, "a photo of a bed")
    print(f"per-call: {(time.time() - t0) / N * 1000:.1f} ms")
    if box is None:
        print("No detection.")
    else:
        rounded = tuple(round(v, 2) for v in box)
        print(f"Detected with confidence {score:.3f} at {rounded}")
