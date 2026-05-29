"""MobileSAM wrapper for whole-image instance segmentation.

Used to localize the target object once the detector says it's present in
the current frame. We run SAM in automatic-mask-generation mode over the
whole RGB image (no box prompt) and let the caller pick which mask is the
target by scoring against the CLIP text embedding of the query. Rationale:
at distance the detector's box is often offset by ~one box width, so
anchoring SAM on that box smears the mask onto walls / floor. Whole-image
auto-gen sidesteps that failure mode entirely — the box only triggers
segmentation, it does not constrain it.
"""

from typing import Dict, List, Union

import numpy as np
import torch
from PIL import Image


class MobileSAMSegmenter:
    """Run MobileSAM in automatic-mask-generation mode on the whole image.

    Returns one mask dict per discovered object (keys: ``segmentation``,
    ``area``, ``bbox``, ``predicted_iou``, ``stability_score``). The caller
    is responsible for picking the target mask (e.g. by mean CLIP cosine
    similarity to the target query inside each mask).
    """

    def __init__(
        self,
        checkpoint: str = "SAM_models/mobile_sam.pt",
        device: str = "cuda",
        points_per_side: int = 16,
        pred_iou_thresh: float = 0.86,
        stability_score_thresh: float = 0.90,
        min_mask_region_area: int = 200,
    ):
        # Lazy import so users without mobile_sam installed don't pay the
        # import cost or hit ModuleNotFoundError unless they actually
        # instantiate the segmenter. The UserWarning filter swallows
        # mobile_sam's timm-registry collision on tiny_vit_5m_224 — that
        # overwrite is intentional (mobile_sam patches the model) and the
        # warning fires every process start otherwise.
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Overwriting tiny_vit_\d+m_\d+ in registry.*",
                category=UserWarning,
            )
            from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry

        sam = sam_model_registry["vit_t"](checkpoint=checkpoint)
        sam.to(device=device)
        sam.eval()
        self.device = device
        self.generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
        )
        # Side channel — populated by bev_cells_from_sam() in main.py
        # whenever a mask clears min_clip_sim. Lets the main loop persist
        # the chosen mask to disk for visualization without expanding
        # plan_one_action's return tuple.
        self.last_mask: "np.ndarray | None" = None
        self.last_box: "tuple | None" = None
        self.last_score: float = 0.0

    @torch.no_grad()
    def segment_all(
        self,
        rgb: Union[torch.Tensor, np.ndarray, Image.Image],
    ) -> List[Dict]:
        arr = _to_uint8(rgb)
        return self.generator.generate(arr)


def _to_uint8(image: Union[torch.Tensor, np.ndarray, Image.Image]) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.asarray(image)
    if arr.dtype != np.uint8:
        if float(arr.max()) <= 1.0 + 1e-3:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB image, got shape {arr.shape}")
    return arr
