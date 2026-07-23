"""Detection gating: detector box → 3D geometry → usability tier → EXPLOIT latch.

Everything between "the detector emitted a box" and "the planner may act on it":

    1. project the box center to a world point / BEV cell (`box_center_world_xz`,
       `bev_cell_from_box_center`),
    2. verify the box against the learned relevance field (the pairwise margin
       gate — this is what suppresses geometric look-alike false positives),
    3. classify it by distance + apparent size into too-close / too-far /
       usable-band (`classify_detection`),
    4. advance the SEARCH→EXPLOIT latch (`DetectionGate`).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from src.perception.utils import unprojection


@dataclass
class Detection:
    """One replan's detection outcome, post field-verify and classification.

    score        — detector confidence (0.0 when the detector was skipped or
                   the box was rejected by the field gate)
    box          — accepted box (xmin, ymin, xmax, ymax) in depth pixel space, or None
    persistent   — usable band: counts toward the latch streak
    investigate  — not too-close: drives the goal cell + confidence weight
    conf_score   — the goal weight MPPI sees in SEARCH (score, or 0.0 if ignored)
    field_score  — pooled field score / pairwise margin (None if the gate didn't run)
    """
    score: float = 0.0
    box: Optional[tuple] = None
    persistent: bool = False
    investigate: bool = False
    conf_score: float = 0.0
    field_score: Optional[float] = None


def box_center_world_xz(perception, cfg, intrinsics, det_box, depth, c2w):
    """Median world (x, z) of a small patch around the center of `det_box`.

    Unprojects only a small window (not the single center pixel, so one depth
    hole at the exact center doesn't drop the result) and returns the median
    world point's BEV-plane coordinates (wx, wz), or None if no valid depth
    near the center. Shared by goal projection and the detection size/distance
    gate. Box pixel coords must be in `depth`'s pixel space.
    """
    if det_box is None or depth is None or c2w is None:
        return None
    xmin, ymin, xmax, ymax = det_box
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    depth_gpu = depth.to(perception.device)
    H_d, W_d = depth_gpu.shape[-2:]
    # Half-window: 5% of the smaller box side, clamped to >=1px.
    half = max(1, int(0.05 * min(xmax - xmin, ymax - ymin)))
    yy, xx = torch.meshgrid(
        torch.arange(H_d, device=perception.device),
        torch.arange(W_d, device=perception.device),
        indexing='ij',
    )
    win = ((xx >= int(cx) - half) & (xx <= int(cx) + half) &
           (yy >= int(cy) - half) & (yy <= int(cy) + half))
    depth_mask = (depth_gpu > cfg.min_sensor_dist) & (depth_gpu < cfg.max_sensor_dist)
    mask = win & depth_mask
    if not bool(mask.any()):
        return None
    world_points = unprojection(
        depth_gpu, intrinsics, c2w.to(perception.device), perception.device, mask=mask)
    if world_points.shape[0] == 0:
        return None
    return float(world_points[:, 0].median()), float(world_points[:, 2].median())


def bev_cell_from_box_center(perception, cfg, intrinsics, det_box, depth, c2w):
    """Single BEV (z_idx, x_idx) cell the *center* of `det_box` projects to.

    Projects only the box-center patch to one world point and returns its lone
    BEV cell, used as the goal verbatim — no similarity argmax, no snap-to-free.
    Relies on LLMDet emitting tight, accurate boxes. Returns (z_idx, x_idx)
    or None.
    """
    wxz = box_center_world_xz(perception, cfg, intrinsics, det_box, depth, c2w)
    if wxz is None:
        return None
    wx, wz = wxz
    return world_to_grid(wx, wz, perception.similarity_grid, cfg.voxel_resolution)


def world_to_grid(x_world: float, z_world: float, ref_grid, res: float):
    """World (x, z) → the BEV (z_idx, x_idx) cell containing it, clamped in-bounds."""
    z_idx = int((z_world - ref_grid.min_z) / res)
    x_idx = int((x_world - ref_grid.min_x) / res)
    z_idx = max(0, min(z_idx, ref_grid.num_z - 1))
    x_idx = max(0, min(x_idx, ref_grid.num_x - 1))
    return (z_idx, x_idx)


def grid_to_world_xz(cell, ref_grid, res: float):
    """BEV (z_idx, x_idx) cell → the world (x, z) of its CENTER."""
    return (ref_grid.min_x + (cell[1] + 0.5) * res,
            ref_grid.min_z + (cell[0] + 0.5) * res)


def classify_detection(perception, cfg, intrinsics, det_box, depth, c2w,
                       agent_pos):
    """Classify a detection by object distance + box size into how it may be used.

    Returns (is_persistent, contributes_confidence):
      - TOO CLOSE — object distance < cfg.detected_min_dist_m OR box covers more
        than cfg.detected_max_box_frac of the frame. The box fills the view and
        carries no usable localization → (False, False): ignored entirely
        (no goal, no confidence weight, no latch).
      - TOO FAR — object distance > cfg.detected_max_dist_m OR box smaller than
        cfg.detected_min_box_frac of the frame. A distant, uncertain sighting →
        (False, True): investigated (projected + cached as the goal, pulls the
        confidence weight) but not persistent — it never counts toward the latch.
      - USABLE BAND (anything else) → (True, True): a persistent detection that
        latches and is cached as the goal, and contributes the confidence weight.

    Each threshold disables at a non-positive value. A missing box or
    unrangeable depth (no valid depth at the box center) → (False, False).
    """
    if det_box is None:
        return (False, False)
    xmin, ymin, xmax, ymax = det_box
    H_img, W_img = depth.shape[-2:]
    box_frac = ((xmax - xmin) * (ymax - ymin)) / float(W_img * H_img)
    wxz = box_center_world_xz(perception, cfg, intrinsics, det_box, depth, c2w)
    if wxz is None:
        return (False, False)
    dist_m = float(np.hypot(wxz[0] - agent_pos[0], wxz[1] - agent_pos[2]))

    box_too_large = cfg.detected_max_box_frac > 0.0 and box_frac > cfg.detected_max_box_frac
    box_too_small = cfg.detected_min_box_frac > 0.0 and box_frac < cfg.detected_min_box_frac
    dist_too_small = cfg.detected_min_dist_m > 0.0 and dist_m < cfg.detected_min_dist_m
    dist_too_large = cfg.detected_max_dist_m > 0.0 and dist_m > cfg.detected_max_dist_m

    if dist_too_small or box_too_large:
        return (False, False)   # too close: ignore entirely
    if dist_too_large or box_too_small:
        return (False, True)    # too far: confidence only, not persistent
    return (True, True)         # usable band


class DetectionGate:
    """Owns the detector, the field-verify gate, and the SEARCH→EXPLOIT latch.

    `detected` latches True once `cfg.detected_persistence` consecutive
    *persistent* (usable-band) detections accrue, and never releases. The caller
    reads it to pick the control mode; the streak state stays in here rather
    than being threaded through the loop.
    """

    def __init__(self, cfg, detector, perception):
        self.cfg = cfg
        self.detector = detector
        self.perception = perception
        self.detected = False
        self.streak = 0

    def should_run_detector(self, replan_idx: int) -> bool:
        """Detector throttle. SEARCH always detects; EXPLOIT reuses the cached
        goal cell, so it only re-detects every `cfg.exploit_redetect_interval`
        replans (<= 0 → never re-detect after latching)."""
        if not self.detected:
            return True
        if self.cfg.exploit_redetect_interval <= 0:
            return False
        return replan_idx % self.cfg.exploit_redetect_interval == 0

    def step(self, sim_iface, rgb, depth, c2w, pos,
             run_detector: bool = True, tag: str = "") -> Detection:
        """Run the detector, verify + classify the box, and advance the latch.

        Classifies the box by object distance + box size into three tiers (see
        `classify_detection`): *too close* → ignored; *too far* → investigate
        (steer + confidence) but not persistent; *usable band* → persistent
        (also counts toward the latch streak).

        When `cfg.field_verify` is on, a detector box must additionally be
        confirmed by the learned relevance field before it counts: the box's
        valid-depth pixels are unprojected to 3D, the field is queried there,
        and the pooled score (`cfg.field_verify_pool`: top-
        `cfg.field_verify_top_frac` mean, or max) must clear
        `cfg.field_verify_threshold` (see PerceptionStack.field_score_in_box).
        A frame that fails the gate is treated as no detection at all — no
        goal, no confidence, no latch.
        """
        cfg, perception = self.cfg, self.perception
        if not run_detector:
            return Detection()

        det_score, det_box = self.detector.detect(rgb, perception.target_query)

        field_score = None
        if cfg.field_verify and det_box is not None:
            field_score = perception.field_score_in_box(
                depth, c2w, sim_iface.intrinsics, det_box,
                top_frac=cfg.field_verify_top_frac,
                min_points=cfg.field_verify_min_points,
                pool=cfg.field_verify_pool)
            if not self._field_accepts(det_score, field_score, tag):
                det_score, det_box = 0.0, None

        det_persistent, det_investigate = classify_detection(
            perception, cfg, sim_iface.intrinsics, det_box, depth, c2w, pos)
        # No separate score gate here: the detector's own floor (llmdet_threshold)
        # already bounds the score of any surviving box, so every usable-band
        # detection counts toward the latch.
        if not self.detected:
            self.streak = self.streak + 1 if det_persistent else 0
            if self.streak >= cfg.detected_persistence:
                self.detected = True
                print(f"{tag}: DETECTED — entering exploit mode "
                      f"(det_score={det_score:.3f})")

        return Detection(
            score=det_score, box=det_box,
            persistent=det_persistent, investigate=det_investigate,
            conf_score=det_score if det_investigate else 0.0,
            field_score=field_score)

    def _field_accepts(self, det_score, field_score, tag) -> bool:
        """Pairwise field-verify gate: worst-case margin over the threshold, and
        (separately) the query channel over the presence floor.

        The presence floor is a distinct "is anything here" conjunct, kept apart
        from the margin threshold so the two scales stay decoupled (0.0 disables
        it; see config.py).
        """
        cfg = self.cfg
        fv = self.perception.last_field_verify
        presence_ok = True
        if (cfg.clipseg_pairwise and fv is not None
                and cfg.field_verify_presence_floor > 0.0):
            presence_ok = fv["presence"] >= cfg.field_verify_presence_floor
        if (field_score is None or field_score < cfg.field_verify_threshold
                or not presence_ok):
            fs = "n/a" if field_score is None else f"{field_score:.3f}"
            why = (f"presence={fv['presence']:.3f} < floor "
                   f"{cfg.field_verify_presence_floor:.2f}"
                   if (field_score is not None
                       and field_score >= cfg.field_verify_threshold)
                   else f"field={fs} < {cfg.field_verify_threshold:.2f}")
            print(f"{tag}: field-verify REJECTED detection "
                  f"(llmdet={det_score:.3f}, {why})")
            return False
        extra = (f", presence={fv['presence']:.3f}"
                 if (fv is not None and fv["margin"] is not None) else "")
        print(f"{tag}: field-verify accepted "
              f"(llmdet={det_score:.3f}, field={field_score:.3f}{extra})")
        return True
