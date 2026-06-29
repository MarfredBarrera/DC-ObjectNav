"""Live perception + MPPI planning loop.

Three cadences in a single process:
    A. train_step              every step                  (fast)
    B. replan + 1 agent step   every REPLAN_INTERVAL       (MPPI is cheap)
    C. refresh buffer + maps   every cfg.hash_buffer_refresh_interval (bottleneck)

Startup: a single observation seeds perception and the first BEV maps are
built directly from the untrained feature field — no spin, no cold-train. The
maps fill in online as the loop trains and cadence C recomputes them.
"""

import argparse
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
os.environ['GLOG_minloglevel'] = '2'
os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['HABITAT_SIM_LOG'] = 'quiet'

import gc
import json
import shutil
import signal
import time
from collections import deque

import numpy as np
import quaternion  # registers numpy.quaternion dtype used by habitat-sim
import torch
import imageio

from src.config import Config
from src.habitat.habitat_utils import (
    get_scene_bounds_from_pathfinder,
    init_simulator,
    spawn_agent_at_random_navpoint,
    spawn_agent_at_pos,
    geodesic_distance,
)
from src.habitat.sim_interface import SimInterface
from src.perception.obj_detection import (
    make_detector, encode_query_with_negatives, target_prob_from_sims,
    SinkGatedDetector,
)
from src.perception.perception_stack import PerceptionStack
from src.perception.segmentation import MobileSAMSegmenter
from src.perception.utils import unprojection
from src.planning.mppi import MPPIPlanner
from src.planning.utils import normalize_sim
from visualize import render_navigation


REPLAN_INTERVAL = 100


def get_agent_heading(agent) -> float:
    """Agent yaw in MPPI's grid frame: atan2(forward_z_world, forward_x_world)."""
    rot = agent.get_state().rotation
    R = quaternion.as_rotation_matrix(rot)
    forward_world = R @ np.array([0.0, 0.0, -1.0])  # Habitat agent forward is local -Z
    return float(np.arctan2(forward_world[2], forward_world[0]))


def world_to_grid(x_world: float, z_world: float, ref_grid, res: float):
    z_idx = int((z_world - ref_grid.min_z) / res)
    x_idx = int((x_world - ref_grid.min_x) / res)
    z_idx = max(0, min(z_idx, ref_grid.num_z - 1))
    x_idx = max(0, min(x_idx, ref_grid.num_x - 1))
    return (z_idx, x_idx)


def _to_numpy(maybe_tensor):
    if maybe_tensor is None:
        return None
    if isinstance(maybe_tensor, torch.Tensor):
        return maybe_tensor.detach().cpu().numpy()
    return maybe_tensor


def bev_cells_from_sim_pixels(perception, cfg, intrinsics, rgb, depth, c2w,
                              target_query, det_box=None, expand_factor=2.0,
                              top_frac=0.05, min_pixels=50):
    """BEV cells from the top-K% most target-similar pixels around the detection.

    The detector's box at distance is often *almost right* — frequently offset
    by ~one box-length from the actual object — so trusting its exact pixels
    smears wall/floor into the goal region. We instead build an expanded
    search window (`expand_factor` × box dims, centered on the box) and pick
    the top-K% of CLIP-similar pixels within it. That forgives offset error
    while still anchoring to where the detector thinks the object is.

    If `det_box` is None or `expand_factor <= 0`, the search runs over the
    full frame (no spatial constraint).

    `top_frac`: fraction of valid pixels to keep. 0.05 = top 5%.
    `min_pixels`: floor on the count so very sparse views still produce a
                  usable mask.
    """
    if depth is None or c2w is None or rgb is None or not target_query:
        return None
    rgb_gpu = rgb.to(perception.device)
    depth_gpu = depth.to(perception.device)
    H_d, W_d = depth_gpu.shape[-2:]

    # Per-pixel similarity to the target text. feats and text_embed are both
    # L2-normalized inside MaskCLIPSemantics, so the dot product is cosine.
    feats = perception.mask_clip.extract_dense_features(rgb_gpu)  # (H, W, 512)
    text_embed = perception.mask_clip.encode_text(target_query)   # (1, 512)
    sim_2d = (feats @ text_embed.T).squeeze(-1)                   # (H, W) in [-1, 1]

    # Spatial window: expanded box around the detection, clipped to image.
    region_mask = torch.ones((H_d, W_d), dtype=torch.bool, device=perception.device)
    if det_box is not None and expand_factor > 0:
        xmin, ymin, xmax, ymax = det_box
        cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        bw, bh = (xmax - xmin), (ymax - ymin)
        ew, eh = bw * expand_factor, bh * expand_factor
        ex0 = int(max(0, np.floor(cx - ew / 2.0)))
        ex1 = int(min(W_d, np.ceil(cx + ew / 2.0)))
        ey0 = int(max(0, np.floor(cy - eh / 2.0)))
        ey1 = int(min(H_d, np.ceil(cy + eh / 2.0)))
        if ex1 <= ex0 or ey1 <= ey0:
            return None
        region_mask = torch.zeros((H_d, W_d), dtype=torch.bool, device=perception.device)
        region_mask[ey0:ey1, ex0:ex1] = True

    depth_mask = (depth_gpu > cfg.min_sensor_dist) & (depth_gpu < cfg.max_sensor_dist)
    valid = depth_mask & region_mask
    if not bool(valid.any()):
        return None
    sim_in_valid = sim_2d[valid]
    n_valid = int(sim_in_valid.numel())
    if n_valid == 0:
        return None
    k = max(min_pixels, int(top_frac * n_valid))
    k = min(k, n_valid)
    thresh = torch.topk(sim_in_valid, k).values.min()
    mask = (sim_2d >= thresh) & valid
    if not bool(mask.any()):
        return None

    world_points = unprojection(
        depth_gpu, intrinsics, c2w.to(perception.device), perception.device, mask=mask,
    )
    if world_points.shape[0] == 0:
        return None

    sg = perception.similarity_grid
    z_idx = ((world_points[:, 2] - sg.min_z) / cfg.voxel_resolution).long().clamp(0, sg.num_z - 1)
    x_idx = ((world_points[:, 0] - sg.min_x) / cfg.voxel_resolution).long().clamp(0, sg.num_x - 1)
    cell_mask = torch.zeros((sg.num_z, sg.num_x), dtype=torch.bool, device=perception.device)
    cell_mask[z_idx, x_idx] = True
    return cell_mask.cpu().numpy()


def bev_cells_from_det_box(perception, cfg, intrinsics, det_box, depth, c2w):
    """BEV (z_idx, x_idx) cells that the pixels inside `det_box` project to.

    Returns a boolean numpy array of shape (num_z, num_x), or None if the box
    is missing / yields no valid depth pixels. Used to constrain goal
    selection in EXPLOIT mode: instead of the global argmax of bev_sim, we
    take the argmax only over cells that actually correspond to the detected
    object in the current frame.
    """
    if det_box is None or depth is None or c2w is None:
        return None
    xmin, ymin, xmax, ymax = det_box
    depth_gpu = depth.to(perception.device)
    H_d, W_d = depth_gpu.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(H_d, device=perception.device),
        torch.arange(W_d, device=perception.device),
        indexing='ij',
    )
    box_mask = (xx >= int(xmin)) & (xx <= int(xmax)) & (yy >= int(ymin)) & (yy <= int(ymax))
    depth_mask = (depth_gpu > cfg.min_sensor_dist) & (depth_gpu < cfg.max_sensor_dist)
    mask = box_mask & depth_mask
    if not bool(mask.any()):
        return None
    world_points = unprojection(depth_gpu, intrinsics, c2w.to(perception.device), perception.device, mask=mask)
    if world_points.shape[0] == 0:
        return None
    sg = perception.similarity_grid
    z_idx = ((world_points[:, 2] - sg.min_z) / cfg.voxel_resolution).long().clamp(0, sg.num_z - 1)
    x_idx = ((world_points[:, 0] - sg.min_x) / cfg.voxel_resolution).long().clamp(0, sg.num_x - 1)
    cell_mask = torch.zeros((sg.num_z, sg.num_x), dtype=torch.bool, device=perception.device)
    cell_mask[z_idx, x_idx] = True
    return cell_mask.cpu().numpy()


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

    Unlike `bev_cells_from_det_box` (which returns every cell the box covers),
    this projects only the box-center patch to one world point and returns its
    lone BEV cell. Used as the goal verbatim — no SAM re-localization, no
    similarity argmax, no snap-to-free. Best paired with a detector that emits
    tight, accurate boxes (e.g. LocateAnything). Returns (z_idx, x_idx) or None.
    """
    wxz = box_center_world_xz(perception, cfg, intrinsics, det_box, depth, c2w)
    if wxz is None:
        return None
    wx, wz = wxz
    sg = perception.similarity_grid
    z_idx = int(np.clip((wz - sg.min_z) / cfg.voxel_resolution, 0, sg.num_z - 1))
    x_idx = int(np.clip((wx - sg.min_x) / cfg.voxel_resolution, 0, sg.num_x - 1))
    return (z_idx, x_idx)


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
        (False, True): may pull the confidence weight but doesn't latch and
        isn't cached as a goal.
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


def bev_cells_from_sam(perception, cfg, intrinsics, rgb, depth, c2w,
                       target_query, segmenter,
                       min_mask_pixels: int = 200,
                       min_clip_sim: float = 0.18):
    """Whole-image SAM → CLIP-scored best mask → BEV cells.

    Run MobileSAM's automatic mask generator over the full RGB frame, score
    every proposal by softmax over its mean CLIP cosine to ``target_query``
    plus the distractor vocabulary (``cfg.det_negative_classes``), pick the
    mask with the highest target probability, and project its pixels (gated
    by valid depth) into BEV. The relative "more pillow than wall" score is
    used because raw CLIP cosines cluster in a narrow ~0.2–0.3 band where
    distractor masks score nearly as high as the target. The detector box is
    only used as a *trigger* upstream — it does NOT constrain SAM.
    Rationale: at distance the box is often a full box-width off, so
    box-prompted SAM inherits that error; auto-gen lets SAM find object
    boundaries from scratch and CLIP picks the right one.

    Returns a (num_z, num_x) bool numpy array, or None if the best mask's
    target probability missed ``cfg.sam_min_target_prob``, its raw cosine
    missed ``min_clip_sim``, or no valid depth pixels remained.
    """
    if rgb is None or depth is None or c2w is None or not target_query or segmenter is None:
        return None
    # Reset the segmenter's side-channel state — the main loop reads these
    # after the call to persist the chosen mask for offline visualization.
    segmenter.last_mask = None
    segmenter.last_box = None
    segmenter.last_score = 0.0
    segmenter.last_prob = 0.0

    rgb_gpu = rgb.to(perception.device)
    depth_gpu = depth.to(perception.device)
    H_d, W_d = depth_gpu.shape[-2:]

    masks = segmenter.segment_all(rgb)
    if not masks:
        return None

    feats = perception.mask_clip.extract_dense_features(rgb_gpu)      # (H, W, 512)
    text_bank = encode_query_with_negatives(
        perception.mask_clip, target_query,
        getattr(cfg, 'det_negative_classes', None))                   # (K, 512)
    sim_maps = (feats @ text_bank.T).permute(2, 0, 1)                 # (K, H, W) in [-1, 1]
    if sim_maps.shape[1:] != (H_d, W_d):
        sim_maps = torch.nn.functional.interpolate(
            sim_maps[None].float(), size=(H_d, W_d),
            mode='bilinear', align_corners=False,
        )[0]

    depth_mask = (depth_gpu > cfg.min_sensor_dist) & (depth_gpu < cfg.max_sensor_dist)

    softmax_temp = float(getattr(cfg, 'sam_softmax_temp', 100.0))
    min_target_prob = float(getattr(cfg, 'sam_min_target_prob', 0.5))

    best_prob = -float('inf')
    best_raw = -float('inf')
    best_seg = None
    for m in masks:
        seg = m['segmentation']  # numpy bool (H_rgb, W_rgb)
        if seg.sum() < min_mask_pixels:
            continue
        seg_t = torch.from_numpy(seg).to(perception.device)
        if seg_t.shape != (H_d, W_d):
            seg_t = torch.nn.functional.interpolate(
                seg_t[None, None].float(), size=(H_d, W_d),
                mode='nearest',
            )[0, 0].bool()
        valid = seg_t & depth_mask
        if not bool(valid.any()):
            continue
        sims = sim_maps[:, valid].mean(dim=1)                         # (K,)
        prob, raw = target_prob_from_sims(sims, softmax_temp)
        if prob > best_prob:
            best_prob = prob
            best_raw = raw
            best_seg = seg_t

    if best_seg is None or best_prob < min_target_prob or best_raw < min_clip_sim:
        return None
    best_score = best_raw  # raw target cosine — keeps sam_score's scale stable

    final_mask = best_seg & depth_mask
    if not bool(final_mask.any()):
        return None

    # Stash the chosen 2D mask + its bbox + score on the segmenter as a
    # side channel for the main loop / visualizer. We use the raw best_seg
    # (not depth-gated) for the visualization overlay since it shows the
    # full SAM contour the user expects to see.
    seg_np = best_seg.detach().cpu().numpy().astype(bool)
    ys, xs = np.where(seg_np)
    if xs.size > 0 and ys.size > 0:
        segmenter.last_mask = seg_np
        segmenter.last_box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        segmenter.last_score = float(best_score)
        segmenter.last_prob = float(best_prob)

    world_points = unprojection(
        depth_gpu, intrinsics, c2w.to(perception.device), perception.device, mask=final_mask,
    )
    if world_points.shape[0] == 0:
        return None

    sg = perception.similarity_grid
    z_idx = ((world_points[:, 2] - sg.min_z) / cfg.voxel_resolution).long().clamp(0, sg.num_z - 1)
    x_idx = ((world_points[:, 0] - sg.min_x) / cfg.voxel_resolution).long().clamp(0, sg.num_x - 1)
    cell_mask = torch.zeros((sg.num_z, sg.num_x), dtype=torch.bool, device=perception.device)
    cell_mask[z_idx, x_idx] = True
    return cell_mask.cpu().numpy()


def closest_cell_in_mask(cell_mask, start_grid):
    """Nearest True cell in a (num_z, num_x) bool grid to `start_grid` (z, x).

    Returns (z_idx, x_idx) of the mask cell closest to the agent — i.e. the
    near face of the projected object footprint — or None if the mask is empty.
    """
    zs, xs = np.where(cell_mask)
    if zs.size == 0:
        return None
    sz, sx = start_grid
    d2 = (zs - sz) ** 2 + (xs - sx) ** 2
    i = int(np.argmin(d2))
    return (int(zs[i]), int(xs[i]))


def snap_to_free(cell, bev_occ, free_val=1):
    """Return `cell` if it is observed-free, else the nearest observed-free cell.

    The near face of the object footprint is usually marked occupied (the
    object is a solid surface in the occupancy grid), so we snap the goal to
    the closest navigable cell — typically the free space directly in front of
    the object on the agent's side. Falls back to `cell` if no free cell exists.
    """
    z, x = int(cell[0]), int(cell[1])
    H, W = bev_occ.shape
    if 0 <= z < H and 0 <= x < W and bev_occ[z, x] == free_val:
        return (z, x)
    fz, fx = np.where(bev_occ == free_val)
    if fz.size == 0:
        return (z, x)
    d2 = (fz - z) ** 2 + (fx - x) ** 2
    i = int(np.argmin(d2))
    return (int(fz[i]), int(fx[i]))


def project_box_goal(perception, sim_iface, cfg, det_box, rgb, depth, c2w,
                     segmenter=None, start_grid=None, bev_occ=None):
    """Project a detection box to a BEV goal cell — the Layer-1 goal logic
    used by `plan_one_action`.

    `box_center` needs only depth + camera pose, so it works before any BEV map
    is meaningful (e.g. on the untrained field at startup). The `sam` path needs
    `bev_occ` + `start_grid` to snap the chosen mask cell to free space; it is
    skipped when those aren't available. Returns a (z, x) cell, or None.
    """
    if det_box is None:
        return None
    if cfg.goal_projection == "box_center":
        return bev_cell_from_box_center(
            perception, cfg, sim_iface.intrinsics, det_box, depth, c2w)
    if segmenter is not None and bev_occ is not None and start_grid is not None:
        det_cells = bev_cells_from_sam(
            perception, cfg, sim_iface.intrinsics,
            rgb, depth, c2w, perception.target_query, segmenter,
            min_mask_pixels=int(getattr(cfg, 'sam_min_mask_pixels', 200)),
            min_clip_sim=float(getattr(cfg, 'sam_min_clip_sim', 0.18)),
        )
        if det_cells is not None and det_cells.any():
            cand = closest_cell_in_mask(det_cells, start_grid)
            if cand is not None:
                return snap_to_free(cand, bev_occ)
    return None


def detect_classify_latch(detector, perception, sim_iface, cfg, segmenter,
                          rgb, depth, c2w, pos, detected, detected_streak,
                          run_detector=True, tag=""):
    """Run the detector, classify the detection, and advance the latch state.

    Classifies the box by object distance + box size into
    three tiers (see `classify_detection`): *too close* → ignored; *too far* →
    investigate (steer + confidence) but not persistent; *usable band* →
    persistent (also counts toward the latch streak). Latches into DETECTED
    once `cfg.detected_persistence` consecutive persistent detections accrue
    (never unlatches).

    Returns (det_score, det_box, det_persistent, det_investigate, conf_score,
    detected, detected_streak).
    """
    if run_detector:
        det_score, det_box = detector.detect(rgb, perception.target_query)
    else:
        det_score, det_box = 0.0, None

    # Reset the SAM side-channel so a mask is only persisted on the frame SAM
    # actually ran (bev_cells_from_sam refills it on a qualifying detection).
    if segmenter is not None:
        segmenter.last_mask = None
        segmenter.last_box = None
        segmenter.last_score = 0.0

    det_persistent, det_investigate = classify_detection(
        perception, cfg, sim_iface.intrinsics, det_box, depth, c2w, pos)
    det_persistent = det_persistent and det_score >= cfg.detected_conf_threshold
    conf_score = det_score if det_investigate else 0.0

    if not detected:
        if det_persistent:
            detected_streak += 1
        else:
            detected_streak = 0
        if detected_streak >= cfg.detected_persistence:
            detected = True
            print(f"{tag}: DETECTED — entering exploit mode "
                  f"(det_score={det_score:.3f})")
    return (det_score, det_box, det_persistent, det_investigate, conf_score,
            detected, detected_streak)


def plan_one_action(perception, sim_iface, mppi, cfg,
                    action_queue: deque, k_max: int = 5,
                    det_score: float = 0.0,
                    detected: bool = False, det_box=None, rgb=None, depth=None, c2w=None,
                    last_box_goal=None,
                    segmenter=None, det_investigate: bool = False):
    """Run MPPI from current pose and return (action, ref_traj, opt_traj).

    action is [v_mps, w_rad_per_s] or None if idling.
    ref_traj / opt_traj are grid-coord lists [(z_idx, x_idx), ...]; None when
    falling back to the queue or idling (no new plan was computed).

    On a successful plan the full MPPI control sequence replaces the queue.
    If every A* goal is unreachable the queue is consumed so the agent keeps
    moving.

    `det_score` is the raw detector confidence for the current frame, used as
    `goal_confidence` in SEARCH mode (MPPI's threshold + hysteresis filter
    noisy detections). `detected=True` overrides it with goal_confidence=1.0,
    saturating the conf curve so `w_ig_conf = 0` and IG is hard-off.
    """
    bev_sim = _to_numpy(perception.similarity_grid.get_2d_map())
    bev_epi = _to_numpy(perception.ugrid.get_2d_map(type='epistemic'))
    bev_occ = _to_numpy(perception.occupancy_grid.get_2d_map())

    if bev_sim is None or bev_epi is None or bev_occ is None:
        print("  plan: missing map(s); idling")
        return (action_queue.popleft() if action_queue else None), None, None, None, None, None

    # IG map by source: 'coverage' = soft observation-count deficit
    # (continuous, keeps a gradient after binary unseen is locally
    # exhausted); 'epistemic' = masked field uncertainty; 'unseen' ignores
    # the map (MPPI builds a binary mask from occupancy internally).
    if cfg.ig_source == 'coverage':
        bev_ig = _to_numpy(perception.occupancy_grid.get_2d_coverage_deficit_map())
    else:
        bev_ig = bev_epi

    pos = sim_iface.agent_position
    heading = get_agent_heading(sim_iface.agent)
    start_grid = world_to_grid(pos[0], pos[2], perception.similarity_grid, cfg.voxel_resolution)

    # Drive straight at the highest-similarity cell, even if it's on an
    # obstacle (the target object itself). MPPI carves a small free disk
    # around the goal and forgives collisions once the rollout has arrived,
    # so the planner can commit instead of orbiting.
    sim_for_goal = bev_sim.copy().astype(np.float32)
    sim_for_goal[bev_occ == 0] = -np.inf  # exclude unseen
    # Goal selection priority:
    #   1. Detection box this frame → argmax of bev_sim restricted to BEV cells
    #      the box projects to.
    #   2. No box this frame, but a previous box-derived goal is cached →
    #      reuse that fixed world cell so the agent commits to the last
    #      sighting instead of chasing a far-away similarity peak.
    #   3. Otherwise → global argmax of observed bev_sim (exploration).
    goal = None
    box_goal = None  # None unless THIS frame's detection produced a fresh goal
    H, W = sim_for_goal.shape
    # Layer 1: project a fresh box-derived goal from THIS frame's detection
    # whenever it's worth investigating (`det_investigate` — anything but a
    # too-close box that fills the frame). This INCLUDES too-far detections, so
    # the agent steers toward and investigates a distant sighting even though it
    # won't latch on it yet (latching needs the usable band; see the caller).
    # Every such detection re-projects, so in SEARCH mode the caller's cache
    # tracks the most recent bounding box. Until a detection is worth
    # investigating the agent explores via the global similarity argmax
    # (Layer 3). SAM runs on the *whole image* (not box-prompted) because the
    # box is often offset by ~one box-width at distance; CLIP picks the mask.
    if det_box is not None and det_investigate:
        # box_center: project the box center straight to one BEV cell, used
        # verbatim (MPPI carves a free disk around the goal and forgives arrival
        # collisions, so a cell on the target surface is fine). sam: pick the
        # near-face mask cell and snap to free space. See project_box_goal.
        cand = project_box_goal(
            perception, sim_iface, cfg, det_box, rgb, depth, c2w,
            segmenter=segmenter, start_grid=start_grid, bev_occ=bev_occ)
        if cand is not None:
            goal = cand
            box_goal = goal
    if goal is None and last_box_goal is not None:
        gz, gx = int(last_box_goal[0]), int(last_box_goal[1])
        if 0 <= gz < H and 0 <= gx < W:
            goal = (gz, gx)
    if goal is None:
        if not np.any(np.isfinite(sim_for_goal)):
            print("  plan: no observed cells, can't pick a goal")
            return (action_queue.popleft() if action_queue else None), None, None, None, None, None
        flat_idx = int(np.argmax(sim_for_goal))
        goal = (flat_idx // W, flat_idx % W)
    # Goal weight (`w_conf` inside MPPI):
    #   EXPLOIT (detected latched True) → pin at 1.0 so the agent commits
    #     hard to the cached max-conf goal and never decays back to IG.
    #   SEARCH → raw per-frame det_score. MPPI's hysteresis
    #     (exploit_conf = max(incoming, prev * conf_decay)) ratchets up on
    #     strong sightings and decays after a stretch of misses, eventually
    #     falling below threshold and re-enabling IG-driven exploration.
    # The cached *goal cell* is independent of this — see Layer 1/2 above —
    # so the planner still aims at the strongest sighting in either mode.
    goal_confidence = 1.0 if detected else float(det_score)

    opt_path, U_opt = mppi.optimize_trajectory(
        start_grid, goal, bev_ig, bev_occ,
        initial_heading=heading,
        intrinsics=sim_iface.intrinsics,
        sensor_height=cfg.sensor_height,
        ig_source=cfg.ig_source,
        goal_confidence=goal_confidence,
    )
    mppi_score = None
    if U_opt is None or U_opt.shape[0] == 0:
        # No safe MPPI plan this replan — idle and let the next replan retry.
        # Goal disconfirmation and freshly observed geometry usually open a new
        # option within a few replans, so we wait rather than commit a forced
        # maneuver.
        print("  plan: MPPI returned no safe control sequence, idling this replan")
        action_queue.clear()
        return None, None, None, None, goal, box_goal

    # Successful plan: replace queue with full MPPI control sequence.
    # MPPI U units: v in [grid cells / mppi_step], w in [rad / mppi_step].
    # sim_iface.step expects [m/s, rad/s].
    action_queue.clear()
    for i in range(U_opt.shape[0]):
        v_mps = float(U_opt[i, 0]) * cfg.voxel_resolution / cfg.mppi_dt
        w_rps = cfg.mppi_w_sign * float(U_opt[i, 1]) / cfg.mppi_dt
        action_queue.append([v_mps, w_rps])

    return action_queue.popleft(), None, opt_path, mppi_score, goal, box_goal


def run(cfg: Config, save_enabled: bool = True,
        visualize: bool = True, viz_output: str = "./figs/nav_history.mp4",
        viz_fps: int = 5,
        start_pos=None, goals=None, success_radius_m: float = 1.0) -> dict:
    """Run one navigation episode.

    `start_pos` is the spawn point (snapped to the navmesh); defaults to the
    historical [0, -3, 2.5]. `goals`, if given, is a list of ground-truth target
    world positions (xyz) used to score the episode — success requires the agent
    to self-stop within `success_radius_m` geodesic of the nearest goal, and SPL
    weights that success by start→nearest-goal geodesic over distance traveled.
    Returns a metrics dict (see bottom of the function)."""
    if start_pos is None:
        start_pos = [0.0, -3.0, 2.5]
    # 1. Init simulator
    sim, agent = init_simulator(
        cfg.scene_path, resolution=cfg.img_width, fov_deg=cfg.fov, sensor_height=cfg.sensor_height,
        agent_radius=cfg.agent_radius,
        agent_height=cfg.agent_height,
    )
    # spawn_agent_at_random_navpoint(sim, agent)
    start_nav = spawn_agent_at_pos(sim, agent, start_pos)  # snapped navmesh start
    scene_bounds = get_scene_bounds_from_pathfinder(sim)

    sim_iface = SimInterface(cfg, sim, agent)
    perception = PerceptionStack(cfg, scene_bounds)  # owns target_query; initialised from cfg

    mppi = MPPIPlanner(cfg, device=cfg.device)
    # LLMDet carries its own knobs (model, attention-sink config); other
    # backends ignore them, so only forward them for "llmdet".
    det_kwargs = {}
    if cfg.detector == 'llmdet':
        det_kwargs = dict(
            model_name=cfg.llmdet_model_name,
            threshold=cfg.llmdet_threshold,
            use_sinks=cfg.llmdet_use_sinks,
            num_sinks=cfg.llmdet_num_sinks,
            sink_init=cfg.llmdet_sink_init,
            sink_special_str=cfg.sink_special_str,
        )
    detector = make_detector(
        cfg.detector, device=cfg.device,
        negative_classes=getattr(cfg, 'det_negative_classes', None),
        **det_kwargs,
    )
    # Neutral attention-sink false-positive gate (Ruis et al., ICLR 2026):
    # wrap the base detector so each fired box is verified against the target
    # query vs. a semantically-neutral sink in CLIP space and dropped if the
    # sink wins. Reuses the perception stack's MaskCLIP (no second CLIP load).
    if cfg.sink_gate:
        detector = SinkGatedDetector(
            detector, perception.mask_clip, device=cfg.device,
            sink_init=cfg.sink_init, sink_num=cfg.sink_num,
            sink_special_str=cfg.sink_special_str,
            softmax_temp=cfg.sink_softmax_temp,
            min_target_prob=cfg.sink_min_target_prob,
            crop_pad=cfg.sink_crop_pad,
            pool=cfg.sink_pool, top_pct=cfg.sink_top_pct,
            seed=cfg.sink_seed,
        )
    # MobileSAM is only consulted when the detector fires (Layer-1 goal
    # update), so a load failure shouldn't kill the run — fall back to the
    # box-based path with a warning. The path is None-guarded downstream.
    segmenter = None
    if bool(getattr(cfg, 'use_mobile_sam', True)):
        try:
            segmenter = MobileSAMSegmenter(
                checkpoint=getattr(cfg, 'sam_checkpoint', 'SAM_models/mobile_sam.pt'),
                device=cfg.device,
                points_per_side=int(getattr(cfg, 'sam_points_per_side', 16)),
                pred_iou_thresh=float(getattr(cfg, 'sam_pred_iou_thresh', 0.86)),
                stability_score_thresh=float(getattr(cfg, 'sam_stability_score_thresh', 0.90)),
                min_mask_region_area=int(getattr(cfg, 'sam_min_mask_region_area', 200)),
            )
            print(f"[init] MobileSAM loaded from {getattr(cfg, 'sam_checkpoint', 'SAM_models/mobile_sam.pt')}")
        except Exception as e:
            print(f"[init] MobileSAM unavailable ({e}); EXPLOIT goals will skip SAM refinement")
    action_queue: deque = deque()

    # Planner state. `detected` latches True (and never releases) once
    # cfg.detected_persistence consecutive persistent detections accrue; while
    # latched MPPI silences IG and pulls hard toward the cached goal.
    # `last_box_goal` caches the goal cell of the most recent investigated
    # detection so a missed frame doesn't drop the agent back to global-argmax
    # exploration.
    detected = False
    detected_streak = 0
    last_box_goal = None

    # Clear per-step artifact dirs from any previous run sharing this
    # output_dir. They hold step-numbered .npy/.png/.npz files; unlike
    # traj_log.jsonl (truncated below) they were never cleared, so a previous,
    # longer run (often a different scene) left stale snapshots that visualize.py
    # then interleaved into the new navigation history. Wipe them so only the
    # current run's maps exist.
    if save_enabled:
        for sub in ("umaps", "occ_maps", "sim_maps", "sam_masks", "rgbs"):
            shutil.rmtree(os.path.join(cfg.output_dir, sub), ignore_errors=True)

    # 2. Startup: a single observation seeds the replay buffer + occupancy, then
    # the first BEV maps are built directly from the untrained feature field —
    # no spin, no cold-train. The maps start as field noise / mostly-unseen and
    # fill in online as the loop trains and cadence C recomputes them.
    print("[init] seeding maps from the untrained feature field (no bootstrap)...")
    rgb, depth, c2w = perception.observe(sim_iface)
    perception.update_replay_buffer(rgb, depth, c2w, sim_iface.intrinsics)
    perception.update_occupancy(depth, c2w, sim_iface.intrinsics)
    super_pts, super_feats = perception.make_super_batch()
    torch.cuda.empty_cache()

    # First maps + grid extent for visualize.py
    perception.update_maps(step=0, save_enabled=save_enabled)
    if save_enabled:
        # Save RGB
        step = 0
        rgb_dir = os.path.join(cfg.output_dir, "rgbs")
        os.makedirs(rgb_dir, exist_ok=True)
        rgb_img = (rgb.numpy() * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(rgb_dir, f"rgb_{step:03d}.png"), rgb_img)

        # Save 2D Sim Map
        perception.save_2d_similarity(step, depth, c2w, sim_iface.intrinsics)

    extent_path = os.path.join(cfg.output_dir, "grid_extent.json")
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(extent_path, 'w') as f:
        json.dump({
            'min_x': perception.similarity_grid.min_x,
            'max_x': perception.similarity_grid.max_x,
            'min_z': perception.similarity_grid.min_z,
            'max_z': perception.similarity_grid.max_z,
            'voxel_resolution': cfg.voxel_resolution,
        }, f)

    traj_log_path = os.path.join(cfg.output_dir, "traj_log.jsonl")
    open(traj_log_path, 'w').close()  # truncate / create fresh

    # 5. Main loop
    print(f"[main] running for {cfg.iterations} iterations "
          f"(replan every {REPLAN_INTERVAL}, refresh every {cfg.hash_buffer_refresh_interval})")
    start_time = time.time()

    # Tracks the last step actually executed by the loop. Used to cap the
    # post-run visualization so it doesn't pull in stale .npy snapshots left
    # over from a previous, longer run (traj_log.jsonl is truncated each run,
    # but the umaps/occ_maps/sim_maps directories are not).
    last_step = 0

    # Episode metrics: accumulate the actual distance the agent travels (sum of
    # per-action displacements) for SPL, and remember whether the agent itself
    # decided to stop (vs. running out the step budget = timeout = failure).
    path_length = 0.0
    prev_pos = np.asarray(start_nav, dtype=np.float64).copy()
    agent_stopped = False

    # Graceful early stop. Two ways to request it; both let the loop break at
    # the next step and fall through to the final map snapshot + visualization
    # below (instead of losing them to a hard kill):
    #   1. Ctrl-C (SIGINT) → sets a flag. A second Ctrl-C hard-aborts in case
    #      something is wedged.
    #   2. Create the sentinel file `<output_dir>/STOP` (e.g. `touch` it, handy
    #      when running under `docker exec` without an attached TTY).
    # A stale STOP from a previous run is removed up front so it doesn't end the
    # new run immediately.
    stop_file = os.path.join(cfg.output_dir, "STOP")
    if os.path.exists(stop_file):
        os.remove(stop_file)
    stop_requested = {"flag": False}

    def _on_sigint(signum, frame):
        if stop_requested["flag"]:
            raise KeyboardInterrupt
        stop_requested["flag"] = True
        print("\n[main] stop requested (Ctrl-C) — finishing the current step, "
              "then saving maps + rendering the visualization. "
              "Press Ctrl-C again to abort immediately.")

    prev_sigint = signal.signal(signal.SIGINT, _on_sigint)

    for step in range(1, cfg.iterations + 1):
        if stop_requested["flag"] or os.path.exists(stop_file):
            reason = "Ctrl-C" if stop_requested["flag"] else f"{stop_file} sentinel"
            print(f"[main] ending early at step {last_step} ({reason})")
            break
        last_step = step
        # A. train every step
        loss = perception.train_step(super_pts, super_feats)

        # B. replan + 1 agent step
        if step % REPLAN_INTERVAL == 0:
            t_plan = time.time()
            pos = sim_iface.agent_position.copy()
            heading = get_agent_heading(sim_iface.agent)

            rgb_cur, depth_cur, c2w_cur = perception.observe(sim_iface)

            # Throttle the detector in EXPLOIT mode: the goal is already pinned
            # to the cached box cell, so skip the (expensive) detector on most
            # replans and reuse the cache. SEARCH mode always detects. A skipped
            # replan reports no detection (det_box=None), so plan_one_action
            # falls through to the cached box-goal (Layer 2). The latch logic is
            # guarded by `if not detected`, so a skipped 0.0 score is harmless.
            replan_idx = step // REPLAN_INTERVAL
            if detected and cfg.exploit_redetect_interval <= 0:
                run_detector = False
            elif detected:
                run_detector = (replan_idx % cfg.exploit_redetect_interval == 0)
            else:
                run_detector = True
            # Detect (subject to the throttle above), classify into too-close /
            # too-far / usable-band, and advance the latch. `det_investigate`
            # (= not too close) drives the goal + confidence; `det_persistent`
            # (usable band over threshold) also drives latching, so the agent
            # approaches a far sighting and commits only once it has closed into
            # the usable band.
            (det_score, det_box, det_persistent, det_investigate, conf_score,
             detected, detected_streak) = detect_classify_latch(
                detector, perception, sim_iface, cfg, segmenter,
                rgb_cur, depth_cur, c2w_cur, pos, detected, detected_streak,
                run_detector=run_detector, tag=f"step {step}")

            # SEARCH-mode goal disconfirmation: if the agent has reached its
            # cached box-goal but nothing is detected there now, the earlier
            # sighting was a false positive (a transient detector spike or a
            # similarity blip). Drop the cache so the planner resumes exploration
            # instead of dwelling next to the spike. EXPLOIT never reaches here
            # (it has latched on a confirmed target), and a fresh investigated
            # detection this frame leaves `det_investigate` True so we keep it.
            if not detected and last_box_goal is not None and not det_investigate:
                agz, agx = world_to_grid(pos[0], pos[2], perception.similarity_grid, cfg.voxel_resolution)
                reach_cells = cfg.stop_distance_m / cfg.voxel_resolution
                if float(np.hypot(agz - last_box_goal[0], agx - last_box_goal[1])) <= reach_cells:
                    print(f"  goal disconfirmed at {last_box_goal} "
                          f"(reached, no detection) — clearing cache")
                    last_box_goal = None

            action, ref_traj, opt_traj, score, goal_cell, box_goal = plan_one_action(
                perception, sim_iface, mppi, cfg, action_queue,
                det_score=conf_score, detected=detected,
                det_box=det_box, rgb=rgb_cur, depth=depth_cur, c2w=c2w_cur,
                last_box_goal=last_box_goal,
                segmenter=segmenter, det_investigate=det_investigate,
            )
            # Cache the goal of the most recent investigated detection.
            # `box_goal` is non-None only for an investigated box (Layer 1 gates
            # on `det_investigate`), so this tracks the latest sighting worth
            # steering toward — including too-far ones — in SEARCH mode; in
            # EXPLOIT the detector is throttled off so box_goal stays None and
            # the cache freezes on the latched object. Fallback paths (cached /
            # global argmax) leave box_goal=None.
            if box_goal is not None:
                last_box_goal = box_goal
                print(f"  cached new goal {box_goal} (det_score={det_score:.3f})")
            # If SAM produced a fresh mask this replan, persist it for the
            # offline visualizer. The mask file is named by step so
            # visualize.py can look up the most recent SAM result at or
            # before each viz frame and overlay it on the saved RGB.
            sam_box_log = None
            sam_score_log = None
            if save_enabled and segmenter is not None and getattr(segmenter, 'last_mask', None) is not None:
                sam_dir = os.path.join(cfg.output_dir, "sam_masks")
                os.makedirs(sam_dir, exist_ok=True)
                # packbits cuts the on-disk size ~8x; unpacked at load time.
                packed = np.packbits(segmenter.last_mask.astype(np.uint8))
                np.savez_compressed(
                    os.path.join(sam_dir, f"sam_mask_{step:06d}.npz"),
                    packed=packed, shape=np.array(segmenter.last_mask.shape, dtype=np.int32),
                )
                sam_box_log = [float(v) for v in segmenter.last_box]
                sam_score_log = float(segmenter.last_score)
            det_backend = detector.backend_for(perception.target_query)
            mode = 'EXPLOIT' if detected else 'SEARCH'
            w_conf = float(getattr(mppi, 'last_w_conf', 0.0))
            # Sink-gate target probability + the box it scored, for this frame's
            # detection (both None when the gate is off, the detector didn't
            # fire, or the crop was tiny). Gate on `run_detector` so a throttled
            # EXPLOIT replan doesn't log the stale side-channel from an earlier
            # detect call. `sink_box` is logged even on rejection so the viz can
            # draw the false positive.
            sink_p = detector.last_target_prob if (cfg.sink_gate and run_detector) else None
            sink_b = detector.last_box if (cfg.sink_gate and run_detector) else None
            sink_str = f" | sink: {sink_p:.2f}" if sink_p is not None else ""
            if goal_cell is not None:
                sg = perception.similarity_grid
                goal_x_m = sg.min_x + goal_cell[1] * cfg.voxel_resolution
                goal_z_m = sg.min_z + goal_cell[0] * cfg.voxel_resolution
                goal_str = f"({goal_x_m:+.2f}, {goal_z_m:+.2f})m"
            else:
                goal_str = "—"
            if action is not None:
                sim_iface.step(action, dt=cfg.mppi_dt)
                cur_pos = np.asarray(sim_iface.agent_position, dtype=np.float64)
                path_length += float(np.linalg.norm(cur_pos - prev_pos))
                prev_pos = cur_pos
                print(f"step {step}: plan {time.time()-t_plan:.2f}s | "
                      f"action [v={action[0]:+.3f} m/s, w={action[1]:+.3f} rad/s] | "
                      f"position: {pos[0]:.2f}, {pos[2]:.2f} | "
                      f"det[{det_backend}]: {det_score:.3f}{sink_str} | "
                      f"goal: {goal_str} | w_conf: {w_conf:.2f} | "
                      f"mode: {mode}")
            else:
                print(f"step {step}: plan {time.time()-t_plan:.2f}s | no action | "
                      f"det[{det_backend}]: {det_score:.3f}{sink_str} | "
                      f"goal: {goal_str} | w_conf: {w_conf:.2f} | "
                      f"mode: {mode}")

            # Termination check: measure distance to whatever goal the planner
            # just aimed at (the max-conf goal — current frame if it beat the
            # cache, else the cached cell). Stop after the traj_log write
            # below so the final replan is captured for visualization.
            stop_now = False
            if detected and goal_cell is not None:
                gz, gx = int(goal_cell[0]), int(goal_cell[1])
                sz, sx = world_to_grid(
                    pos[0], pos[2], perception.similarity_grid, cfg.voxel_resolution,
                )
                dist_m = float(np.hypot(gz - sz, gx - sx)) * cfg.voxel_resolution
                if dist_m <= cfg.stop_distance_m:
                    print(f"step {step}: TARGET REACHED "
                          f"(dist={dist_m:.2f}m <= {cfg.stop_distance_m:.2f}m)")
                    stop_now = True

            with open(traj_log_path, 'a') as _f:
                # Convert score to cost: higher score is better, so cost is -score.
                cost = -float(score) if score is not None else None
                _f.write(json.dumps({
                    'step': step,
                    'pos': [float(pos[0]), float(pos[2])],
                    'heading': float(heading),
                    'action': [float(action[0]), float(action[1])] if action is not None else [0.0, 0.0],
                    'ref_traj': [[int(p[0]), int(p[1])] for p in ref_traj] if ref_traj else [],
                    'opt_traj': [[int(p[0]), int(p[1])] for p in opt_traj] if opt_traj else [],
                    'mppi_cost': cost,
                    'det_conf': float(det_score),
                    'det_box': [float(v) for v in det_box] if det_box is not None else None,
                    'sam_box': sam_box_log,
                    'sam_score': sam_score_log,
                    'goal': [int(goal_cell[0]), int(goal_cell[1])] if goal_cell is not None else None,
                    'mode': mode,
                    'w_conf': w_conf,
                    'sink_prob': float(sink_p) if sink_p is not None else None,
                    'sink_box': [float(v) for v in sink_b] if sink_b is not None else None,
                }) + '\n')

            if stop_now:
                agent_stopped = True
                sim_iface.step([0.0, 0.0], dt=cfg.mppi_dt)
                break

        # C. refresh buffer + maps (slowest cadence — bottleneck)
        if step % cfg.hash_buffer_refresh_interval == 0:
            t_refresh = time.time()
            rgb, depth, c2w = perception.observe(sim_iface)
            perception.update_replay_buffer(rgb, depth, c2w, sim_iface.intrinsics)
            perception.update_occupancy(depth, c2w, sim_iface.intrinsics)

            if save_enabled:
                # Save RGB
                rgb_dir = os.path.join(cfg.output_dir, "rgbs")
                os.makedirs(rgb_dir, exist_ok=True)
                rgb_img = (rgb.numpy() * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(rgb_dir, f"rgb_{step:03d}.png"), rgb_img)

                # Save 2D Sim Map
                perception.save_2d_similarity(step, depth, c2w, sim_iface.intrinsics)

            if super_pts is not None:
                del super_pts, super_feats
                torch.cuda.empty_cache()
            super_pts, super_feats = perception.make_super_batch()

            perception.update_maps(step=step, save_enabled=save_enabled)
            gc.collect()
            torch.cuda.empty_cache()
            print(f"step {step}: refresh+maps {time.time()-t_refresh:.2f}s | "
                  f"history {perception.buffer_size}/{perception.buffer_capacity} pts")

        # if step % 100 == 0:
        #     print(f"  step {step:05d} | loss {loss:.5f} | t {time.time()-start_time:.1f}s")

    # Restore the default Ctrl-C behavior so the (potentially long) finalization
    # + visualization below can be aborted normally.
    signal.signal(signal.SIGINT, prev_sigint)

    # Snapshot maps + RGB + 2D-sim aligned to last_step so the visualizer's
    # final frame is up-to-date. The C cadence only fires every refresh
    # interval, and early termination via `stop_now` or Ctrl-C can leave it
    # 100+ steps behind the actual last action. Skipped when last_step already
    # matches the most recent refresh tick — same files would just be overwritten.
    if save_enabled and last_step % cfg.hash_buffer_refresh_interval != 0:
        print(f"[main] saving final maps at step {last_step}...")
        rgb_f, depth_f, c2w_f = perception.observe(sim_iface)
        perception.update_replay_buffer(rgb_f, depth_f, c2w_f, sim_iface.intrinsics)
        perception.update_occupancy(depth_f, c2w_f, sim_iface.intrinsics)
        rgb_dir = os.path.join(cfg.output_dir, "rgbs")
        os.makedirs(rgb_dir, exist_ok=True)
        rgb_img = (rgb_f.numpy() * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(rgb_dir, f"rgb_{last_step:03d}.png"), rgb_img)
        perception.save_2d_similarity(last_step, depth_f, c2w_f, sim_iface.intrinsics)
        perception.update_maps(step=last_step, save_enabled=True)

    perception.save_models()

    # Episode scoring. Compute geodesics while the pathfinder is still alive
    # (before sim.close()). Without ground-truth `goals` we can only report the
    # self-reported subset (the agent's own stop decision + distance traveled).
    final_pos = np.asarray(sim_iface.agent_position, dtype=np.float64)
    if goals:
        l_geo = min(geodesic_distance(sim.pathfinder, start_nav, g) for g in goals)
        d_final = min(geodesic_distance(sim.pathfinder, final_pos, g) for g in goals)
        success = bool(agent_stopped and d_final <= success_radius_m)
        if not success:
            spl = 0.0
        elif l_geo > 0.0 and np.isfinite(l_geo):
            spl = float(l_geo / max(path_length, l_geo))
        else:
            # Spawned already on the goal (l == 0) or goal unreachable on the
            # navmesh but the agent reported success — credit a perfect path.
            spl = 1.0
        print(f"[eval] success={success} spl={spl:.3f} | "
              f"l_geo={l_geo:.2f}m path={path_length:.2f}m "
              f"final_geo={d_final:.2f}m stopped={agent_stopped}")
        metrics = {
            'success': success, 'spl': spl,
            'l_geodesic': float(l_geo), 'final_geodesic': float(d_final),
            'path_length': float(path_length), 'agent_stopped': bool(agent_stopped),
            'steps': int(last_step), 'scene': cfg.scene_path, 'query': cfg.target_query,
        }
    else:
        metrics = {
            'success': None, 'spl': None,
            'l_geodesic': None, 'final_geodesic': None,
            'path_length': float(path_length), 'agent_stopped': bool(agent_stopped),
            'steps': int(last_step), 'scene': cfg.scene_path, 'query': cfg.target_query,
        }

    sim.close()
    print("[main] done.")

    if visualize:
        print(f"[viz] rendering navigation video up to step {last_step}...")
        try:
            render_navigation(cfg, viz_output, fps=viz_fps, max_step=last_step)
        except Exception as e:
            print(f"[viz] visualization failed: {e}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None, help="Override target query")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device index")
    parser.add_argument("--no-save", action="store_true", default=False,
                        help="Skip saving BEV maps to disk during the live loop")
    parser.add_argument("--ig-source", type=str, choices=["unseen", "epistemic", "coverage"], default="epistemic",
                        help="Information gain source for MPPI (unseen or epistemic)")
    parser.add_argument("--detector", type=str,
                        choices=["yolo", "coco_yolo", "hybrid", "grounding_dino",
                                 "locate_anything", "llmdet"],
                        default="hybrid",
                        help="Detector backend: yolo (YOLO-Worldv2), coco_yolo "
                             "(closed-set YOLOv8), hybrid (COCO→YOLOv8 else "
                             "YOLO-Worldv2), grounding_dino, locate_anything, or "
                             "llmdet (LLMDet + attention sinks)")
    parser.add_argument("--no-visualize", action="store_true", default=False,
                        help="Skip rendering nav_history video after the run")
    parser.add_argument("--viz-output", type=str, default="./figs/nav_history.mp4",
                        help="Output path for the post-run navigation video")
    parser.add_argument("--viz-fps", type=int, default=5,
                        help="FPS for the post-run navigation video")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    cfg = Config("config/config.yaml")
    if args.query is not None:
        cfg.target_query = args.query
    cfg.ig_source = args.ig_source
    cfg.detector = args.detector

    run(cfg, save_enabled=not args.no_save,
        visualize=not args.no_visualize,
        viz_output=args.viz_output,
        viz_fps=args.viz_fps)
