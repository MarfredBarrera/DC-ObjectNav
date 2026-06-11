"""Live perception + MPPI planning loop.

Three cadences in a single process:
    A. train_step              every step                  (fast)
    B. replan + 1 agent step   every REPLAN_INTERVAL       (MPPI is cheap)
    C. refresh buffer + maps   every cfg.hash_buffer_refresh_interval (bottleneck)

Bootstrap: spin a full circle observing each frame, then cold-train for
BOOTSTRAP_TRAIN_STEPS so the first map is meaningful before the first plan.
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
    spawn_agent_at_pos
)
from src.habitat.sim_interface import SimInterface
from src.perception.obj_detection import (
    make_detector, encode_query_with_negatives, target_prob_from_sims,
)
from src.perception.perception_stack import PerceptionStack
from src.perception.segmentation import MobileSAMSegmenter
from src.perception.utils import unprojection
from src.planning.mppi import MPPIPlanner
from src.planning.utils import normalize_sim
from visualize import render_navigation


REPLAN_INTERVAL = 100
SPIN_FRAMES = 36                       # 36 * 10° = full circle
SPIN_OMEGA = np.deg2rad(10) / 0.1      # 10° per 0.1s sim step
BOOTSTRAP_TRAIN_STEPS = 2000


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


def plan_one_action(perception, sim_iface, mppi, cfg,
                    action_queue: deque, k_max: int = 5,
                    det_score: float = 0.0,
                    detected: bool = False, det_box=None, rgb=None, depth=None, c2w=None,
                    last_box_goal=None, last_box_conf: float = -1.0,
                    segmenter=None):
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
    bev_sim = _to_numpy(perception.similarity_grid.get_2d_map(min_y=0.1, max_y=1.5))
    bev_epi = _to_numpy(perception.ugrid.get_2d_map(type='epistemic'))
    bev_occ = _to_numpy(perception.occupancy_grid.get_2d_map())

    if bev_sim is None or bev_epi is None or bev_occ is None:
        print("  plan: missing map(s); idling")
        return (action_queue.popleft() if action_queue else None), None, None, None, None, None

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
    # Layer 1: only spend a fresh SAM + MaskCLIP pass when this frame's
    # detection actually beats the cached max. Otherwise the cached goal
    # already represents the strongest sighting and we should aim at it
    # (see Layer 2). This makes the agent pursue the max-confidence goal
    # at all times — weak transient detections can't displace a strong one.
    # SAM runs on the *whole image* (not box-prompted) because the box is
    # often offset by ~one box-width at distance; CLIP picks the right mask.
    if det_box is not None and det_score > last_box_conf and segmenter is not None:
        det_cells = bev_cells_from_sam(
            perception, cfg, sim_iface.intrinsics,
            rgb, depth, c2w, perception.target_query,
            segmenter,
            min_mask_pixels=int(getattr(cfg, 'sam_min_mask_pixels', 200)),
            min_clip_sim=float(getattr(cfg, 'sam_min_clip_sim', 0.18)),
        )
        if det_cells is not None and det_cells.any():
            # Goal = the mask cell closest to the agent (near face of the
            # object's projected footprint), not the argmax of the smeared
            # similarity field. If that cell sits on the object (occupied),
            # snap to the nearest navigable free cell so the agent stops in
            # front of the target rather than inside it.
            cand = closest_cell_in_mask(det_cells, start_grid)
            if cand is not None:
                goal = snap_to_free(cand, bev_occ)
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
        start_grid, goal, bev_epi, bev_occ,
        initial_heading=heading,
        intrinsics=sim_iface.intrinsics,
        sensor_height=cfg.sensor_height,
        ig_source=cfg.ig_source,
        goal_confidence=goal_confidence,
    )
    mppi_score = None
    if U_opt is None or U_opt.shape[0] == 0:
        # No safe MPPI plan this replan — spin in place to scan for new
        # options. Rotating doesn't translate the agent, so it's always safe
        # in a free start cell, and the heading change usually reveals new
        # geometry that makes the next replan more likely to find a safe plan.
        print("  plan: MPPI returned no safe control sequence, replanning")
        action_queue.clear()
        return [0.0, -np.pi/4], None, None, None, goal, box_goal

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
        viz_fps: int = 5) -> None:
    # 1. Init simulator
    sim, agent = init_simulator(
        cfg.scene_path, resolution=cfg.img_width, fov_deg=cfg.fov, sensor_height=cfg.sensor_height,
        agent_radius=cfg.agent_radius,
        agent_height=cfg.agent_height,
    )
    # spawn_agent_at_random_navpoint(sim, agent)
    spawn_agent_at_pos(sim, agent, [0.0, 0.0, -5.0])
    scene_bounds = get_scene_bounds_from_pathfinder(sim)

    sim_iface = SimInterface(cfg, sim, agent)
    perception = PerceptionStack(cfg, scene_bounds)  # owns target_query; initialised from cfg

    mppi = MPPIPlanner(cfg, device=cfg.device)
    detector = make_detector(
        cfg.detector, device=cfg.device,
        negative_classes=getattr(cfg, 'det_negative_classes', None),
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

    # 2. Bootstrap A: spin in a full circle, observe each frame
    print(f"[bootstrap A] spinning {SPIN_FRAMES} frames (~360°)...")
    for f in range(SPIN_FRAMES):
        sim_iface.step([0.0, SPIN_OMEGA])
        rgb, depth, c2w = perception.observe(sim_iface)
        perception.update_replay_buffer(rgb, depth, c2w, sim_iface.intrinsics)
        perception.update_occupancy(depth, c2w, sim_iface.intrinsics)
        
        if save_enabled:
            # Save periodic bootstrap views
            rgb_dir = os.path.join(cfg.output_dir, "rgbs")
            os.makedirs(rgb_dir, exist_ok=True)
        #     rgb_img = (rgb.numpy() * 255).astype(np.uint8)
        #     imageio.imwrite(os.path.join(rgb_dir, f"bootstrap_{f:03d}.png"), rgb_img)
            
        torch.cuda.empty_cache()

        print(f"  bootstrap frame {f + 1}/{SPIN_FRAMES}")
    super_pts, super_feats = perception.make_super_batch()

    # 3. Bootstrap B: cold-train so first BEV maps have signal
    print(f"[bootstrap B] cold-training {BOOTSTRAP_TRAIN_STEPS} steps...")
    t0 = time.time()
    for s in range(BOOTSTRAP_TRAIN_STEPS):
        loss = perception.train_step(super_pts, super_feats)
        if s % 200 == 0:
            print(f"  bootstrap step {s:04d} | loss {loss:.5f} | t {time.time()-t0:.1f}s")

    # 4. Bootstrap C: build first maps, save grid extent for visualize.py
    print("[bootstrap C] computing first BEV maps...")
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

    # Two-phase planner state. `detected` latches to True (and never releases)
    # once `det_score >= cfg.detected_conf_threshold` for
    # `cfg.detected_persistence` consecutive replans. While latched, MPPI
    # silences IG and pulls hard toward the BEV similarity peak.
    detected = False
    detected_streak = 0

    # Best goal cell derived from a detection bounding box so far, with its
    # detector confidence. Persists across replans so a missing detection
    # this frame doesn't drop the agent back to global-argmax exploration.
    # Only overwritten when a NEW detection beats `last_box_conf` — a noisy
    # low-conf detection can't displace an earlier strong sighting.
    last_box_goal = None
    last_box_conf = -1.0

    # Tracks the last step actually executed by the loop. Used to cap the
    # post-run visualization so it doesn't pull in stale .npy snapshots left
    # over from a previous, longer run (traj_log.jsonl is truncated each run,
    # but the umaps/occ_maps/sim_maps directories are not).
    last_step = 0

    for step in range(1, cfg.iterations + 1):
        last_step = step
        # A. train every step
        loss = perception.train_step(super_pts, super_feats)

        # B. replan + 1 agent step
        if step % REPLAN_INTERVAL == 0:
            t_plan = time.time()
            pos = sim_iface.agent_position.copy()
            heading = get_agent_heading(sim_iface.agent)

            rgb_cur, depth_cur, c2w_cur = perception.observe(sim_iface)
            det_score, det_box = detector.detect(rgb_cur, perception.target_query)

            # Clear last replan's SAM side-channel so a mask is only persisted
            # on the frame SAM actually ran. `bev_cells_from_sam` resets these
            # too, but it only runs on a fresh qualifying detection — without
            # this reset the previous mask lingers and gets re-saved/re-logged
            # every replan, leaving a stale overlay on frames with no detection.
            if segmenter is not None:
                segmenter.last_mask = None
                segmenter.last_box = None
                segmenter.last_score = 0.0

            # Latch into DETECTED once we get `detected_persistence` consecutive
            # detections above threshold. Never unlatches — see comment above.
            if not detected:
                if det_score >= cfg.detected_conf_threshold:
                    detected_streak += 1
                else:
                    detected_streak = 0
                if detected_streak >= cfg.detected_persistence:
                    detected = True
                    print(f"step {step}: DETECTED — entering exploit mode "
                          f"(det_score={det_score:.3f})")

            action, ref_traj, opt_traj, score, goal_cell, box_goal = plan_one_action(
                perception, sim_iface, mppi, cfg, action_queue,
                det_score=det_score, detected=detected,
                det_box=det_box, rgb=rgb_cur, depth=depth_cur, c2w=c2w_cur,
                last_box_goal=last_box_goal, last_box_conf=last_box_conf,
                segmenter=segmenter,
            )
            # Cache only when THIS frame's box yielded a valid goal AND its
            # confidence beats the best we've seen. Fallback paths (cached /
            # global argmax) return goal_cell but leave box_goal=None, so they
            # never touch the cache.
            if box_goal is not None and det_score > last_box_conf:
                last_box_goal = box_goal
                last_box_conf = float(det_score)
                print(f"  cached new goal {box_goal} (conf {det_score:.3f})")
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
            if goal_cell is not None:
                sg = perception.similarity_grid
                goal_x_m = sg.min_x + goal_cell[1] * cfg.voxel_resolution
                goal_z_m = sg.min_z + goal_cell[0] * cfg.voxel_resolution
                goal_str = f"({goal_x_m:+.2f}, {goal_z_m:+.2f})m"
            else:
                goal_str = "—"
            if action is not None:
                sim_iface.step(action, dt=cfg.mppi_dt)
                print(f"step {step}: plan {time.time()-t_plan:.2f}s | "
                      f"action [v={action[0]:+.3f} m/s, w={action[1]:+.3f} rad/s] | "
                      f"position: {pos[0]:.2f}, {pos[2]:.2f} | "
                      f"det[{det_backend}]: {det_score:.3f} | "
                      f"goal: {goal_str} | w_conf: {w_conf:.2f} | "
                      f"mode: {mode}")
            else:
                print(f"step {step}: plan {time.time()-t_plan:.2f}s | no action | "
                      f"det[{det_backend}]: {det_score:.3f} | "
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
                }) + '\n')

            if stop_now:
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

    # Snapshot maps + RGB + 2D-sim aligned to last_step so the visualizer's
    # final frame is up-to-date. The C cadence only fires every refresh
    # interval, and early termination via `stop_now` can leave it 100+ steps
    # behind the actual last action. Skipped when last_step already matches
    # the most recent refresh tick — same files would just be overwritten.
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
    sim.close()
    print("[main] done.")

    if visualize:
        print(f"[viz] rendering navigation video up to step {last_step}...")
        try:
            render_navigation(cfg, viz_output, fps=viz_fps, max_step=last_step)
        except Exception as e:
            print(f"[viz] visualization failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None, help="Override target query")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device index")
    parser.add_argument("--no-save", action="store_true", default=False,
                        help="Skip saving BEV maps to disk during the live loop")
    parser.add_argument("--ig-source", type=str, choices=["unseen", "epistemic"], default="epistemic",
                        help="Information gain source for MPPI (unseen or epistemic)")
    parser.add_argument("--detector", type=str,
                        choices=["yolo", "coco_yolo", "hybrid", "grounding_dino"],
                        default="hybrid",
                        help="Detector backend: yolo (YOLO-Worldv2), coco_yolo "
                             "(closed-set YOLOv8), hybrid (COCO→YOLOv8 else "
                             "YOLO-Worldv2), or grounding_dino")
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
