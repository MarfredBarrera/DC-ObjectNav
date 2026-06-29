from typing import Optional
import yaml
from pathlib import Path


class Config:
    def __init__(self, yaml_path: Optional[str] = None):
        """
        Initialize configuration from YAML file or use defaults.
        
        Args:
            yaml_path: Path to YAML configuration file. If None, uses default values.
        """
        # Visualization
        self.viz_interval = 500
        self.vmax_epi = 0.5
        # Set all default values first
        # Paths
        self.output_dir = "/workspace/DCON/output/current_scene"
        self.output_name = "model.pt"

        # Habitat Settings
        self.scene_path = "/workspace/DCON/gibson_scenes/Anaheim.glb"
        self.img_width = 720
        self.img_height = 720
        self.fov = 90
        self.data_queue_size = 30
        self.training_queue_size = 10
        self.sensor_height = 1.0
        self.min_sensor_dist = 0.00
        self.max_sensor_dist = 10.0
        # Navmesh agent radius (m). Habitat's `pathfinder.try_step` (used to
        # move the agent in SimInterface) clamps motion to the navmesh, which
        # insets every wall by this radius — so narrow gaps the BEV considers
        # open can be sealed in the navmesh, wedging the agent. Recomputed on
        # load (overrides the scene's prebaked .navmesh). Smaller = the agent
        # squeezes through tighter gaps. <=0 keeps the scene's baked navmesh.
        self.agent_radius = 0.1
        self.agent_height = 1.5
        
        # Semantics Settings (MaskCLIP)
        self.maskclip_model_name = "ViT-B/16"
        self.maskclip_input_size = 448
        self.target_query = "green plant"
        
        # Training Settings
        self.iterations = 10000
        self.device = "cuda"
        self.gpu_indices = "0"
        
        # Camera / Data
        self.near_plane = 0.01
        
        # Hyperparameters
        self.ssim_weight = 0.2
        self.l1_weight = 0.8
        self.scale_reg = 0.01
        self.uncertainty_weight = 0.1
        self.uncertainty_dim = 16
        
        # Learning Rates
        self.lr_means = 1.6e-4
        self.lr_scales = 0.005
        self.lr_quats = 0.001
        self.lr_opacities = 5e-2
        self.lr_sh0 = 2.5e-3
        self.lr_shN = 2.5e-3 / 20
        self.lr_uncertainty = 1e-3
        
        # Strategy Settings
        self.refine_start_iter = 100
        self.refine_stop_iter = 9500
        self.refine_every = 100
        self.reset_every = 1000
        self.grow_grad2d = 0.0002
        self.prune_opa = 0.005
        
        # Toggles
        self.train_uncertainty = True
        
        # HashGrid Configuration
        self.hash_n_levels = 16
        self.hash_n_features_per_level = 4
        self.hash_log2_hashmap_size = 21
        self.hash_base_resolution = 16
        self.hash_per_level_scale = 1.5
        
        # HashGrid MLP
        self.hash_n_neurons = 128
        self.hash_n_hidden_layers = 3
        self.hash_feature_dim = 512
        
        # HashGrid Training
        self.hash_lr = 1e-3
        self.hash_per_image_batch_size = 4096
        self.hash_train_batch_size = 8192
        self.hash_inference_batch_size = 16384
        self.hash_replay_buffer_size = 10  # legacy: still used by offline.py's own buffer
        self.hash_buffer_refresh_interval = 200
        self.hash_per_frame_cache_size = 8192*4
        # Flat history buffer: bounded ring of (pt, feat) rows. Memory cost is
        # capacity * (3 + hash_feature_dim) * 4 bytes (~412MB at 200k & 512-dim).
        self.history_buffer_capacity = 500_000
        self.hash_train_every_n_steps = 1
        self.hash_warmup_steps = 0

        # Ensemble
        self.ensemble_num_models = 3

        # Grid
        self.voxel_resolution = 0.05
        self.grid_max_height = 2.0
        # Height band (world-Y meters, ABSOLUTE coordinates) collapsed into the
        # 2D BEV maps — similarity, occupancy, coverage deficit, and the
        # epistemic uncertainty volume all reduce over this slice. Defaults
        # assume the floor sits near Y=0. min must sit ~0.2 m ABOVE the floor
        # surface: the occupancy grid marks the floor plane itself occupied and
        # it bleeds up ~one voxel row, so a min flush with the floor makes the
        # whole walkable area read as obstacle (0.2 is what OccupancyGrid used
        # before the band was unified). For a multi-floor scene, set the band to
        # the target floor (floor+0.2 .. floor+1.5) so upper floors don't bleed
        # in; raise grid_max_height if the band exceeds 2.0 m. min must be >= the
        # scene's min_y; max is clipped by grid_max_height.
        self.bev_height_min = 0.2
        self.bev_height_max = 1.5
        # Soft coverage accumulator (OccupancyGrid.coverage). Each depth
        # update adds min(1, (ref_dist/d)^2) to every voxel it confirms —
        # close views count fully, distant ones fractionally. The IG-facing
        # deficit is exp(-coverage/tau): tau = how many full-quality views a
        # voxel needs before it stops attracting exploration (~63% drained
        # after tau views, ~95% after 3*tau).
        self.coverage_ref_dist_m = 2.0
        self.coverage_tau = 3.0
        # Zero epistemic/aleatoric uncertainty at observed-FREE voxels when
        # building maps. The field only trains on surface points, so free-air
        # uncertainty is init noise ("phantom traces" over traversed rooms),
        # not signal. Occupied keeps real surface uncertainty; unseen keeps
        # the frontier signal. Also switches the BEV reduction to max-over-Y.
        self.mask_free_epistemic = True

        # Planning
        self.mppi_dt = 0.1
        self.mppi_w_sign = -1.0  # flip if Habitat ω rotates opposite of MPPI's heading convention
        self.mppi_max_w_rps = 2.0  # rad/s clamp — prevents huge spins from sharp A* turns
        self.mppi_min_v_mps = 0.0
        self.mppi_max_v_mps = 1.0
        self.mppi_horizon = 150           # rollout length (steps)
        self.mppi_goal_carve_radius = 1   # free disk carved around the goal cell for collision
        # Collision-check subsampling: number of intermediate points checked
        # along each waypoint-to-waypoint segment (in addition to the
        # waypoints). The agent can move >1 cell per horizon step, so a
        # waypoint-only check tunnels through thin walls that fall between two
        # consecutive waypoints. Set to N so the spacing between checks is
        # ~(cells per step)/(N+1); 0 disables (waypoints only).
        self.mppi_collision_substeps = 4

        self.mppi_lambda = 1.0     # softmax temperature: high = flatter weights, wider sampling
        self.mppi_w_ig = 30.0      # information-gain reward: high = chase uncertainty
        self.mppi_w_goal = 1.0     # goal-distance pull: low early = pure IG exploration

        # Detection-confidence → goal-pull weighting, with hysteresis. The
        # latched exploit_conf ratchets up on a sighting and decays on misses;
        # below threshold the goal pull is off (pure IG), above it the concave
        # curve w = scale*(a^conf - 1)/(a - 1) saturates quickly (a < 1).
        self.mppi_conf_weight_a = 0.1
        self.mppi_conf_weight_scale = 100.0
        self.mppi_conf_decay = 0.98
        self.mppi_conf_threshold = 0.1
        # Control-noise stddev is anchored to half the actuator limits inside
        # optimize_trajectory; no scalar knob. Trajectory- and action-level
        # annealing (mppi_anneal_beta_traj / mppi_anneal_beta_action) shape
        # the envelope across iters and horizon index.

        # DIAL-MPC dual annealing. Per-(iter, horizon-step) noise scaling
        #   factor(it, h) = exp(-it / (β_traj * N) - (H - h) / (β_action * H))
        # for MPPI iter `it` ∈ [0, N) and horizon step `h` ∈ [0, H). Smaller
        # β → more aggressive annealing. β_traj shrinks variance across iters
        # (iter 0 widest, iter N-1 narrowest); β_action shrinks variance across
        # horizon (step 0 narrowest, step H-1 widest). The action-level schedule
        # pairs naturally with the warm-start: early horizon steps are inherited
        # from a known-safe plan and get small perturbation; tail steps are
        # zero-padded and explore widely.
        self.mppi_num_iters = 5
        self.mppi_anneal_beta_traj = 3.0
        self.mppi_anneal_beta_action = 1.5

        # Visualization
        self.viz_interval = 500

        self.detector = "hybrid"

        # Canonical distractor vocabulary for false-positive suppression.
        # Used at two levels:
        #   1) YOLO-World: registered as competing classes alongside the
        #      target query; only boxes whose winning class is the target
        #      are accepted, so salient walls/doors get claimed by their
        #      own class instead of leaking into the target's.
        #   2) SAM mask scoring: per-mask CLIP similarity is converted to a
        #      softmax over [target + distractors] ("more pillow than wall")
        #      instead of a raw cosine, which is far more discriminative in
        #      CLIP's compressed similarity range.
        # Stored as bare nouns; the CLIP path prepends "a " to match the
        # repo's "a [object]" prompt convention.
        self.det_negative_classes = [
            "wall", "door", "window", "floor", "ceiling",
            "curtain", "cabinet", "shelf", "picture",
        ]

        # Detection-based termination. Once `det_score >= detected_conf_threshold`
        # for `detected_persistence` consecutive replans, latch into DETECTED
        # mode (IG off, goal pull saturated via goal_confidence=1.0). Terminate
        # the run when the agent is within `stop_distance_m` of the BEV
        # similarity peak.
        self.detected_conf_threshold = 0.5
        self.detected_persistence = 1
        self.stop_distance_m = 1.5

        # Classify each detection by BOTH object distance and box size (see
        # classify_detection in main.py). Box fractions are detector box area /
        # image area; distances (m) are to the box-center world point. Each
        # threshold disables at <=0.
        #   TOO CLOSE  (dist < detected_min_dist_m OR box > detected_max_box_frac)
        #     → ignored entirely: no goal, no confidence weight, no latch (the
        #       box fills the frame and carries no usable localization).
        #   TOO FAR    (dist > detected_max_dist_m OR box < detected_min_box_frac)
        #     → may contribute the confidence weight but is NOT persistent: it
        #       doesn't latch and isn't cached as a goal.
        #   USABLE BAND (anything else) → persistent: latches (after
        #     `detected_persistence` consecutive), is cached as the goal, and
        #     contributes the confidence weight.
        self.detected_min_box_frac = 0.01
        self.detected_max_box_frac = 0.95
        self.detected_min_dist_m = 0.1
        self.detected_max_dist_m = 3.0

        # Detector cadence once latched into EXPLOIT (`detected`) mode. The goal
        # is pinned to the cached box cell and committed hard there, so
        # re-running an expensive detector (e.g. LocateAnything ~1 s/call) every
        # replan buys little. While detected, run the detector only once every
        # `exploit_redetect_interval` replans (<=0 → never re-detect after
        # latching; the cached goal is reused for the rest of the run). A
        # periodic value > 0 lets the goal refine as the agent closes in and
        # the box gets more accurate. SEARCH mode always detects every replan.
        self.exploit_redetect_interval = 0

        # How a fresh detection box is turned into the MPPI goal cell:
        #   "box_center" — project the *center pixel* of the detector's box
        #     straight to one BEV cell and use it as the goal verbatim (no SAM,
        #     no similarity argmax, no free-cell snapping). Suited to detectors
        #     that already emit tight, accurate boxes (e.g. LocateAnything).
        #   "sam" — re-localize with MobileSAM whole-image auto-gen + MaskCLIP
        #     best-mask scoring, then take the mask cell closest to the agent
        #     (the legacy path; see the MobileSAM block below).
        self.goal_projection = "box_center"

        # MobileSAM goal refinement. When the detector fires AND the new
        # det_score beats the cached one, we re-localize the target by:
        #   1) running MobileSAM auto-mask-gen on the whole image,
        #   2) picking the mask with highest mean CLIP-similarity to the
        #      target query,
        #   3) projecting that mask's pixels to BEV cells.
        # Set use_mobile_sam=False to fall back to the old box-based path.
        # min_clip_sim is a floor on the best mask's CLIP score; below this
        # we discard the segmentation as not actually the target.
        self.use_mobile_sam = True
        self.sam_checkpoint = "SAM_models/mobile_sam.pt"
        self.sam_points_per_side = 16
        self.sam_pred_iou_thresh = 0.86
        self.sam_stability_score_thresh = 0.90
        self.sam_min_mask_region_area = 200
        self.sam_min_mask_pixels = 200
        self.sam_min_clip_sim = 0.18
        # Softmax-over-distractors acceptance for the chosen mask (see
        # det_negative_classes). `sam_softmax_temp` is the CLIP-style logit
        # scale applied to the cosine vector before softmax; the best mask
        # is accepted only if its target probability clears
        # `sam_min_target_prob` AND its raw target cosine clears
        # `sam_min_clip_sim` (absolute floor — a mask can win the softmax
        # while matching nothing at all).
        self.sam_softmax_temp = 100.0
        self.sam_min_target_prob = 0.5

        # Neutral attention-sink false-positive gate (Ruis et al., "Fantastic
        # Tractor-Dogs...", ICLR 2026), realized as a post-hoc CLIP check on the
        # detector's box (see SinkGatedDetector). When enabled, every box the
        # base detector emits is cropped, mean-CLIP-encoded, and compared
        # against the target query and a semantically-NEUTRAL sink via a sharp
        # softmax; if the sink out-scores the query (target prob <
        # sink_min_target_prob) the box is dropped as "none of the above". This
        # is the faithful adaptation of the paper's mechanism to a generative
        # grounding detector (LocateAnything), which has no class-prompt softmax
        # to redirect internally. The sink is NOT a semantic negative — the
        # paper shows real negatives ("wall", "door") don't suppress false
        # positives; det_negative_classes is unused by this gate.
        #   sink_init: "mean" (mean CLIP encoding over a generic category-spread
        #     vocab — output-space analog of the paper's mean-of-vocabulary
        #     init; best default for CLIP), "special" (encode sink_special_str,
        #     the paper's best for LLM-trained detectors), or "random".
        #   sink_min_target_prob: target must win at least this softmax mass
        #     over [query, sinks] to survive (0.5 with one sink ≈ "query beats
        #     sink"); raise to gate harder.
        # Cost: one CLIP forward per *fired* detection — negligible against
        # LocateAnything (~1 s/call), ~3x overhead on a ~15 ms YOLO-World.
        self.sink_gate = True
        self.sink_init = "mean"
        self.sink_num = 1
        self.sink_special_str = "[()]"
        self.sink_softmax_temp = 100.0
        self.sink_min_target_prob = 0.4
        self.sink_crop_pad = 1.0
        # Crop pooling for the gate: "mean" pools the whole crop into one CLIP
        # feature; "top_pct" ranks crop patches by query similarity and pools the
        # top `sink_top_pct` fraction of those embeddings into one feature scored
        # against the whole bank, rescuing small/partial-object boxes the mean-pool
        # gate over-rejects (the object's patches no longer get averaged away) — but
        # only the query selects patches, so it over-fires on background/walls.
        # "top_pct_pertext" is the symmetric middle ground: query AND each sink
        # average their own best `sink_top_pct` patches, so the neutral sink can
        # reject walls while small objects still survive.
        self.sink_pool = "top_pct"
        self.sink_top_pct = 0.05
        self.sink_seed = 0
        # LLMDet detector (detector="llmdet") with the paper's training-free
        # attention sinks built into its own vision-language fusion layers — the
        # faithful Ruis et al. (ICLR 2026) method, unlike the post-hoc cropped-
        # CLIP sink_gate above. MUST use the `iSEE-Laboratory/llmdet_*` weights
        # (model_type "mm-grounding-dino", loads natively as MMGroundingDino); the
        # `fushh7/*_hf` weights declare plain "grounding-dino" and load with a
        # broken (non-discriminative) contrastive head — do NOT use them.
        # `llmdet_use_sinks` toggles the sinks for an A/B. Defaults tuned by a
        # 60-frame sweep (87 COCO-YOLO-oracle true positives, 600 out-of-domain
        # false positives), reported as true-positive retention at a target
        # background FP-rejection:
        #   * model=large: clearly best (no-sink separation 0.32 vs tiny 0.24).
        #     An earlier 3-frame test wrongly favored tiny — noise. base/large
        #     repos: iSEE-Laboratory/llmdet_{base,large}.
        #   * num_sinks=48: sinks help large at every operating point, most at the
        #     aggressive end — at 99% FP-rejection TP-retention 0.54->0.71, at 95%
        #     0.74->0.87. (8/24 over-suppress; `sink_init` "special"=[()] vs "mean"
        #     is INERT — the BERT text encoder recontextualizes the [unused]
        #     tokens, washing out the word-embedding init.)
        #   * threshold=0.42: for large+sinks48 this is the ~95% FP-rejection
        #     point (~85-87% TP-retention). Raise to ~0.47 for ~99% FP-rejection
        #     (~71% TP-retention) toward the paper's near-elimination.
        self.llmdet_model_name = "iSEE-Laboratory/llmdet_large"
        self.llmdet_threshold = 0.42
        self.llmdet_use_sinks = True
        self.llmdet_num_sinks = 48
        self.llmdet_sink_init = "special"
        # Visualization-only: when rendering nav_history.mp4, overlay the
        # most recent saved SAM mask whose step is within this many ticks
        # of the current viz frame. Larger = more frames get an overlay
        # but the mask may be stale (from an older viewpoint).
        self.sam_lookback_steps = 200

        # Load from YAML if path provided
        if yaml_path is not None:
            self._load_from_yaml(yaml_path)
    
    def _load_from_yaml(self, yaml_path: str):
        """Load configuration from a YAML file and override defaults."""
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Paths
        if 'paths' in yaml_data:
            paths = yaml_data['paths']
            if 'output_dir' in paths:
                self.output_dir = paths['output_dir']
            if 'output_name' in paths:
                self.output_name = paths['output_name']
        # Habitat Settings
        if 'habitat' in yaml_data:
            habitat = yaml_data['habitat']
            if 'scene_path' in habitat:
                self.scene_path = habitat['scene_path']
            if 'img_width' in habitat:
                self.img_width = habitat['img_width']
            if 'img_height' in habitat:
                self.img_height = habitat['img_height']
            if 'fov' in habitat:
                self.fov = habitat['fov']
            if 'data_queue_size' in habitat:
                self.data_queue_size = habitat['data_queue_size']
            if 'sensor_height' in habitat:
                self.sensor_height = habitat['sensor_height']
            if 'min_sensor_dist' in habitat:
                self.min_sensor_dist = habitat['min_sensor_dist']
            if 'max_sensor_dist' in habitat:
                self.max_sensor_dist = habitat['max_sensor_dist']
            if 'agent_radius' in habitat:
                self.agent_radius = habitat['agent_radius']
            if 'agent_height' in habitat:
                self.agent_height = habitat['agent_height']
            if 'training_queue_size' in habitat:
                self.training_queue_size = habitat['training_queue_size']
        
        # Semantics (MaskCLIP)
        if 'semantics' in yaml_data:
            sem = yaml_data['semantics']
            if 'maskclip_model_name' in sem:
                self.maskclip_model_name = sem['maskclip_model_name']
            if 'maskclip_input_size' in sem:
                self.maskclip_input_size = sem['maskclip_input_size']
            if 'target_query' in sem:
                self.target_query = sem['target_query']
        
        # Training
        if 'training' in yaml_data:
            train = yaml_data['training']
            if 'iterations' in train:
                self.iterations = train['iterations']
            if 'device' in train:
                self.device = train['device']
            if 'gpu_indices' in train:
                self.gpu_indices = train['gpu_indices']
            if 'near_plane' in train:
                self.near_plane = train['near_plane']
        
        # Hyperparameters
        if 'hyperparameters' in yaml_data:
            hyper = yaml_data['hyperparameters']
            if 'ssim_weight' in hyper:
                self.ssim_weight = hyper['ssim_weight']
            if 'l1_weight' in hyper:
                self.l1_weight = hyper['l1_weight']
            if 'scale_reg' in hyper:
                self.scale_reg = hyper['scale_reg']
            if 'uncertainty_weight' in hyper:
                self.uncertainty_weight = hyper['uncertainty_weight']
            if 'uncertainty_dim' in hyper:
                self.uncertainty_dim = hyper['uncertainty_dim']
        
        # Learning Rates
        if 'learning_rates' in yaml_data:
            lr = yaml_data['learning_rates']
            if 'means' in lr:
                self.lr_means = lr['means']
            if 'scales' in lr:
                self.lr_scales = lr['scales']
            if 'quats' in lr:
                self.lr_quats = lr['quats']
            if 'opacities' in lr:
                self.lr_opacities = lr['opacities']
            if 'sh0' in lr:
                self.lr_sh0 = lr['sh0']
            if 'shN' in lr:
                self.lr_shN = lr['shN']
            if 'uncertainty' in lr:
                self.lr_uncertainty = lr['uncertainty']
        
        # Strategy
        if 'strategy' in yaml_data:
            strat = yaml_data['strategy']
            if 'refine_start_iter' in strat:
                self.refine_start_iter = strat['refine_start_iter']
            if 'refine_stop_iter' in strat:
                self.refine_stop_iter = strat['refine_stop_iter']
            if 'refine_every' in strat:
                self.refine_every = strat['refine_every']
            if 'reset_every' in strat:
                self.reset_every = strat['reset_every']
            if 'grow_grad2d' in strat:
                self.grow_grad2d = strat['grow_grad2d']
            if 'prune_opa' in strat:
                self.prune_opa = strat['prune_opa']
        
        # Toggles
        if 'toggles' in yaml_data:
            toggles = yaml_data['toggles']
            if 'train_uncertainty' in toggles:
                self.train_uncertainty = toggles['train_uncertainty']
        
        # HashGrid
        if 'hashgrid' in yaml_data:
            hg = yaml_data['hashgrid']
            if 'n_levels' in hg:
                self.hash_n_levels = hg['n_levels']
            if 'n_features_per_level' in hg:
                self.hash_n_features_per_level = hg['n_features_per_level']
            if 'log2_hashmap_size' in hg:
                self.hash_log2_hashmap_size = hg['log2_hashmap_size']
            if 'base_resolution' in hg:
                self.hash_base_resolution = hg['base_resolution']
            if 'per_level_scale' in hg:
                self.hash_per_level_scale = hg['per_level_scale']
            if 'n_neurons' in hg:
                self.hash_n_neurons = hg['n_neurons']
            if 'n_hidden_layers' in hg:
                self.hash_n_hidden_layers = hg['n_hidden_layers']
            if 'feature_dim' in hg:
                self.hash_feature_dim = hg['feature_dim']
            if 'lr' in hg:
                self.hash_lr = hg['lr']
            if 'per_image_batch_size' in hg:
                self.hash_per_image_batch_size = hg['per_image_batch_size']
            if 'train_batch_size' in hg:
                self.hash_train_batch_size = hg['train_batch_size']
            if 'inference_batch_size' in hg:
                self.hash_inference_batch_size = hg['inference_batch_size']
            if 'replay_buffer_size' in hg:
                self.hash_replay_buffer_size = hg['replay_buffer_size']
            if 'buffer_refresh_interval' in hg:
                self.hash_buffer_refresh_interval = hg['buffer_refresh_interval']
            if 'per_frame_cache_size' in hg:
                self.hash_per_frame_cache_size = hg['per_frame_cache_size']
            if 'history_buffer_capacity' in hg:
                self.history_buffer_capacity = hg['history_buffer_capacity']
            if 'train_every_n_steps' in hg:
                self.hash_train_every_n_steps = hg['train_every_n_steps']
            if 'warmup_steps' in hg:
                self.hash_warmup_steps = hg['warmup_steps']
        
        # Ensemble
        if 'ensemble' in yaml_data:
            ensemble = yaml_data['ensemble']
            if 'num_models' in ensemble:
                self.ensemble_num_models = ensemble['num_models']

        # Grid
        if 'grid' in yaml_data:
            grid = yaml_data['grid']
            if 'voxel_resolution' in grid:
                self.voxel_resolution = grid['voxel_resolution']
            if 'max_height' in grid:
                self.grid_max_height = grid['max_height']
            for key in ('coverage_ref_dist_m', 'coverage_tau', 'mask_free_epistemic',
                        'bev_height_min', 'bev_height_max'):
                if key in grid:
                    setattr(self, key, grid[key])

        # Planning
        if 'planning' in yaml_data:
            planning = yaml_data['planning']
            if 'mppi_dt' in planning:
                self.mppi_dt = planning['mppi_dt']
            if 'mppi_w_sign' in planning:
                self.mppi_w_sign = planning['mppi_w_sign']
            if 'mppi_max_w_rps' in planning:
                self.mppi_max_w_rps = planning['mppi_max_w_rps']
            if 'mppi_min_v_mps' in planning:
                self.mppi_min_v_mps = planning['mppi_min_v_mps']
            if 'mppi_max_v_mps' in planning:
                self.mppi_max_v_mps = planning['mppi_max_v_mps']
            for key in (
                'mppi_num_iters',
                'mppi_anneal_beta_traj', 'mppi_anneal_beta_action',
                'mppi_collision_substeps',
            ):
                if key in planning:
                    setattr(self, key, planning[key])

        # Detection / termination
        if 'detection' in yaml_data:
            det = yaml_data['detection']
            for key in ('detector', 'detected_conf_threshold', 'detected_persistence',
                        'stop_distance_m', 'det_negative_classes', 'goal_projection',
                        'exploit_redetect_interval',
                        'detected_min_box_frac', 'detected_max_box_frac',
                        'detected_min_dist_m', 'detected_max_dist_m',
                        'sink_gate', 'sink_init', 'sink_num', 'sink_special_str',
                        'sink_softmax_temp', 'sink_min_target_prob',
                        'sink_crop_pad', 'sink_pool', 'sink_top_pct',
                        'sink_seed',
                        'llmdet_model_name', 'llmdet_threshold',
                        'llmdet_use_sinks', 'llmdet_num_sinks', 'llmdet_sink_init'):
                if key in det:
                    setattr(self, key, det[key])

        # MobileSAM goal refinement
        if 'sam' in yaml_data:
            sam = yaml_data['sam']
            for key in (
                'use_mobile_sam', 'sam_checkpoint',
                'sam_points_per_side', 'sam_pred_iou_thresh',
                'sam_stability_score_thresh', 'sam_min_mask_region_area',
                'sam_min_mask_pixels', 'sam_min_clip_sim',
                'sam_softmax_temp', 'sam_min_target_prob',
                'sam_lookback_steps',
            ):
                if key in sam:
                    setattr(self, key, sam[key])