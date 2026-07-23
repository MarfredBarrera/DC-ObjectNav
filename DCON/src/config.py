from typing import Optional
import yaml


class Config:
    # YAML section → keys loaded from it. A dict entry maps yaml key →
    # attribute name; a list entry means the yaml key IS the attribute name.
    # Every attribute here must have a default in __init__ (no getattr
    # fallbacks anywhere in the codebase).
    _YAML_SCHEMA = {
        'paths': ['output_dir'],
        'habitat': ['scene_path', 'img_width', 'img_height', 'fov',
                    'sensor_height', 'min_sensor_dist', 'max_sensor_dist',
                    'agent_radius', 'agent_height', 'max_spawn_snap_m'],
        'semantics': ['target_query'],
        'training': ['iterations', 'device'],
        'hashgrid': {
            'n_levels': 'hash_n_levels',
            'n_features_per_level': 'hash_n_features_per_level',
            'log2_hashmap_size': 'hash_log2_hashmap_size',
            'base_resolution': 'hash_base_resolution',
            'per_level_scale': 'hash_per_level_scale',
            'n_neurons': 'hash_n_neurons',
            'n_hidden_layers': 'hash_n_hidden_layers',
            'feature_dim': 'hash_feature_dim',
            'lr': 'hash_lr',
            'train_batch_size': 'hash_train_batch_size',
            'inference_batch_size': 'hash_inference_batch_size',
            'buffer_refresh_interval': 'hash_buffer_refresh_interval',
            'per_frame_cache_size': 'hash_per_frame_cache_size',
            'history_buffer_capacity': 'history_buffer_capacity',
        },
        'grid': {
            'voxel_resolution': 'voxel_resolution',
            'max_height': 'grid_max_height',
            'bev_height_min': 'bev_height_min',
            'bev_height_max': 'bev_height_max',
            'mask_free_epistemic': 'mask_free_epistemic',
        },
        'planning': ['mppi_dt', 'mppi_w_sign', 'mppi_max_w_rps',
                     'mppi_min_v_mps', 'mppi_max_v_mps', 'mppi_horizon',
                     'mppi_collision_substeps', 'mppi_lambda', 'mppi_w_ig',
                     'mppi_w_goal', 'mppi_conf_weight_a',
                     'mppi_conf_weight_scale', 'mppi_conf_decay',
                     'mppi_conf_threshold', 'mppi_num_iters',
                     'mppi_anneal_beta_traj', 'mppi_anneal_beta_action',
                     'mppi_occupied_cell_cost', 'mppi_unseen_cell_cost',
                     'discrete_actions', 'discrete_forward_m',
                     'discrete_turn_deg', 'discrete_lookahead_m',
                     'max_agent_steps', 'ddppo_checkpoint_path'],
        'detection': ['detected_persistence',
                      'stop_distance_m', 'exploit_redetect_interval',
                      'detected_min_box_frac', 'detected_max_box_frac',
                      'detected_min_dist_m', 'detected_max_dist_m',
                      'sink_special_str', 'llmdet_model_name',
                      'llmdet_threshold', 'llmdet_use_sinks',
                      'llmdet_num_sinks', 'llmdet_sink_init',
                      'clipseg_model_name',
                      'background_terms', 'distractor_objects',
                      'clipseg_pairwise',
                      'field_verify', 'field_verify_threshold',
                      'field_verify_presence_floor',
                      'field_verify_top_frac', 'field_verify_min_points',
                      'field_verify_pool', 'field_verify_save_frames'],
        'visualization': ['viz_interval', 'vmax_epi'],
    }

    def __init__(self, yaml_path: Optional[str] = None):
        """
        Initialize configuration from YAML file or use defaults.

        Args:
            yaml_path: Path to YAML configuration file. If None, uses default values.
        """
        # Paths
        self.output_dir = "/workspace/DCON/output/current_scene"

        # Habitat Settings
        self.scene_path = "/workspace/DCON/benchmarks/gibson_scenes/Anaheim.glb"
        self.img_width = 720
        self.img_height = 720
        self.fov = 90
        self.sensor_height = 1.25
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
        # Max distance (m) pathfinder.snap_point may move a requested spawn
        # point. Normal navmesh snapping is a few cm (settling onto the
        # walkable surface); a much larger snap means the requested point
        # isn't actually connected to this scene's navmesh (a disconnected
        # floor, broken stairs in the reconstruction, or a bad annotation)
        # and got matched to an unrelated, possibly different-floor location
        # instead. main.run() raises rather than running a doomed episode
        # against the wrong floor's geometry for the full step budget.
        self.max_spawn_snap_m = 1.0

        # Semantics Settings (dense supervision comes from CLIPSeg — see
        # clipseg_model_name under Detection and CLIPSegSemantics).
        self.target_query = "green plant"

        # Training Settings
        self.iterations = 10000
        self.device = "cuda"

        # HashGrid Configuration
        self.hash_n_levels = 16
        self.hash_n_features_per_level = 4
        self.hash_log2_hashmap_size = 21
        self.hash_base_resolution = 16
        self.hash_per_level_scale = 1.5

        # HashGrid MLP
        self.hash_n_neurons = 128
        self.hash_n_hidden_layers = 3
        # Scalar CLIPSeg relevance score, not a MaskCLIP embedding (see
        # CLIPSegSemantics / PerceptionStack) -- the field regresses a single
        # query-conditioned score per 3D point.
        self.hash_feature_dim = 1

        # HashGrid Training
        self.hash_lr = 1e-3
        self.hash_train_batch_size = 8192
        self.hash_inference_batch_size = 16384
        self.hash_buffer_refresh_interval = 200
        self.hash_per_frame_cache_size = 8192*4
        # Flat history buffer: bounded ring of (pt, feat) rows. Memory cost is
        # capacity * (3 + hash_feature_dim) * 4 bytes (~8MB at 500k & scalar feature_dim=1).
        self.history_buffer_capacity = 500_000

        # Grid
        self.voxel_resolution = 0.05
        self.grid_max_height = 2.0
        # Height band (world-Y meters, ABSOLUTE coordinates) collapsed into the
        # 2D BEV maps — similarity, occupancy, and the epistemic uncertainty
        # volume all reduce over this slice. Defaults
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
        # Zero epistemic/aleatoric uncertainty at observed-FREE voxels when
        # building maps. The field only trains on surface points, so free-air
        # uncertainty is init noise ("phantom traces" over traversed rooms),
        # not signal. Occupied keeps real surface uncertainty; unseen keeps
        # the frontier signal. Also switches the BEV reduction to max-over-Y.
        self.mask_free_epistemic = True

        # Planning
        self.mppi_dt = 0.1
        self.mppi_w_sign = -1.0  # flip if Habitat ω rotates opposite of MPPI's heading convention
        self.mppi_max_w_rps = 2.0  # rad/s clamp — prevents huge spins from sharp turns
        self.mppi_min_v_mps = 0.0
        self.mppi_max_v_mps = 1.0
        self.mppi_horizon = 150           # rollout length (steps)
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
        # Per-cell cost multiplier for traversing OCCUPIED cells in the
        # obstacle-aware goal-distance field (planning/utils.goal_distance_field).
        # High enough that crossing a wall (~thickness × this) always loses to
        # any indoor detour, low enough that a goal buried a few cells inside
        # the target object's surface blob still seeds a finite wavefront.
        # At 0.05 m resolution, 50.0 ⇒ crossing a 15 cm wall "costs" 7.5 m of
        # free-space travel.
        self.mppi_occupied_cell_cost = 50.0
        # Per-cell cost multiplier for traversing UNSEEN cells in the goal-
        # distance field, applied in EXPLOIT only (SEARCH keeps 1.0 so
        # exploration still enters unseen space). >1 makes the committed
        # approach prefer observed-free routes and take an unseen shortcut
        # only when the known route is that factor longer — unseen space may
        # hide a wall, and discovering one mid-shortcut forces a backtrack
        # that burns SPL. 1.0 disables.
        self.mppi_unseen_cell_cost = 3.0

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

        # Continuous→discrete action transformation. When `discrete_actions` is
        # True the agent no longer executes MPPI's continuous [v, w] command;
        # instead a low-level tracking controller converts each replan's plan
        # into ONE Habitat ObjectNav primitive — MOVE_FORWARD (`discrete_forward_m`),
        # TURN_LEFT/RIGHT (`discrete_turn_deg`), or STOP — so SR/SPL are directly
        # comparable to VLFM / Goal-Oriented Semantic Exploration baselines.
        # The controller looks `discrete_lookahead_m` ahead along the MPPI
        # optimized path, and turns toward that bearing whenever the heading
        # error exceeds half a turn (nearest-primitive rounding), else steps
        # forward. `max_agent_steps` is the per-episode primitive budget
        # (exhausting it without self-stopping = timeout = failure); turns and
        # forwards both count, matching the Habitat ObjectNav challenge.
        self.discrete_actions = False
        # 25 cm / 30° is the ObjectNav-challenge convention. These govern ONLY
        # the SEARCH tracking controller — DD-PPO's EXPLOIT stepping uses its
        # own checkpoint-coupled magnitudes (DDPPO_FORWARD_M / DDPPO_TURN_DEG
        # in src/planning/ddppo_policy.py, 25 cm / 10°).
        self.discrete_forward_m = 0.25
        self.discrete_turn_deg = 30.0
        self.discrete_lookahead_m = 0.5
        self.max_agent_steps = 500

        # EXPLOIT control: pretrained DD-PPO PointNav policy (depth-only,
        # SE-ResNeXt101 + 2-layer LSTM1024), replacing the A* + waypoint
        # controller — see src/planning/ddppo_policy.py and its use in
        # main.py's EXPLOIT branch.
        self.ddppo_checkpoint_path = "ddppo_weights/gibson-2plus-se-resneXt101-lstm1024.pth"

        # Visualization
        self.viz_interval = 500
        self.vmax_epi = 0.25

        # Detection-based termination. Once a usable-band detection recurs for
        # `detected_persistence` consecutive replans, latch into DETECTED mode
        # (IG off, goal pull saturated via goal_confidence=1.0). There is no
        # separate latch score threshold — the detector's own floor
        # (`llmdet_threshold`) bounds the score of every surviving box.
        # Terminate the run when the agent is within `stop_distance_m` of the
        # goal (obstacle-aware field distance).
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
        self.detected_max_box_frac = 0.97
        self.detected_min_dist_m = 0.05
        self.detected_max_dist_m = 3.0

        # Detector cadence once latched into EXPLOIT (`detected`) mode. The goal
        # is pinned to the cached box cell and committed hard there, so
        # re-running the detector every replan buys little. While detected, run
        # the detector only once every `exploit_redetect_interval` replans
        # (<=0 → never re-detect after latching; the cached goal is reused for
        # the rest of the run). A periodic value > 0 lets the goal refine as
        # the agent closes in and the box gets more accurate. SEARCH mode
        # always detects every replan.
        self.exploit_redetect_interval = 0

        # Neutral special-character string the LLMDet attention sinks encode for
        # the "special" init (see llmdet_sink_init below).
        self.sink_special_str = "[()]"
        # LLMDet detector with the paper's training-free attention sinks built
        # into its own vision-language fusion layers (the faithful Ruis et al.,
        # ICLR 2026 method). MUST use the `iSEE-Laboratory/llmdet_*` weights
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

        # The distractor vocabulary is the bank of COMPETING relevancy classes
        # for the pairwise CLIPSeg field (`clipseg_pairwise`, the chosen
        # verification method). It has two parts, assembled by
        # src.perception.distractor_gen.build_distractor_vocabulary:
        #
        #   1. `background_terms` — generic scene-background phrases (wall,
        #      floor, ceiling, window, ...). These are the negative/background
        #      prompts of the standard CLIP relevancy formulation: contrasting
        #      the query against a background bank is what turns a raw
        #      "target-like?" similarity into a relevancy score, and it absorbs
        #      the structurally-high background level that would otherwise tax
        #      every box's margin. Query-independent, always included.
        #   2. object confusers — objects a detector visually MISTAKES for the
        #      target that are structurally similar but DISTINCT-MATERIAL (an
        #      armchair/bench vs a chair, a trash can vs a potted plant, a
        #      framed picture vs a tv). These are the dominant FP mode and the
        #      whole point of a competing-class gate: a single-frame look-alike
        #      box fails the margin because the confuser channel out-scores the
        #      query. Same-material siblings (a bathtub/sink for a toilet) are
        #      deliberately EXCLUDED — a real target scores high on them too, so
        #      they collapse the margin and kill recall (empirically toilet SR
        #      82%->64%). The static `distractor_objects` list below (one list
        #      for every query; a per-target LLM-generated variant existed and
        #      was removed — see git history).
        #
        # Adding competitors lowers true-target margins, so
        # `field_verify_threshold` is calibrated against this vocabulary (see
        # detector_pairwise_field_maxj.yaml); a materially different bank
        # warrants a re-sweep.
        #
        # Phrases sharing a content word with the query are dropped per query
        # (semantics.filter_distractors) so the target never competes with
        # itself; synonyms are NOT caught (query "a sofa" keeps "a couch").
        # Keep the combined bank moderate — every phrase costs one CLIPSeg
        # decoder pass per frame (one extra channel in the batched forward).
        self.background_terms = [
            "a wall", "the floor", "the ceiling", "a window",
            "a door", "a picture",
        ]
        self.distractor_objects = [
            "a chair", "a couch", "a bed", "a potted plant", "a toilet",
            "a tv", "a table", "a cabinet", "a bench", "a trash can",
        ]
        # Pairwise-logit CLIPSeg field (`clipseg_pairwise`, the canonical
        # verification method): the field regresses one sigmoid CHANNEL PER
        # TERM ([query] + filtered `distractor_objects`; hash_feature_dim is
        # overridden to 1+K at PerceptionStack init). Contrast happens at
        # VERIFY time on the multi-view-converged channels: field-verify pools
        # the top-frac in-box cells selected by the QUERY channel, reads every
        # channel at those same cells, and scores the box by the worst-case
        # margin  presence_q - max_i presence_i  (margin-of-means; the earlier
        # contrastive sigmoid-x-softmax-share scalar target and the 2026-07
        # mean-of-margins scalar target were both removed — git history).
        # `field_score` IS that margin in this mode — it lives in [-1, 1], so
        # a log-only calibration gate needs field_verify_threshold <= -1.0,
        # not 0.0. `field_verify_presence_floor` is a separate "is anything
        # here" conjunct on the query channel (0.0 disables): margin answers
        # "more couch than bed?", the floor answers "couch-like at all?" —
        # kept separate so the two scales stay decoupled.
        self.clipseg_pairwise = False
        self.field_verify_presence_floor = 0.0

        # CLIPSeg model powering CLIPSegSemantics (the pairwise relevance
        # field's per-pixel supervision signal — see semantics.py).
        self.clipseg_model_name = "CIDAS/clipseg-rd64-refined"

        # Field verification of detections (main.py detect_classify_latch):
        # when a detector box fires, unproject its valid-depth pixels to 3D,
        # forward-pass the trained CLIPSeg relevance field there, pool the top
        # `field_verify_top_frac` fraction of point scores, and count the frame
        # as a detection only if the pooled score clears
        # `field_verify_threshold`. The field aggregates CLIPSeg evidence
        # across every observed viewpoint, so a single-frame look-alike that
        # fools the detector reads low until the map itself says "target here".
        # Cold-start caveat: an unobserved/untrained region reads ~0.02 (the
        # COLD_START_BIAS sigmoid shift), so the gate also delays latching
        # until the field has actually trained on the region — first sightings
        # of a true target are rejected until its CLIPSeg evidence lands in
        # the map (typically one buffer refresh + some train steps later).
        # Boxes with fewer than `field_verify_min_points` valid-depth pixels
        # can't be verified and are rejected. The pooled score is logged as
        # `field_score` in traj_log.jsonl for every verified frame.
        self.field_verify = False
        self.field_verify_threshold = 0.30
        # Pooling over the in-box point scores: "topk" = mean of the top
        # `field_verify_top_frac` fraction (robust to background in a loose
        # box, not a single-point statistic); "max" = the single highest point
        # score (most permissive — one well-corroborated point passes the box,
        # so thresholds must sit higher than for topk).
        self.field_verify_pool = "topk"
        self.field_verify_top_frac = 0.10
        self.field_verify_min_points = 20
        # Calibration aid: dump the exact rgb_cur frame (with the detector box
        # drawn) to <output_dir>/field_verify_frames/ whenever field_score is
        # computed, so a field_score sample logged in traj_log.jsonl can be
        # visually verified against the actual image instead of a proxy.
        # Only meaningful when the box wasn't reset by a real (>0) gate —
        # calibration runs use field_verify_threshold=0.0 so it never is.
        self.field_verify_save_frames = False

        # Load from YAML if path provided
        if yaml_path is not None:
            self.apply_yaml(yaml_path)

    def apply_yaml(self, yaml_path: str):
        """Override current values from a YAML file, driven by _YAML_SCHEMA.
        Callable repeatedly — later files overlay earlier ones (used by
        eval_scene.py --agent_config to stack an agent/sensor profile on the
        base config)."""
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f) or {}

        for section, keys in self._YAML_SCHEMA.items():
            sub = yaml_data.get(section) or {}
            renames = keys if isinstance(keys, dict) else {k: k for k in keys}
            for yaml_key, attr in renames.items():
                if yaml_key in sub:
                    setattr(self, attr, sub[yaml_key])
