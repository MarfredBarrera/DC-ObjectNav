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
        self.vmax_epi = 0.002
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
        self.hash_replay_buffer_size = 10
        self.hash_buffer_refresh_interval = 200
        self.hash_per_frame_cache_size = 8192
        self.hash_train_every_n_steps = 1
        self.hash_warmup_steps = 0

        # Ensemble
        self.ensemble_num_models = 3

        # Grid
        self.voxel_resolution = 0.05
        self.grid_max_height = 2.0

        # Planning
        self.mppi_dt = 0.5
        self.mppi_w_sign = -1.0  # flip if Habitat ω rotates opposite of MPPI's heading convention
        self.mppi_max_w_rps = 2.0  # rad/s clamp — prevents huge spins from sharp A* turns
        self.mppi_min_v_mps = 0.0
        self.mppi_max_v_mps = 1.0

        # MPPI exploration→exploitation schedule. progress=0 uses *_start,
        # progress=1 uses *_end, linearly interpolated.
        self.mppi_lambda_start = 1.0     # softmax temperature: high = flatter weights, wider sampling
        self.mppi_lambda_end = 1.0
        self.mppi_w_ig_start = 30.0      # information-gain reward: high = chase uncertainty
        self.mppi_w_ig_end = 30.0
        self.mppi_w_goal_start = 0.0     # goal-distance pull: low early = pure IG exploration
        self.mppi_w_goal_end = 0.0       # bump end value to pull rollouts toward goal late
        self.mppi_cov_scale_start = 4.0  # scalar on noise covariance: high = explore wider controls
        self.mppi_cov_scale_end = 4.0

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
        self.mppi_anneal_beta_traj = 1.0
        self.mppi_anneal_beta_action = 1.0

        # Visualization
        self.viz_interval = 1000
        
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
                'mppi_lambda_start', 'mppi_lambda_end',
                'mppi_w_ig_start', 'mppi_w_ig_end',
                'mppi_w_goal_start', 'mppi_w_goal_end',
                'mppi_cov_scale_start', 'mppi_cov_scale_end',
                'mppi_num_iters',
                'mppi_anneal_beta_traj', 'mppi_anneal_beta_action',
            ):
                if key in planning:
                    setattr(self, key, planning[key])