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
        # Set all default values first
        # Paths
        self.output_dir = "/workspace/DCON/output/current_scene"
        self.output_name = "model.pt"
        self.SAM_checkpoint_path = "/workspace/DCON/SAM_models/sam_vit_b_01ec64.pth"

        # Habitat Settings
        self.scene_path = "/workspace/DCON/gibson_scenes/Anaheim.glb"
        self.img_width = 720
        self.img_height = 720
        self.fov = 90
        self.data_queue_size = 30
        self.training_queue_size = 10
        
        # Semantics Settings
        self.SAM_model_type = "vit_b"
        self.CLIP_model_name = "openai/clip-vit-base-patch32"
        self.CLIP_label_batch_size = 64
        self.points_per_side = 32
        self.pred_iou_thresh = 0.86
        self.stability_score_thresh = 0.92
        self.crop_n_layers = 1
        self.crop_n_points_downscale_factor = 2
        self.min_mask_region_area = 100
        
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
        self.hash_train_every_n_steps = 1
        self.hash_warmup_steps = 0

        # Ensemble
        self.ensemble_num_models = 3
        
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
            if 'sam_checkpoint' in paths:
                self.SAM_checkpoint_path = paths['sam_checkpoint']
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
            if 'training_queue_size' in habitat:
                self.training_queue_size = habitat['training_queue_size']
        
        # Semantics
        if 'semantics' in yaml_data:
            sem = yaml_data['semantics']
            if 'sam_model_type' in sem:
                self.SAM_model_type = sem['sam_model_type']
            if 'clip_model_name' in sem:
                self.CLIP_model_name = sem['clip_model_name']
            if 'clip_label_batch_size' in sem:
                self.CLIP_label_batch_size = sem['clip_label_batch_size']
            if 'points_per_side' in sem:
                self.points_per_side = sem['points_per_side']
            if 'pred_iou_thresh' in sem:
                self.pred_iou_thresh = sem['pred_iou_thresh']
            if 'stability_score_thresh' in sem:
                self.stability_score_thresh = sem['stability_score_thresh']
            if 'crop_n_layers' in sem:
                self.crop_n_layers = sem['crop_n_layers']
            if 'crop_n_points_downscale_factor' in sem:
                self.crop_n_points_downscale_factor = sem['crop_n_points_downscale_factor']
            if 'min_mask_region_area' in sem:
                self.min_mask_region_area = sem['min_mask_region_area']
        
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
            if 'train_every_n_steps' in hg:
                self.hash_train_every_n_steps = hg['train_every_n_steps']
            if 'warmup_steps' in hg:
                self.hash_warmup_steps = hg['warmup_steps']
        
        # Ensemble
        if 'ensemble' in yaml_data:
            ensemble = yaml_data['ensemble']
            if 'num_models' in ensemble:
                self.ensemble_num_models = ensemble['num_models']