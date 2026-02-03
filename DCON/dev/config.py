from dataclasses import dataclass

@dataclass
class Config:
    # Paths
    scene_dir: str = "/workspace/DCON/output/current_scene"
    output_name: str = "model.pt"
    SAM_checkpoint_path: str = "/workspace/DCON/SAM_models/sam_vit_b_01ec64.pth"

    # Semantics Settings
    SAM_model_type = "vit_b"
    CLIP_model_name = "openai/clip-vit-base-patch32"
    CLIP_label_batch_size: int = 64

    points_per_side: int = 32
    pred_iou_thresh: float = 0.86
    stability_score_thresh: float = 0.92
    crop_n_layers: int = 1
    crop_n_points_downscale_factor: int = 2
    min_mask_region_area: int = 100  # Filter very small noise
    
    # Training Settings
    iterations: int = 1000
    device: str = "cuda"
    gpu_indices: str = "1,2"
    
    # Camera / Data
    near_plane: float = 0.01
    
    # Hyperparameters
    ssim_weight: float = 0.2
    l1_weight: float = 0.8
    scale_reg: float = 0.01
    uncertainty_weight: float = 0.1
    uncertainty_dim: int = 16 
    
    # Learning Rates
    lr_means: float = 1.6e-4
    lr_scales: float = 0.005
    lr_quats: float = 0.001
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20  
    lr_uncertainty: float = 1e-3

    # Strategy Settings
    refine_start_iter: int = 100
    refine_stop_iter: int = 10000 - 500
    refine_every: int = 100
    reset_every: int = 1000
    grow_grad2d: float = 0.0002
    prune_opa: float = 0.005

    # Toggles
    train_uncertainty: bool = True

    # HashGrid Configuration
    hash_n_levels: int = 16              # Number of resolution levels
    hash_n_features_per_level: int = 2   # Features per level (2-8 typical)
    hash_log2_hashmap_size: int = 19     # Hash table size (2^19 entries)
    hash_base_resolution: int = 16       # Coarsest resolution
    hash_per_level_scale: float = 1.5    # Growth factor between levels
    
    # HashGrid MLP
    hash_n_neurons: int = 64             # Hidden layer width
    hash_n_hidden_layers: int = 2        # Number of hidden layers
    hash_feature_dim: int = 512          # Output feature dimension (match CLIP)
    
    # HashGrid Training
    hash_lr: float = 1e-2                  # Learning rate
    hash_train_batch_size: int = 8192      # Points per training step
    hash_inference_batch_size: int = 16384 # Points per inference batch
    
    # HashGrid Training Schedule
    hash_train_every_n_steps: int = 10   # Train HashGrid every N steps
    hash_warmup_steps: int = 0        # Steps before starting HashGrid training


