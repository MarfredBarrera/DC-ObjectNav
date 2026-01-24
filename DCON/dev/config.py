from dataclasses import dataclass

@dataclass
class Config:
    # Paths
    scene_dir: str = "/workspace/DCON/output/current_scene"
    output_name: str = "model.pt"
    
    # Training Settings
    iterations: int = 10000
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