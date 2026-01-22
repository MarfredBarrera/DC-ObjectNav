import os
import json
import math
import time
import torch
import numpy as np
import imageio.v2 as imageio
import cv2
import torch.nn.functional as F
import tinycudann as tcnn

import semantics

from dataclasses import dataclass, field
from typing import Dict, Tuple, List

from torchmetrics.image import StructuralSimilarityIndexMeasure
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy


# -----------------------------------------------------------------------------
# Utility Functions (Math & SH)
# -----------------------------------------------------------------------------
def rgb_to_sh(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def eval_sh(coeffs, dirs):
    """
    Evaluates spherical harmonics for a batch of directions.
    """
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    # Degree 0 (Constant)
    C0 = 0.28209479177387814
    result = C0 * coeffs[..., 0]
    
    if coeffs.shape[1] > 1:
        # Degree 1
        C1 = 0.4886025119029199
        result += C1 * (-y * coeffs[..., 1] + z * coeffs[..., 2] - x * coeffs[..., 3])
    return result

def unprojection(depth, intrinsics, device):    
    fx, fy, cx, cy, H, W = intrinsics
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    z = depth
    x_c = (x - cx) * z / fx
    y_c = (y - cy) * z / fy
    return x_c, y_c, z


# -----------------------------------------------------------------------------
# 1. Config Class
# -----------------------------------------------------------------------------
@dataclass
class Config:
    # Paths
    scene_dir: str = "/workspace/DCON/output/current_scene"
    output_name: str = "model.pt"
    
    # Training Settings
    iterations: int = 10000
    device: str = "cuda"
    gpu_indices: str = "4,5"
    
    # Camera / Data
    near_plane: float = 0.01
    
    # Hyperparameters
    ssim_weight: float = 0.2
    l1_weight: float = 0.8
    scale_reg: float = 0.01
    uncertainty_weight: float = 0.1
    uncertainty_dim: int = 16  # Degree 3 (16 coeffs) or Degree 1 (4 coeffs)
    
    # Learning Rates
    lr_means: float = 1.6e-4
    lr_scales: float = 0.005
    lr_quats: float = 0.001
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20  # Target LR after warmup
    lr_uncertainty: float = 1e-3

    # Strategy Settings
    refine_start_iter: int = 100
    refine_stop_iter: int = 10000 - 500
    refine_every: int = 100
    reset_every: int = 1000
    grow_grad2d: float = 0.0002
    prune_opa: float = 0.005

    # include trained uncertainty
    train_uncertainty: bool = True

# -----------------------------------------------------------------------------
# 2. Runner Class (Engine)
# -----------------------------------------------------------------------------
class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        # Set Environment
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cfg.gpu_indices
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"
        
        self.device = self.cfg.device
        self.splats = torch.nn.ParameterDict()
        self.optimizers = {}
        self.strategy = None
        self.strategy_state = None
        
        # Load Data
        print(f"Loading data from {self.cfg.scene_dir}...")
        self.gt_images, self.gt_depths, self.c2ws, self.intrinsics = self._load_scene_data()
        self.num_cameras = len(self.gt_images)
        self.fx, self.fy, self.cx, self.cy, self.H, self.W = self.intrinsics

        self.semantic_extractor = semantics.SemanticFeatureExtractor(device=self.device)
        
        # Initialize Geometry
        self._init_parameters()
        self._setup_optimizers_and_strategy()
        
        # Metrics
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.K = torch.tensor([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], device=self.device)

    def _load_scene_data(self):
        json_path = os.path.join(self.cfg.scene_dir, "transforms.json")
        with open(json_path, 'r') as f:
            meta = json.load(f)

        frames = meta['frames']
        img_0 = imageio.imread(os.path.join(self.cfg.scene_dir, frames[0]['file_path']))
        H, W = img_0.shape[:2]
        
        fov_x = meta['camera_angle_x']
        fx = 0.5 * W / math.tan(0.5 * fov_x)
        fy = fx
        cx, cy = W / 2.0, H / 2.0

        gt_images = []
        gt_depths = []
        c2w_matrices = []

        # Coordinate Conversion: OpenGL (Habitat) -> OpenCV (GSplat)
        convert_mat = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ])

        print(f"Processing {len(frames)} frames...")
        for frame in frames:
            # Load RGB
            rgb_path = os.path.join(self.cfg.scene_dir, frame['file_path'])
            rgb = imageio.imread(rgb_path)
            gt_images.append(torch.from_numpy(rgb).float().to(self.device) / 255.0)

            # Load Depth
            depth_name = os.path.basename(frame['file_path']).replace("rgb", "depth").replace(".png", ".npy")
            depth_path = os.path.join(self.cfg.scene_dir, "depth_data", depth_name)
            depth = np.load(depth_path)
            if depth.shape[:2] != (H, W):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            gt_depths.append(torch.from_numpy(depth).float().to(self.device))

            # Convert Pose
            c2w_hab = np.array(frame['transform_matrix'])
            c2w_cv = c2w_hab @ convert_mat
            c2w_matrices.append(torch.from_numpy(c2w_cv).float().to(self.device))

        return torch.stack(gt_images), torch.stack(gt_depths), torch.stack(c2w_matrices), (fx, fy, cx, cy, H, W)

    def _init_parameters(self):
        # Init Point Cloud from Depth
        points_list = []
        colors_list = []
        num_init_frames = 10
        indices = np.linspace(0, len(self.gt_images)-1, num_init_frames, dtype=int)
        
        print("Initializing point cloud from depth...")
        for idx in indices:
            depth = self.gt_depths[idx]
            color = self.gt_images[idx]
            c2w = self.c2ws[idx]
            mask = (depth > 0.1) & (depth < 10.0)

            x_c, y_c, z_c = unprojection(depth, self.intrinsics, self.device)
            x_c, y_c, z_c = x_c[mask], y_c[mask], z_c[mask]

            cam_points = torch.stack([x_c, y_c, z_c, torch.ones_like(z_c)], dim=1)
            world_points = (c2w @ cam_points.T).T
            
            points_list.append(world_points[:, :3])
            colors_list.append(color[mask])

        full_points = torch.cat(points_list, dim=0)
        full_colors = torch.cat(colors_list, dim=0)
        
        # Downsample if too large
        if full_points.shape[0] > 100_000:
            indices = torch.randperm(full_points.shape[0])[:100_000]
            full_points = full_points[indices]
            full_colors = full_colors[indices]

        N = len(full_points)
        print(f"Initialized {N} points.")

        # Create Parameters
        params = {
            "means": torch.nn.Parameter(full_points.contiguous()),
            "scales": torch.nn.Parameter(torch.ones(N, 3, device=self.device).contiguous() * -2.5),
            "quats": torch.nn.Parameter(torch.zeros(N, 4, device=self.device).contiguous()),
            "opacities": torch.nn.Parameter(torch.ones(N, device=self.device).contiguous() * 0.5),
            "sh0": torch.nn.Parameter(rgb_to_sh(full_colors).unsqueeze(1).contiguous()), 
            "shN": torch.nn.Parameter(torch.zeros(N, 15, 3, device=self.device).contiguous()),
        }
        
        if self.cfg.train_uncertainty:
            params["uncertainty_sh"] = torch.nn.Parameter(torch.zeros(N, self.cfg.uncertainty_dim, device=self.device).contiguous())
        
        with torch.no_grad():
            params["quats"][:, 0] = 1.0 

        self.splats = torch.nn.ParameterDict(params).to(self.device)

    def _setup_optimizers_and_strategy(self):
        self.optimizers = {
            "means": torch.optim.Adam([{"params": self.splats["means"], "lr": self.cfg.lr_means, "name": "means"}], eps=1e-15),
            "scales": torch.optim.Adam([{"params": self.splats["scales"], "lr": self.cfg.lr_scales, "name": "scales"}], eps=1e-15),
            "quats": torch.optim.Adam([{"params": self.splats["quats"], "lr": self.cfg.lr_quats, "name": "quats"}], eps=1e-15),
            "opacities": torch.optim.Adam([{"params": self.splats["opacities"], "lr": self.cfg.lr_opacities, "name": "opacities"}], eps=1e-15),
            "sh0": torch.optim.Adam([{"params": self.splats["sh0"], "lr": self.cfg.lr_sh0, "name": "sh0"}], eps=1e-15),
            "shN": torch.optim.Adam([{"params": self.splats["shN"], "lr": 0, "name": "shN"}], eps=1e-15),
        }
        
        if self.cfg.train_uncertainty:
            self.optimizers["uncertainty_sh"] = torch.optim.Adam([{"params": self.splats["uncertainty_sh"], "lr": self.cfg.lr_uncertainty, "name": "uncertainty_sh"}], eps=1e-15)

        self.strategy = DefaultStrategy(
            verbose=True,
            refine_start_iter=self.cfg.refine_start_iter,
            refine_stop_iter=self.cfg.refine_stop_iter,
            refine_every=self.cfg.refine_every,
            reset_every=self.cfg.reset_every,
            grow_grad2d=self.cfg.grow_grad2d,
            prune_opa=self.cfg.prune_opa
        )
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state()
        self.scheduler_means = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizers["means"], gamma=0.01 ** (1.0 / self.cfg.iterations)
        )

    def _compute_uncertainty_loss(self, info, c2w):
        # Identify visible gaussians
        visible_mask = (info["radii"] > 0).squeeze(0)
        
        if not visible_mask.any():
            return 0.0

        u_coeffs = self.splats["uncertainty_sh"][visible_mask]
        vis_means = self.splats["means"][visible_mask]
        
        # Camera center
        cam_center = c2w[:3, 3] 
        dir_front = vis_means - cam_center
        dir_front = F.normalize(dir_front, dim=-1)
        
        # Forward view -> Certain (0)
        u_front = torch.sigmoid(eval_sh(u_coeffs, dir_front))
        
        # Backward view -> Uncertain (1)
        u_back = torch.sigmoid(eval_sh(u_coeffs, -dir_front))
        
        # Loss
        lambda_u = 0.5
        return (1 - lambda_u) * u_front.mean() + lambda_u * (1.0 - u_back).mean()

    def train(self):
        print(f"Starting training for {self.cfg.iterations} iterations...")
        start_time = time.time()

        for step in range(self.cfg.iterations):
            # Select Camera
            cam_idx = torch.randint(0, self.num_cameras, (1,)).item()
            gt_image = self.gt_images[cam_idx]
            c2w = self.c2ws[cam_idx]
            w2c = torch.inverse(c2w)

            # Warmup shN learning rate
            if step == 1000:
                for param_group in self.optimizers["shN"].param_groups:
                    param_group['lr'] = self.cfg.lr_shN

            # Rasterize
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)
            sh_degree = min(step // 1000, 3)
            
            renders, alphas, info = rasterization(
                means=self.splats["means"],
                quats=self.splats["quats"] / self.splats["quats"].norm(dim=-1, keepdim=True),
                scales=torch.exp(self.splats["scales"]),
                opacities=torch.sigmoid(self.splats["opacities"]),
                colors=colors,
                viewmats=w2c[None, ...],
                Ks=self.K[None, ...],
                width=self.W,
                height=self.H,
                sh_degree=sh_degree,
                packed=False,
                near_plane=self.cfg.near_plane
            )
            
            rgb_render = renders[0]

            self.strategy.step_pre_backward(params=self.splats, optimizers=self.optimizers, state=self.strategy_state, step=step, info=info)

            # Standard Loss
            l1loss = F.l1_loss(rgb_render, gt_image)
            ssimloss = 1.0 - self.ssim_metric(rgb_render.permute(2, 0, 1).unsqueeze(0), gt_image.permute(2, 0, 1).unsqueeze(0))
            loss = l1loss * self.cfg.l1_weight + ssimloss * self.cfg.ssim_weight

            # Uncertainty Loss
            if self.cfg.train_uncertainty:
                loss_uncertainty = self._compute_uncertainty_loss(info, c2w)
                loss += self.cfg.uncertainty_weight * loss_uncertainty
            
            # Scale Reg
            loss += self.cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

            loss.backward()
            
            for opt in self.optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
                
            self.strategy.step_post_backward(params=self.splats, optimizers=self.optimizers, state=self.strategy_state, step=step, info=info)
            self.scheduler_means.step()

            if step % 100 == 0:
                num_gs = len(self.splats["means"])
                print(f"Step {step:04d} | GS: {num_gs} | Loss: {loss.item():.5f} | Time: {time.time()-start_time:.1f}s")

    def save_model(self):
        print("Saving model...")
        save_dict = {k: v.detach() for k, v in self.splats.items()}
        torch.save({"splats": save_dict}, os.path.join(self.cfg.scene_dir, self.cfg.output_name))
        print(f"Model saved to {os.path.join(self.cfg.scene_dir, self.cfg.output_name)}")

    def test_semantics(self, img_index):
        color_image = self.gt_images[img_index]
        depth_image = self.gt_depths[img_index]

        c2w = self.c2ws[img_index]

        points, features = self.semantic_extractor.unproject_and_label(color_image, depth_image, self.intrinsics, c2w)
        return points, features



# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Create Configuration
    config = Config(
        scene_dir="/workspace/DCON/output/current_scene",
        iterations=10000,
        gpu_indices="4,5"
    )

    # Initialize Runner
    runner = Runner(config)
    
    # Run Training
    # runner.train()

    points, features = runner.test_semantics(img_index=20)

    print(f"Generated labeled point cloud:")
    print(f"Points Shape: {points.shape}")      # (N, 3)
    print(f"Features Shape: {features.shape}")  # (N, 768) for ViT-B/16

    
    # Save Output
    runner.save_model()