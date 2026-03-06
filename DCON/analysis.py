import os
import json
import math
import time
import torch
import numpy as np
import imageio.v2 as imageio
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from collections import deque
import pickle

# Custom Imports
from dev.config import Config
from dev.gaussians import GaussianSplatting
from dev.semantics import SAM_CLIP_Semantics
from dev.utils import unprojection
from dev.hashgrid import HashGrid
from dev.recorder import BEVGrid

class Visualizer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        # Environment Setup
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cfg.gpu_indices
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"
        self.device = self.cfg.device

        # 1. Load Data
        print(f"Loading data from {self.cfg.output_dir}...")
        self.gt_images, self.gt_depths, self.c2ws, self.intrinsics_tuple = self._load_scene_data()
        self.fx, self.fy, self.cx, self.cy, self.H, self.W = self.intrinsics_tuple
        self.num_cameras = len(self.gt_images)

        # 2. Semantics
        self.sam_clip = SAM_CLIP_Semantics(self.cfg, device=self.device)

        # 3. Ensemble Models
        self.ensemble_models = self.load_ensemble()

        # 4. BEV Grid
        self.bev_grid = BEVGrid(cfg, self.ensemble_models)

    def _load_scene_data(self):
        json_path = os.path.join(self.cfg.output_dir, "transforms.json")
        with open(json_path, 'r') as f:
            meta = json.load(f)

        frames = meta['frames']
        img_0 = imageio.imread(os.path.join(self.cfg.output_dir, frames[0]['file_path']))
        H, W = img_0.shape[:2]
        
        fov_x = meta['camera_angle_x']
        fx = 0.5 * W / math.tan(0.5 * fov_x)
        fy = fx
        cx, cy = W / 2.0, H / 2.0

        gt_images, gt_depths, c2w_matrices = [], [], []

        # Habitat (OpenGL) -> GSplat (OpenCV)
        convert_mat = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])

        for frame in frames:
            # RGB
            rgb_path = os.path.join(self.cfg.output_dir, frame['file_path'])
            rgb = imageio.imread(rgb_path)
            gt_images.append(torch.from_numpy(rgb).float().to(self.device) / 255.0)

            # Depth
            depth_name = os.path.basename(frame['file_path']).replace("rgb", "depth").replace(".png", ".npy")
            depth_path = os.path.join(self.cfg.output_dir, "depth_data", depth_name)
            depth = np.load(depth_path)
            if depth.shape[:2] != (H, W):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            gt_depths.append(torch.from_numpy(depth).float().to(self.device))

            # Pose
            c2w_hab = np.array(frame['transform_matrix'])
            c2w_cv = c2w_hab @ convert_mat
            c2w_matrices.append(torch.from_numpy(c2w_cv).float().to(self.device))

        return torch.stack(gt_images), torch.stack(gt_depths), torch.stack(c2w_matrices), (fx, fy, cx, cy, H, W)
    

    def visualize_2d_similarity(self, img_index, text_query, save_path=None, 
                                vmin=0.6, overlay_alpha=0.6):
        """
        Visualize similarity scores as a 2D heatmap overlay on the RGB image.
        
        Args:
            img_index: Index of the camera/image to visualize
            text_query: Text query for semantic similarity (e.g., "a pillow")
            save_path: Optional path to save the figure
            overlay_alpha: Alpha value for heatmap overlay (0-1)
            
        Returns:
            similarity_map: (H, W) numpy array of similarity scores
        """
        # Get image and camera data
        rgb_image = self.gt_images[img_index].cpu().numpy()
        depth = self.gt_depths[img_index]
        c2w = self.c2ws[img_index]
        intrinsics = self.intrinsics_tuple
        
        # Get predicted features from HashGrid
        pred_features = self.hashgrid.get_hashgrid_features(depth, c2w, intrinsics)
        
        # Query similarity
        similarity_map = self.sam_clip.query(pred_features, text_query)
        similarity_np = similarity_map.cpu().numpy()
        
        # Handle invalid values
        similarity_np = np.nan_to_num(similarity_np, nan=0.5, posinf=1.0, neginf=0.0)
        similarity_np = np.clip(similarity_np, 0.0, 1.0)
        
        # Print statistics
        print(f"Similarity Map Statistics for '{text_query}':")
        print(f"  Range: [{similarity_np.min():.4f}, {similarity_np.max():.4f}]")
        print(f"  Mean: {similarity_np.mean():.4f}, Std: {similarity_np.std():.4f}")

        # scaling
        vis_data = similarity_np - similarity_np.min()
        vis_data = vis_data / (vis_data.max() + 1e-8)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        
        # Original image
        axes[0].imshow(rgb_image)
        axes[0].set_title("Original RGB Image", fontsize=14)
        axes[0].axis('off')
        
        # Heatmap overlay
        axes[1].imshow(rgb_image)
        heatmap = axes[1].imshow(vis_data, cmap='jet', alpha=overlay_alpha, vmin=vmin, vmax=1)
        axes[1].set_title(f"Similarity Overlay: '{text_query}'", fontsize=14)
        axes[1].axis('off')
        plt.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)
        
        axes[2].imshow(vis_data, cmap='jet', vmin=vmin, vmax=1)
        axes[2].set_title(f"Similarity Heatmap: '{text_query}'", fontsize=14)
        axes[2].axis('off')
        plt.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
        
        return similarity_np

    def load_ensemble(self):
        """Loads the 3 ensemble HashGrid models from the output directory."""
        ensemble_models = []
        ensemble_dir = os.path.join(self.cfg.output_dir, "ensemble")
        
        if not os.path.exists(ensemble_dir):
            print(f"Error: Ensemble directory not found at {ensemble_dir}")
            return

        print("Loading Ensemble Models...")
        for i in range(self.cfg.ensemble_num_models):
            model_path = os.path.join(ensemble_dir, f"hashgrid_ensemble_{i}.pt")
            if os.path.exists(model_path):
                # Initialize a new HashGrid instance
                model = HashGrid(self.cfg, device=self.device)
                model.load(model_path)
                ensemble_models.append(model)
                print(f"  -> Loaded Ensemble Model {i}")
            else:
                print(f"  -> Warning: Model {i} not found at {model_path}")

        return ensemble_models


    def get_ensemble_variance(self, img_index):
        """
        Computes the per-pixel semantic variance across the ensemble models.
        High variance indicates high uncertainty (model disagreement).
        """
        # Load models if not already loaded
        if not hasattr(self, 'ensemble_models') or not self.ensemble_models:
            self.load_ensemble()
            
        if not self.ensemble_models:
            return None

        # Get scene data
        depth = self.gt_depths[img_index]
        c2w = self.c2ws[img_index]
        intrinsics = self.intrinsics_tuple

        feature_stack = []
        var_stack = []

        # 1. Query all models
        with torch.no_grad():
            for model in self.ensemble_models:
                # features shape: (H, W, Feature_Dim)
                features, var = model.get_hashgrid_features(depth, c2w, intrinsics, return_uncertainty=True)
                feature_stack.append(features)
                var_stack.append(var)


        # Stack shape: (Num_Models, H, W, Feature_Dim)
        stack = torch.stack(feature_stack, dim=0) 
        # var_stack shape: (Num_Models, H, W)
        var_stack = torch.stack(var_stack, dim=0)

        # 2. Compute Epistemic Uncertainty: Var(ensemble_mean)
        # Calculate variance across the ensemble dimension (dim=0)
        # We then take the mean across the feature dimension (dim=-1) to get a scalar per pixel
        # variance_map shape: (H, W)
        epistemic_map = torch.var(stack, dim=0).mean(dim=-1)

        # 3. Compute Aleatoric Uncertainty: Mean(predicted_variance)
        # aleatoric_map shape: (H, W)
        aleatoric_map = torch.mean(var_stack, dim=0)

        # 3. Compute Mean Feature
        # mean map shape: (H,W,Feature_Dim)
        mean_map = torch.mean(stack, dim=0)

        return epistemic_map, aleatoric_map, mean_map


    def plot_similarity_and_uncertainty(self, img_index, text_query, save_path=None):
        """
        Plots both the similarity map and the uncertainty map side by side for analysis.
        """
        # Get image and camera data
        rgb_image = self.gt_images[img_index].cpu().numpy()
        depth = self.gt_depths[img_index]
        c2w = self.c2ws[img_index]
        intrinsics = self.intrinsics_tuple
        

        ## Mean ensemble plotting
        epistemic_map, aleatoric_map, pred_features_mean = self.get_ensemble_variance(img_index)

        # Query similarity
        similarity_map = self.sam_clip.query(pred_features_mean, text_query)
        similarity_np = similarity_map.cpu().numpy()
        
        # Handle invalid values
        similarity_np = np.nan_to_num(similarity_np, nan=0.5, posinf=1.0, neginf=0.0)
        similarity_np = np.clip(similarity_np, 0.0, 1.0)
        
        # Print statistics
        print(f"Similarity Map Statistics for '{text_query}':")
        print(f"  Range: [{similarity_np.min():.4f}, {similarity_np.max():.4f}]")
        print(f"  Mean: {similarity_np.mean():.4f}, Std: {similarity_np.std():.4f}")

        # scaling
        vis_data = similarity_np - similarity_np.min()
        vis_data = vis_data / (vis_data.max() + 1e-8)

        ## Epistemic Variance Plotting
        # Prepare for plotting
        epi_var_np = epistemic_map.cpu().numpy()
        ale_var_np = aleatoric_map.cpu().numpy()
        rgb_image = self.gt_images[img_index].cpu().numpy()
        
        # Handle outliers for better visualization contrast
        # We clip the top 2% of variance values to avoid hot pixels washing out the map
        v_min = epi_var_np.min()
        # v_max = np.percentile(epi_var_np, 98) 
        v_max = epi_var_np.max()
        # v_max = var_np.max()
        epi_var_np_clipped = np.clip(epi_var_np, v_min, v_max)
        ale_var_np_clipped = np.clip(ale_var_np, ale_var_np.min(), np.percentile(ale_var_np, 98))

        print(f"Uncertainty Stats | Min: {v_min:.6f} | Max: {v_max:.6f} | Mean: {epi_var_np.mean():.6f}")

        # Plotting
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # 1. RGB
        axes[0, 0].imshow(rgb_image)
        axes[0, 0].set_title(f"RGB Input (Frame {img_index})", fontsize=14)
        axes[0, 0].axis('off')

        # 2. Similarity Map (Mean Ensemble)
        im = axes[0, 1].imshow(vis_data, cmap='jet', vmin=0.6, vmax=1)
        axes[0, 1].set_title(rf"Similarity Map (Ensemble Mean $\mu_\varepsilon)$ for '{text_query}'", fontsize=14)
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # 2. Aleatoric Uncertainty
        im = axes[1, 1].imshow(ale_var_np_clipped, cmap='magma', vmin=ale_var_np_clipped.min(), vmax=ale_var_np_clipped.max())
        axes[1, 1].set_title(r"Aleatoric Uncertainty: $\mathbb{E}[\sigma^2_\theta]$", fontsize=14)
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

        # 3. Epistemic Uncertainty (Magma is good for 'intensity/heat')
        im = axes[1, 0].imshow(epi_var_np_clipped, cmap='magma', vmin=v_min, vmax=v_max)
        axes[1, 0].set_title(r"Epistemic Uncertainty: $\mathbb{V}[\mu_\theta]$", fontsize=14)
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)



        plt.tight_layout(pad=2.0)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved variance visualization to {save_path}")
            
        plt.show()

    def load_bev_maps(self, step):
        """Load the BEV uncertainty maps for a specific training step."""
        umaps_dir = os.path.join(self.cfg.output_dir, "umaps")
        epi_path = os.path.join(umaps_dir, f"bev_epistemic_uncertainty_{step}.npy")
        ale_path = os.path.join(umaps_dir, f"bev_aleatoric_uncertainty_{step}.npy")
        
        if os.path.exists(epi_path) and os.path.exists(ale_path):
            bev_epi_umap = np.load(epi_path)
            bev_ale_umap = np.load(ale_path)

            bev_epi_2d = bev_epi_umap.reshape(self.bev_grid.bev_height, self.bev_grid.bev_width)
            bev_ale_2d = bev_ale_umap.reshape(self.bev_grid.bev_height, self.bev_grid.bev_width)
            print(f"Loaded BEV maps for step {step} from {umaps_dir}")
            return bev_epi_2d, bev_ale_2d
        else:
            print(f"BEV maps for step {step} not found in {umaps_dir}")
            return None, None
        
    def visualize_bev_map(self, u_maps):
        """Visualize the BEV uncertainty maps."""
        bev_epi_2d, bev_ale_2d = u_maps
        
        if bev_epi_2d is not None and bev_ale_2d is not None:
            # Reshape from flattened (N,) to 2D (bev_height, bev_width)
            # bev_epi_2d = bev_epi_umap.reshape(self.bev_grid.bev_height, self.bev_grid.bev_width)
            # bev_ale_2d = bev_ale_umap.reshape(self.bev_grid.bev_height, self.bev_grid.bev_width)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Epistemic Uncertainty Map
            im1 = axes[0].imshow(bev_epi_2d, cmap='magma', origin='lower', aspect='equal')
            axes[0].set_title(r"Epistemic Uncertainty: $\mathbb{V}[\mu_\theta]$", fontsize=10)
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

            # Add statistics text
            epi_stats_text = (
                f'Min: {bev_epi_2d.min():.6f}\n'
                f'Max: {bev_epi_2d.max():.6f}\n'
                f'Mean: {bev_epi_2d.mean():.6f}\n'
                f'Std: {bev_epi_2d.std():.6f}'
            )
            axes[0].text(
                0.01, 1.05, epi_stats_text,
                transform=axes[0].transAxes,
                fontsize=10,
                verticalalignment='bottom',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

            im2 = axes[1].imshow(bev_ale_2d, cmap='magma', origin='lower', aspect='equal')
            axes[1].set_title(r"Aleatoric Uncertainty: $\mathbb{E}[\sigma^2_\theta]$", fontsize=10)
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

            # Add statistics text
            ale_stats_text = (
                f'Min: {bev_ale_2d.min():.6f}\n'
                f'Max: {bev_ale_2d.max():.6f}\n'
                f'Mean: {bev_ale_2d.mean():.6f}\n'
                f'Std: {bev_ale_2d.std():.6f}'
            )
            axes[1].text(
                0.01, 1.05, ale_stats_text,
                transform=axes[1].transAxes,
                fontsize=10,
                verticalalignment='bottom',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

            plt.tight_layout()
            plt.show()
        else:
            print("Unable to visualize BEV maps due to missing data.")


if __name__ == "__main__":
    config = Config("./config/config.yaml")
    
    # Example 1: Analyze 2D similarity and uncertainty from a single viewpoint
    visualizer = Visualizer(config)

    bev_maps = visualizer.load_bev_maps(step=50000)
    visualizer.visualize_bev_map(bev_maps)
    # visualizer.bev_grid.forward_pass(height_filter=(0.1, 2.0))
    # visualizer.bev_grid.visualize_bev_map(show=True)
    