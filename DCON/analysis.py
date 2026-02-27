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
from collections import deque
import pickle

# Custom Imports
from dev.config import Config
from dev.gaussians import GaussianSplatting
from dev.semantics import SAM_CLIP_Semantics
from dev.utils import unprojection
from dev.hashgrid import HashGrid
from dev.recorder import BEVGrid


class Runner:
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
        self.bev_grid = BEVGrid(self.cfg, self.ensemble_models)

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

    def visualize_ensemble_variance(self, img_index, save_path=None, overlay_alpha=0.6):
        """
        Visualizes the uncertainty (variance) of the semantic field.
        """
        epistemic_map, aleatoric_map, mean_map = self.get_ensemble_variance(img_index)
        
        if epistemic_map is None:
            print("Could not compute variance (ensemble not loaded).")
            return

        # Prepare for plotting
        var_np = epistemic_map.cpu().numpy()
        rgb_image = self.gt_images[img_index].cpu().numpy()
        
        # Handle outliers for better visualization contrast
        # We clip the top 2% of variance values to avoid hot pixels washing out the map
        v_min = var_np.min()
        v_max = np.percentile(var_np, 98) 
        var_np_clipped = np.clip(var_np, v_min, v_max)

        # Normalize to 0-1 for the overlay
        var_norm = (var_np_clipped - v_min) / (v_max - v_min + 1e-8)

        print(f"Uncertainty Stats | Min: {v_min:.6f} | Max: {v_max:.6f} | Mean: {var_np.mean():.6f}")

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))

        # 1. RGB
        axes[0].imshow(rgb_image)
        axes[0].set_title(f"RGB Input (Frame {img_index})", fontsize=14)
        axes[0].axis('off')

        # 2. Heatmap (Magma is good for 'intensity/heat')
        im = axes[1].imshow(var_np_clipped, cmap='magma', vmin=v_min, vmax=v_max)
        axes[1].set_title("Ensemble Variance (Uncertainty)", fontsize=14)
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # 3. Overlay
        axes[2].imshow(rgb_image)
        axes[2].imshow(var_norm, cmap='magma', alpha=overlay_alpha)
        axes[2].set_title("Uncertainty Overlay", fontsize=14)
        axes[2].axis('off')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved variance visualization to {save_path}")
            
        plt.show()

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


def load_bev_uncertainty_snapshots(pickle_path):
    """
    Load BEV uncertainty snapshots from pickle file.
    
    Args:
        pickle_path: Path to the recorder_snapshots.pkl file
        
    Returns:
        snapshots: List of snapshot dictionaries
    """
    with open(pickle_path, 'rb') as f:
        snapshots = pickle.load(f)
    print(f"Loaded {len(snapshots)} snapshots from {pickle_path}")
    return snapshots


def visualize_bev_uncertainty(pickle_path, snapshot_idx=None, height_filter=None, 
                               save_path=None, cmap='magma', vmin=None, vmax=None,
                               show_trajectory=True, config=None):
    """
    Visualize BEV uncertainty map from recorded snapshots with optional filtering.
    
    Args:
        pickle_path: Path to the recorder_snapshots.pkl file
        snapshot_idx: Index of snapshot to visualize (None for last snapshot)
        height_filter: Tuple (min_height, max_height) to filter by agent height/z-position
                      If None, no height filtering is applied
        save_path: Optional path to save the figure
        cmap: Colormap for uncertainty visualization (default: 'magma')
        vmin, vmax: Optional colorbar limits
        show_trajectory: If True, overlay agent trajectory on the map
        config: Config object to load scene bounds (optional)
        
    Returns:
        uncertainty_map: (H, W) numpy array of the uncertainty map
    """
    # Load snapshots
    snapshots = load_bev_uncertainty_snapshots(pickle_path)
    
    if len(snapshots) == 0:
        print("No snapshots found!")
        return None
    
    # Load scene bounds if config provided
    bev_resolution = 0.01  # Default
    bev_min_x, bev_max_x, bev_min_z, bev_max_z = None, None, None, None
    
    if config is not None:
        bev_resolution = config.bev_resolution
        transforms_path = os.path.join(config.output_dir, "transforms.json")
        if os.path.exists(transforms_path):
            with open(transforms_path, 'r') as f:
                transforms = json.load(f)
            if 'scene_bounds' in transforms:
                bounds = transforms['scene_bounds']
                bev_min_x, bev_max_x = bounds['min'][0], bounds['max'][0]
                bev_min_z, bev_max_z = bounds['min'][2], bounds['max'][2]
    
    # Select snapshot
    if snapshot_idx is None:
        snapshot_idx = -1  # Last snapshot by default
    
    snapshot = snapshots[snapshot_idx]
    uncertainty_map = snapshot['uncertainty_map']
    count_map = snapshot['count_map']
    step = snapshot['step']
    agent_mat = snapshot['agent_mat']
    
    print(f"\nVisualizing snapshot at step {step}")
    print(f"  Cells with data: {(count_map > 0).sum()} / {count_map.size}")
    
    # Apply height filter if specified
    if height_filter is not None:
        min_h, max_h = height_filter
        # Filter based on agent/camera z-position across snapshots
        filtered_uncertainty = np.zeros_like(uncertainty_map)
        filtered_counts = np.zeros_like(count_map)
        
        for snap in snapshots:
            snap_agent_pos = snap['agent_mat']
            # agent_mat is a 4x4 transformation matrix stored as nested list
            # Position is in the last column: [x, y, z] = [mat[0][3], mat[1][3], mat[2][3]]
            # Extract height (y-coordinate) for filtering
            if isinstance(snap_agent_pos, list) and len(snap_agent_pos) == 4:
                # It's a 4x4 matrix - extract y position (height) from [1][3]
                agent_height = snap_agent_pos[1][3]
            elif isinstance(snap_agent_pos, list) and len(snap_agent_pos) >= 3:
                # It's a simple [x, y, z] list
                agent_height = snap_agent_pos[1]
            else:
                agent_height = 0.0  # Default if format unknown
            
            if min_h <= agent_height <= max_h:
                filtered_uncertainty += snap['uncertainty_map'] * (snap['count_map'] > 0)
                filtered_counts += (snap['count_map'] > 0).astype(int)
        
        # Average the filtered data
        mask = filtered_counts > 0
        uncertainty_map = np.where(mask, filtered_uncertainty / np.maximum(filtered_counts, 1), 0)
        count_map = filtered_counts
        print(f"Applied height filter [{min_h}, {max_h}] m")
    
    # Mask out unexplored areas (where count_map == 0)
    masked_uncertainty = np.ma.masked_where(count_map == 0, uncertainty_map)
    
    # Auto-scale vmin/vmax if not provided
    if vmin is None:
        vmin = np.percentile(uncertainty_map[count_map > 0], 2) if (count_map > 0).any() else 0
    if vmax is None:
        vmax = np.percentile(uncertainty_map[count_map > 0], 98) if (count_map > 0).any() else uncertainty_map.max()
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Set up extent for world coordinates if bounds are available
    extent = None
    if bev_min_x is not None:
        extent = [bev_min_x, bev_max_x, bev_min_z, bev_max_z]
    
    # Plot 1: Uncertainty Map
    im1 = axes[0].imshow(masked_uncertainty, cmap=cmap, vmin=vmin, vmax=vmax, 
                         origin='lower', interpolation='nearest', extent=extent)
    axes[0].set_title(f'BEV Uncertainty Map (Step {step})', fontsize=14, fontweight='bold')
    if extent:
        axes[0].set_xlabel('X World Coordinate (m)', fontsize=12)
        axes[0].set_ylabel('Z World Coordinate (m)', fontsize=12)
        axes[0].axhline(0, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
        axes[0].axvline(0, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
    else:
        axes[0].set_xlabel('X Grid Cell', fontsize=12)
        axes[0].set_ylabel('Z Grid Cell', fontsize=12)
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='Uncertainty')
    
    # Overlay trajectory if requested
    if show_trajectory and bev_min_x is not None:
        # Extract agent positions from snapshots and plot them
        agent_positions = []
        for snap in snapshots:
            agent_mat = snap['agent_mat']
            # Handle both 4x4 matrix (nested list) and [x,y,z] vector (flat list)
            if isinstance(agent_mat, list) and len(agent_mat) == 4 and isinstance(agent_mat[0], list):
                # Extract position from 4x4 matrix
                x, z = agent_mat[0][3], agent_mat[2][3]
                agent_positions.append([x, z])
            elif isinstance(agent_mat, (list, np.ndarray)) and len(agent_mat) >= 3:
                # Extract position from [x, y, z] vector
                x, z = agent_mat[0], agent_mat[2]
                agent_positions.append([x, z])
        
        if agent_positions:
            agent_positions = np.array(agent_positions)
            # Plot trajectory on both maps
            for ax in axes:
                ax.plot(agent_positions[:, 0], agent_positions[:, 1], 'c-', 
                       linewidth=2, alpha=0.7, label='Trajectory')
                ax.scatter(agent_positions[-1, 0], agent_positions[-1, 1], 
                          c='cyan', s=150, marker='*', label='Current Position', 
                          edgecolors='white', linewidths=2, zorder=10)
                ax.scatter(agent_positions[0, 0], agent_positions[0, 1], 
                          c='yellow', s=100, marker='o', label='Start Position', 
                          edgecolors='white', linewidths=2, zorder=10)
            axes[0].legend(loc='upper right', fontsize=10)
    
    # Plot 2: Coverage/Count Map
    im2 = axes[1].imshow(count_map, cmap='viridis', origin='lower', interpolation='nearest', extent=extent)
    axes[1].set_title(f'BEV Coverage Map (Observation Counts)', fontsize=14, fontweight='bold')
    if extent:
        axes[1].set_xlabel('X World Coordinate (m)', fontsize=12)
        axes[1].set_ylabel('Z World Coordinate (m)', fontsize=12)
        axes[1].axhline(0, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
        axes[1].axvline(0, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
    else:
        axes[1].set_xlabel('X Grid Cell', fontsize=12)
        axes[1].set_ylabel('Z Grid Cell', fontsize=12)
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Visit Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    return uncertainty_map


if __name__ == "__main__":
    config = Config("./config/config.yaml")
    
    # Example 1: Analyze 2D similarity and uncertainty from a single viewpoint
    runner = Runner(config)
    runner.plot_similarity_and_uncertainty(
        img_index=13, 
        text_query="a pillow", 
        save_path="output/current_scene/similarity_uncertainty.png"
    )
    
    # # Example 2: Visualize BEV uncertainty map from recorded snapshots
    # pickle_path = os.path.join(config.output_dir, "recorder_snapshots.pkl")
    # visualize_bev_uncertainty(
    #     pickle_path=pickle_path,
    #     snapshot_idx=-1,
    #     config=config,
    #     height_filter=(0.0,1.75),
    #     save_path="output/current_scene/bev_uncertainty.png",
    #     show_trajectory=False
    # )