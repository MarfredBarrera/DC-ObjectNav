import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
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
        self.bev_grid = BEVGrid(cfg, ensemble=self.ensemble_models)

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

    def load_bev_maps(self, step):
        """Load the BEV uncertainty maps for a specific training step."""
        umaps_dir = os.path.join(self.cfg.output_dir, "umaps")
        epi_path = os.path.join(umaps_dir, f"bev_epistemic_uncertainty_{step}.npy")
        ale_path = os.path.join(umaps_dir, f"bev_aleatoric_uncertainty_{step}.npy")
        
        if os.path.exists(epi_path) and os.path.exists(ale_path):
            bev_epi_umap = np.load(epi_path)
            bev_ale_umap = np.load(ale_path)


            # print(f"Loaded BEV maps for step {step} from {umaps_dir}")
            return bev_epi_umap, bev_ale_umap
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
            extent = [self.bev_grid.bev_min_x, self.bev_grid.bev_max_x, self.bev_grid.bev_min_z, self.bev_grid.bev_max_z]

            # Epistemic Uncertainty Map
            im1 = axes[0].imshow(bev_epi_2d, cmap='magma', origin='lower', aspect='equal', extent=extent)
            axes[0].set_xlabel('X Position (m)', fontsize=12)
            axes[0].set_ylabel('Z Position (m)', fontsize=12)
            axes[0].set_title(r"Epistemic Uncertainty: $\mathbb{V}[\mu_\theta]$", fontsize=10)
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

            im2 = axes[1].imshow(bev_ale_2d, cmap='magma', origin='lower', aspect='equal', extent=extent)
            axes[1].set_xlabel('X Position (m)', fontsize=12)
            axes[1].set_ylabel('Z Position (m)', fontsize=12)
            axes[1].set_title(r"Aleatoric Uncertainty: $\mathbb{E}[\sigma^2_\theta]$", fontsize=10)
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
            
            # Add grid
            axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

            plt.tight_layout()
            plt.show()
        else:
            print("Unable to visualize BEV maps due to missing data.")

    def get_submap_avg(self, bev_map, extent, center_x, center_z, submap_size=1):
        """
        Extract a submap around (center_x, center_z) and compute its average value.
        
        Args:
            bev_map: 2D numpy array (height, width)
            extent: [min_x, max_x, min_z, max_z] in meters
            center_x: X coordinate in meters
            center_z: Z coordinate in meters
            submap_size: Size of submap in grid cells (default: 1)
        
        Returns:
            avg_value: Average value in the submap
        """
        min_x, max_x, min_z, max_z = extent
        height, width = bev_map.shape
        
        # Convert world coordinates (meters) to grid indices
        grid_x = int((center_x - min_x) / (max_x - min_x) * width)
        grid_z = int((center_z - min_z) / (max_z - min_z) * height)
        
        half_size = submap_size // 2
        x_start = max(grid_x - half_size, 0)
        x_end = min(grid_x + half_size + 1, width)
        z_start = max(grid_z - half_size, 0)
        z_end = min(grid_z + half_size + 1, height)
        
        submap = bev_map[z_start:z_end, x_start:x_end]
        avg_value = submap.mean()
        return avg_value
    
    def viz_map_history(self, grid_params, save_path='bev_history.mp4', fps=2, format='mp4'):
        """
        Create an animated visualization of BEV maps across training steps.
        
        Args:
            grid_params: Tuple of (extent, grid_size, box) where extent is [min_x, max_x, min_z, max_z],
                        grid_size is (bev_width, bev_height),
                        and box is (center_x, center_z, size)
            save_path: Path to save the animation (default: 'bev_history.mp4')
            fps: Frames per second for the animation (default: 2)
            format: 'mp4' or 'gif' (default: 'mp4')
        """

        # unexplored box
        extent, grid_size, box = grid_params
        center_x, center_z, side_length = box
        min_x, max_x, min_z, max_z = extent
        bev_width, bev_height = grid_size

        # explored box
        explored_side = side_length * 2
        center_x_explored = -5
        center_z_explored = -2
        explored_box = (center_x_explored, center_z_explored, explored_side)
        
        epochs = list(range(0, self.cfg.iterations + 1, self.cfg.viz_interval))
        
        print(f"Creating animation for {len(epochs)} training steps...")
        
        # Collect data for all epochs
        epi_avgs = []
        ale_avgs = []
        epi_expl_avgs = []
        ale_expl_avgs = []
        frames_data = []
        
        for step in epochs:
            bev_maps = self.load_bev_maps(step)
            if bev_maps[0] is not None:
                epi_map, ale_map = bev_maps
                epi_avg = self.get_submap_avg(epi_map, extent, center_x, center_z, side_length)
                ale_avg = self.get_submap_avg(ale_map, extent, center_x, center_z, side_length)
                epi_expl_avg = self.get_submap_avg(epi_map, extent, center_x_explored, center_z_explored, explored_side)
                ale_expl_avg = self.get_submap_avg(ale_map, extent, center_x_explored, center_z_explored, explored_side)


                epi_avgs.append(epi_avg)
                ale_avgs.append(ale_avg)
                epi_expl_avgs.append(epi_expl_avg)
                ale_expl_avgs.append(ale_expl_avg)
                frames_data.append((step, epi_map, ale_map))
        
        if not frames_data:
            print("No BEV maps found!")
            return
        
        # Calculate global color scale limits for consistent visualization
        all_epi_maps = np.array([epi_map for _, epi_map, _ in frames_data])
        all_ale_maps = np.array([ale_map for _, _, ale_map in frames_data])
        
        # Use log scale for both epistemic and aleatoric for consistency
        # Avoid zero/negative values by using a small epsilon or percentile floor
        epi_vmin = max(np.percentile(all_epi_maps, 1), 1e-10)  # Use 1st percentile to avoid zeros
        epi_vmax = np.percentile(all_epi_maps, 90)
        
        ale_vmin = max(np.percentile(all_ale_maps, 1), 1e-10)  # Use 1st percentile to avoid zeros
        ale_vmax = np.percentile(all_ale_maps, 98)
        
        print(f"Epistemic scale: [{epi_vmin:.6e}, {epi_vmax:.6e}] (1st-95th percentile)")
        print(f"Aleatoric scale: [{ale_vmin:.6e}, {ale_vmax:.6e}] (1st-98th percentile)")
        
        # Create frames
        frames = []
        for idx, (step, epi_map, ale_map) in enumerate(frames_data):
            fig = plt.figure(figsize=(14, 10))
            gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
            
            # Top row: BEV maps
            ax_epi = fig.add_subplot(gs[0, 0])
            ax_ale = fig.add_subplot(gs[0, 1])
            
            # Epistemic map (with logarithmic scale for better color distribution)
            im1 = ax_epi.imshow(epi_map, cmap='magma', origin='lower', aspect='equal', extent=extent,
                               vmin=epi_vmin, vmax=epi_vmax)
            ax_epi.set_xlabel('X Position (m)', fontsize=12)
            ax_epi.set_ylabel('Z Position (m)', fontsize=12)
            ax_epi.set_title(r"Epistemic Uncertainty: $\mathbb{V}[\mu_\theta]$", fontsize=10)
            plt.colorbar(im1, ax=ax_epi, fraction=0.046, pad=0.04)
            ax_epi.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # Draw submap box (side_length is in meters, same units as extent)
            box_x_size = side_length
            box_z_size = side_length
            
            from matplotlib.patches import Rectangle
            epi_rect = Rectangle((center_x - box_x_size/2, center_z - box_z_size/2), 
                           box_x_size, box_z_size,
                           linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
            ax_epi.add_patch(epi_rect)

            epi_explored_rect = Rectangle((center_x_explored - explored_side/2, center_z_explored - explored_side/2),
                                        explored_side, explored_side,
                                        linewidth=2, edgecolor='green', facecolor='none', linestyle='--')
            ax_epi.add_patch(epi_explored_rect)

            ale_explored_rect = Rectangle((center_x_explored - explored_side/2, center_z_explored - explored_side/2),
                                        explored_side, explored_side,
                                        linewidth=2, edgecolor='green', facecolor='none', linestyle='--')
            ax_ale.add_patch(ale_explored_rect)

            ale_rect = Rectangle((center_x - box_x_size/2, center_z - box_z_size/2), 
                           box_x_size, box_z_size,
                           linewidth=2, edgecolor='red', facecolor='none', linestyle='--')

            ax_ale.add_patch(ale_rect)
            
            # Stats text
            epi_stats_text = (
                f'Step: {step}\n'
                f'Min: {epi_map.min():.6f}\n'
                f'Max: {epi_map.max():.6f}\n'
                f'Mean: {epi_map.mean():.6f}'
            )
            ax_epi.text(0.01, 1.05, epi_stats_text,
                       transform=ax_epi.transAxes, fontsize=9,
                       verticalalignment='bottom', horizontalalignment='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Aleatoric map (with logarithmic scale for better color distribution)
            im2 = ax_ale.imshow(ale_map, cmap='magma', origin='lower', aspect='equal', extent=extent,
                               vmin=ale_vmin, vmax=ale_vmax)
            ax_ale.set_xlabel('X Position (m)', fontsize=12)
            ax_ale.set_ylabel('Z Position (m)', fontsize=12)
            ax_ale.set_title(r"Aleatoric Uncertainty: $\mathbb{E}[\sigma^2_\theta]$", fontsize=10)
            plt.colorbar(im2, ax=ax_ale, fraction=0.046, pad=0.04)
            ax_ale.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            ale_stats_text = (
                f'Step: {step}\n'
                f'Min: {ale_map.min():.6f}\n'
                f'Max: {ale_map.max():.6f}\n'
                f'Mean: {ale_map.mean():.6f}'
            )
            ax_ale.text(0.01, 1.05, ale_stats_text,
                       transform=ax_ale.transAxes, fontsize=9,
                       verticalalignment='bottom', horizontalalignment='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Bottom row: Time series plots (separate for epistemic and aleatoric)
            ax_epi_time = fig.add_subplot(gs[1, 0])
            ax_ale_time = fig.add_subplot(gs[1, 1])
            
            # Plot up to current step
            current_epochs = epochs[:idx+1]
            current_epi_avgs = epi_avgs[:idx+1]
            current_ale_avgs = ale_avgs[:idx+1]
            current_epi_expl_avgs = epi_expl_avgs[:idx+1]
            current_ale_expl_avgs = ale_expl_avgs[:idx+1]
            
            # Epistemic uncertainty time series
            ax_epi_time.plot(current_epochs, current_epi_avgs, 
                            marker='o', color='red', linewidth=2, markersize=6)
            ax_epi_time.plot(current_epochs, current_epi_expl_avgs, 
                            marker='s', color='green', linewidth=2, markersize=6, label='Explored Patch')
            ax_epi_time.set_xlabel('Training Step', fontsize=12)
            ax_epi_time.set_ylabel('Avg Epistemic Uncertainty', fontsize=12)
            ax_epi_time.set_title(f'Epistemic (Submap at [{center_x:.1f}, {center_z:.1f}])', fontsize=10)
            ax_epi_time.grid(True, alpha=0.3)
            ax_epi_time.set_xlim(epochs[0], epochs[-1])
            
            # Set y-limits based on epistemic data range for consistency
            epi_min, epi_max = min(epi_avgs), max(epi_avgs)
            epi_margin = (epi_max - epi_min) * 0.1 if epi_max > epi_min else 0.1
            ax_epi_time.set_ylim(epi_min - epi_margin, epi_max + epi_margin)
            
            # Aleatoric uncertainty time series
            ax_ale_time.plot(current_epochs, current_ale_avgs, 
                            marker='s', color='blue', linewidth=2, markersize=6)
            ax_ale_time.plot(current_epochs, current_ale_expl_avgs, 
                            marker='s', color='green', linewidth=2, markersize=6, label='Explored Patch')
            ax_ale_time.set_xlabel('Training Step', fontsize=12)
            ax_ale_time.set_ylabel('Avg Aleatoric Uncertainty', fontsize=12)
            ax_ale_time.set_title(f'Aleatoric (Submap at [{center_x:.1f}, {center_z:.1f}])', fontsize=10)
            ax_ale_time.grid(True, alpha=0.3)
            ax_ale_time.set_xlim(epochs[0], epochs[-1])
            ax_ale_time.legend(loc='upper right', fontsize=9)
            
            # Set y-limits based on aleatoric data range for consistency
            ale_min, ale_max = min(ale_avgs), max(ale_avgs)
            ale_margin = (ale_max - ale_min) * 0.1 if ale_max > ale_min else 0.1
            ax_ale_time.set_ylim(ale_min - ale_margin, ale_max + ale_margin)
            
            # Render frame to numpy array
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            
            plt.close(fig)
            print(f"  Rendered frame {idx+1}/{len(frames_data)} (step {step})")
        
        # Save animation
        if format.lower() == 'gif':
            save_path = save_path.replace('.mp4', '.gif')
            imageio.mimsave(save_path, frames, fps=fps)
        else:
            save_path = save_path.replace('.gif', '.mp4')
            imageio.mimsave(save_path, frames, fps=fps, codec='libx264')
        
        print(f"Animation saved to: {save_path}")
        
        # Also show final plots (separate for better visualization)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Epistemic uncertainty
        ax1.plot(epochs, epi_avgs, marker='o', color='red', linewidth=2, markersize=6, label='Unexplored Patch')
        ax1.plot(epochs, epi_expl_avgs, marker='s', color='green', linewidth=2, markersize=6, label='Explored Patch')
        ax1.set_xlabel('Training Step', fontsize=12)
        ax1.set_ylabel('Average Epistemic Uncertainty', fontsize=12)
        ax1.set_title('Epistemic Uncertainty Over Training', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=10)

        
        # Aleatoric uncertainty
        ax2.plot(epochs, ale_avgs, marker='s', color='blue', linewidth=2, markersize=6, label='Unexplored Patch')
        ax2.plot(epochs, ale_expl_avgs, marker='s', color='green', linewidth=2, markersize=6, label='Explored Patch')
        ax2.set_xlabel('Training Step', fontsize=12)
        ax2.set_ylabel('Average Aleatoric Uncertainty', fontsize=12)
        ax2.set_title('Aleatoric Uncertainty Over Training', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    config = Config("./config/config.yaml")
    
    visualizer = Visualizer(config)
    
    # # Example 1: Visualize a single BEV map
    # bev_maps = visualizer.load_bev_maps(step=50000)
    # visualizer.visualize_bev_map(bev_maps)
    
    # Example 2: Create animated history of BEV maps over training
    # Setup grid and submap bounds
    extent = [visualizer.bev_grid.bev_min_x, visualizer.bev_grid.bev_max_x, 
              visualizer.bev_grid.bev_min_z, visualizer.bev_grid.bev_max_z]
    grid_size = (visualizer.bev_grid.bev_width, visualizer.bev_grid.bev_height)
    center_x, center_z = 4, -3 # Submap center in meters
    side_length = 1  # Submap size in meters
    box = (center_x, center_z, side_length)
    grid_params = (extent, grid_size, box)
    
    # Create animation (MP4 or GIF)
    visualizer.viz_map_history(grid_params, save_path='bev_history.mp4', fps=2, format='mp4')


