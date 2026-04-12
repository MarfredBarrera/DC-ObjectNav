import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import matplotlib
matplotlib.use('Agg')
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
import time
import torch
import numpy as np
import imageio.v2 as imageio
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, ListedColormap
from collections import deque

# Habitat Imports
import habitat_sim
import habitat_sim.utils.common as utils
import habitat_sim.physics as physics

# Custom Imports
from src.config import Config
from src.semantics import SAM_CLIP_Semantics
from src.featurefield import FeatureField
from src.grid import UncertaintyGrid


def get_scene_bounds(scene_path):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False
    sim_cfg.load_semantic_mesh = False
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    try:
        bounds = sim.pathfinder.get_bounds()
        return [np.array(bounds[0]).tolist(), np.array(bounds[1]).tolist()]
    finally:
        sim.close()

class Visualizer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        # Environment Setup
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cfg.gpu_indices
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"
        self.device = self.cfg.device
        self.scene_bounds = get_scene_bounds(self.cfg.scene_path)

        # 1. Ensemble Models
        self.ensemble_models = self.load_ensemble()

        # 2. BEV Grid
        self.bev_grid = UncertaintyGrid(cfg, ensemble=self.ensemble_models, scene_bounds=self.scene_bounds)

    # Legacy camera data loading removed.
    

    def load_ensemble(self):
        """Loads the 3 ensemble HashGrid models from the output directory."""
        ensemble_models = []
        ensemble_dir = os.path.join(self.cfg.output_dir, "ensemble")
        
        if not os.path.exists(ensemble_dir):
            print(f"Error: Ensemble directory not found at {ensemble_dir}")
            return

        print("Loading Ensemble Models...")
        for i in range(self.cfg.ensemble_num_models):
            model_path = os.path.join(ensemble_dir, f"featurefield_ensemble_{i}.pt")
            if os.path.exists(model_path):
                # Initialize a new FeatureField instance
                model = FeatureField(self.cfg, self.scene_bounds, device=self.device)
                model.load(model_path)
                ensemble_models.append(model)
                print(f"  -> Loaded Ensemble Model {i}")
            else:
                print(f"  -> Warning: Model {i} not found at {model_path}")

        return ensemble_models

    # Transpose logic removed as discretization is now unified.
        
    def visualize_bev_map(self, u_maps):
        """Visualize the BEV uncertainty maps."""
        bev_epi_2d, bev_ale_2d = u_maps
        
        if bev_epi_2d is not None and bev_ale_2d is not None:
            # Reshape from flattened (N,) to 2D (bev_height, bev_width)
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            extent = [self.bev_grid.min_x, self.bev_grid.max_x, self.bev_grid.min_z, self.bev_grid.max_z]

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
            plt.savefig("./bev_maps.png")
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
        num_x, num_z = grid_size

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
        
        # Track shapes to debug inhomogeneous arrays
        shapes_seen = {}
        
        for step in epochs:
            bev_maps = self.load_bev_maps(step)
            if bev_maps[0] is not None:
                epi_map, ale_map = bev_maps
                
                # Track shapes
                epi_shape = epi_map.shape
                if epi_shape not in shapes_seen:
                    shapes_seen[epi_shape] = []
                shapes_seen[epi_shape].append(step)
                
                epi_avg = self.get_submap_avg(epi_map, extent, center_x, center_z, side_length)
                ale_avg = self.get_submap_avg(ale_map, extent, center_x, center_z, side_length)
                epi_expl_avg = self.get_submap_avg(epi_map, extent, center_x_explored, center_z_explored, explored_side)
                ale_expl_avg = self.get_submap_avg(ale_map, extent, center_x_explored, center_z_explored, explored_side)


                epi_avgs.append(epi_avg)
                ale_avgs.append(ale_avg)
                epi_expl_avgs.append(epi_expl_avg)
                ale_expl_avgs.append(ale_expl_avg)
                frames_data.append((step, epi_map, ale_map))
        
        # Report shape inconsistencies
        if len(shapes_seen) > 1:
            print(f"\nWARNING: Found {len(shapes_seen)} different shapes across BEV maps:")
            for shape, steps in shapes_seen.items():
                print(f"  Shape {shape}: steps {steps}")
        else:
            print(f"\nAll BEV maps have consistent shape: {list(shapes_seen.keys())[0]}")
        
        if not frames_data:
            print("No BEV maps found!")
            return
        
        # Calculate global color scale limits for consistent visualization
        # Concatenate flattened arrays to handle maps with different shapes
        all_epi_values = np.concatenate([epi_map.flatten() for _, epi_map, _ in frames_data])
        all_ale_values = np.concatenate([ale_map.flatten() for _, _, ale_map in frames_data])
        
        # Use log scale for both epistemic and aleatoric for consistency
        # Avoid zero/negative values by using a small epsilon or percentile floor
        epi_vmin = max(np.percentile(all_epi_values, 1), 1e-10)  # Use 1st percentile to avoid zeros
        epi_vmax = np.percentile(all_epi_values, 90)
        
        ale_vmin = max(np.percentile(all_ale_values, 1), 1e-10)  # Use 1st percentile to avoid zeros
        ale_vmax = np.percentile(all_ale_values, 98)
        
        print(f"Epistemic scale: [{epi_vmin:.6e}, {epi_vmax:.6e}] (1st-95th percentile)")
        print(f"Aleatoric scale: [{ale_vmin:.6e}, {ale_vmax:.6e}] (1st-98th percentile)")
        
        # Create frames
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
        
        ax_epi = fig.add_subplot(gs[0, 0])
        ax_ale = fig.add_subplot(gs[0, 1])
        ax_epi_time = fig.add_subplot(gs[1, 0])
        ax_ale_time = fig.add_subplot(gs[1, 1])

        from matplotlib.patches import Rectangle
        box_x_size = box_z_size = side_length
        
        # Initialize plot objects with dummy data
        im_epi = ax_epi.imshow(np.zeros((10, 10)), cmap='magma', origin='lower', extent=extent, vmin=epi_vmin, vmax=epi_vmax)
        im_ale = ax_ale.imshow(np.zeros((10, 10)), cmap='magma', origin='lower', extent=extent, vmin=ale_vmin, vmax=ale_vmax)
        
        rect_epi_unexpl = Rectangle((center_x - box_x_size/2, center_z - box_z_size/2), box_x_size, box_z_size, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
        rect_epi_expl = Rectangle((center_x_explored - explored_side/2, center_z_explored - explored_side/2), explored_side, explored_side, linewidth=2, edgecolor='green', facecolor='none', linestyle='--')
        ax_epi.add_patch(rect_epi_unexpl)
        ax_epi.add_patch(rect_epi_expl)

        rect_ale_unexpl = Rectangle((center_x - box_x_size/2, center_z - box_z_size/2), box_x_size, box_z_size, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
        rect_ale_expl = Rectangle((center_x_explored - explored_side/2, center_z_explored - explored_side/2), explored_side, explored_side, linewidth=2, edgecolor='green', facecolor='none', linestyle='--')
        ax_ale.add_patch(rect_ale_unexpl)
        ax_ale.add_patch(rect_ale_expl)

        line_epi_unexpl, = ax_epi_time.plot([], [], marker='o', color='red', linewidth=2, markersize=6)
        line_epi_expl, = ax_epi_time.plot([], [], marker='s', color='green', linewidth=2, markersize=6, label='Explored Patch')
        line_ale_unexpl, = ax_ale_time.plot([], [], marker='s', color='blue', linewidth=2, markersize=6)
        line_ale_expl, = ax_ale_time.plot([], [], marker='s', color='green', linewidth=2, markersize=6, label='Explored Patch')

        # Static labels/configs
        ax_epi.set_title(r"Epistemic Uncertainty: $\mathbb{V}[\mu_\theta]$")
        ax_ale.set_title(r"Aleatoric Uncertainty: $\mathbb{E}[\sigma^2_\theta]$")
        plt.colorbar(im_epi, ax=ax_epi, fraction=0.046, pad=0.04)
        plt.colorbar(im_ale, ax=ax_ale, fraction=0.046, pad=0.04)
        
        ax_epi_time.set_xlim(epochs[0], epochs[-1])
        ax_ale_time.set_xlim(epochs[0], epochs[-1])
        epi_min, epi_max = min(epi_avgs), max(epi_avgs)
        ale_min, ale_max = min(ale_avgs), max(ale_avgs)
        ax_epi_time.set_ylim(epi_min - (epi_max-epi_min)*0.1 - 0.1, epi_max + (epi_max-epi_min)*0.1 + 0.1)
        ax_ale_time.set_ylim(ale_min - (ale_max-ale_min)*0.1 - 0.1, ale_max + (ale_max-ale_min)*0.1 + 0.1)
        
        epi_text = ax_epi.text(0.01, 1.05, '', transform=ax_epi.transAxes, fontsize=9, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ale_text = ax_ale.text(0.01, 1.05, '', transform=ax_ale.transAxes, fontsize=9, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        frames = []
        for idx, (step, epi_map, ale_map) in enumerate(frames_data):
            im_epi.set_data(epi_map)
            im_ale.set_data(ale_map)
            
            line_epi_unexpl.set_data(epochs[:idx+1], epi_avgs[:idx+1])
            line_epi_expl.set_data(epochs[:idx+1], epi_expl_avgs[:idx+1])
            line_ale_unexpl.set_data(epochs[:idx+1], ale_avgs[:idx+1])
            line_ale_expl.set_data(epochs[:idx+1], ale_expl_avgs[:idx+1])
            
            epi_text.set_text(f'Step: {step}\nMin: {epi_map.min():.6f}\nMax: {epi_map.max():.6f}\nMean: {epi_map.mean():.6f}')
            ale_text.set_text(f'Step: {step}\nMin: {ale_map.min():.6f}\nMax: {ale_map.max():.6f}\nMean: {ale_map.mean():.6f}')

            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            
            print(f"  Rendered frame {idx+1}/{len(frames_data)} (step {step})")
        
        plt.close(fig)

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

    # visualize_cam_similarity logic removed as it depends on transforms.json.

    def visualize_occupancy(self, step, height_range=None):
        """
        Visualize the occupancy map for a specific training step.
        """
        # Load occupancy map
        occ_maps_dir = os.path.join(self.cfg.output_dir, "occ_maps")
        occ_path = os.path.join(occ_maps_dir, f"bev_occupancy_{step}.npy")
        
        if not os.path.exists(occ_path):
            print(f"Occupancy map for step {step} not found at {occ_path}")
            return
        
        occupancy_map = np.load(occ_path)
        print(f"Loaded occupancy map: {occupancy_map.shape}")
        
        # Get grid extent from grid
        extent = [self.bev_grid.min_x, self.bev_grid.max_x, 
                  self.bev_grid.min_z, self.bev_grid.max_z]
        
        # Determine height range for display
        if height_range is None:
            height_range = (self.bev_grid.min_y, self.bev_grid.max_y)
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use a discrete colormap: 0=Gray (Unseen), 1=White (Free), 2=Black (Occupied)
        occ_cmap = ListedColormap(['#808080', '#FFFFFF', '#000000'])
        im = ax.imshow(occupancy_map, cmap=occ_cmap, origin='lower', aspect='equal', extent=extent, vmin=0, vmax=2)
        
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Z Position (m)', fontsize=12)
        ax.set_title(f"Occupancy Map (Step {step})\nHeight Range: [{height_range[0]:.2f}, {height_range[1]:.2f}] m", fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add statistics
        unseen_cells = np.sum(occupancy_map == 0)
        free_cells = np.sum(occupancy_map == 1)
        occupied_cells = np.sum(occupancy_map == 2)
        total_cells = occupancy_map.size
        occupancy_rate = occupied_cells / total_cells * 100 if total_cells > 0 else 0
        exploration_rate = (free_cells + occupied_cells) / total_cells * 100 if total_cells > 0 else 0
        
        stats_text = (
            f'Occupied Cells: {occupied_cells:,}\n'
            f'Free Cells: {free_cells:,}\n'
            f'Unseen Cells: {unseen_cells:,}\n'
            f'Exploration Rate: {exploration_rate:.2f}%\n'
            f'Occupancy Rate: {occupancy_rate:.2f}%\n'
            f'Resolution: {self.bev_grid.resolution}m/cell\n'
            f'Grid Size: {occupancy_map.shape[1]} × {occupancy_map.shape[0]}'
        )
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('State (0=Unseen, 1=Free, 2=Occupied)')
        cbar.set_ticks([0.33, 1.0, 1.66])
        cbar.set_ticklabels(['Unseen', 'Free', 'Occupied'])
        plt.tight_layout()
        plt.show()
        
        return occupancy_map
    
    def visualize_sim_map(self, step):
        """
        Visualize the similarity map for a specific training step.
        """
        sim_maps_dir = os.path.join(self.cfg.output_dir, "sim_maps")
        sim_path = os.path.join(sim_maps_dir, f"bev_similarity_{step}.npy")
        
        if not os.path.exists(sim_path):
            print(f"Similarity map for step {step} not found at {sim_path}")
            return
        
        sim_map = np.load(sim_path)
        print(f"Loaded similarity map: {sim_map.shape}")

        # Exclude all zero scores and normalize
        non_zero_mask = sim_map > 0
        if np.any(non_zero_mask):
            vmin_orig = sim_map[non_zero_mask].min()
            vmax_orig = sim_map[non_zero_mask].max()
            num_zeros = np.sum(~non_zero_mask)
            
            # Normalize non-zero scores to [0, 1]
            sim_map_normalized = np.zeros_like(sim_map)
            sim_map_normalized[non_zero_mask] = (sim_map[non_zero_mask] - vmin_orig) / (vmax_orig - vmin_orig)
            vmin = 0
            vmax = 1
        else:
            sim_map_normalized = sim_map
            vmin = 0
            vmax = 1
            print("All scores are zero!")
        
        # Get grid extent from BEV grid
        extent = [self.bev_grid.min_x, self.bev_grid.max_x, 
                  self.bev_grid.min_z, self.bev_grid.max_z]
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(sim_map_normalized, cmap='jet', origin='lower', aspect='equal', extent=extent, vmin=vmin, vmax=vmax)
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Z Position (m)', fontsize=12)
        ax.set_title(f"Similarity Map (Step {step})", fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Normalized Similarity Score')
        plt.tight_layout()
        plt.show()
        plt.savefig("./figs/similarity_map.png")

        return sim_map_normalized
    
    def load_umaps(self, step):
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
        
    def load_occmap(self,step):
        # Load occupancy map
        occ_maps_dir = os.path.join(self.cfg.output_dir, "occ_maps")
        occ_path = os.path.join(occ_maps_dir, f"bev_occupancy_{step}.npy")
        
        if not os.path.exists(occ_path):
            print(f"Occupancy map for step {step} not found at {occ_path}")
            return
        
        occupancy_map = np.load(occ_path)
        return occupancy_map
    
    def load_simmap(self,step):
        sim_maps_dir = os.path.join(self.cfg.output_dir, "sim_maps")
        sim_path = os.path.join(sim_maps_dir, f"bev_similarity_{step}.npy")
        
        if not os.path.exists(sim_path):
            print(f"Similarity map for step {step} not found at {sim_path}")
            return
        
        sim_map = np.load(sim_path)
        return sim_map
    
        epi_map, sim_plot, occ_map, extent = self._prepare_all_maps_data(step)
        if epi_map is None: return

        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        self._render_all_maps_to_axes(axes, step, epi_map, sim_plot, occ_map, extent)
        
        plt.subplots_adjust(hspace=0.05)
        plt.savefig(f"./all_maps_step_{step}.png", bbox_inches='tight')
        plt.show()

    def _prepare_all_maps_data(self, step):
        """Helper to load, align, and normalize all maps for a given step."""
        epi_map, _ = self.load_umaps(step)
        sim_map = self.load_simmap(step)
        occ_map = self.load_occmap(step)

        if epi_map is None or sim_map is None or occ_map is None:
            return None, None, None, None

        expected_shape = (self.bev_grid.num_z, self.bev_grid.num_x)

        def align_map(map_2d, map_name):
            if map_2d.shape == expected_shape:
                return map_2d
            if map_2d.shape == (expected_shape[1], expected_shape[0]):
                return map_2d.T
            return None # Should not happen with current grid system

        epi_map = align_map(epi_map, "epistemic")
        sim_map = align_map(sim_map, "similarity")
        occ_map = align_map(occ_map, "occupancy")
        
        if epi_map is None or sim_map is None or occ_map is None:
            return None, None, None, None

        # Normalize similarity
        non_zero_mask = sim_map > 0
        if np.any(non_zero_mask):
            sim_min, sim_max = sim_map[non_zero_mask].min(), sim_map[non_zero_mask].max()
            sim_plot = np.zeros_like(sim_map)
            if sim_max > sim_min:
                sim_plot[non_zero_mask] = (sim_map[non_zero_mask] - sim_min) / (sim_max - sim_min)
            else:
                sim_plot[non_zero_mask] = 1.0
        else:
            sim_plot = sim_map

        extent = [self.bev_grid.min_x, self.bev_grid.max_x, 
                  self.bev_grid.min_z, self.bev_grid.max_z]
        
        return epi_map, sim_plot, occ_map, extent

    def _render_all_maps_to_axes(self, axes, step, epi_map, sim_plot, occ_map, extent):
        """Helper to render maps onto provided axes."""
        # 1) Epistemic uncertainty
        im_epi = axes[0].imshow(epi_map, cmap='magma', origin='lower', aspect='equal', extent=extent)
        axes[0].set_xlabel('X Position (m)', fontsize=12)
        axes[0].set_ylabel('Z Position (m)', fontsize=12)
        axes[0].set_title(rf"Uncertainty: $\mathbb{{V}}[\mu_\theta]$ (Step {step})", fontsize=11)
        axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        plt.colorbar(im_epi, ax=axes[0], fraction=0.046, pad=0.02, shrink=0.5, label='Epistemic Uncertainty')

        epi_stats = f"Min: {epi_map.min():.6f}\nMax: {epi_map.max():.6f}\nMean: {epi_map.mean():.6f}"
        axes[0].text(0.02, 0.98, epi_stats, transform=axes[0].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 2) Occupancy
        occ_cmap = ListedColormap(['#808080', '#FFFFFF', '#000000'])
        im_occ = axes[1].imshow(occ_map, cmap=occ_cmap, origin='lower', aspect='equal', extent=extent, vmin=0, vmax=2)
        axes[1].set_xlabel('X Position (m)', fontsize=12)
        axes[1].set_ylabel('Z Position (m)', fontsize=12)
        axes[1].set_title(f"Occupancy Map (Step {step})", fontsize=11)
        axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        cbar_occ = plt.colorbar(im_occ, ax=axes[1], fraction=0.046, pad=0.02, shrink=0.5)
        cbar_occ.set_ticks([0.33, 1.0, 1.66])
        cbar_occ.set_ticklabels(['Unseen', 'Free', 'Occupied'])

        occupied_cells = np.sum(occ_map == 2)
        exploration_rate = (np.sum(occ_map >= 1) / occ_map.size) * 100
        occ_stats = f"Explored: {exploration_rate:.2f}%\nOccupied: {occupied_cells:,}"
        axes[1].text(0.02, 0.98, occ_stats, transform=axes[1].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 3) Similarity
        im_sim = axes[2].imshow(sim_plot, cmap='jet', origin='lower', aspect='equal', extent=extent, vmin=0, vmax=1)
        axes[2].set_xlabel('X Position (m)', fontsize=12)
        axes[2].set_ylabel('Z Position (m)', fontsize=12)
        axes[2].set_title(f"Similarity Map (Step {step})", fontsize=11)
        axes[2].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        plt.colorbar(im_sim, ax=axes[2], fraction=0.046, pad=0.02, shrink=0.5, label='Normalized Similarity')

    def viz_all_maps_history(self, save_path='simulation_history.mp4', fps=2):
        """Create an MP4 animation of all maps across the entire simulation."""
        epochs = list(range(0, self.cfg.iterations + 1, self.cfg.viz_interval))
        print(f"Generating full simulation history animation ({len(epochs)} steps)...")
        
        frames = []
        for step in epochs:
            epi_map, sim_plot, occ_map, extent = self._prepare_all_maps_data(step)
            if epi_map is None:
                continue
                
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            self._render_all_maps_to_axes(axes, step, epi_map, sim_plot, occ_map, extent)
            plt.subplots_adjust(wspace=0.3)
            
            # Capture frame
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            plt.close(fig)
            print(f"  Rendered step {step}")

        if frames:
            imageio.mimsave(save_path, frames, fps=fps, codec='libx264')
            print(f"Animation saved to: {save_path}")
        else:
            print("No maps found to animate.")






if __name__ == "__main__":
    config = Config("./config/config.yaml")
    visualizer = Visualizer(config)

    # step = 10000

    # epi_map, ale_map = visualizer.load_umaps(step)
    # visualizer.visualize_bev_map((epi_map, ale_map))

    # epi_map, sim_plot, occ_map, extent = visualizer._prepare_all_maps_data(step)
    # fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # visualizer._render_all_maps_to_axes(axes, step, epi_map, sim_plot, occ_map, extent)
    # plt.subplots_adjust(wspace=0.3)
    # plt.savefig(f'./figs/all_maps_step_{step}.png')  
    visualizer.viz_all_maps_history(save_path='./figs/full_simulation_history.mp4', fps=2)
