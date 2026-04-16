import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"

# Silence habitat-sim warnings and logs
os.environ['GLOG_minloglevel'] = '2'
os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['HABITAT_SIM_LOG'] = 'quiet'

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# User Dev Imports
from src.perception.grid import UncertaintyGrid, SimilarityGrid
from src.config import Config
from src.perception.semantics import SAM_CLIP_Semantics
from src.perception.featurefield import FeatureField
from src.habitat.habitat_utils import (
    make_cfg,
    get_scene_bounds_from_pathfinder,
    spawn_agent_at_random_navpoint,
    init_simulator,
)
from src.habitat.sim_interface import SimInterface
from src.planning.path_explorer import PathExplorer

import habitat_sim



# --------------------------------------------------------
# Planner
# --------------------------------------------------------

class Planner:
    """Planning interface that reads pre-built maps from a PerceptionStack.

    Parameters
    ----------
    cfg:
        Project Config object.
    sim_iface:
        A :class:`src.sim_interface.SimInterface` providing robot control.
    scene_bounds:
        Two-element list ``[min_corner, max_corner]``.
    """

    def __init__(self, cfg: Config, sim_iface: SimInterface, scene_bounds: list):
        self.cfg = cfg
        self.device = cfg.device
        self.scene_bounds = scene_bounds  # stored so load_ensemble can use it

        # SimInterface for robot control and sensor reads
        self.sim_iface = sim_iface

        # Semantics and Ensemble Models
        self.sam_clip = SAM_CLIP_Semantics(cfg, device=self.device)
        self.ensemble_models = [
            FeatureField(cfg, scene_bounds=scene_bounds, device=self.device)
            for _ in range(cfg.ensemble_num_models)
        ]

        self.ugrid = UncertaintyGrid(cfg, ensemble=self.ensemble_models, scene_bounds=scene_bounds)
        self.sim_map = SimilarityGrid(cfg, ensemble=self.ensemble_models, sam_clip=self.sam_clip, scene_bounds=scene_bounds)
        
        self.bev_epi_map = None
        self.bev_ale_map = None
        self.bev_sim_map = None
        self.bev_occ_map = None
        
        self.explorer = PathExplorer(cfg, device=self.device)
    
    def load_umap(self, step=20000):
        umaps_dir = os.path.join(self.cfg.output_dir, "umaps")
        epi_path = os.path.join(umaps_dir, f"bev_epistemic_uncertainty_{step}.npy")
        ale_path = os.path.join(umaps_dir, f"bev_aleatoric_uncertainty_{step}.npy")
        
        if os.path.exists(epi_path) and os.path.exists(ale_path):
            self.bev_epi_map = np.load(epi_path)
            self.bev_ale_map = np.load(ale_path)
            return self.bev_epi_map, self.bev_ale_map
        else:
            print(f"BEV maps for step {step} not found in {umaps_dir}")
            return None, None

    def load_sim_map(self, step=20000):
        sim_map_dir = os.path.join(self.cfg.output_dir, "sim_maps")
        sim_map_path = os.path.join(sim_map_dir, f"bev_similarity_{step}.npy")
        
        if os.path.exists(sim_map_path):
            self.bev_sim_map = np.load(sim_map_path)
            return self.bev_sim_map
        else:
            print(f"BEV map for step {step} not found in {sim_map_dir}")
            print(f"sim_map_path: {sim_map_path}")
            return None
            
    def load_occ_map(self, step=20000):
        occ_map_dir = os.path.join(self.cfg.output_dir, "occ_maps")
        occ_map_path = os.path.join(occ_map_dir, f"bev_occupancy_{step}.npy")
        
        if os.path.exists(occ_map_path):
            self.bev_occ_map = np.load(occ_map_path)
            return self.bev_occ_map
        else:
            print(f"BEV map for step {step} not found in {occ_map_dir}")
            return None
        
    def get_sim_map(self):
        self.sim_map.compute_similarity_map("a photo of a pillow")
        return self.sim_map.get_2d_map(min_y=0.1, max_y=1.5).cpu().numpy()
    
    def set_sim_map(self, sim_map):
        self.bev_sim_map = sim_map

    def set_umap(self, umap):
        if umap[0] is not None:
            self.bev_epi_map, self.bev_ale_map = umap
            
    def set_occ_map(self, occ_map):
        self.bev_occ_map = occ_map
        
    def load_ensemble(self):
        """Loads the ensemble FeatureField models from the output directory."""
        ensemble_models = []
        ensemble_dir = os.path.join(self.cfg.output_dir, "ensemble")
        
        if not os.path.exists(ensemble_dir):
            print(f"Error: Ensemble directory not found at {ensemble_dir}")
            return

        print("Loading Ensemble Models...")
        for i in range(self.cfg.ensemble_num_models):
            model_path = os.path.join(ensemble_dir, f"featurefield_ensemble_{i}.pt")
            if os.path.exists(model_path):
                # Initialise using the same scene_bounds as the grid members
                model = FeatureField(self.cfg, scene_bounds=self.scene_bounds, device=self.device)
                model.load(model_path)
                ensemble_models.append(model)
                print(f"  -> Loaded Ensemble Model {i}")
            else:
                print(f"  -> Warning: Model {i} not found at {model_path}")

        return ensemble_models

    def viz_umap(self):

        bev_epi_2d = self.bev_epi_map
        if bev_epi_2d is not None:
            if isinstance(bev_epi_2d, torch.Tensor): bev_epi_2d = bev_epi_2d.cpu().numpy()
            fig, axes = plt.subplots(figsize=(12,6))
            extent = [self.ugrid.min_x, self.ugrid.max_x, self.ugrid.min_z, self.ugrid.max_z]

            # Epistemic Uncertainty Map
            im1 = axes.imshow(bev_epi_2d, cmap='magma', origin='lower', aspect='equal', extent=extent)
            axes.set_xlabel('X Position (m)', fontsize=12)
            axes.set_ylabel('Z Position (m)', fontsize=12)
            axes.set_title(r"Epistemic Uncertainty: $\mathbb{V}[\mu_\theta]$", fontsize=10)
            plt.colorbar(im1, ax=axes, fraction=0.046, pad=0.04)

            # Add statistics text
            epi_stats_text = (
                f'Min: {bev_epi_2d.min():.6f}\n'
                f'Max: {bev_epi_2d.max():.6f}\n'
                f'Mean: {bev_epi_2d.mean():.6f}\n'
                f'Std: {bev_epi_2d.std():.6f}'
            )
            axes.text(
                0.01, 1.05, epi_stats_text,
                transform=axes.transAxes,
                fontsize=10,
                verticalalignment='bottom',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            axes.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            plt.tight_layout()
            plt.show()
        else:
            print("Unable to visualize BEV maps due to missing data.")

    def viz_sim_map(self):
        bev_sim_2d = self.bev_sim_map
        if bev_sim_2d is not None:
            if isinstance(bev_sim_2d, torch.Tensor): bev_sim_2d = bev_sim_2d.cpu().numpy()
            fig, axes = plt.subplots(figsize=(12,6))
            extent = [self.sim_map.min_x, self.sim_map.max_x, self.sim_map.min_z, self.sim_map.max_z]

            def normalize_sim(m):
                if m is None: return None
                non_zero_mask = m > 0
                if not np.any(non_zero_mask): return m
                m_min, m_max = m[non_zero_mask].min(), m[non_zero_mask].max()
                m_norm = np.zeros_like(m)
                if m_max > m_min:
                    m_norm[non_zero_mask] = (m[non_zero_mask] - m_min) / (m_max - m_min)
                else:
                    m_norm[non_zero_mask] = 1.0
                return m_norm

            bev_sim_2d = normalize_sim(bev_sim_2d)

            # Similarity Map
            im1 = axes.imshow(bev_sim_2d, cmap='jet', origin='lower', aspect='equal', extent=extent, vmin=0, vmax=1)
            axes.set_xlabel('X Position (m)', fontsize=12)
            axes.set_ylabel('Z Position (m)', fontsize=12)
            axes.set_title(r"Similarity Map: $sim_\theta(x)$", fontsize=10)
            plt.colorbar(im1, ax=axes, fraction=0.046, pad=0.04)

            # Add statistics text
            sim_stats_text = (
                f'Min: {bev_sim_2d.min():.6f}\n'
                f'Max: {bev_sim_2d.max():.6f}\n'
                f'Mean: {bev_sim_2d.mean():.6f}\n'
                f'Std: {bev_sim_2d.std():.6f}'
            )
            axes.text(
                0.01, 1.05, sim_stats_text,
                transform=axes.transAxes,
                fontsize=10,
                verticalalignment='bottom',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            axes.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            plt.tight_layout()
            plt.show()
            plt.savefig("./sim_map.png")
        else:
            print("Unable to visualize BEV maps due to missing data.")


    def viz_occ_map(self):
        bev_occ_2d = self.bev_occ_map
        if bev_occ_2d is not None:
            fig, axes = plt.subplots(figsize=(12,6))
            extent = [self.sim_map.min_x, self.sim_map.max_x, self.sim_map.min_z, self.sim_map.max_z]

            im1 = axes.imshow(bev_occ_2d, cmap='gray', origin='lower', aspect='equal', extent=extent)
            axes.set_xlabel('X Position (m)', fontsize=12)
            axes.set_ylabel('Z Position (m)', fontsize=12)
            axes.set_title(r"Occupancy Map", fontsize=10)
            plt.colorbar(im1, ax=axes, fraction=0.046, pad=0.04)

            axes.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            plt.tight_layout()
            plt.show()
        else:
            print("Unable to visualize Occupancy map due to missing data.")

    def plan_paths(self, start_pos, alpha=1.0, beta=1.0, gamma=1.0, num_modes=3):
        """
        Generate and score trajectories from start_pos.
        start_pos: (z, x) tuple of starting grid coordinates
        """
        if self.bev_sim_map is None or self.bev_epi_map is None or self.bev_occ_map is None:
            print("Cannot plan paths, missing maps. Ensure sim, epi, and occ maps are loaded.")
            return [], None
            
        # 1. Get Candidate Goals from GMM
        print("Finding candidate goals using GMM...")
        goals = self.explorer.get_gmm_goals(self.bev_sim_map, num_modes=num_modes)
        print(f"Testing {len(goals)} potential goals.")
        
        # 2. Plan paths using A*
        trajectories = []
        for goal in goals:
            path = self.explorer.astar(self.bev_occ_map, start_pos, goal)
            if path is not None:
                trajectories.append(path)
            else:
                print(f"Goal {goal} is unreachable.")
                
        # 3. Score paths
        scores, best_idx = self.explorer.score_trajectories(
            trajectories, self.bev_sim_map, self.bev_epi_map, self.bev_occ_map,
            alpha, beta, gamma
        )
        
        return scores, best_idx

    def world_to_grid(self, x, z):
        """Converts world coordinates (x, z) in meters to grid indices (z_idx, x_idx)."""
        res = self.cfg.voxel_resolution
        x_idx = int((x - self.sim_map.min_x) / res)
        z_idx = int((z - self.sim_map.min_z) / res)
        return (z_idx, x_idx)

    def grid_to_world(self, z_idx, x_idx):
        """Converts grid indices (z_idx, x_idx) to world coordinates (x, z) in meters."""
        res = self.cfg.voxel_resolution
        x = self.sim_map.min_x + x_idx * res
        z = self.sim_map.min_z + z_idx * res
        return (x, z)

    def viz_paths(self, scores, best_idx):
        if not scores or best_idx is None:
            print("No paths to visualize.")
            return

        best_score = scores[best_idx]
        best_traj = best_score['traj']
        seen_mask = best_score['seen_mask']

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        extent = [self.sim_map.min_x, self.sim_map.max_x, self.sim_map.min_z, self.sim_map.max_z]

        # 1. Occupancy with all paths - Custom Colormap (Unseen, Free, Occupied)
        occ_cmap = ListedColormap(['#808080', '#FFFFFF', '#000000'])
        axes[0].imshow(self.bev_occ_map, cmap=occ_cmap, origin='lower', aspect='equal', extent=extent, vmin=0, vmax=2)
        axes[0].set_title("A* Routes over Occupancy")
        for s in scores:
            route = s['traj']
            if route:
                # Convert grid coords to world coords for plotting
                coords = [self.grid_to_world(pt[0], pt[1]) for pt in route]
                x_vals = [c[0] for c in coords]
                z_vals = [c[1] for c in coords]
                alpha = 1.0 if s['idx'] == best_idx else 0.3
                color = 'red' if s['idx'] == best_idx else 'blue'
                axes[0].plot(x_vals, z_vals, color=color, alpha=alpha, linewidth=2)
                axes[0].scatter(x_vals[-1], z_vals[-1], marker='x', color='green', s=100) # Goal

        # 2. Similarity map + best trajectory - 'jet' Colormap
        # Normalize for visualization if not already
        sim_vis = self.bev_sim_map.copy()
        def normalize_sim(m):
            if m is None: return None
            non_zero_mask = m > 0
            if not np.any(non_zero_mask): return m
            m_min, m_max = m[non_zero_mask].min(), m[non_zero_mask].max()
            m_norm = np.zeros_like(m)
            if m_max > m_min:
                m_norm[non_zero_mask] = (m[non_zero_mask] - m_min) / (m_max - m_min)
            else:
                m_norm[non_zero_mask] = 1.0
            return m_norm

        sim_vis = normalize_sim(sim_vis)
        axes[1].imshow(sim_vis, cmap='jet', origin='lower', aspect='equal', extent=extent, vmin=0, vmax=1)
        axes[1].set_title("Best Trajectory over Similarity Map")
        best_coords = [self.grid_to_world(pt[0], pt[1]) for pt in best_traj]
        x_best = [c[0] for c in best_coords]
        z_best = [c[1] for c in best_coords]
        axes[1].plot(x_best, z_best, color='white', linewidth=3, alpha=0.8)
        axes[1].scatter(x_best[-1], z_best[-1], marker='*', color='white', s=200)

        # 3. Uncertainty Map + seen mask FOV shadow - 'magma' Colormap
        axes[2].imshow(self.bev_epi_map, cmap='magma', origin='lower', aspect='equal', extent=extent)
        axes[2].set_title(f"Information Gain Sweep (IG={best_score['ig']:.2f})")
        axes[2].plot(x_best, z_best, color='cyan', linewidth=3)
        # Overlay FOV Mask in green alpha
        mask_rgba = np.zeros((*seen_mask.shape, 4))
        mask_rgba[seen_mask] = [0, 1, 0, 0.3] # Green transparent
        axes[2].imshow(mask_rgba, origin='lower', aspect='equal', extent=extent)

        for ax in axes:
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Z (m)')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.show()
        plt.savefig("./planner_viz.png")

    def step(self, u: list, dt: float = 0.1) -> None:
        """Advance the simulator by one step via SimInterface.

        Parameters
        ----------
        u:
            ``[forward_velocity, yaw_angular_velocity]``
        dt:
            Integration time-step in seconds.
        """
        self.sim_iface.step(u, dt)

    def get_observations(self):
        """Read sensor data via SimInterface.

        Returns
        -------
        (rgb, depth, c2w) — same semantics as SimInterface.get_observations()
        """
        return self.sim_iface.get_observations()



        






# --------------------------------------------------------
# Main Execution
# --------------------------------------------------------
if __name__ == "__main__":

    cfg = Config("config/config.yaml")

    # 1. Init Habitat simulator
    sim, agent = init_simulator(cfg.scene_path, resolution=cfg.img_width, fov_deg=cfg.fov)
    spawn_agent_at_random_navpoint(sim, agent)
    scene_bounds = get_scene_bounds_from_pathfinder(sim)

    # 2. Build interfaces
    sim_iface = SimInterface(cfg, sim, agent)
    planner = Planner(cfg, sim_iface, scene_bounds)

    planner.load_ensemble()
    planner.set_umap(planner.load_umap(step=30000))
    planner.set_sim_map(planner.load_sim_map(step=30000))
    planner.set_occ_map(planner.load_occ_map(step=30000))
    
    # planner.viz_umap()
    planner.viz_sim_map()
    # planner.viz_occ_map()
    
    # Test planning
    start_world = (-1.0, -6.5)
    start_point = planner.world_to_grid(*start_world)
    
    print(f"Testing path planner from world {start_world} -> grid {start_point}")
    scores, best_idx = planner.plan_paths(start_point, num_modes=10,gamma=0,beta=100.0, alpha=0.0)
    
    if best_idx is not None:
        best_score = scores[best_idx]
        print(f"Best path found (Score: {best_score['score']:.4f})")
        planner.viz_paths(scores, best_idx)
        
    sim.close()