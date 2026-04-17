import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
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
from src.planning.pathfinder import PathFinder
from src.visualization.visualizer import Visualizer

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
        
        self.pathfinder = PathFinder(cfg, device=self.device)
    
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
        
    def load_ensemble(self, pretrained=True):
        """Loads the ensemble FeatureField models from the output directory."""
        ensemble_models = []
        ensemble_dir = os.path.join(self.cfg.output_dir, "ensemble")

        if pretrained:
            for i in range(self.cfg.ensemble_num_models):
                model_path = os.path.join(ensemble_dir, f"pretrained/pretrained_{i}.pt")
                if os.path.exists(model_path):
                    model = FeatureField(self.cfg, scene_bounds=self.scene_bounds, device=self.device)
                    model.load(model_path)
                    ensemble_models.append(model)
                    print(f"  -> Loaded Pretrained Ensemble Model")
                else:
                    print(f"  -> Warning: Pretrained Ensemble Model not found at {model_path}")
            return ensemble_models

        else:   
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
            fig, axes = plt.subplots(1, 1, figsize=(12,6))
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
        goals = self.pathfinder.get_gmm_goals(self.bev_sim_map, num_modes=num_modes)
        print(f"Testing {len(goals)} potential goals.")
        
        # 2. Plan paths using A*
        trajectories = []
        for goal in goals:
            path = self.pathfinder.astar(self.bev_occ_map, start_pos, goal)
            if path is not None:
                trajectories.append(path)
            else:
                print(f"Goal {goal} is unreachable.")
                
        # 3. Score paths
        scores, best_idx, _ = self.pathfinder.score_trajectories(
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

    visualizer = Visualizer(cfg) 
    frames = []

    for k in range(int(cfg.iterations/cfg.viz_interval)+1):
        print(k)
        step = k*cfg.viz_interval
        planner.load_ensemble()
        umap = planner.load_umap(step=step)
        omap = planner.load_occ_map(step=step)
        smap = planner.load_sim_map(step=step)

        planner.set_umap(umap)
        planner.set_occ_map(omap)
        planner.set_sim_map(smap)
        
        # planner.viz_umap()
        # planner.viz_sim_map()
        # planner.viz_occ_map()
        
        # Test planning
        start_world = (-1.0, -6.5)
        start_point = planner.world_to_grid(*start_world)
        
        print(f"Testing path planner from world {start_world} -> grid {start_point}")
        scores, best_idx = planner.plan_paths(start_point, num_modes=10,gamma=0,beta=100.0, alpha=0.0)
        
        
        if best_idx is not None:
            best_score = scores[best_idx]
            print(f"Best path found (Score: {best_score['score']:.4f})")

            extent = [planner.sim_map.min_x, planner.sim_map.max_x, planner.sim_map.min_z, planner.sim_map.max_z]
            fig = visualizer.plot_planner_paths(planner.bev_occ_map, planner.bev_sim_map, planner.bev_epi_map,
            extent, scores, best_idx, save_path=f"./figs/planner_viz_{step}.png", grid_to_world_fn=planner.grid_to_world)
            frames.append(visualizer.fig_to_numpy(fig))

    visualizer.create_video(frames, "./figs/planner_viz.mp4", fps=2)

    sim.close()