import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"

# Silence habitat-sim warnings and logs
os.environ['GLOG_minloglevel'] = '2'
os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['HABITAT_SIM_LOG'] = 'quiet'

import torch
import numpy as np
import matplotlib.pyplot as plt

# User Dev Imports
from src.grid import UncertaintyGrid, SimilarityGrid
from src.config import Config
from src.semantics import SAM_CLIP_Semantics
from src.featurefield import FeatureField
from src.habitat_utils import (
    make_cfg,
    get_scene_bounds_from_pathfinder,
    spawn_agent_at_random_navpoint,
    init_simulator,
)
from src.sim_interface import SimInterface

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

    def set_umap(self, step=20000):
        epi_map, _ = self.load_umap(step=step)
        self.ugrid.bev_epi_umap = epi_map
    
    def load_umap(self, step=20000):
        umaps_dir = os.path.join(self.cfg.output_dir, "umaps")
        epi_path = os.path.join(umaps_dir, f"bev_epistemic_uncertainty_{step}.npy")
        ale_path = os.path.join(umaps_dir, f"bev_aleatoric_uncertainty_{step}.npy")
        
        if os.path.exists(epi_path) and os.path.exists(ale_path):
            bev_epi_umap = np.load(epi_path)
            bev_ale_umap = np.load(ale_path)
            return bev_epi_umap, bev_ale_umap
        else:
            print(f"BEV maps for step {step} not found in {umaps_dir}")
            return None, None
    
    def get_sim_map(self):
        return self.sim_map.compute_similarity_map("a photo of a pillow", height_filter = (0.1,1.5))
    
    def set_sim_map(self, sim_map):
        self.sim_map.bev_similarity_map = sim_map
        
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

        bev_epi_2d = self.ugrid.bev_epi_umap
        if bev_epi_2d is not None:
            fig, axes = plt.subplots(figsize=(12,6))
            extent = [self.ugrid.bev_min_x, self.ugrid.bev_max_x, self.ugrid.bev_min_z, self.ugrid.bev_max_z]

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
        bev_sim_2d = self.sim_map.bev_similarity_map
        if bev_sim_2d is not None:
            fig, axes = plt.subplots(figsize=(12,6))
            extent = [self.sim_map.bev_min_x, self.sim_map.bev_max_x, self.sim_map.bev_min_z, self.sim_map.bev_max_z]

            # Normalize for visualization, excluding bottom 10%
            if bev_sim_2d.size > 0:
                vmin = np.percentile(bev_sim_2d, 10)
                vmax = bev_sim_2d.max()
            else:
                vmin, vmax = 0, 1

            # Similarity Map
            im1 = axes.imshow(bev_sim_2d, cmap='viridis', origin='lower', aspect='equal', extent=extent, vmin=vmin, vmax=vmax)
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
        else:
            print("Unable to visualize BEV maps due to missing data.")


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
    # planner.set_umap(step=30000)
    # planner.viz_umap()
    planner.set_sim_map(planner.get_sim_map())
    planner.viz_sim_map()

    sim.close()