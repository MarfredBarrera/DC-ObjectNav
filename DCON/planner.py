import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
import gc
import json
import math
import random
import time
import torch
import numpy as np
import cv2
import magnum as mn
import matplotlib.pyplot as plt

# User Dev Imports
from src.grid import UncertaintyGrid, SimilarityGrid
from src.config import Config
from src.gaussians import GaussianSplatting
from src.semantics import SAM_CLIP_Semantics
from src.utils import unprojection
from src.featurefield import FeatureField

# Habitat Imports
import habitat_sim
import habitat_sim.utils.common as utils
import habitat_sim.physics as physics


# Silence habitat-sim warnings and logs
os.environ['GLOG_minloglevel'] = '2'
os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['HABITAT_SIM_LOG'] = 'quiet'



# --------------------------------------------------------
# Habitat Configuration & Helpers
# --------------------------------------------------------
def get_camera_matrix(agent):
    # 1. Get the state of the specific sensor 'rgb'
    state = agent.get_state().sensor_states['rgb']
    # 2. Extract Rotation (Quaternion) and Translation (Vector)
    rot_quat = state.rotation
    translation = state.position
    # 3. Convert Quaternion to 3x3 Rotation Matrix
    rot_mat = utils.quat_to_magnum(rot_quat).to_matrix()
    rot_mat = np.array(rot_mat)
    # 4. Build 4x4 Matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rot_mat
    transform_matrix[:3, 3] = translation
    
    return transform_matrix

def make_cfg(scene_filepath):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_filepath
    sim_cfg.enable_physics = False
    sim_cfg.load_semantic_mesh = False

    # Define Sensors
    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "rgb"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [512, 512] 
    rgb_sensor.position = [0.0, 1.5, 0.0]
    rgb_sensor.orientation = [0.0, 0.0, 0.0]

    # Add depth sensor
    depth_sensor = habitat_sim.CameraSensorSpec()
    depth_sensor.uuid = "depth"
    depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor.resolution = [512, 512]
    depth_sensor.position = [0.0, 1.5, 0.0]  
    depth_sensor.orientation = [0.0, 0.0, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor, depth_sensor]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])



class Planner:
    def __init__(self, cfg: Config, sim, agent):
        self.cfg = cfg
        self.device = self.cfg.device
        
        # Simulation References
        self.sim = sim
        self.agent = agent
        self.frames_processed = 0
        
        # Velocity Controller Setup for continuous kinematics
        self.vel_control = physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        
        # Habitat (OpenGL: -Z forward, Y up) -> GSplat (OpenCV: +Z forward, -Y up)
        self.convert_mat = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])

        # Intrinsics (Calculated from Habitat Config)
        self.H, self.W = 512, 512
        fov_x = math.radians(90.0)
        self.fx = 0.5 * self.W / math.tan(0.5 * fov_x)
        self.fy = self.fx
        self.cx, self.cy = self.W / 2.0, self.H / 2.0
        self.intrinsics_tuple = (self.fx, self.fy, self.cx, self.cy, self.H, self.W)

        # Semantics and Ensemble Models
        self.sam_clip = SAM_CLIP_Semantics(self.cfg, device=self.device)
        self.ensemble_models = [
            FeatureField(self.cfg, device=self.device) 
            for _ in range(self.cfg.ensemble_num_models)
        ]

        self.ugrid = UncertaintyGrid(cfg, ensemble=self.ensemble_models)
        self.sim_map = SimilarityGrid(cfg, ensemble=self.ensemble_models, sam_clip=self.sam_clip)

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
                # Initialize a new FeatureField instance
                model = FeatureField(self.cfg, device=self.device)
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


    def step_simulator(self, u, dt=0.1):
        """
        Advances the agent kinematically using the control input u.
        u: [forward_velocity, yaw_angular_velocity]
        """
        lin_vel, ang_vel = u[0], u[1]
        
        # 1. Apply Continuous Velocity Commands (Habitat forward is -Z)
        self.vel_control.linear_velocity = np.array([0.0, 0.0, -lin_vel])
        self.vel_control.angular_velocity = np.array([0.0, ang_vel, 0.0])
        
        # 2. Integrate kinematics using Magnum bindings
        agent_state = self.agent.get_state()
        magnum_rotation = utils.quat_to_magnum(agent_state.rotation)
        magnum_translation = mn.Vector3(agent_state.position)
        rigid_state = habitat_sim.RigidState(magnum_rotation, magnum_translation)
        
        new_rigid_state = self.vel_control.integrate_transform(dt, rigid_state)
        
        agent_state.position = np.array(new_rigid_state.translation)
        agent_state.rotation = utils.quat_from_magnum(new_rigid_state.rotation) 
        self.agent.set_state(agent_state)

    def get_observations(self):
        """
        Grabs the current sensor data from the simulator and formats it for PyTorch.
        """
        obs = self.sim.get_sensor_observations()
        rgb_rgba = obs["rgb"]
        depth = obs["depth"]

        # Format for PyTorch
        rgb_rgb = rgb_rgba[..., :3] # Strip Alpha channel
        rgb_tensor = torch.from_numpy(rgb_rgb).float() / 255.0
        depth_tensor = torch.from_numpy(depth).float()

        c2w_hab = get_camera_matrix(self.agent)
        c2w_cv = c2w_hab @ self.convert_mat
        c2w_tensor = torch.from_numpy(c2w_cv).float()
        
        self.frames_processed += 1

        return rgb_tensor, depth_tensor, c2w_tensor



        






# --------------------------------------------------------
# Main Execution
# --------------------------------------------------------
if __name__ == "__main__":
    
    # 1. Init Habitat sim
    scene = "/workspace/DCON/gibson_scenes/Anaheim.glb"
    sim_config = make_cfg(scene)

    try:
        sim = habitat_sim.Simulator(sim_config)
    except Exception as e:
        print(f"Error loading simulator: {e}")
        exit()

    agent = sim.initialize_agent(0)

    if sim.pathfinder.is_loaded:
        nav_point = sim.pathfinder.get_random_navigable_point()
        initial_state = habitat_sim.AgentState()
        initial_state.position = nav_point
        agent.set_state(initial_state)
        print(f"Agent spawned at: {nav_point}")
    else:
        print("Warning: No navmesh found. Agent spawned at origin.")

    # 2. Init runner
    cfg = Config("config/config.yaml")
    planner = Planner(cfg, sim, agent)
    planner.load_ensemble()
    # planner.set_umap(step=30000)
    # planner.viz_umap()
    planner.set_sim_map(planner.get_sim_map())
    planner.viz_sim_map()
    

    sim.close()