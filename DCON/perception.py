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

# Habitat Imports
import habitat_sim
import habitat_sim.utils.common as utils
import habitat_sim.physics as physics

# User Dev Imports
from src.grid import UncertaintyGrid
from src.config import Config
from src.gaussians import GaussianSplatting
from src.semantics import SAM_CLIP_Semantics
from src.utils import unprojection
from src.featurefield import FeatureField

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



class Runner:
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

    def sample_rgb(self):
        """Extracts features and unprojects points from the current agent observation."""
        # 1. Fetch current frame from Habitat
        rgb, depth, c2w_hash = self.get_observations()

        # 2. Move specific tensors to GPU for processing
        depth = depth.to(self.device)
        c2w_hash = c2w_hash.to(self.device)

        rgb_np = (rgb.numpy() * 255).astype(np.uint8)

        # 3. Feature extraction on GPU
        clip_features = self.sam_clip.extract_dense_features(rgb_np)
        
        # 4. Unprojection on GPU
        mask = (depth > 0.1) & (depth < 10.0)
        world_points = unprojection(depth, self.intrinsics_tuple, c2w_hash, self.device, mask=mask)
        gt_features = clip_features[mask]

        # 5. Filter zero-norm features
        valid_mask = gt_features.norm(dim=-1) > 1e-6
        world_points = world_points[valid_mask]
        gt_features = gt_features[valid_mask]

        # Return CPU tensors to save VRAM
        return world_points.cpu(), gt_features.cpu()

    def _super_batch(self, global_point_buffer, staging_size, recent_sample_portion):
        """
        Prepare a super-batch for training by sampling from the global point buffer.
        
        Args:
            global_point_buffer: List of (points, features) tuples from observation history
            staging_size: Total number of points to stage on GPU
            recent_sample_portion: Fraction of staging_size to sample from most recent frame
            
        Returns:
            super_points_gpu: Concatenated points tensor on GPU (or None if insufficient data)
            super_features_gpu: Concatenated features tensor on GPU (or None if insufficient data)
        """
        gpu_points_chunks = []
        gpu_features_chunks = []
        
        # A. Sample from the most recent frame
        recent_points, recent_features = global_point_buffer[-1]
        staging_size_recent = int(staging_size * recent_sample_portion)
        
        if recent_points.shape[0] > 0:
            num_samples_recent = min(staging_size_recent, recent_points.shape[0])
            indices = torch.randint(0, recent_points.shape[0], (num_samples_recent,))
            
            gpu_points_chunks.append(recent_points[indices].to(self.device))
            gpu_features_chunks.append(recent_features[indices].to(self.device))

        # B. Sample from historical frames
        staging_size_history = staging_size - staging_size_recent
        history_buffer = global_point_buffer[:-1]
        
        if history_buffer and staging_size_history > 0:
            points_per_frame = staging_size_history // len(history_buffer)
            for pts, fts in history_buffer:
                if pts.shape[0] > 0:
                    num_to_sample = min(points_per_frame, pts.shape[0])
                    indices = torch.randint(0, pts.shape[0], (num_to_sample,))
                    
                    gpu_points_chunks.append(pts[indices].to(self.device))
                    gpu_features_chunks.append(fts[indices].to(self.device))
        
        # C. Concatenate into GPU super-batch
        if not gpu_points_chunks:
            print("Warning: Cannot create super-batch, not enough data.")
            return None, None
        else:
            super_points_gpu = torch.cat(gpu_points_chunks, dim=0)
            super_features_gpu = torch.cat(gpu_features_chunks, dim=0)
            print(f"Staged {super_points_gpu.shape[0]} points to GPU.")
            return super_points_gpu, super_features_gpu

    def train_ensemble(self, save_enabled=False):
        viz_interval = self.cfg.viz_interval
        mini_batch_size = self.cfg.hash_train_batch_size
        
        recent_sample_portion = 0.2
        global_point_buffer = []
        max_buffer_frames = self.cfg.hash_replay_buffer_size
        min_frames_to_start = 3

        print(f"Initializing history buffer (start training after {min_frames_to_start} frames)...")
        for i in range(min_frames_to_start):
            # Advance Simulator: Spinning in a circle test [lin_vel=0, ang_vel=2.5]
            u = [0.0, 3.0]
            self.step_simulator(u)
            
            # Gather state observations
            world_points, gt_features = self.sample_rgb()
            global_point_buffer.append((world_points, gt_features))
            print(f"  Buffered frame {i+1}/{min_frames_to_start}")
            torch.cuda.empty_cache()

        refresh_interval = self.cfg.hash_buffer_refresh_interval
        staging_size = mini_batch_size * refresh_interval

        super_points_gpu = None
        super_features_gpu = None

        start_time = time.time()

        for step in range(self.cfg.iterations + 1):
            if step % refresh_interval == 0:
                if step > 0:
                    print(f"Refreshing history buffer...")
                    if len(global_point_buffer) >= max_buffer_frames:
                        old_points, old_features = global_point_buffer.pop(0)
                        del old_points, old_features
                        gc.collect() 
                        torch.cuda.empty_cache() 

                    # Control Policy Query (Spinning for now)
                    u = [0.0, 3.0]
                    self.step_simulator(u)
                    
                    new_points, new_features = self.sample_rgb()
                    global_point_buffer.append((new_points, new_features))
                    gc.collect()
                    torch.cuda.empty_cache() 
                    
                    buffer_status = f"{len(global_point_buffer)}/{max_buffer_frames}"
                    print(f"Buffer updated (frames: {buffer_status}, total frames processed: {self.frames_processed})")

                if super_points_gpu is not None:
                    del super_points_gpu, super_features_gpu
                    torch.cuda.empty_cache()
                
                print(f"\n--- Staging new super-batch for steps {step}-{step+refresh_interval-1} ---")
                
                # Prepare super-batch
                super_points_gpu, super_features_gpu = self._super_batch(
                    global_point_buffer, staging_size, recent_sample_portion
                )

                gc.collect()
                torch.cuda.empty_cache()

            if super_points_gpu is None or super_points_gpu.shape[0] < mini_batch_size:
                continue

            # Batch Sampling from GPU Super-batch
            batch_idx = torch.randint(0, super_points_gpu.shape[0], (mini_batch_size,), device=self.device)
            batch_points = super_points_gpu[batch_idx]
            batch_features = super_features_gpu[batch_idx]

            # Forward / Train Step
            loss = 0
            for model in self.ensemble_models:
                train_loss = model.train_step(batch_points, batch_features)
                if train_loss is not None:
                    loss += train_loss
            avg_loss = loss / self.cfg.ensemble_num_models

            if step % 100 == 0:
                print(f"Step {step:04d} | Train Loss: {avg_loss:.5f} | Time: {time.time()-start_time:.1f}s")

            if save_enabled and step >= 0 and step % viz_interval == 0:
                self.save_uncertainty_snapshot(step)

    def save_uncertainty_snapshot(self, step):
        """
        Compute and save uncertainty maps at the current training step.
        
        Args:
            step: Current training iteration number
        """
        elapsed = self.ugrid.compute_and_save_uncertainty_snapshot(
            iteration=step,
            height_filter=(0.1, 2.0)
        )
        print(f"Uncertainty snapshot time: {elapsed:.6f}s")

    def save_models(self):
        for i, model in enumerate(self.ensemble_models):
            ensemble_path = os.path.join(self.cfg.output_dir, f"ensemble/featurefield_ensemble_{i}.pt")
            os.makedirs(os.path.dirname(ensemble_path), exist_ok=True)
            model.save(ensemble_path)
            print(f"Saved Ensemble Model {i} to {ensemble_path}")

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
    runner_config = Config("config/config.yaml")
    runner = Runner(runner_config, sim, agent)

    print("\nStarting Online Training. The agent will spin and sample frames automatically...")
    
    # 3. Train
    runner.train_ensemble(save_enabled=True)
    runner.save_models()

    # Cleanup
    cv2.destroyAllWindows()
    sim.close()
    print("Simulation and Training Complete.")