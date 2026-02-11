import os
import json
import math
import time
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque
import queue

# Custom imports
from dev.config import Config
from dev.semantics import SAM_CLIP_Semantics
from dev.utils import unprojection
from dev.hashgrid import HashGrid
from dev.visualizer import Visualizer

import threading

# Habitat imports
import habitat_sim
import habitat_sim.utils.common as utils

# Silence habitat-sim warnings and logs
os.environ['GLOG_minloglevel'] = '2'
os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['HABITAT_SIM_LOG'] = 'quiet'
os.environ['CUDA_VISIBLE_DEVICES'] = '2' 


class HabitatSim:
    def __init__(self, cfg: Config, scene_path: str):
        self.cfg = cfg
        self.scene_path = scene_path
        
        # Setup device
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cfg.gpu_indices
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"
        self.device = self.cfg.device
        
        # Output directory
        self.output_dir = cfg.scene_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/rgbs", exist_ok=True)
        os.makedirs(f"{self.output_dir}/depth_data", exist_ok=True)
        os.makedirs(f"{self.output_dir}/depth_vis", exist_ok=True)
        
        # Camera parameters
        self.IMG_WIDTH = 720
        self.IMG_HEIGHT = 720
        self.FOV_DEG = 90.0
        
        # Initialize Habitat simulator
        self._init_simulator()
        
        # Initialize intrinsics
        fov_rad = np.deg2rad(self.FOV_DEG)
        self.fx = (self.IMG_WIDTH / 2) / np.tan(fov_rad / 2)
        self.fy = self.fx
        self.cx = self.IMG_WIDTH / 2
        self.cy = self.IMG_HEIGHT / 2
        self.intrinsics_tuple = (self.fx, self.fy, self.cx, self.cy, self.IMG_HEIGHT, self.IMG_WIDTH)
        
        # Data storage
        self.rgb_imgs = []
        self.depth_imgs = []
        self.pose_matrices = []
        self.frame_count = 0
        
        # Initialize semantics and hashgrid
        print("Initializing SAM-CLIP...")
        self.sam_clip = SAM_CLIP_Semantics(self.cfg, device=self.device)
        
        print("Initializing HashGrid...")
        self.hashgrid = HashGrid(self.cfg, device=self.device)
        
        # Training state
        self.replay_buffer = deque(maxlen=self.cfg.hash_replay_buffer_size)
        self.training_step = 0
        self.training_active = False
        self.training_thread = None
        self.data_queue = queue.Queue(maxsize=10)
        
        # Visualization
        self.visualizer = None
        self.last_viz_update = 0
        self.viz_update_interval = 5  # Update visualization every 5 frames
        
        print("\nInitialization complete!")

    def _init_simulator(self):
        # Create Habitat-Sim configuration
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = self.scene_path
        sim_cfg.enable_physics = False
        sim_cfg.load_semantic_mesh = False

        # RGB Sensor
        rgb_sensor = habitat_sim.CameraSensorSpec()
        rgb_sensor.uuid = "rgb"
        rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor.resolution = [self.IMG_WIDTH, self.IMG_HEIGHT]
        rgb_sensor.position = [0.0, 1.5, 0.0]
        rgb_sensor.orientation = [0.0, 0.0, 0.0]

        # Depth Sensor
        depth_sensor = habitat_sim.CameraSensorSpec()
        depth_sensor.uuid = "depth"
        depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor.resolution = [self.IMG_WIDTH, self.IMG_HEIGHT]
        depth_sensor.position = [0.0, 1.5, 0.0]
        depth_sensor.orientation = [0.0, 0.0, 0.0]

        # Agent Configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_sensor, depth_sensor]
        agent_cfg.action_space = {
            "move_forward": habitat_sim.ActionSpec(
                "move_forward", habitat_sim.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.ActionSpec(
                "turn_left", habitat_sim.ActuationSpec(amount=10.0)
            ),
            "turn_right": habitat_sim.ActionSpec(
                "turn_right", habitat_sim.ActuationSpec(amount=10.0)
            ),
        }

        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

        try:
            self.simulator = habitat_sim.Simulator(cfg)
            self.agent = self.simulator.initialize_agent(0)
            
            # Set initial position
            if self.simulator.pathfinder.is_loaded:
                nav_point = self.simulator.pathfinder.get_random_navigable_point()
                agent_state = habitat_sim.AgentState()
                agent_state.position = nav_point
                self.agent.set_state(agent_state)
                print(f"Agent spawned at: {nav_point}")
                
                # Get and set scene bounds for HashGrid
                bounds = self.simulator.pathfinder.get_bounds()
                self.hashgrid.bounds_min = torch.tensor(bounds[0], device=self.device, dtype=torch.float32)
                self.hashgrid.bounds_max = torch.tensor(bounds[1], device=self.device, dtype=torch.float32)
            else:
                print("Warning: No navmesh found. Agent spawned at origin.")
                
        except Exception as e:
            print(f"Error loading simulator: {e}")
            raise e
        
    def _get_camera_matrix(self):
        """Get camera transformation matrix from Habitat agent"""
        state = self.agent.get_state().sensor_states['rgb']
        rot_quat = state.rotation
        translation = state.position
        
        rot_mat = utils.quat_to_magnum(rot_quat).to_matrix()
        rot_mat = np.array(rot_mat)
        
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rot_mat
        transform_matrix[:3, 3] = translation
        
        return transform_matrix
    
    def run_exploration(self):
        """Exploration loop with keyboard controls"""
        print("\n" + "="*40)
        print(" COMMANDS:")
        print("  [w]    : Move Forward")
        print("  [a]    : Turn Left")
        print("  [d]    : Turn Right")
        print("  [t]    : Toggle training on/off")
        print("  [Q] or [ESC]  : Quit")
        print("="*40 + "\n")

        self.start_training()

        while True:
            obs = self.simulator.get_sensor_observations()    
            rgb = obs["rgb"]
            depth = obs["depth"]

            current_matrix = self._get_camera_matrix()

            # Window display of RGB
            cv2_img = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
            small_img = cv2.resize(cv2_img, (512, 512))
            
            # Add info overlay
            info_img = small_img.copy()
            cv2.putText(info_img, f"Frame: {self.frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(info_img, f"Buffer: {len(self.replay_buffer)}/{self.cfg.hash_replay_buffer_size}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(info_img, f"Training: {'ON' if self.training_active else 'OFF'}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 255, 0) if self.training_active else (0, 0, 255), 2)
            cv2.putText(info_img, f"Train Step: {self.training_step}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Habitat Agent View", info_img)

            try:
                world_points, gt_features, rgb_np, depth_np, c2w_cv = self._process_frame(
                                rgb, depth, current_matrix)
                
                # Add to training queue (move to CPU to avoid GPU memory issues)
                self.data_queue.put_nowait((world_points.cpu(), gt_features.cpu()))
                
                # Save RGB and depth
                self.rgb_imgs.append(cv2_img)
                self.depth_imgs.append(depth)
                self.pose_matrices.append(current_matrix)
                
            except queue.Full:
                print("Queue full, skipping frame")
            except Exception as e:
                print(f"Frame processing error: {e}")
                import traceback
                traceback.print_exc()

            print("Waiting for input...")
            key = cv2.waitKey(0)
            
            print(f"Key pressed: {key}")
            
            if key == ord('q'):
                break
            elif key == ord('w'):
                self.simulator.step("move_forward")
                print("Action: Move Forward")
            elif key == ord('a'):
                self.simulator.step("turn_left")
                print("Action: Turn Left")
            elif key == ord('d'):
                self.simulator.step("turn_right")
                print("Action: Turn Right")
            elif key == ord('t'):
                if self.training_active:
                    self.stop_training()
                else:
                    self.start_training()

            self.frame_count += 1

    def save_data(self):
        """Save all collected data and models"""
        print(f"\nSaving {self.frame_count} frames and transforms...")
        
        # Calculate scene bounds
        if self.simulator.pathfinder.is_loaded:
            bounds = self.simulator.pathfinder.get_bounds()
            scene_bounds = {
                "min": np.array(bounds[0]).tolist(),
                "max": np.array(bounds[1]).tolist()
            }
        else:
            all_positions = np.array([pose[:3, 3] for pose in self.pose_matrices])
            scene_min = all_positions.min(axis=0)
            scene_max = all_positions.max(axis=0)
            padding = (scene_max - scene_min) * 0.2
            scene_min -= padding
            scene_max += padding
            scene_bounds = {
                "min": scene_min.tolist(),
                "max": scene_max.tolist()
            }
        
        # Save frames and create transforms.json
        frames_data = []
        fov_rad = np.deg2rad(self.FOV_DEG)
        
        for idx, (cv2_img, depth, pose) in enumerate(zip(self.rgb_imgs, self.depth_imgs, self.pose_matrices)):
            # Save RGB and depth
            rgb_rel_path = f"rgbs/rgb_{idx:03d}.png"
            cv2.imwrite(f"{self.output_dir}/{rgb_rel_path}", cv2_img)
            np.save(f"{self.output_dir}/depth_data/depth_{idx:03d}.npy", depth)
            
            # Depth visualization
            depth_vis = np.clip(depth * 255 / 10.0, 0, 255).astype(np.uint8)
            cv2.imwrite(f"{self.output_dir}/depth_vis/depth_vis_{idx:03d}.png", depth_vis)
            
            frames_data.append({
                "file_path": rgb_rel_path,
                "transform_matrix": pose.tolist()
            })
        
        # Create transforms.json
        json_data = {
            "camera_angle_x": fov_rad,
            "fl_x": self.fx,
            "fl_y": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "w": self.IMG_WIDTH,
            "h": self.IMG_HEIGHT,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "scene_bounds": scene_bounds,
            "frames": frames_data
        }
        
        with open(f"{self.output_dir}/transforms.json", "w") as f:
            json.dump(json_data, f, indent=4)
        
        print(f"Saved transforms.json with {len(frames_data)} frames")
        
        # Save HashGrid model
        hashgrid_path = f"{self.output_dir}/hashgrid_model.pt"
        self.hashgrid.save(hashgrid_path)
        print(f"Saved HashGrid model to {hashgrid_path}")

    def _process_frame(self, rgb, depth, pose_matrix):
        """Process and extract features from a frame"""
        # Convert RGB for CLIP - remove alpha channel if present
        if rgb.shape[-1] == 4:
            rgb_np = (rgb[:, :, :3] * 255).astype(np.uint8)
        else:
            rgb_np = (rgb * 255).astype(np.uint8)
        
        # Extract CLIP features (returns torch tensor)
        print(f"Extracting features from {rgb_np.shape} image...")
        clip_features = self.sam_clip.extract_dense_features(rgb_np)
        
        # clip_features is already a torch tensor on GPU with shape (H, W, feature_dim)
        # No resizing needed - same as training.py
        
        # Convert depth to tensor
        depth_tensor = torch.from_numpy(depth).float().to(self.device)
        
        # Habitat (OpenGL) -> GSplat (OpenCV) coordinate conversion
        convert_mat = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
        c2w_cv = pose_matrix @ convert_mat
        c2w_tensor = torch.from_numpy(c2w_cv).float().to(self.device)
        
        # Unproject to world coordinates
        mask = (depth_tensor > 0.1) & (depth_tensor < 10.0)
        world_points = unprojection(depth_tensor, self.intrinsics_tuple, c2w_tensor, self.device, mask=mask)
        gt_features = clip_features[mask]
        
        # Filter zero-norm features
        valid_mask = gt_features.norm(dim=-1) > 1e-6
        world_points = world_points[valid_mask]
        gt_features = gt_features[valid_mask]
        
        print(f"Extracted {world_points.shape[0]} valid points with features")
        
        return world_points, gt_features, rgb_np, depth, c2w_cv

    def _training_worker(self):
        """Background thread for continuous training"""
        print("Training worker started...")
        batch_size = self.cfg.hash_train_batch_size
        
        while self.training_active:
            # Check if we have enough data
            if len(self.replay_buffer) < 3:
                time.sleep(0.1)
                continue
            
            # Get data from queue if available
            try:
                new_data = self.data_queue.get_nowait()
                if new_data is not None:
                    # Move data to GPU for training
                    world_points_cpu, gt_features_cpu = new_data
                    world_points_gpu = world_points_cpu.to(self.device)
                    gt_features_gpu = gt_features_cpu.to(self.device)
                    self.replay_buffer.append((world_points_gpu, gt_features_gpu))
                    print(f"Buffer size: {len(self.replay_buffer)}")
            except queue.Empty:
                pass
            
            # Prepare training batch from replay buffer
            world_points = torch.cat([x[0] for x in self.replay_buffer], dim=0)
            gt_features = torch.cat([x[1] for x in self.replay_buffer], dim=0)
            
            if world_points.shape[0] < batch_size:
                time.sleep(0.1)
                continue
            
            # Sample batch
            batch_idx = torch.randperm(world_points.shape[0], device=self.device)[:batch_size]
            batch_points = world_points[batch_idx]
            batch_features = gt_features[batch_idx]
            
            # Training step
            loss = self.hashgrid.train_step(batch_points, batch_features)
            
            if loss is not None and self.training_step % 50 == 0:
                print(f"Training Step {self.training_step:04d} | Loss: {loss:.5f}")
            
            self.training_step += 1
            
            # Small sleep to prevent GPU overload
            time.sleep(0.01)

    def start_training(self):
        """Start background training thread"""
        if not self.training_active:
            self.training_active = True
            self.training_thread = threading.Thread(target=self._training_worker, daemon=True)
            self.training_thread.start()
            print("Background training started!")

    def stop_training(self):
        """Stop background training thread"""
        if self.training_active:
            self.training_active = False
            if self.training_thread:
                self.training_thread.join(timeout=2.0)
            print("Background training stopped!")

    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_training()
        cv2.destroyAllWindows()
        self.simulator.close()

def main():
    config = Config("./config/config.yaml")
    runner = HabitatSim(config, scene_path="/workspace/DCON/gibson_scenes/Anaheim.glb")
    
    try:
        runner.run_exploration()
    finally:
        runner.save_data()
        runner.cleanup()
        print("\n" + "="*60)
        print("Session complete! Data saved and ready for analysis.")
        print("="*60)


if __name__ == "__main__":
    main()