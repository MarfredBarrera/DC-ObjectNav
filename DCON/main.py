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
        self.output_dir = cfg.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/rgbs", exist_ok=True)
        os.makedirs(f"{self.output_dir}/depth_data", exist_ok=True)
        os.makedirs(f"{self.output_dir}/depth_vis", exist_ok=True)
        
        # Camera parameters
        self.IMG_WIDTH = cfg.img_width
        self.IMG_HEIGHT = cfg.img_height
        self.FOV_DEG = cfg.fov

        # Initialize semantics and hashgrid
        self.sam_clip = SAM_CLIP_Semantics(self.cfg, device=self.device)
        
        print("Initializing HashGrid...")
        self.hashgrid = HashGrid(self.cfg, device=self.device)

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
        
        # Training state
        self.replay_buffer = deque(maxlen=self.cfg.hash_replay_buffer_size)
        self.training_step = 0
        self.training_active = False
        self.extraction_thread = None
        self.training_thread = None
        self.data_queue = queue.Queue(maxsize=self.cfg.data_queue_size)
        self.training_queue = queue.Queue(maxsize=self.cfg.data_queue_size)
        
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
    
    def _create_telemetry_panel(self, width=400, height=512):
        """Create a telemetry information panel"""
        panel = np.ones((height, width, 3), dtype=np.uint8) * 40  # Dark background
        
        # Title
        cv2.putText(panel, "TELEMETRY", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.line(panel, (10, 40), (width - 10, 40), (100, 100, 100), 1)
        
        # Frame info
        y_pos = 70
        cv2.putText(panel, "Frame Count:", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(panel, f"{self.frame_count}", (width - 100, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Buffer info
        y_pos += 35
        cv2.putText(panel, "Replay Buffer:", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        buffer_text = f"{len(self.replay_buffer)}/{self.cfg.hash_replay_buffer_size}"
        cv2.putText(panel, buffer_text, (width - 120, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Buffer bar
        bar_x, bar_y, bar_w, bar_h = 10, y_pos + 10, width - 20, 20
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 1)
        if self.cfg.hash_replay_buffer_size > 0:
            fill_w = int(bar_w * len(self.replay_buffer) / self.cfg.hash_replay_buffer_size)
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 200, 255), -1)
        
        # Training status
        y_pos += 55
        cv2.putText(panel, "Training Status:", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        status_text = "ON" if self.training_active else "OFF"
        status_color = (0, 255, 0) if self.training_active else (0, 0, 255)
        cv2.putText(panel, status_text, (width - 80, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Training step
        y_pos += 35
        cv2.putText(panel, "Training Step:", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(panel, f"{self.training_step}", (width - 120, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Queue status
        y_pos += 35
        cv2.putText(panel, "Data Queue:", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(panel, f"{self.data_queue.qsize()}/{self.cfg.data_queue_size}", (width - 100, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_pos += 35
        cv2.putText(panel, "Training Queue:", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(panel, f"{self.training_queue.qsize()}/{self.cfg.training_queue_size}", (width - 100, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Separator
        y_pos += 25
        cv2.line(panel, (10, y_pos), (width - 10, y_pos), (100, 100, 100), 1)
        
        # Camera position
        y_pos += 30
        state = self.agent.get_state()
        pos = state.position
        cv2.putText(panel, "Camera Position:", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_pos += 25
        cv2.putText(panel, f"X: {pos[0]:6.2f}", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
        cv2.putText(panel, f"Y: {pos[1]:6.2f}", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
        cv2.putText(panel, f"Z: {pos[2]:6.2f}", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # # Controls reminder
        # y_pos = height - 120
        # cv2.line(panel, (10, y_pos - 10), (width - 10, y_pos - 10), (100, 100, 100), 1)
        # cv2.putText(panel, "CONTROLS:", (10, y_pos),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        # y_pos += 25
        # cv2.putText(panel, "[W] Forward", (10, y_pos),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        # y_pos += 20
        # cv2.putText(panel, "[A] Turn Left", (10, y_pos),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        # y_pos += 20
        # cv2.putText(panel, "[D] Turn Right", (10, y_pos),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        # y_pos += 20
        # cv2.putText(panel, "[T] Toggle Training", (10, y_pos),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        # y_pos += 20
        # cv2.putText(panel, "[Q] Quit", (10, y_pos),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return panel
    
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

            # Create agent view and telemetry panel
            cv2_img = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
            agent_view = cv2.resize(cv2_img, (640, 512))
            
            # Add simple label to agent view
            cv2.putText(agent_view, "AGENT VIEW", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Create telemetry panel
            telemetry_panel = self._create_telemetry_panel(width=400, height=512)
            
            # Combine horizontally
            combined_view = np.hstack([agent_view, telemetry_panel])
            
            cv2.imshow("Habitat Explorer - Agent View & Telemetry", combined_view)

            try:
                if self.frame_count % self.viz_update_interval == 0:
                    self.data_queue.put_nowait((rgb,depth,current_matrix))
                    
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

            # Non-blocking key detection (1ms wait)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("Quit requested")
                break
            elif key == ord('w'):
                self.simulator.step("move_forward")
                print("Action: Move Forward")
                self.frame_count += 1
            elif key == ord('a'):
                self.simulator.step("turn_left")
                print("Action: Turn Left")
                self.frame_count += 1
            elif key == ord('d'):
                self.simulator.step("turn_right")
                print("Action: Turn Right")
                self.frame_count += 1
            elif key == ord('t'):
                if self.training_active:
                    self.stop_training()
                else:
                    self.start_training()
            else:
                self.frame_count += 1
            
            # # Small delay to prevent CPU spinning
            # time.sleep(0.01)

            time.sleep(0.2)

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
        # Habitat returns uint8 [0, 255] RGBA
        # Convert to RGB only (remove alpha channel)
        if rgb.shape[-1] == 4:
            rgb_np = rgb[:, :, :3]  # Just remove alpha, keep as uint8
        else:
            rgb_np = rgb
            
        # Ensure it's uint8 in [0, 255] for SAM
        if rgb_np.dtype != np.uint8:
            if rgb_np.max() <= 1.0:
                rgb_np = (rgb_np * 255).astype(np.uint8)
            else:
                rgb_np = rgb_np.astype(np.uint8)
        
        # Clear GPU cache before heavy SAM processing
        torch.cuda.empty_cache()
        
        # Extract CLIP features (returns torch tensor on GPU)
        print(f"Extracting features from {rgb_np.shape} image...")
        try:
            clip_features = self.sam_clip.extract_dense_features(rgb_np)
            
            # IMMEDIATELY move to CPU to free GPU memory for SAM
            clip_features_cpu = clip_features.cpu()
            del clip_features
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA OOM during feature extraction: {e}")
            print("Try reducing SAM settings in config or killing other GPU processes")
            torch.cuda.empty_cache()
            raise
        
        # Convert depth to CPU tensor first
        depth_tensor = torch.from_numpy(depth).float()
        
        # Habitat (OpenGL) -> GSplat (OpenCV) coordinate conversion
        convert_mat = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
        c2w_cv = pose_matrix @ convert_mat
        c2w_cpu = torch.from_numpy(c2w_cv).float()
        
        # Do unprojection on CPU to avoid GPU memory issues
        mask = (depth_tensor > 0.1) & (depth_tensor < 10.0)
        
        # Move only what we need to GPU for unprojection
        depth_gpu = depth_tensor.to(self.device)
        c2w_gpu = c2w_cpu.to(self.device)
        
        world_points = unprojection(depth_gpu, self.intrinsics_tuple, c2w_gpu, self.device, mask=mask)
        
        # Free GPU tensors immediately
        del depth_gpu, c2w_gpu
        torch.cuda.empty_cache()
        
        # Move world_points to CPU, apply mask on CPU
        world_points_cpu = world_points.cpu()
        del world_points
        torch.cuda.empty_cache()
        
        # Apply mask to features on CPU
        gt_features_cpu = clip_features_cpu[mask]
        del clip_features_cpu
        
        # Filter zero-norm features on CPU
        valid_mask = gt_features_cpu.norm(dim=-1) > 1e-6
        world_points_final = world_points_cpu[valid_mask]
        gt_features_final = gt_features_cpu[valid_mask]
        
        del world_points_cpu, gt_features_cpu        
        # Return CPU tensors - they'll be moved to GPU in training worker
        return world_points_final, gt_features_final, rgb_np, depth, c2w_cv
    
    def _extract_semantics(self):

        while self.training_active:
            try:
                new_data = self.data_queue.get_nowait()
                if new_data is not None:
                    rgb, depth, current_matrix = new_data
                    world_points_cpu, gt_features_cpu, _, _, _ = self._process_frame(
                        rgb, depth, current_matrix)
                    self.training_queue.put_nowait((world_points_cpu, gt_features_cpu))
                    print(f"Enqueued {world_points_cpu.shape[0]} points for training")
                    time.sleep(0.1)
            except queue.Empty:
                pass

    def _training_worker(self):
        """Background thread for continuous training"""
        print("Training worker started...")
        batch_size = self.cfg.hash_train_batch_size
        
        while self.training_active:
            # Get data from queue if available
            try:
                new_training_data = self.training_queue.get_nowait()
                if new_training_data is not None:
                    world_points_cpu, gt_features_cpu = new_training_data
                    world_points_gpu = world_points_cpu.to(self.device)
                    gt_features_gpu = gt_features_cpu.to(self.device)

                    self.replay_buffer.append((world_points_gpu, gt_features_gpu))
                    print(f"Buffer size: {len(self.replay_buffer)}/{self.cfg.hash_replay_buffer_size}")
            except queue.Empty:
                pass
            
            # Check if we have enough data in buffer
            if len(self.replay_buffer) < 3:
                time.sleep(0.1)
                continue
            
            # Prepare training batch from replay buffer
            try:
                world_points = torch.cat([x[0] for x in self.replay_buffer], dim=0)
                gt_features = torch.cat([x[1] for x in self.replay_buffer], dim=0)
            except Exception as e:
                print(f"Error concatenating buffer: {e}")
                continue
            
            if world_points.shape[0] < batch_size:
                print(f"Not enough points for batch: {world_points.shape[0]} < {batch_size}")
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
            self.extraction_thread = threading.Thread(target=self._extract_semantics, daemon=True)
            self.extraction_thread.start()
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

    def stop_extraction(self):
        """Stop background extraction thread"""
        if self.extraction_thread:
            self.extraction_thread.join(timeout=2.0)
            print("Background extraction stopped!")
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_training()
        self.stop_extraction()
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