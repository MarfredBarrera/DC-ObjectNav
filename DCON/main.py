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
import threading
import random

# Custom imports
from dev.config import Config
from dev.semantics import SAM_CLIP_Semantics
from dev.utils import unprojection
from dev.hashgrid import HashGrid
from dev.visualizer import Visualizer

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
        # CHANGED: Replaced deque with a standard list to hold ALL history on CPU
        self.global_point_buffer = [] 
        self.training_step = 0
        self.training_active = False
        self.extraction_thread = None
        self.training_thread = None
        
        # Queues
        self.data_queue = queue.Queue(maxsize=self.cfg.data_queue_size)
        self.training_queue = queue.Queue(maxsize=self.cfg.data_queue_size)
        
        # Thread safety
        self.buffer_lock = threading.Lock()
        
        # Visualization
        self.viz_active = False
        self.last_viz_update = 0
        self.viz_thread = None
        self.viz_update_interval = 5 
        
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
                   cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 2)
        cv2.line(panel, (10, 40), (width - 10, 40), (100, 100, 100), 1)
        
        # Frame info
        y_pos = 70
        cv2.putText(panel, "Frame Count:", (10, y_pos),
                   cv2.FONT_HERSHEY_TRIPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(panel, f"{self.frame_count}", (width - 100, y_pos),
                   cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255), 2)
        
        # Buffer info
        y_pos += 35
        cv2.putText(panel, "Total Samples:", (10, y_pos),
                   cv2.FONT_HERSHEY_TRIPLEX, 0.6, (200, 200, 200), 1)
        with self.buffer_lock:
            buf_len = len(self.global_point_buffer)
        cv2.putText(panel, f"{buf_len}", (width - 120, y_pos),
                   cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255), 2)
        
        # Training status
        y_pos += 55
        cv2.putText(panel, "Training Status:", (10, y_pos),
                   cv2.FONT_HERSHEY_TRIPLEX, 0.6, (200, 200, 200), 1)
        status_text = "ON" if self.training_active else "OFF"
        status_color = (0, 255, 0) if self.training_active else (0, 0, 255)
        cv2.putText(panel, status_text, (width - 80, y_pos),
                   cv2.FONT_HERSHEY_TRIPLEX, 0.6, status_color, 2)
        
        # Training step
        y_pos += 35
        cv2.putText(panel, "Training Step:", (10, y_pos),
                   cv2.FONT_HERSHEY_TRIPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(panel, f"{self.training_step}", (width - 120, y_pos),
                   cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255), 2)
        
        # Queue status
        y_pos += 35
        cv2.putText(panel, "Data Queue:", (10, y_pos),
                   cv2.FONT_HERSHEY_TRIPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(panel, f"{self.data_queue.qsize()}/{self.cfg.data_queue_size}", (width - 100, y_pos),
                   cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255), 2)
        
        y_pos += 35
        cv2.putText(panel, "Training Queue:", (10, y_pos),
                   cv2.FONT_HERSHEY_TRIPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(panel, f"{self.training_queue.qsize()}/{self.cfg.training_queue_size}", (width - 100, y_pos),
                   cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255), 2)
        
        # Separator
        y_pos += 25
        cv2.line(panel, (10, y_pos), (width - 10, y_pos), (100, 100, 100), 1)
        
        # Camera position
        y_pos += 30
        state = self.agent.get_state()
        pos = state.position
        cv2.putText(panel, "Camera Position:", (10, y_pos),
                   cv2.FONT_HERSHEY_TRIPLEX, 0.6, (200, 200, 200), 1)
        y_pos += 25
        cv2.putText(panel, f"X: {pos[0]:6.2f}", (20, y_pos),
                   cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
        cv2.putText(panel, f"Y: {pos[1]:6.2f}", (20, y_pos),
                   cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
        cv2.putText(panel, f"Z: {pos[2]:6.2f}", (20, y_pos),
                   cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
        
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
            agent_view = cv2.resize(cv2_img, (720, 720))
            
            cv2.putText(agent_view, "AGENT VIEW", (10, 30),
                       cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)
            
            telemetry_panel = self._create_telemetry_panel(width=400, height=720)
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

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:
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
            
            time.sleep(0.5)

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
        """Process and extract features from a frame, strictly returning CPU tensors"""
        if rgb.shape[-1] == 4:
            rgb_np = rgb[:, :, :3]
        else:
            rgb_np = rgb
            
        if rgb_np.dtype != np.uint8:
            if rgb_np.max() <= 1.0:
                rgb_np = (rgb_np * 255).astype(np.uint8)
            else:
                rgb_np = rgb_np.astype(np.uint8)
        
        torch.cuda.empty_cache()
        
        try:
            # clip_features comes from SAM as a GPU tensor usually
            clip_features = self.sam_clip.extract_dense_features(rgb_np)
            clip_features_cpu = clip_features.cpu()
            del clip_features
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA OOM: {e}")
            torch.cuda.empty_cache()
            raise
        
        depth_tensor = torch.from_numpy(depth).float()
        
        convert_mat = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
        c2w_cv = pose_matrix @ convert_mat
        c2w_cpu = torch.from_numpy(c2w_cv).float()
        
        depth_mask = (depth_tensor > 0.1) & (depth_tensor < 10.0)
        valid_indices = torch.nonzero(depth_mask.flatten(), as_tuple=False).squeeze(1)

        mask = torch.zeros(depth_tensor.numel(), dtype=torch.bool)
        mask[valid_indices] = True
        mask = mask.view_as(depth_tensor)
        
        # Temp move to GPU for fast unprojection
        depth_gpu = depth_tensor.to(self.device)
        c2w_gpu = c2w_cpu.to(self.device)
        
        world_points = unprojection(
            depth_gpu,
            self.intrinsics_tuple,
            c2w_gpu,
            self.device,
            mask=mask.to(self.device)
        )
        
        # Move immediately back to CPU
        world_points_cpu = world_points.cpu()
        del world_points, depth_gpu, c2w_gpu
        torch.cuda.empty_cache()
        
        gt_features_cpu = clip_features_cpu[mask]
        del clip_features_cpu
        
        valid_mask = gt_features_cpu.norm(dim=-1) > 1e-6
        world_points_final = world_points_cpu[valid_mask]
        gt_features_final = gt_features_cpu[valid_mask]
        
        # Returns all CPU tensors
        return world_points_final, gt_features_final, rgb_np, depth, c2w_cv
    
    def _extract_semantics(self):
        """gets frames from data queue, extracts semantics, then pushes result to training queue"""
        while self.training_active:
            try:
                new_data = self.data_queue.get_nowait()
                if new_data is not None:
                    rgb, depth, current_matrix = new_data
                    world_points_cpu, gt_features_cpu, _, _, _ = self._process_frame(
                        rgb, depth, current_matrix)
                    
                    # Push CPU tensors to the training queue
                    self.training_queue.put_nowait((world_points_cpu, gt_features_cpu))
                    time.sleep(0.01)
            except queue.Empty:
                time.sleep(0.1)

    def _sample_from_global_history(self, batch_size, num_frames_to_sample=4):
        """
        Randomly samples a batch from the global history buffer.
        
        Args: 
        batch_size (int): Number of points to sample in total.
        num_frames_to_sample (int): Number of frames to sample from the global history.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: Sampled points and features tensors.
        """
        with self.buffer_lock:
            total_frames = len(self.global_point_buffer)
            if total_frames == 0:
                return None, None
            
            # 1. Pick random frames
            sample_count = min(total_frames, num_frames_to_sample)
            frame_indices = np.random.choice(total_frames, size=sample_count, replace=False)
            
            # 2. Pre-calculate how many points to take from each frame
            # (Uniformly distribute the batch size across selected frames)
            points_per_frame = batch_size // sample_count
            remainder = batch_size % sample_count
            
            batch_points_list = []
            batch_features_list = []
            
            for i, idx in enumerate(frame_indices):
                pts, fts = self.global_point_buffer[idx]
                n_available = pts.shape[0]
                
                # Calculate how many to take for this frame
                n_take = points_per_frame + (1 if i < remainder else 0)
                
                # If frame has fewer points than we want, take all of them
                if n_available <= n_take:
                    batch_points_list.append(pts)
                    batch_features_list.append(fts)
                else:
                    # Randomly sample INDICES on CPU, then slice
                    # This is faster than concating huge arrays then slicing
                    rand_idx = torch.randperm(n_available)[:n_take]
                    batch_points_list.append(pts[rand_idx])
                    batch_features_list.append(fts[rand_idx])
        
        # 3. Concatenate smaller chunks (Much faster)
        batch_points = torch.cat(batch_points_list, dim=0)
        batch_features = torch.cat(batch_features_list, dim=0)
        
        # 4. Move to GPU
        return batch_points.to(self.device), batch_features.to(self.device)


    def _training_worker(self):
        """
        Background thread for continuous training.
        Optimized with a GPU Staging Buffer (Super-Batch) to reduce CPU-GPU transfer overhead.
        """
        print("Training worker started...")
        
        # Configuration
        mini_batch_size = self.cfg.hash_train_batch_size
        steps_per_stage = 50  # How many steps to train on one GPU chunk
        staging_size = mini_batch_size * steps_per_stage  # Size of the "Super-Batch"
        
        while self.training_active:
            # --- PHASE 1: Ingest New Data (CPU) ---
            # Always drain the queue first so our history is up to date
            data_added = 0
            while not self.training_queue.empty():
                try:
                    new_training_data = self.training_queue.get_nowait()
                    if new_training_data is not None:
                        with self.buffer_lock:
                            self.global_point_buffer.append(new_training_data)
                        data_added += 1
                except queue.Empty:
                    break
        

            # Check if we have enough data to start
            with self.buffer_lock:
                buffer_len = len(self.global_point_buffer)

            if buffer_len < 1:
                time.sleep(0.1)
                continue

            # --- PHASE 2: Create Super-Batch (CPU -> GPU) ---
            # We pull a large diversified chunk of history to the GPU.
            # This is the "expensive" step, but we only do it once every 50 steps.
            
            # Note: We sample from many frames (e.g., 32) to ensure the agent 
            # doesn't forget the past while learning the new room.
            super_points_gpu, super_features_gpu = self._sample_from_global_history(
                batch_size=staging_size, 
                num_frames_to_sample=10 
            )
            
            if super_points_gpu is None:
                time.sleep(0.1)
                continue

            # --- PHASE 3: Fast Inner Loop (GPU Only) ---
            # Train for multiple steps on the resident GPU data.
            # This loop is extremely fast because there is no PCIe transfer.
            
            total_super_samples = super_points_gpu.shape[0]
            
            # If we don't have enough data for a full stage, just train once
            current_stage_steps = steps_per_stage if total_super_samples >= mini_batch_size else 1
            
            for _ in range(current_stage_steps):
                if not self.training_active:
                    break

                # 1. Randomly slice a mini-batch from the Super-Batch (Instant)
                # We use randint to pick random indices from the GPU tensor
                batch_idx = torch.randint(0, total_super_samples, (mini_batch_size,), device=self.device)
                
                batch_p = super_points_gpu[batch_idx]
                batch_f = super_features_gpu[batch_idx]
                
                # 2. Train Step
                loss = self.hashgrid.train_step(batch_p, batch_f)
                
                # 3. Logging
                if loss is not None and self.training_step % 50 == 0:
                    print(f"Training Step {self.training_step:04d} | Loss: {loss:.5f}")
                
                self.training_step += 1
                
                # Tiny sleep to allow other GPU operations (like rendering) to sneak in
                time.sleep(0.002)

        
    def _offline_training(self):
        print("Starting offline training with Super-Batch Staging...")
        
        # Configuration
        batch_size = self.cfg.hash_train_batch_size
        # How many training steps to run on one GPU chunk before fetching new CPU data
        steps_per_stage = 50  
        # Size of the chunk to move to GPU (amortizes the transfer cost)
        staging_size = batch_size * steps_per_stage
        
        target_step = self.training_step + self.cfg.iterations
        
        # Ensure pending data is in global buffer
        while not self.training_queue.empty():
            try:
                self.global_point_buffer.append(self.training_queue.get_nowait())
            except:
                break
                
        print(f"Total history size: {len(self.global_point_buffer)} frames")
        if len(self.global_point_buffer) == 0:
            print("No data to train on!")
            return

        # Main Loop
        while self.training_step < target_step:
            
            # --- PHASE 1: Heavy Lifting (CPU -> GPU) ---
            # We do this expensive part only once every 'steps_per_stage' iterations
            
            # Sample a massive chunk from history
            # (Reusing the improved sampling logic discussed previously)
            super_points, super_features = self._sample_from_global_history(
                batch_size=staging_size,
                num_frames_to_sample=len(self.global_point_buffer)//3 # Sample from many frames for diversity
            )
            
            if super_points is None:
                break
                
            # Move big chunk to GPU once
            super_points = super_points.to(self.device)
            super_features = super_features.to(self.device)
            
            total_super_samples = super_points.shape[0]
            
            # --- PHASE 2: Fast Training (GPU only) ---
            # Run multiple steps on this GPU-resident data
            # This mimics the speed of your "old" code
            
            for _ in range(steps_per_stage):
                if self.training_step >= target_step:
                    break
                
                # Fast GPU slicing (Instant)
                # Randomly sample indices from our local GPU super-batch
                batch_idx = torch.randint(0, total_super_samples, (batch_size,), device=self.device)
                
                batch_p = super_points[batch_idx]
                batch_f = super_features[batch_idx]
                
                # Train
                loss = self.hashgrid.train_step(batch_p, batch_f)
                
                if loss is not None and self.training_step % 100 == 0:
                    print(f"Offline Step {self.training_step:04d} | Loss: {loss:.5f}")

                self.training_step += 1

    def _visualize_score(self):
        pass

    def start_training(self):
        if not self.training_active:
            self.training_active = True
            self.extraction_thread = threading.Thread(target=self._extract_semantics, daemon=True)
            self.extraction_thread.start()
            self.training_thread = threading.Thread(target=self._training_worker, daemon=True)
            self.training_thread.start()
            print("Background training started!")

    def stop_training(self):
        if self.training_active:
            self.training_active = False
            if self.training_thread:
                self.training_thread.join(timeout=2.0)
            print("Background training stopped!")

    def stop_extraction(self):
        if self.extraction_thread:
            self.extraction_thread.join(timeout=2.0)
            print("Background extraction stopped!")

    def start_viz(self):
        if not self.viz_active:
            self.viz_active = True
            self.viz_thread = threading.Thread(target=self._visualize_score, daemon = True)
            self.viz_thread.start()
            print("Visualization thread started!")
    
    def cleanup(self):
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
        runner.stop_extraction()
        runner.stop_training()
        runner._offline_training()
        runner.save_data()
        runner.cleanup()
        print("\n" + "="*60)
        print("Session complete! Data saved and ready for analysis.")
        print("="*60)

if __name__ == "__main__":
    main()