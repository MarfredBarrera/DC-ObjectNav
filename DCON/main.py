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
import argparse

# Custom imports
from dev.config import Config
from dev.semantics import SAM_CLIP_Semantics
from dev.utils import unprojection
from dev.hashgrid import HashGrid
from dev.visualizer import Visualizer

# Habitat imports
import habitat_sim
import habitat_sim.utils.common as utils

# Silence warnings and logs
os.environ['GLOG_minloglevel'] = '2'
os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['HABITAT_SIM_LOG'] = 'quiet'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

class HabitatSim:
       # Coordinate conversion matrix (Habitat to standard/world)
    CONVERT_MAT = np.array([
        [1,  0,  0,  0], 
        [0, -1,  0,  0], 
        [0,  0, -1,  0], 
        [0,  0,  0,  1]
    ], dtype=np.float32)

    def __init__(self, cfg: Config, scene_path: str, offline: bool = False):
        self.cfg = cfg
        self.scene_path = scene_path
        self.offline = offline
        
        # Setup device
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cfg.gpu_indices
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"
        self.device = self.cfg.device
        
        # Output directories
        self.output_dir = cfg.output_dir
        for sub_dir in ["rgbs", "depth_data", "depth_vis", "ensemble"]:
            os.makedirs(os.path.join(self.output_dir, sub_dir), exist_ok=True)
        
        # Camera parameters
        self.IMG_WIDTH = cfg.img_width
        self.IMG_HEIGHT = cfg.img_height
        self.FOV_DEG = cfg.fov

        # Initialize semantics and hashgrid
        self.sam_clip = SAM_CLIP_Semantics(self.cfg, device=self.device)
        self.hashgrid = HashGrid(self.cfg, device=self.device)

        # Initialize ensemble models
        self.ensemble_models = [
            HashGrid(self.cfg, device=self.device) 
            for _ in range(self.cfg.ensemble_num_models)
        ]

        # Initialize Simulator State
        self.simulator = None
        self.agent = None

        # Multithreading & Training State
        self.global_point_buffer = [] 
        self.buffer_lock = threading.Lock()
        
        self.training_active = False
        self.extraction_thread = None
        
        self.ensemble_training_active = False
        self.ensemble_training_thread = None
        self.ensemble_steps = [0] * self.cfg.ensemble_num_models
                
        # Data storage & Queues
        self.rgb_imgs = []
        self.depth_imgs = []
        self.pose_matrices = []
        self.frame_count = 0
        self.offline_data_stream = []

        self.data_queue = queue.Queue(maxsize=self.cfg.data_queue_size)
        self.training_queue = queue.Queue(maxsize=self.cfg.data_queue_size)
            
        # Visualization
        self.viz_active = False
        self.last_viz_update = 0
        self.viz_thread = None
        self.viz_update_interval = 2 
        self.camera_on = True

        # Initialize Mode
        if not self.offline:
            self._init_simulator()
            # Initialize intrinsics
            fov_rad = np.deg2rad(self.FOV_DEG)
            self.fx = (self.IMG_WIDTH / 2) / np.tan(fov_rad / 2)
            self.fy = self.fx
            self.cx = self.IMG_WIDTH / 2
            self.cy = self.IMG_HEIGHT / 2
            self.intrinsics_tuple = (self.fx, self.fy, self.cx, self.cy, self.IMG_HEIGHT, self.IMG_WIDTH)
        else:
            self._load_offline_data()

        print("\nInitialization complete!")

    def _init_simulator(self):
        """Sets up the Habitat simulator and the agent's sensors."""
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = self.scene_path
        sim_cfg.enable_physics = False
        sim_cfg.load_semantic_mesh = False

        # Configure Sensors
        def create_sensor_spec(uuid, sensor_type):
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid = uuid
            spec.sensor_type = sensor_type
            spec.resolution = [self.IMG_WIDTH, self.IMG_HEIGHT]
            spec.position = [0.0, 1.5, 0.0]
            spec.orientation = [0.0, 0.0, 0.0]
            return spec

        rgb_sensor = create_sensor_spec("rgb", habitat_sim.SensorType.COLOR)
        depth_sensor = create_sensor_spec("depth", habitat_sim.SensorType.DEPTH)

        # Agent Configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_sensor, depth_sensor]
        agent_cfg.action_space = {
            "move_forward": habitat_sim.ActionSpec("move_forward", habitat_sim.ActuationSpec(amount=0.25)),
            "turn_left": habitat_sim.ActionSpec("turn_left", habitat_sim.ActuationSpec(amount=10.0)),
            "turn_right": habitat_sim.ActionSpec("turn_right", habitat_sim.ActuationSpec(amount=10.0)),
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
        
    def _get_camera_matrix(self) -> np.ndarray:
        """Get camera transformation matrix from the Habitat agent."""
        state = self.agent.get_state().sensor_states['rgb']
        rot_mat = np.array(utils.quat_to_magnum(state.rotation).to_matrix())
        
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rot_mat
        transform_matrix[:3, 3] = state.position
        
        return transform_matrix
    
    def _create_telemetry_panel(self, width: int = 400, height: int = 512) -> np.ndarray:
        """Create a telemetry information panel using OpenCV."""
        panel = np.ones((height, width, 3), dtype=np.uint8) * 40  # Dark background
        
        def add_text(text: str, y: int, x: int = 10, color=(200, 200, 200), scale=0.6, bold=False):
            thickness = 2 if bold else 1
            cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_TRIPLEX, scale, color, thickness)

        # Title
        add_text("TELEMETRY", 30, color=(255, 255, 255), scale=0.8, bold=True)
        cv2.line(panel, (10, 40), (width - 10, 40), (100, 100, 100), 1)
        
        with self.buffer_lock:
            buf_len = len(self.global_point_buffer)
            
        status_text = "ON" if self.ensemble_training_active else "OFF"
        status_color = (0, 255, 0) if self.ensemble_training_active else (0, 0, 255)
        max_step = max(self.ensemble_steps) if self.ensemble_steps else 0

        # Info rows
        y_pos = 70
        add_text("Frame Count:", y_pos)
        add_text(f"{self.frame_count}", y_pos, x=width - 100, color=(255, 255, 255), bold=True)
        
        y_pos += 35
        add_text("Total Samples:", y_pos)
        add_text(f"{buf_len}", y_pos, x=width - 120, color=(255, 255, 255), bold=True)
        
        y_pos += 55
        add_text("Ensemble Training:", y_pos)
        add_text(status_text, y_pos, x=width - 80, color=status_color, bold=True)
        
        y_pos += 35
        add_text("Ensemble Step:", y_pos)
        add_text(f"{max_step}", y_pos, x=width - 120, color=(255, 255, 255), bold=True)
        
        y_pos += 35
        add_text("Data Queue:", y_pos)
        add_text(f"{self.data_queue.qsize()}/{self.cfg.data_queue_size}", y_pos, x=width - 100, color=(255, 255, 255), bold=True)
        
        y_pos += 35
        add_text("Training Queue:", y_pos)
        add_text(f"{self.training_queue.qsize()}/{self.cfg.training_queue_size}", y_pos, x=width - 100, color=(255, 255, 255), bold=True)
        
        # Separator
        y_pos += 25
        cv2.line(panel, (10, y_pos), (width - 10, y_pos), (100, 100, 100), 1)
        
        # Camera position
        if self.agent:
            y_pos += 30
            pos = self.agent.get_state().position
            add_text("Camera Position:", y_pos)
            add_text(f"X: {pos[0]:6.2f}", y_pos + 25, x=20, scale=0.5)
            add_text(f"Y: {pos[1]:6.2f}", y_pos + 45, x=20, scale=0.5)
            add_text(f"Z: {pos[2]:6.2f}", y_pos + 65, x=20, scale=0.5)
        
        return panel
    
    def _load_offline_data(self):
        """Loads transforms.json and images for offline training."""
        print(f"Loading offline data from {self.output_dir}...")
        json_path = os.path.join(self.output_dir, "transforms.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"transforms.json not found at {json_path}")
            
        with open(json_path, 'r') as f:
            meta = json.load(f)
            
        # Load intrinsics
        self.fx, self.fy = meta['fl_x'], meta['fl_y']
        self.cx, self.cy = meta['cx'], meta['cy']
        self.IMG_WIDTH, self.IMG_HEIGHT = meta['w'], meta['h']
        self.intrinsics_tuple = (self.fx, self.fy, self.cx, self.cy, self.IMG_HEIGHT, self.IMG_WIDTH)
        
        # Load bounds if available
        if 'scene_bounds' in meta:
            bounds = meta['scene_bounds']
            self.hashgrid.bounds_min = torch.tensor(bounds['min'], device=self.device, dtype=torch.float32)
            self.hashgrid.bounds_max = torch.tensor(bounds['max'], device=self.device, dtype=torch.float32)
            
        frames = meta['frames']
        print(f"Found {len(frames)} frames. Loading into memory...")
        
        for i, frame in enumerate(frames):
            # Load RGB
            rgb_path = os.path.join(self.output_dir, frame['file_path'])
            rgb = cv2.imread(rgb_path)
            if rgb is None:
                print(f"Warning: Could not load {rgb_path}")
                continue
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            
            # Load Depth
            depth_name = os.path.basename(frame['file_path']).replace("rgb", "depth").replace(".png", ".npy")
            depth_path = os.path.join(self.output_dir, "depth_data", depth_name)
            if not os.path.exists(depth_path):
                print(f"Warning: Could not load {depth_path}")
                continue
            depth = np.load(depth_path)
            
            # Pose
            pose_matrix = np.array(frame['transform_matrix'])
            self.offline_data_stream.append((rgb, depth, pose_matrix))
                
            if i % 10 == 0:
                print(f"Loaded {i}/{len(frames)} frames", end='\r')
                
        print(f"\nLoaded {len(self.offline_data_stream)} frames for simulation.")

    def save_models(self):
        """Save only the ensemble models."""
        for i, model in enumerate(self.ensemble_models):
            ensemble_path = os.path.join(self.output_dir, f"ensemble/hashgrid_ensemble_{i}.pt")
            model.save(ensemble_path)
            print(f"Saved Ensemble Model {i} to {ensemble_path}")

    def save_data(self):
        """Save all collected frames and models."""
        print(f"\nSaving {self.frame_count} frames and transforms...")
        
        # Calculate scene bounds
        if self.simulator and self.simulator.pathfinder.is_loaded:
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
            scene_bounds = {
                "min": (scene_min - padding).tolist(),
                "max": (scene_max + padding).tolist()
            }
        
        frames_data = []
        fov_rad = np.deg2rad(self.FOV_DEG)
        
        for idx, (cv2_img, depth, pose) in enumerate(zip(self.rgb_imgs, self.depth_imgs, self.pose_matrices)):
            # Save RGB and depth
            rgb_rel_path = f"rgbs/rgb_{idx:03d}.png"
            cv2.imwrite(os.path.join(self.output_dir, rgb_rel_path), cv2_img)
            np.save(os.path.join(self.output_dir, f"depth_data/depth_{idx:03d}.npy"), depth)
            
            # Depth visualization
            depth_vis = np.clip(depth * 255 / 10.0, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(self.output_dir, f"depth_vis/depth_vis_{idx:03d}.png"), depth_vis)
            
            frames_data.append({
                "file_path": rgb_rel_path,
                "transform_matrix": pose.tolist()
            })
        
        # Create transforms.json
        json_data = {
            "camera_angle_x": fov_rad,
            "fl_x": self.fx, "fl_y": self.fy,
            "cx": self.cx, "cy": self.cy,
            "w": self.IMG_WIDTH, "h": self.IMG_HEIGHT,
            "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
            "scene_bounds": scene_bounds,
            "frames": frames_data
        }
        
        transforms_path = os.path.join(self.output_dir, "transforms.json")
        with open(transforms_path, "w") as f:
            json.dump(json_data, f, indent=4)
        
        print(f"Saved transforms.json with {len(frames_data)} frames")
        self.save_models()

    def _process_frame(self, rgb: np.ndarray, depth: np.ndarray, pose_matrix: np.ndarray):
        """Process and extract features from a frame, strictly returning CPU tensors."""
        # Clean RGB Image format
        rgb_np = rgb[:, :, :3] if rgb.shape[-1] == 4 else rgb
        if rgb_np.dtype != np.uint8:
            rgb_np = (rgb_np * 255).astype(np.uint8) if rgb_np.max() <= 1.0 else rgb_np.astype(np.uint8)
        
        torch.cuda.empty_cache()
        
        try:
            clip_features = self.sam_clip.extract_dense_features(rgb_np)
            clip_features_cpu = clip_features.cpu()
            del clip_features
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA OOM during feature extraction: {e}")
            torch.cuda.empty_cache()
            raise
        
        depth_tensor = torch.from_numpy(depth).float()
        c2w_cv = pose_matrix @ self.CONVERT_MAT
        c2w_cpu = torch.from_numpy(c2w_cv).float()
        
        # Filter valid depth regions
        depth_mask = (depth_tensor > 0.1) & (depth_tensor < 10.0)
        valid_indices = torch.nonzero(depth_mask.flatten(), as_tuple=False).squeeze(1)

        mask = torch.zeros(depth_tensor.numel(), dtype=torch.bool)
        mask[valid_indices] = True
        mask = mask.view_as(depth_tensor)
        
        # Temp move to GPU for fast unprojection
        depth_gpu = depth_tensor.to(self.device)
        c2w_gpu = c2w_cpu.to(self.device)
        
        world_points = unprojection(
            depth_gpu, self.intrinsics_tuple, c2w_gpu, self.device, mask=mask.to(self.device)
        )
        
        # Move immediately back to CPU
        world_points_cpu = world_points.cpu()
        del world_points, depth_gpu, c2w_gpu
        torch.cuda.empty_cache()
        
        gt_features_cpu = clip_features_cpu[mask]
        del clip_features_cpu
        
        # Return only points with valid features
        valid_mask = gt_features_cpu.norm(dim=-1) > 1e-6
        return world_points_cpu[valid_mask], gt_features_cpu[valid_mask], rgb_np, depth, c2w_cv
    
    def _extract_semantics(self):
        """Gets frames from data queue, extracts semantics, and pushes result to training queue."""
        while self.training_active:
            try:
                new_data = self.data_queue.get_nowait()
                if new_data is not None:
                    rgb, depth, current_matrix = new_data
                    world_points_cpu, gt_features_cpu, _, _, _ = self._process_frame(
                        rgb, depth, current_matrix
                    )
                    
                    self.training_queue.put_nowait((world_points_cpu, gt_features_cpu))
                    
                    # Clean up to free memory
                    del rgb, depth, current_matrix
                    time.sleep(0.01)
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                print(f"\\nError in extraction: {e}")
                time.sleep(0.5)

    def _sample_from_global_history(self, batch_size: int, num_frames_to_sample: int = 4):
        """Randomly samples a batch from the global history buffer."""
        with self.buffer_lock:
            total_frames = len(self.global_point_buffer)
            if total_frames == 0:
                return None, None
            
            sample_count = min(total_frames, num_frames_to_sample)
            frame_indices = np.random.choice(total_frames, size=sample_count, replace=False)
            
            points_per_frame = batch_size // sample_count
            remainder = batch_size % sample_count
            
            batch_points_list, batch_features_list = [], []
            
            for i, idx in enumerate(frame_indices):
                pts, fts = self.global_point_buffer[idx]
                n_available = pts.shape[0]
                n_take = points_per_frame + (1 if i < remainder else 0)

                if n_available <= n_take:
                    batch_points_list.append(pts)
                    batch_features_list.append(fts)
                else:
                    rand_idx = torch.randperm(n_available)[:n_take]
                    batch_points_list.append(pts[rand_idx])
                    batch_features_list.append(fts[rand_idx])
     
        batch_points = torch.cat(batch_points_list, dim=0)
        batch_features = torch.cat(batch_features_list, dim=0)
        return batch_points.to(self.device), batch_features.to(self.device)

    def _ensemble_training_worker(self):
        """Trains all ensemble models sequentially on a single thread."""
        print("Ensemble training worker started...")
        mini_batch_size = self.cfg.hash_train_batch_size
        steps_per_stage = 50
        staging_size = mini_batch_size * steps_per_stage
        max_buffer_size = 200  # Limit buffer to prevent OOM

        while self.ensemble_training_active:
            # Phase 1: Ingest New Data (during online training)
            if self.training_active:
                while not self.training_queue.empty():
                    try:
                        new_training_data = self.training_queue.get_nowait()
                        if new_training_data:
                            with self.buffer_lock:
                                self.global_point_buffer.append(new_training_data)
                                # Limit buffer size to prevent OOM
                                if len(self.global_point_buffer) > max_buffer_size:
                                    # Remove oldest 20% when buffer is full
                                    n_remove = max_buffer_size // 5
                                    self.global_point_buffer = self.global_point_buffer[n_remove:]
                    except queue.Empty:
                        break

            # Check if there is enough history to sample from
            with self.buffer_lock:
                buffer_len = len(self.global_point_buffer)

            if buffer_len < 5:
                time.sleep(1.0)
                continue

            # Phase 2: Sample a Super-Batch from CPU history
            super_points_gpu, super_features_gpu = self._sample_from_global_history(
                batch_size=staging_size, num_frames_to_sample=15
            )

            if super_points_gpu is None:
                time.sleep(0.1)
                continue

            total_super_samples = super_points_gpu.shape[0]

            # Phase 3: Sequential Training Loop
            for _ in range(steps_per_stage):
                if not self.ensemble_training_active:
                    break

                batch_idx = torch.randint(0, total_super_samples, (mini_batch_size,), device=self.device)
                batch_p = super_points_gpu[batch_idx]
                batch_f = super_features_gpu[batch_idx]
                
                for model_idx, model in enumerate(self.ensemble_models):
                    loss = model.train_step(batch_p, batch_f)
                    
                    if loss is not None and self.ensemble_steps[model_idx] % 100 == 0:
                        print(f"Ensemble Model {model_idx} | Step {self.ensemble_steps[model_idx]:04d} | Loss: {loss:.5f}")

                    self.ensemble_steps[model_idx] += 1

                time.sleep(0.002)
            
            # Clean up GPU memory after each super-batch
            del super_points_gpu, super_features_gpu
            torch.cuda.empty_cache()

    def run_exploration(self):
        """Exploration loop with keyboard controls."""
        print("\n" + "="*40)
        print(" COMMANDS:")
        print("  [w]    : Move Forward")
        print("  [a]    : Turn Left")
        print("  [d]    : Turn Right")
        print("  [t]    : Toggle ensemble training on/off")
        print("  [c]    : Toggle camera on/off")
        print("  [q/ESC]: Quit")
        print("="*40 + "\n")

        self.start_training()
        self.start_ensemble_training()

        try:
            while True:
                obs = self.simulator.get_sensor_observations()    
                rgb = obs["rgb"]
                depth = obs["depth"]
                current_matrix = self._get_camera_matrix()

                cv2_img = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
                agent_view = cv2.resize(cv2_img, (720, 720))
                
                cv2.putText(agent_view, "AGENT VIEW", (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)
                
                telemetry_panel = self._create_telemetry_panel(width=400, height=720)
                combined_view = np.hstack([agent_view, telemetry_panel])
                
                cv2.imshow("Habitat Explorer - Agent View & Telemetry", combined_view)

                try:
                    if self.frame_count % self.viz_update_interval == 0 and self.camera_on:
                        self.data_queue.put_nowait((rgb, depth, current_matrix))
                        self.rgb_imgs.append(cv2_img)
                        self.depth_imgs.append(depth)
                        self.pose_matrices.append(current_matrix)
                except queue.Full:
                    print("Queue full, skipping frame")
                except Exception as e:
                    print(f"Frame processing error: {e}")

                key = cv2.waitKey(1) & 0xFF
                
                if key in (ord('q'), 27):
                    print("Quit requested")
                    break
                elif key == ord('w'):
                    self.simulator.step("move_forward")
                    self.frame_count += 1
                elif key == ord('a'):
                    self.simulator.step("turn_left")
                    self.frame_count += 1
                elif key == ord('d'):
                    self.simulator.step("turn_right")
                    self.frame_count += 1
                elif key == ord('t'):
                    if self.ensemble_training_active:
                        self.stop_ensemble_training()
                    else:
                        self.start_ensemble_training()
                elif key == ord('c'):
                    self.camera_on = not self.camera_on
                    print(f"Camera turned {'on' if self.camera_on else 'off'}")
                else:
                    self.frame_count += 1
                
                time.sleep(0.5)
        finally:
            self.cleanup()

    def run_offline_simulation(self, headless=True):
        """Simulate online exploration using offline data."""
        print("\n" + "="*40)
        print(" STARTING OFFLINE SIMULATION")
        print("="*40 + "\n")

        self.start_training()
        self.start_ensemble_training()
        
        total_frames = len(self.offline_data_stream)
        
        try:
            for i, (rgb, depth, pose) in enumerate(self.offline_data_stream):
                # Skip visualization in headless mode to save memory
                if not headless:
                    cv2_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    cv2.putText(cv2_img, f"Offline Stream: {i+1}/{total_frames}", (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(cv2_img, f"Queue: {self.data_queue.qsize()}", (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Habitat Explorer - Offline Simulation", cv2_img)
                    
                    if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                        print("Simulation stopped by user")
                        break
                
                # Progress indicator
                if i % 10 == 0:
                    print(f"Processing frame {i+1}/{total_frames} | Queue: {self.data_queue.qsize()}/{self.cfg.data_queue_size}", end='\r')
                
                try:
                    self.data_queue.put((rgb, depth, pose), timeout=5.0)
                except queue.Full:
                    print(f"\nWarning: Data queue full at frame {i}, skipping frame")
                
                # Slow down to prevent overwhelming the processing pipeline
                time.sleep(0.1)
                
            print(f"\nFinished streaming {total_frames} frames.")
            
            while not self.data_queue.empty():
                time.sleep(0.5)
                print(f"Waiting for processing... Data Q: {self.data_queue.qsize()}", end='\r')
            print("\nData queue drained.")
        finally:
            if not headless:
                cv2.destroyAllWindows()
            self.cleanup()

    def _visualize_score(self):
        """Placeholder for visualizing exploration scores."""
        pass

    def start_training(self):
        """Start the extraction thread for processing new frames."""
        if not self.training_active:
            self.training_active = True
            self.extraction_thread = threading.Thread(target=self._extract_semantics, daemon=True)
            self.extraction_thread.start()
            print("Extraction started!")

    def start_ensemble_training(self):
        """Start a single thread that trains all ensemble models sequentially."""
        if not self.ensemble_training_active:
            self.ensemble_training_active = True
            self.ensemble_steps = [0] * len(self.ensemble_models)

            self.ensemble_training_thread = threading.Thread(
                target=self._ensemble_training_worker,
                name="ensemble-training-worker",
                daemon=True,
            )
            self.ensemble_training_thread.start()
            print(f"Ensemble training started! ({len(self.ensemble_models)} models sequential)")
    
    def continue_offline_training(self):
        """Continue ensemble training offline until target iterations are reached."""
        target_step = self.cfg.iterations
        
        while not self.training_queue.empty():
            try:
                new_data = self.training_queue.get_nowait()
                if new_data is not None:
                    with self.buffer_lock:
                        self.global_point_buffer.append(new_data)
            except:
                break
        
        print(f"\nStarting offline ensemble training...")
        print(f"Total history size: {len(self.global_point_buffer)} frames")
        print(f"Current max steps: {max(self.ensemble_steps)}, Target: {target_step}")
        
        if not self.global_point_buffer:
            print("No data to train on!")
            return
        
        if not self.ensemble_training_active:
            self.start_ensemble_training()
        
        while self.ensemble_training_active and max(self.ensemble_steps) < target_step:
            current_max = max(self.ensemble_steps)
            if current_max % 500 == 0 and current_max > 0:
                print(f"Offline training progress: {current_max}/{target_step}")
            time.sleep(1.0)
        
        self.stop_ensemble_training()
        print(f"Offline training complete! Final steps: {self.ensemble_steps}")

    def stop_training(self):
        """Stop the extraction process."""
        if self.training_active:
            self.training_active = False
            print("Extraction stopped!")

    def stop_extraction(self):
        if self.extraction_thread:
            self.extraction_thread.join(timeout=2.0)
            print("Extraction stopped!")

    def stop_ensemble_training(self):
        """Signal the ensemble worker thread to stop and wait for it to finish."""
        if self.ensemble_training_active:
            self.ensemble_training_active = False
            if self.ensemble_training_thread:
                self.ensemble_training_thread.join(timeout=2.0)
                if self.ensemble_training_thread.is_alive():
                    print(f"Warning: ensemble worker did not stop within timeout.")
            self.ensemble_training_thread = None
            print("Ensemble training stopped!")

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
        if self.simulator:
            self.simulator.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Skip exploration and train on existing data")
    args = parser.parse_args()

    config = Config("./config/config.yaml")
    runner = HabitatSim(config, scene_path="/workspace/DCON/gibson_scenes/Anaheim.glb", offline=args.offline)
    
    try:
        if not args.offline:
            runner.run_exploration()
        else:
            runner.run_offline_simulation(headless=True)
    finally:
        runner.stop_training()
        runner.stop_extraction()
        runner.continue_offline_training()
        if args.offline:
            runner.save_models()
        else:
            runner.save_data()
        runner.cleanup()
        print("\n" + "="*60)
        print("Session complete! Data saved and ready for analysis.")
        print("="*60)

if __name__ == "__main__":
    main()