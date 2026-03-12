import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
import gc
import json
import tracemalloc
import math
import random
import time
import torch
import numpy as np
import imageio.v2 as imageio
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import deque

from src.recorder import BEVGrid
from src.config import Config
from src.gaussians import GaussianSplatting
from src.semantics import SAM_CLIP_Semantics
from src.utils import unprojection
from src.hashgrid import HashGrid

class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.cfg.gpu_indices
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"
        self.device = self.cfg.device

        # Data Loading
        print(f"Loading data from {self.cfg.output_dir}...")
        self.gt_images, self.gt_depths, self.c2ws, self.intrinsics_tuple = self._load_scene_data()
        self.fx, self.fy, self.cx, self.cy, self.H, self.W = self.intrinsics_tuple
        self.num_cameras = len(self.gt_images)

        # Semantics and Ensemble Models
        self.sam_clip = SAM_CLIP_Semantics(self.cfg, device=self.device)
        self.ensemble_models = [
            HashGrid(self.cfg, device=self.device) 
            for _ in range(self.cfg.ensemble_num_models)
        ]

        self.recorder = BEVGrid(cfg, ensemble=self.ensemble_models)
        
        # Sequential sampling: track current image index
        self.current_image_idx = 0


    def _load_scene_data(self):
        json_path = os.path.join(self.cfg.output_dir, "transforms.json")
        with open(json_path, 'r') as f:
            meta = json.load(f)

        frames = meta['frames']
        img_0 = imageio.imread(os.path.join(self.cfg.output_dir, frames[0]['file_path']))
        H, W = img_0.shape[:2]
        
        fov_x = meta['camera_angle_x']
        fx = 0.5 * W / math.tan(0.5 * fov_x)
        fy = fx
        cx, cy = W / 2.0, H / 2.0

        gt_images, gt_depths, c2w_matrices = [], [], []

        # Habitat (OpenGL) -> GSplat (OpenCV)
        convert_mat = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])

        for frame in frames:
            # RGB
            rgb_path = os.path.join(self.cfg.output_dir, frame['file_path'])
            rgb = imageio.imread(rgb_path)
            gt_images.append(torch.from_numpy(rgb).float() / 255.0)

            # Depth
            depth_name = os.path.basename(frame['file_path']).replace("rgb", "depth").replace(".png", ".npy")
            depth_path = os.path.join(self.cfg.output_dir, "depth_data", depth_name)
            depth = np.load(depth_path)
            if depth.shape[:2] != (H, W):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            gt_depths.append(torch.from_numpy(depth).float())

            # Pose
            c2w_hab = np.array(frame['transform_matrix'])
            c2w_cv = c2w_hab @ convert_mat
            c2w_matrices.append(torch.from_numpy(c2w_cv).float())

        return torch.stack(gt_images), torch.stack(gt_depths), torch.stack(c2w_matrices), (fx, fy, cx, cy, H, W)
    
    def sample_rgb(self, idx=None):

        if idx is None:
            # Use sequential sampling instead of random
            idx = self.current_image_idx
            self.current_image_idx = (self.current_image_idx + 1) % len(self.gt_images)

        # Move data for current frame to GPU for processing
        depth = self.gt_depths[idx].to(self.device)
        rgb = self.gt_images[idx]  # Stays on CPU for sam_clip
        c2w_hash = self.c2ws[idx].to(self.device)

        rgb_np = (rgb.numpy() * 255).astype(np.uint8)

        # Feature extraction on GPU
        clip_features = self.sam_clip.extract_dense_features(rgb_np)
        
        # Unprojection on GPU
        mask = (depth > 0.1) & (depth < 10.0)
        world_points = unprojection(depth, self.intrinsics_tuple, c2w_hash, self.device, mask=mask)
        gt_features = clip_features[mask]

        # Filter zero-norm features
        valid_mask = gt_features.norm(dim=-1) > 1e-6
        world_points = world_points[valid_mask]
        gt_features = gt_features[valid_mask]

        # Return CPU tensors to save VRAM
        return world_points.cpu(), gt_features.cpu()
    
    def train_ensemble(self, save_enabled=False):

        viz_interval = self.cfg.viz_interval
        mini_batch_size = self.cfg.hash_train_batch_size
        
        # Portion of batch to be sampled from the most recent image
        recent_sample_portion = 0.2
        
        # Use a list to store history on CPU
        global_point_buffer = []
        
        # Limit buffer size to avoid excessive RAM usage
        max_buffer_frames = self.cfg.hash_replay_buffer_size
        
        min_frames_to_start = 3

        print(f"Initializing history buffer (start training after {min_frames_to_start} frames)...")
        for i in range(min_frames_to_start):
            # sample_rgb returns CPU tensors
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
            # --- Staging and Refresh Logic ---
            if step % refresh_interval == 0:
                # 1. Refresh CPU buffer (if not the very first step)
                if step > 0:
                    print(f"Refreshing history buffer...")
                    
                    # If buffer is full, explicitly remove oldest element BEFORE sampling new data
                    if len(global_point_buffer) >= max_buffer_frames:
                        old_points, old_features = global_point_buffer.pop(0)
                        del old_points, old_features
                        gc.collect() # Force collection of old tensors before new allocation
                        torch.cuda.empty_cache() # Clear GPU cache if any GPU tensors were involved in old_points/features

                    new_points, new_features = self.sample_rgb()
                    global_point_buffer.append((new_points, new_features))
                    gc.collect()
                    torch.cuda.empty_cache() # Clear GPU cache again
                    
                    buffer_status = f"{len(global_point_buffer)}/{max_buffer_frames}"
                    print(f"Buffer updated (frames: {buffer_status}, total frames processed: {self.current_image_idx})")

                # 2. Stage a new super-batch to the GPU
                if super_points_gpu is not None:
                    del super_points_gpu, super_features_gpu
                    torch.cuda.empty_cache()
                
                print(f"\n--- Staging new super-batch for steps {step}-{step+refresh_interval-1} ---")

                # --- New memory-efficient staging ---
                # A. First, calculate the total size of the super-batch without moving anything to GPU
                total_points_to_stage = 0
                
                # From recent frame
                recent_points, recent_features = global_point_buffer[-1]
                staging_size_recent = int(staging_size * recent_sample_portion)
                if recent_points.shape[0] > 0:
                    num_samples_recent = min(staging_size_recent, recent_points.shape[0])
                    total_points_to_stage += num_samples_recent

                # From historical frames (sample from ALL frames)
                staging_size_history = staging_size - staging_size_recent
                history_buffer = global_point_buffer[:-1]
                frames_to_sample_from = []
                points_per_frame = 0
                k = 0
                if history_buffer and staging_size_history > 0:
                    k = len(history_buffer)  # Use all historical frames
                    if k > 0:
                        points_per_frame = staging_size_history // k
                        frames_to_sample_from = history_buffer  # Sample from all frames
                        for pts, fts in frames_to_sample_from:
                            if pts.shape[0] > 0:
                                num_to_sample = min(points_per_frame, pts.shape[0])
                                total_points_to_stage += num_to_sample
                
                if total_points_to_stage == 0:
                    print("Warning: Cannot create super-batch, not enough data. Skipping stage.")
                    super_points_gpu = None
                    super_features_gpu = None
                else:
                    # B. Pre-allocate the final tensors on the GPU
                    super_points_gpu = torch.empty(total_points_to_stage, 3, device=self.device, dtype=torch.float32)
                    super_features_gpu = torch.empty(total_points_to_stage, self.cfg.hash_feature_dim, device=self.device, dtype=torch.float32)
                    
                    current_pos = 0

                    # C. Iterate again, sample, move chunk to GPU, copy, and delete chunk
                    # From recent frame
                    if recent_points.shape[0] > 0:
                        num_samples = min(staging_size_recent, recent_points.shape[0])
                        indices = torch.randint(0, recent_points.shape[0], (num_samples,))
                        
                        points_chunk_gpu = recent_points[indices].to(self.device)
                        features_chunk_gpu = recent_features[indices].to(self.device)
                        
                        super_points_gpu[current_pos : current_pos + num_samples] = points_chunk_gpu
                        super_features_gpu[current_pos : current_pos + num_samples] = features_chunk_gpu
                        current_pos += num_samples
                        
                        del points_chunk_gpu, features_chunk_gpu

                    # From historical frames
                    if history_buffer and staging_size_history > 0 and k > 0:
                        for pts, fts in frames_to_sample_from:
                            if pts.shape[0] > 0:
                                num_to_sample = min(points_per_frame, pts.shape[0])
                                indices = torch.randint(0, pts.shape[0], (num_to_sample,))
                                
                                points_chunk_gpu = pts[indices].to(self.device)
                                features_chunk_gpu = fts[indices].to(self.device)

                                super_points_gpu[current_pos : current_pos + num_to_sample] = points_chunk_gpu
                                super_features_gpu[current_pos : current_pos + num_to_sample] = features_chunk_gpu
                                current_pos += num_to_sample

                                del points_chunk_gpu, features_chunk_gpu
                    
                    print(f"Staged {super_points_gpu.shape[0]} points to GPU.")

                # 4. Clean up and empty cache
                gc.collect()
                torch.cuda.empty_cache()

            # --- Batch Sampling from GPU Super-batch ---
            if super_points_gpu is None or super_points_gpu.shape[0] < mini_batch_size:
                continue

            batch_idx = torch.randint(0, super_points_gpu.shape[0], (mini_batch_size,), device=self.device)
            batch_points = super_points_gpu[batch_idx]
            batch_features = super_features_gpu[batch_idx]

            # --- Training ---
            loss = 0
            for model in self.ensemble_models:
                train_loss = model.train_step(batch_points, batch_features)
                if train_loss is not None:
                    loss += train_loss
            avg_loss = loss / self.cfg.ensemble_num_models

            # Logging
            if step % 100 == 0:
                print(f"Step {step:04d} | Train Loss: {avg_loss:.5f} | Time: {time.time()-start_time:.1f}s")

            # --- Visualization/Saving ---
            if save_enabled and step > 0 and step % viz_interval == 0:
                self.save_uncertainty_snapshot(step)


    def save_uncertainty_snapshot(self, step):
        """
        Compute and save uncertainty maps at the current training step.
        
        Args:
            step: Current training iteration number
        """
        elapsed = self.recorder.compute_and_save_uncertainty_snapshot(
            iteration=step,
            height_filter=(0.1, 2.0)
        )
        print(f"Uncertainty snapshot time: {elapsed:.6f}s")





    def save_models(self):
        """Save only the ensemble models."""
        for i, model in enumerate(self.ensemble_models):
            ensemble_path = os.path.join(self.cfg.output_dir, f"ensemble/hashgrid_ensemble_{i}.pt")
            os.makedirs(os.path.dirname(ensemble_path), exist_ok=True)
            model.save(ensemble_path)
            print(f"Saved Ensemble Model {i} to {ensemble_path}")


        


if __name__ == "__main__":
    config = Config("config/config.yaml")
    runner = Runner(config)

    runner.train_ensemble(save_enabled=True)
    runner.save_models()
