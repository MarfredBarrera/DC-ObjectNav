import os
import json
import math
import time
import torch
import numpy as np
import imageio.v2 as imageio
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import deque

from dev.recorder import BEVGrid
from dev.config import Config
from dev.gaussians import GaussianSplatting
from dev.semantics import SAM_CLIP_Semantics
from dev.utils import unprojection
from dev.hashgrid import HashGrid

class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cfg.gpu_indices
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

        self.recorder = BEVGrid(cfg, self.ensemble_models)


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
            gt_images.append(torch.from_numpy(rgb).float().to(self.device) / 255.0)

            # Depth
            depth_name = os.path.basename(frame['file_path']).replace("rgb", "depth").replace(".png", ".npy")
            depth_path = os.path.join(self.cfg.output_dir, "depth_data", depth_name)
            depth = np.load(depth_path)
            if depth.shape[:2] != (H, W):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            gt_depths.append(torch.from_numpy(depth).float().to(self.device))

            # Pose
            c2w_hab = np.array(frame['transform_matrix'])
            c2w_cv = c2w_hab @ convert_mat
            c2w_matrices.append(torch.from_numpy(c2w_cv).float().to(self.device))

        return torch.stack(gt_images), torch.stack(gt_depths), torch.stack(c2w_matrices), (fx, fy, cx, cy, H, W)
    
    def sample_rgb(self, idx=None):

        if idx is None:
            idx = torch.randint(0, len(self.gt_images), (1,)).item()
        depth = self.gt_depths[idx]
        rgb = self.gt_images[idx]
        c2w_hash = self.c2ws[idx]
        
        rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)
        clip_features = self.sam_clip.extract_dense_features(rgb_np)
        
        mask = (depth > 0.1) & (depth < 10.0)
        world_points = unprojection(depth, self.intrinsics_tuple, c2w_hash, self.device, mask=mask)
        gt_features = clip_features[mask]

        # Filter zero-norm features
        valid_mask = gt_features.norm(dim=-1) > 1e-6
        world_points = world_points[valid_mask]
        gt_features = gt_features[valid_mask]

        return world_points, gt_features
    
    def train_ensemble(self):

        buf_size = self.cfg.hash_replay_buffer_size
        replay_buffer = deque(maxlen=buf_size)
        refresh_interval = self.cfg.hash_buffer_refresh_interval 

        print(f"Initializing replay buffer with {buf_size} samples...")
        for i in range(buf_size):
            world_points, gt_features = self.sample_rgb()
            valid_mask = gt_features.norm(dim=-1) > 1e-6  # Remove near-zero norm features
            world_points = world_points[valid_mask]
            gt_features = gt_features[valid_mask]

            replay_buffer.append((world_points, gt_features))
            print(f"  Buffered sample {i+1}/{buf_size}")
            
            # Free memory after each sample
            if i % 3 == 2:  # Every 3 samples
                torch.cuda.empty_cache()

        world_points = torch.cat([x[0] for x in replay_buffer], dim=0)
        gt_features = torch.cat([x[1] for x in replay_buffer], dim=0)

        torch.cuda.empty_cache()
        batch_size = min(self.cfg.hash_train_batch_size, world_points.shape[0])

        start_time = time.time()

        for step in range(self.cfg.iterations):
            # Sample a batch from concatenated data
            batch_indx = torch.randperm(world_points.shape[0], device=world_points.device)[:batch_size]
            batch_points = world_points[batch_indx]
            batch_features = gt_features[batch_indx]

            loss = 0
            for model in self.ensemble_models:
                loss += model.train_step(batch_points, batch_features)
            avg_loss = loss/self.cfg.ensemble_num_models

            # Logging
            if step % 100 == 0:
                print(f"Step {step:04d} | Train Loss: {avg_loss:.5f} | Time: {time.time()-start_time:.1f}s")
            
            # Buffer refresh
            if step > 0 and step % refresh_interval == 0:
                
                # Free old concatenated tensors before refresh
                del world_points, gt_features
                torch.cuda.empty_cache()
                
                # Sample new data and add to buffer
                sample_points, sample_features = self.sample_rgb()
                valid_mask = sample_features.norm(dim=-1) > 1e-6
                sample_points = sample_points[valid_mask]
                sample_features = sample_features[valid_mask]
                
                # Add to buffer - oldest entry automatically removed due to maxlen
                replay_buffer.append((sample_points, sample_features))
                
                # Refresh concatenated training data from updated buffer
                world_points = torch.cat([x[0] for x in replay_buffer], dim=0)
                gt_features = torch.cat([x[1] for x in replay_buffer], dim=0)
                batch_size = min(self.cfg.hash_train_batch_size, world_points.shape[0])
                
                torch.cuda.empty_cache()
                print(f"Buffer updated")

                ### Save uncertainty map
                self.uncertainty_snapshot(step)

    def uncertainty_snapshot(self, step):
        """Save the BEV uncertainty maps at a specific training step."""
        self.recorder.iteration_num = step
        self.recorder.forward_pass()
        self.recorder.save_bev_maps()





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

    runner.train_ensemble()
    runner.save_models()

