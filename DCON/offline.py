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

# Custom Imports
from dev.config import Config
from dev.gaussians import GaussianSplatting
from dev.semantics import SAM_CLIP_Semantics
from dev.utils import unprojection
from dev.hashgrid import HashGrid
from dev.visualizer import Visualizer

class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        # Environment Setup
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cfg.gpu_indices
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"
        self.device = self.cfg.device

        # 1. Load Data
        print(f"Loading data from {self.cfg.output_dir}...")
        self.gt_images, self.gt_depths, self.c2ws, self.intrinsics_tuple = self._load_scene_data()
        self.fx, self.fy, self.cx, self.cy, self.H, self.W = self.intrinsics_tuple
        self.num_cameras = len(self.gt_images)

        # 2. Semantics
        self.sam_clip = SAM_CLIP_Semantics(self.cfg, device=self.device)

        # 3. HashGrid
        self.hashgrid = HashGrid(self.cfg, device=self.device, transforms_json=os.path.join(self.cfg.output_dir, "transforms.json"))

        bounds = self.hashgrid.load_scene_bounds_from_json(os.path.join(self.cfg.output_dir, "transforms.json"))

        self.bounds_min = bounds[0]
        self.bounds_max = bounds[1]
        
        # # Prepare data for Model Initialization
        # init_points, init_colors = self._create_initial_point_cloud()
        
        # # Create GSplat Model
        # intrinsics_dict = {'fx': self.fx, 'fy': self.fy, 'cx': self.cx, 'cy': self.cy, 'H': self.H, 'W': self.W}
        # self.gs_model = GaussianSplatting(self.cfg, init_points, init_colors, intrinsics_dict)

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

    def _create_initial_point_cloud(self):
        """Generates the initial point cloud from depth maps."""
        points_list = []
        colors_list = []
        num_init_frames = 10
        indices = np.linspace(0, len(self.gt_images)-1, num_init_frames, dtype=int)
        
        print("Initializing point cloud from depth...")
        for idx in indices:
            depth = self.gt_depths[idx]
            color = self.gt_images[idx]
            c2w = self.c2ws[idx]
            mask = (depth > 0.1) & (depth < 10.0)

            world_points = unprojection(depth, self.intrinsics_tuple, c2w, self.device, mask=mask)
            points_list.append(world_points)
            colors_list.append(color[mask])

        full_points = torch.cat(points_list, dim=0)
        full_colors = torch.cat(colors_list, dim=0)
        
        if full_points.shape[0] > 100_000:
            indices = torch.randperm(full_points.shape[0])[:100_000]
            full_points = full_points[indices]
            full_colors = full_colors[indices]
            
        return full_points, full_colors

    def run_training(self):
        print(f"Starting training for {self.cfg.iterations} iterations...")
        start_time = time.time()

        for step in range(self.cfg.iterations):
            cam_idx = torch.randint(0, self.num_cameras, (1,)).item()
            gt_image = self.gt_images[cam_idx]
            c2w = self.c2ws[cam_idx]
            
            # Delegate step to the Model
            loss_val, num_gs = self.gs_model.step(step, gt_image, c2w)

            if step % 100 == 0:
                print(f"Step {step:04d} | GS: {num_gs} | Loss: {loss_val:.5f} | Time: {time.time()-start_time:.1f}s")
    
    def sample_rgb(self):

        idx = torch.randint(0, len(self.gt_images), (1,)).item()

        depth = self.gt_depths[idx]
        rgb = self.gt_images[idx]
        c2w_hash = self.c2ws[idx]
        
        rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)
        clip_features = self.sam_clip.extract_dense_features(rgb_np)
        
        # Compute world points and features
        mask = (depth > 0.1) & (depth < 10.0)
        world_points = unprojection(depth, self.intrinsics_tuple, c2w_hash, self.device, mask=mask)
        gt_features = clip_features[mask]

        # Filter zero-norm features
        valid_mask = gt_features.norm(dim=-1) > 1e-6
        world_points = world_points[valid_mask]
        gt_features = gt_features[valid_mask]

        return world_points, gt_features

    def train_feature_field(self):

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

            loss = self.hashgrid.train_step(batch_points, batch_features)
            if loss is None:
                continue
            
            # Logging
            if step % 100 == 0:
                print(f"Step {step:04d} | Train Loss: {loss:.5f} | Time: {time.time()-start_time:.1f}s")
            
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



    def save_results(self):
        save_path = os.path.join(self.cfg.output_dir, self.cfg.output_name)
        print(f"Saving model to {save_path}...")
        self.gs_model.save(save_path)


# After training, add this diagnostic:
def diagnose_hashgrid(runner, clip_idx=30, text_query="a pillow"):
    """Check if HashGrid can reconstruct training image."""
    
    # Get ground truth
    depth = runner.gt_depths[clip_idx]
    rgb = runner.gt_images[clip_idx]
    c2w = runner.c2ws[clip_idx]
    
    rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)
    gt_features = runner.sam_clip.extract_dense_features(rgb_np)
    
    # Get predictions
    pred_features = runner.hashgrid.get_hashgrid_features(depth, c2w, runner.intrinsics_tuple)
    
    # Compute correlation
    mask = (depth > 0.1) & (depth < 10.0)
    gt_flat = gt_features[mask]
    pred_flat = pred_features[mask]
    
    # Normalize both
    gt_norm = gt_flat / (gt_flat.norm(dim=-1, keepdim=True) + 1e-8)
    pred_norm = pred_flat / (pred_flat.norm(dim=-1, keepdim=True) + 1e-8)
    
    cosine_sim = (gt_norm * pred_norm).sum(dim=-1)
    
    print(f"Ground Truth/Predicted Features Cosine similarity stats:")
    print(f"  Mean: {cosine_sim.mean():.4f}")
    print(f"  Std: {cosine_sim.std():.4f}")
    print(f"  Min: {cosine_sim.min():.4f}")
    print(f"  Max: {cosine_sim.max():.4f}")
    
    # Test with a text query
    gt_sim = runner.sam_clip.query(gt_features, text_query)
    pred_sim = runner.sam_clip.query(pred_features, text_query)
    
    visualize_similarity(runner, gt_sim, clip_idx, text_query)
    visualize_similarity(runner, pred_sim, clip_idx, text_query)

def visualize_similarity(runner, similarity_map, img_index, text_query="a pillow"):
    """
    Visualizes the similarity map with better diagnostics.
    """
    # Get original image
    rgb_image = runner.gt_images[img_index].cpu().numpy()

    # Convert to numpy and handle invalid values
    similarity_np = similarity_map.cpu().numpy()
    
    # Replace NaN and inf values with 0
    similarity_np = np.nan_to_num(similarity_np, nan=0.5, posinf=1.0, neginf=0.0)
    
    # Clip to valid range [0, 1] before scaling
    similarity_np = np.clip(similarity_np, 0.0, 1.0)
    
    # Convert to uint8
    similarity_np = (similarity_np * 255).astype(np.uint8)
    
    # Print statistics
    print(f"Score range: [{similarity_np.min():.4f}, {similarity_np.max():.4f}]")
    print(f"Mean: {similarity_np.mean():.4f}, Std: {similarity_np.std():.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    vis_data = similarity_np - similarity_np.min()
    vis_data = vis_data / (vis_data.max() + 1e-8)
    
    # Original image
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original RGB Image")
    axes[0].axis('off')
    
    # Heatmap overlay
    axes[1].imshow(rgb_image)
    heatmap = axes[1].imshow(vis_data, cmap='jet', alpha=0.6, vmin=0.6, vmax=1)
    axes[1].set_title(f"Similarity Overlay: {text_query}")
    axes[1].axis('off')
    plt.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Pure heatmap
    heatmap_pure = axes[2].imshow(vis_data, cmap='jet', vmin=0.6, vmax=1)
    axes[2].set_title(f"Heatmap: {text_query}")
    axes[2].axis('off')
    plt.colorbar(heatmap_pure, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

        


if __name__ == "__main__":
    config = Config("config/config.yaml")
    runner = Runner(config)

    runner.train_feature_field()
    runner.hashgrid.save("hashgrid_model.pt")
    # runner.hashgrid.load("hashgrid_model.pt")

    text_query = "a pillow"
    clip_idx = 12
    # diagnose_hashgrid(runner, clip_idx=clip_idx, text_query=text_query)
    
    visualizer = Visualizer(runner)
    visualizer.visualize_2d_similarity(clip_idx, text_query)

    # visualizer.create_birds_eye_view(text_query, 
    #                                  num_cameras=100, 
    #                                  grid_resolution=0.01, 
    #                                  aggregation='median', 
    #                                  save_path='figs/bev_map.png', 
    #                                  colormap='jet', 
    #                                  vmin=0.6)