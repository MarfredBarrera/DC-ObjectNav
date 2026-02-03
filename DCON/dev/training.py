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

# Custom Imports
from config import Config
from gaussians import GaussianSplatting
import semantics
from utils import unprojection

class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        # Environment Setup
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cfg.gpu_indices
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"
        self.device = self.cfg.device

        # 1. Load Data
        print(f"Loading data from {self.cfg.scene_dir}...")
        self.gt_images, self.gt_depths, self.c2ws, self.intrinsics_tuple = self._load_scene_data()
        self.fx, self.fy, self.cx, self.cy, self.H, self.W = self.intrinsics_tuple
        self.num_cameras = len(self.gt_images)

        # 2. Init Sub-Modules
        self.clip_labels = semantics.SAM_CLIP_Semantics(self.cfg, device=self.device)
        
        # Prepare data for Model Initialization
        init_points, init_colors = self._create_initial_point_cloud()
        
        # Create GSplat Model
        intrinsics_dict = {'fx': self.fx, 'fy': self.fy, 'cx': self.cx, 'cy': self.cy, 'H': self.H, 'W': self.W}
        self.gs_model = GaussianSplatting(self.cfg, init_points, init_colors, intrinsics_dict)

    def _load_scene_data(self):
        json_path = os.path.join(self.cfg.scene_dir, "transforms.json")
        with open(json_path, 'r') as f:
            meta = json.load(f)

        frames = meta['frames']
        img_0 = imageio.imread(os.path.join(self.cfg.scene_dir, frames[0]['file_path']))
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
            rgb_path = os.path.join(self.cfg.scene_dir, frame['file_path'])
            rgb = imageio.imread(rgb_path)
            gt_images.append(torch.from_numpy(rgb).float().to(self.device) / 255.0)

            # Depth
            depth_name = os.path.basename(frame['file_path']).replace("rgb", "depth").replace(".png", ".npy")
            depth_path = os.path.join(self.cfg.scene_dir, "depth_data", depth_name)
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

            world_points = unprojection(depth, self.intrinsics_tuple, c2w, self.device)
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

    def save_results(self):
        save_path = os.path.join(self.cfg.scene_dir, self.cfg.output_name)
        print(f"Saving model to {save_path}...")
        self.gs_model.save(save_path)


def visualize_similarity(runner, feature_map, text_query, img_index):
    """
    Visualizes the similarity map with better diagnostics.
    """
    # Get similarity with debug info
    sim_map = runner.clip_labels.query(feature_map, text_query)
    sim_np = sim_map.cpu().numpy()
    
    print(f"\n[VISUALIZATION] After normalization to [0,1]:")
    print(f"  Range: [{sim_np.min():.4f}, {sim_np.max():.4f}]")
    print(f"  Mean: {sim_np.mean():.4f}, Std: {sim_np.std():.4f}")
    
    # Get original image
    rgb_image = runner.gt_images[img_index].cpu().numpy()

    vis_data = (sim_np - sim_np.min()) / (sim_np.max() - sim_np.min() + 1e-8)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Heatmap overlay
    axes[1].imshow(rgb_image)
    # Mask values below 0.1 to make them transparent
    # vis_data_masked = np.ma.masked_where(vis_data < 0.05, vis_data)

    hm = axes[1].imshow(vis_data, cmap='jet', alpha=0.6, vmin=0.5, vmax=1)
    axes[1].set_title(f"Similarity Overlay\n'{text_query}'")
    axes[1].axis('off')
    plt.colorbar(hm, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Pure heatmap
    hm_pure = axes[2].imshow(vis_data, cmap='jet', vmin=0.5, vmax=1)
    axes[2].set_title("Pure Heatmap")
    axes[2].axis('off')
    plt.colorbar(hm_pure, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    save_path = f"sam_clip_{img_index}_{text_query.replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {save_path}")
    plt.show()

        


if __name__ == "__main__":
    config = Config()
    runner = Runner(config)

    # runner.run_training()
    # runner.save_results()

    text_query="a microwave"
    img_index=283

    image = runner.gt_images[img_index].cpu().numpy()


    time_start = time.time()
    feature_map = runner.clip_labels.extract_dense_features(image)
    time_end = time.time()
    print(f"Feature extraction time: {time_end - time_start:.2f} seconds")
    sim_map = runner.clip_labels.query(feature_map, text_query)
    print(f"Similarity time: {time.time() - time_end:.2f} seconds")

    visualize_similarity(runner, feature_map, text_query, img_index)