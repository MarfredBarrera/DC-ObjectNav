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
        self.clip_labels = semantics.SemanticFeatureExtractor(device=self.device)
        
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

            # x_c, y_c, z_c = unprojection(depth, self.intrinsics_tuple, self.device)
            # x_c, y_c, z_c = x_c[mask], y_c[mask], z_c[mask]

            # cam_points = torch.stack([x_c, y_c, z_c, torch.ones_like(z_c)], dim=1)
            # world_points = (c2w @ cam_points.T).T
            
            # points_list.append(world_points[:, :3])

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
    def compute_similarity_and_visualize(self, img_index, text_query, method="similarity", invert_similarity=False, save_path=None):
        """
        Extracts dense features for a text query and visualizes the heatmap.
        
        Args:
            img_index: Index of the frame to process
            text_query: Text query string (e.g., "a pillow")
            method: "similarity" (cosine similarity, related objects get high scores) or 
                    "segmentation" (binary mask, only exact object gets high scores)
            invert_similarity: If True, invert similarity scores (1 - score). Use this if 
                              target objects are showing up as blue/low instead of red/high.
            save_path: Optional path to save the visualization
        Returns:
            similarity_map: torch.Tensor (H, W) with scores
        """
        print(f"\n=== Processing Frame {img_index} with query: '{text_query}' (method: {method}) ===")
        
        rgb_image = self.gt_images[img_index]
        
        # Choose extraction method
        if method == "segmentation":
            similarity_map = self.clip_labels.extract_dense_features_segmentation(rgb_image, text_query)
            title_suffix = "(Segmentation)"
        elif method == "similarity":
            similarity_map = self.clip_labels.extract_dense_features_similarity(rgb_image, text_query)
            title_suffix = "(Semantic Similarity)"
            
            # Option to invert if consistently backwards
            if invert_similarity:
                similarity_map = 1.0 - similarity_map
                title_suffix += " [INVERTED]"
                print("Applied inversion: 1 - score")
        else:
            raise ValueError(f"Unknown method: {method}. Use 'similarity' or 'segmentation'")
        
        # Convert to numpy for statistics and visualization
        similarity_np = similarity_map.cpu().numpy()
        
        # Print statistics to diagnose
        print(f"Raw score range: [{similarity_np.min():.4f}, {similarity_np.max():.4f}]")
        print(f"Mean: {similarity_np.mean():.4f}, Std: {similarity_np.std():.4f}")
        
        # DEBUG: Check a sample of high and low scoring pixels
        flat = similarity_np.flatten()
        high_idx = np.argmax(flat)
        low_idx = np.argmin(flat)
        high_y, high_x = np.unravel_index(high_idx, similarity_np.shape)
        low_y, low_x = np.unravel_index(low_idx, similarity_np.shape)
        print(f"Highest score pixel at ({high_y}, {high_x}): {flat[high_idx]:.4f}")
        print(f"Lowest score pixel at ({low_y}, {low_x}): {flat[low_idx]:.4f}")
        
        # For similarity method with low variance, apply adaptive contrast stretching
        if method == "similarity" and similarity_np.std() < 0.05:
            print("Warning: Low variance detected. Applying adaptive contrast stretching.")
            # Stretch the narrow range to full [0, 1] for visualization
            vis_data = (similarity_np - similarity_np.min()) / (similarity_np.max() - similarity_np.min() + 1e-8)
            vmin, vmax = 0, 1
        else:
            vis_data = similarity_np
            vmin, vmax = 0, 1
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(rgb_image.cpu().numpy())
        axes[0].set_title("Original RGB Image")
        axes[0].axis('off')
        
        # Heatmap overlay
        axes[1].imshow(rgb_image.cpu().numpy())
        heatmap = axes[1].imshow(vis_data, cmap='jet', alpha=0.6, vmin=vmin, vmax=vmax)
        axes[1].set_title(f"'{text_query}'\n{title_suffix}")
        axes[1].axis('off')
        plt.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Pure heatmap
        heatmap_pure = axes[2].imshow(vis_data, cmap='jet', vmin=vmin, vmax=vmax)
        axes[2].set_title(f"Heatmap")
        axes[2].axis('off')
        plt.colorbar(heatmap_pure, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path is None:
            suffix = "_inverted" if invert_similarity else ""
            save_path = f"{method}_{img_index}_{text_query.replace(' ', '_')}{suffix}.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
        plt.close()
        
        return similarity_map


if __name__ == "__main__":
    config = Config()
    runner = Runner(config)
    
    # # Method 1: Segmentation (works perfectly)
    # seg_map = runner.compute_similarity_and_visualize(
    #     img_index=20, 
    #     text_query="a stool",
    #     method="segmentation"
    # )

    runner.run_training()
    runner.save_results()

    text_query="a couch"
    img_index=321

    
    # Method 2: Semantic Similarity (normal)
    sim_map = runner.compute_similarity_and_visualize(
        img_index=img_index, 
        text_query=text_query,
        method="similarity"
    )
    
    # Method 3: Semantic Similarity (INVERTED - try this if target is blue)
    sim_map_inverted = runner.compute_similarity_and_visualize(
        img_index=img_index, 
        text_query=text_query,
        method="similarity",
        invert_similarity=True
    )