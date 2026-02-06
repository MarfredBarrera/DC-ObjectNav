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
from hashgrid import HashGrid

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

        # 2. Semantics
        self.sam_clip = semantics.SAM_CLIP_Semantics(self.cfg, device=self.device)
        self.clipseg = semantics.CLIPSeg(device=self.device)

        # 3. HashGrid
        self.hashgrid = HashGrid(self.cfg, device=self.device, transforms_json=os.path.join(self.cfg.scene_dir, "transforms.json"))
        
        # # Prepare data for Model Initialization
        # init_points, init_colors = self._create_initial_point_cloud()
        
        # # Create GSplat Model
        # intrinsics_dict = {'fx': self.fx, 'fy': self.fy, 'cx': self.cx, 'cy': self.cy, 'H': self.H, 'W': self.W}
        # self.gs_model = GaussianSplatting(self.cfg, init_points, init_colors, intrinsics_dict)

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

    def train_feature_field(self):
        print(f"Starting training for {self.cfg.iterations} iterations...")
        
        # Print HashGrid architecture
        print("\n=== HASHGRID CONFIGURATION ===")
        print(f"Feature dim: {self.hashgrid.feature_dim}")
        print(f"Encoding dim: {self.hashgrid.encoding_dim}")
        print(f"n_levels: {self.cfg.hash_n_levels}")
        print(f"n_features_per_level: {self.cfg.hash_n_features_per_level}")
        print(f"log2_hashmap_size: {self.cfg.hash_log2_hashmap_size}")
        print(f"base_resolution: {self.cfg.hash_base_resolution}")
        print(f"n_neurons: {self.cfg.hash_n_neurons}")
        print(f"n_hidden_layers: {self.cfg.hash_n_hidden_layers}")
        
        # Extract features
        clip_idx = 30
        depth = self.gt_depths[clip_idx]
        rgb = self.gt_images[clip_idx]
        c2w_hash = self.c2ws[clip_idx]
        
        rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)
        clip_features = self.sam_clip.extract_dense_features(rgb_np)
        
        print(f"\n=== CLIP FEATURES ===")
        print(f"Shape: {clip_features.shape}")
        print(f"Feature dim: {clip_features.shape[-1]}")
        print(f"Data type: {clip_features.dtype}")
        print(f"Min: {clip_features.min():.4f}, Max: {clip_features.max():.4f}")
        print(f"Mean: {clip_features.mean():.4f}, Std: {clip_features.std():.4f}")
        
        # Check if features are already normalized
        feature_norms = torch.norm(clip_features.reshape(-1, clip_features.shape[-1]), dim=-1)
        print(f"Feature norms - Min: {feature_norms.min():.4f}, Max: {feature_norms.max():.4f}, Mean: {feature_norms.mean():.4f}")
        
        # Precompute training data
        world_points = unprojection(depth, self.intrinsics_tuple, c2w_hash, self.device)
        fx, fy, cx, cy, H, W = self.intrinsics_tuple
        mask = (depth > 0.1) & (depth < 10.0)
        gt_features = clip_features[mask]


        # FILTER OUT ZERO-NORM FEATURES
        feature_norms = gt_features.norm(dim=-1)
        valid_mask = feature_norms > 1e-6  # Remove near-zero norm features
        
        world_points = world_points[valid_mask]
        gt_features = gt_features[valid_mask]
        
        print(f"Total training points: {world_points.shape[0]}")
        print(f"Filtered out {(~valid_mask).sum().item()} zero-norm features")
        
        print(f"\n=== SCENE GEOMETRY ===")
        print(f"Total points: {world_points.shape[0]}")
        print(f"Point cloud bounds:")
        print(f"  X: [{world_points[:, 0].min():.3f}, {world_points[:, 0].max():.3f}]")
        print(f"  Y: [{world_points[:, 1].min():.3f}, {world_points[:, 1].max():.3f}]")
        print(f"  Z: [{world_points[:, 2].min():.3f}, {world_points[:, 2].max():.3f}]")
        print(f"Scene bounds: {self.hashgrid.scene_bounds}")
        
        # Check normalized positions
        normalized_pos = self.hashgrid.normalize_positions(world_points[:1000])
        print(f"\nNormalized positions (first 1000):")
        print(f"  Min: {normalized_pos.min():.4f}, Max: {normalized_pos.max():.4f}")
        print(f"  Mean: {normalized_pos.mean():.4f}")
        
        # TEST: Can the network output diverse values at all?
        print("\n=== NETWORK OUTPUT TEST ===")
        with torch.no_grad():
            test_points = torch.rand(1000, 3, device=self.device)  # Random points in [0,1]
            test_output = self.hashgrid.model(test_points)
            print(f"Random input -> Output range: [{test_output.min():.4f}, {test_output.max():.4f}]")
            print(f"Output std: {test_output.std():.4f}")
            print(f"Output mean: {test_output.mean():.4f}")


            # After creating hashgrid, before training
        print("\n=== TESTING NETWORK CAPACITY ===")
        test_pts = torch.rand(100, 3, device=self.device)
        test_target = torch.randn(100, 512, device=self.device)
        test_target = test_target / (test_target.norm(dim=-1, keepdim=True) + 1e-8)

        test_optimizer = torch.optim.Adam(self.hashgrid.model.parameters(), lr=1e-2)

        for i in range(100):
            pred = self.hashgrid.model(test_pts)
            pred_norm = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
            loss = 1.0 - (pred_norm * test_target).sum(dim=-1).mean()
            
            test_optimizer.zero_grad()
            loss.backward()
            test_optimizer.step()
            
            if i % 20 == 0:
                print(f"Test step {i}: loss={loss.item():.4f}, output_range=[{pred.min():.4f}, {pred.max():.4f}]")        


        print(f"Starting training for {self.cfg.iterations} iterations...")
        start_time = time.time()
        
        # EXTRACT FEATURES ONCE
        clip_idx = 30
        depth = self.gt_depths[clip_idx]
        rgb = self.gt_images[clip_idx]
        c2w_hash = self.c2ws[clip_idx]
        
        rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)
        clip_features = self.sam_clip.extract_dense_features(rgb_np)
        
        print(f"Extracted CLIP features: {clip_features.shape}")
        print(f"CLIP feature dim: {clip_features.shape[-1]}")
        print(f"HashGrid output dim: {self.hashgrid.feature_dim}")
        
        # CHECK IF DIMENSIONS MATCH!
        if clip_features.shape[-1] != self.hashgrid.feature_dim:
            raise ValueError(f"Dimension mismatch! CLIP: {clip_features.shape[-1]} vs HashGrid: {self.hashgrid.feature_dim}")
        
        # Precompute all training data
        world_points = unprojection(depth, self.intrinsics_tuple, c2w_hash, self.device)
        fx, fy, cx, cy, H, W = self.intrinsics_tuple
        mask = (depth > 0.1) & (depth < 10.0)
        gt_features = clip_features[mask]
        
        print(f"Total training points: {world_points.shape[0]}")
        
        # CHECK SCENE BOUNDS
        print(f"Point cloud bounds:")
        print(f"  X: [{world_points[:, 0].min():.3f}, {world_points[:, 0].max():.3f}]")
        print(f"  Y: [{world_points[:, 1].min():.3f}, {world_points[:, 1].max():.3f}]")
        print(f"  Z: [{world_points[:, 2].min():.3f}, {world_points[:, 2].max():.3f}]")
        print(f"HashGrid bounds: {self.hashgrid.scene_bounds}")
        
        # Use learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.hashgrid.optimizer, mode='min', factor=0.5, patience=500
        )
        
        best_loss = float('inf')
    
        for step in range(self.cfg.iterations):
            # Sample batch
            num_points = world_points.shape[0]
            if num_points > self.cfg.hash_train_batch_size:
                indices = torch.randperm(num_points, device=self.device)[:self.cfg.hash_train_batch_size]
                batch_points = world_points[indices]
                batch_features = gt_features[indices]
            else:
                batch_points = world_points
                batch_features = gt_features
            
            # Forward pass - NO normalization
            pred_features = self.hashgrid.forward(batch_points, normalize=False)

            pred_norm = self.hashgrid.safe_normalize(pred_features)
            batch_features_norm = self.hashgrid.safe_normalize(batch_features)

            # MSE loss
            loss = ((pred_norm - batch_features_norm) ** 2).mean()
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"NaN detected at step {step}!")
                print(f"Pred features: min={pred_features.min():.4f}, max={pred_features.max():.4f}, mean={pred_features.mean():.4f}")
                print(f"Pred norms: min={pred_features.norm(dim=-1).min():.4f}, max={pred_features.norm(dim=-1).max():.4f}")
                print(f"GT norms: min={batch_features.norm(dim=-1).min():.4f}, max={batch_features.norm(dim=-1).max():.4f}")
                continue
            
            # Backward pass
            self.hashgrid.optimizer.zero_grad()
            loss.backward()
            
            # Check gradients
            total_norm = torch.nn.utils.clip_grad_norm_(self.hashgrid.model.parameters(), max_norm=1.0)
            if torch.isnan(total_norm):
                print(f"NaN gradients! Skipping step {step}...")
                continue
            
            self.hashgrid.optimizer.step()
            
            if step % 100 == 0:
                with torch.no_grad():
                    all_pred = self.hashgrid.forward(world_points, normalize=False)
                    all_pred_norm = self.hashgrid.safe_normalize(all_pred)
                    all_gt_norm = gt_features / (gt_features.norm(dim=-1, keepdim=True) + 1e-8)
                    val_loss = 1.0 - (all_pred_norm * all_gt_norm).sum(dim=-1).mean()
                    
                    # Track best
                    if val_loss < best_loss:
                        best_loss = val_loss
                        
                scheduler.step(val_loss)
                
                print(f"Step {step:04d} | Train Loss: {loss.item():.5f} | Val Loss: {val_loss.item():.5f} | Best: {best_loss:.5f} | LR: {self.hashgrid.optimizer.param_groups[0]['lr']:.2e} | Time: {time.time()-start_time:.1f}s")



    def save_results(self):
        save_path = os.path.join(self.cfg.scene_dir, self.cfg.output_name)
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
    pred_features = runner.hashgrid.render_feature_map(depth, c2w, runner.intrinsics_tuple)
    
    # Compute correlation
    mask = (depth > 0.1) & (depth < 10.0)
    gt_flat = gt_features[mask]
    pred_flat = pred_features[mask]
    
    # Normalize both
    gt_norm = gt_flat / (gt_flat.norm(dim=-1, keepdim=True) + 1e-8)
    pred_norm = pred_flat / (pred_flat.norm(dim=-1, keepdim=True) + 1e-8)
    
    cosine_sim = (gt_norm * pred_norm).sum(dim=-1)
    
    print(f"Cosine similarity stats:")
    print(f"  Mean: {cosine_sim.mean():.4f}")
    print(f"  Std: {cosine_sim.std():.4f}")
    print(f"  Min: {cosine_sim.min():.4f}")
    print(f"  Max: {cosine_sim.max():.4f}")
    
    # Test with a text query
    gt_sim = runner.sam_clip.query(gt_features, text_query)
    pred_sim = runner.hashgrid.query_similarity(
        depth, c2w, runner.intrinsics_tuple,
        text_query,
        runner.sam_clip.clip_processor,
        runner.sam_clip.clip_model
    )
    
    visualize_similarity(runner, gt_sim, clip_idx, text_query)
    visualize_similarity(runner, pred_sim, clip_idx, text_query)

def visualize_similarity(runner, similarity_map, img_index, text_query="a pillow"):
    """
    Visualizes the similarity map with better diagnostics.
    """
    # # Get similarity with debug info
    # sim_map = runner.clip_labels.query(feature_map, text_query)
    
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

    # vis_data = (similarity_np - similarity_np.min()) / (similarity_np.max() - similarity_np.min() + 1e-8)
    
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
    heatmap = axes[1].imshow(vis_data, cmap='jet', alpha=0.6, vmin=0, vmax=1)
    axes[1].set_title(f"Similarity Overlay")
    axes[1].axis('off')
    plt.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Pure heatmap
    heatmap_pure = axes[2].imshow(vis_data, cmap='jet', vmin=0, vmax=1)
    axes[2].set_title(f"Heatmap")
    axes[2].axis('off')
    plt.colorbar(heatmap_pure, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

        


if __name__ == "__main__":
    config = Config()
    runner = Runner(config)

    # runner.train_feature_field()
    # runner.hashgrid.save("hashgrid_model.pt")
    runner.hashgrid.load("hashgrid_model.pt")

    text_query = "a pillow"
    diagnose_hashgrid(runner, clip_idx=30, text_query=text_query)