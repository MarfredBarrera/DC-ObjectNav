import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from utils import unprojection
import cv2


class Visualizer:
    """
    Visualizer for feature field outputs.
    Supports 2D image similarity visualization and bird's-eye view relevancy maps.
    """
    
    def __init__(self, runner, device="cuda"):
        """
        Args:
            runner: Runner object containing hashgrid, cameras, and scene data
            device: torch device
        """
        self.runner = runner
        self.device = device
        self.hashgrid = runner.hashgrid
        self.sam_clip = runner.sam_clip
        
    def visualize_2d_similarity(self, img_index, text_query, save_path=None, 
                                vmin=0.6, overlay_alpha=0.6):
        """
        Visualize similarity scores as a 2D heatmap overlay on the RGB image.
        
        Args:
            img_index: Index of the camera/image to visualize
            text_query: Text query for semantic similarity (e.g., "a pillow")
            save_path: Optional path to save the figure
            overlay_alpha: Alpha value for heatmap overlay (0-1)
            
        Returns:
            similarity_map: (H, W) numpy array of similarity scores
        """
        # Get image and camera data
        rgb_image = self.runner.gt_images[img_index].cpu().numpy()
        depth = self.runner.gt_depths[img_index]
        c2w = self.runner.c2ws[img_index]
        intrinsics = self.runner.intrinsics_tuple
        
        # Get predicted features from HashGrid
        pred_features = self.hashgrid.get_hashgrid_features(depth, c2w, intrinsics)
        
        # Query similarity
        similarity_map = self.sam_clip.query(pred_features, text_query)
        similarity_np = similarity_map.cpu().numpy()
        
        # Handle invalid values
        similarity_np = np.nan_to_num(similarity_np, nan=0.5, posinf=1.0, neginf=0.0)
        similarity_np = np.clip(similarity_np, 0.0, 1.0)
        
        # Print statistics
        print(f"Similarity Map Statistics for '{text_query}':")
        print(f"  Range: [{similarity_np.min():.4f}, {similarity_np.max():.4f}]")
        print(f"  Mean: {similarity_np.mean():.4f}, Std: {similarity_np.std():.4f}")

        # scaling
        vis_data = similarity_np - similarity_np.min()
        vis_data = vis_data / (vis_data.max() + 1e-8)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(rgb_image)
        axes[0].set_title("Original RGB Image", fontsize=14)
        axes[0].axis('off')
        
        # Heatmap overlay
        axes[1].imshow(rgb_image)
        heatmap = axes[1].imshow(vis_data, cmap='jet', alpha=overlay_alpha, vmin=vmin, vmax=1)
        axes[1].set_title(f"Similarity Overlay: '{text_query}'", fontsize=14)
        axes[1].axis('off')
        plt.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)
        
        axes[2].imshow(vis_data, cmap='jet', vmin=vmin, vmax=1)
        axes[2].set_title(f"Similarity Heatmap: '{text_query}'", fontsize=14)
        axes[2].axis('off')
        plt.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
        
        return similarity_np
    
    def create_birds_eye_view(self, text_query, grid_resolution=0.1, height_range=None,
                               aggregation='max', save_path=None, colormap='jet', vmin=0.6,
                               num_cameras=50, batch_size=5, downsample_points=True,
                               max_points_per_camera=50000):
        """
        Create a bird's-eye view (top-down) relevancy map by projecting 3D semantic features
        onto a 2D ground plane.
        
        Memory-efficient implementation that aggregates directly into grid without concatenating.
        
        Args:
            text_query: Text query for semantic similarity
            grid_resolution: Size of each grid cell in meters (default: 0.1m = 10cm)
            height_range: Tuple (min_height, max_height) to filter points by height.
                         If None, uses all points. Useful for filtering ground/ceiling.
            aggregation: How to aggregate multiple points in same cell ('max', 'mean', 'median')
            save_path: Optional path to save the figure
            colormap: Matplotlib colormap name
            num_cameras: Number of camera views to sample (default: 20)
            batch_size: Number of cameras to process at once (default: 5)
            downsample_points: Whether to downsample points per camera (default: True)
            max_points_per_camera: Max points to keep per camera if downsampling (default: 50000)
            
        Returns:
            bev_map: (H, W) numpy array of bird's-eye view relevancy scores
            grid_info: dict with grid metadata (bounds, resolution, etc.)
        """
        print(f"Generating bird's-eye view for query: '{text_query}'")
        
        # randomly sample camera indices
        num_samples = min(len(self.runner.gt_images), num_cameras)
        # indices = np.linspace(0, len(self.runner.gt_images)-1, num_samples, dtype=int)
        indices = np.random.choice(len(self.runner.gt_images), num_samples, replace=False)
        
        print(f"Sampling {num_samples} camera views in batches of {batch_size}...")
        
        # First pass: determine grid bounds
        print("Pass 1: Determining scene bounds...")
        x_coords_list = []
        y_coords_list = []
        
        for batch_start in range(0, len(indices), batch_size):
            batch_end = min(batch_start + batch_size, len(indices))
            batch_indices = indices[batch_start:batch_end]
            
            for idx in batch_indices:
                depth = self.runner.gt_depths[idx]
                c2w = self.runner.c2ws[idx]
                intrinsics = self.runner.intrinsics_tuple
                
                # Get 3D world points
                from utils import unprojection
                world_points = unprojection(depth, intrinsics, c2w, self.device)
                
                # Filter by height early (Y is up)
                if height_range is not None:
                    min_h, max_h = height_range
                    height_mask = (world_points[:, 1] >= min_h) & (world_points[:, 1] <= max_h)
                    world_points = world_points[height_mask]
                
                # Downsample for bounds estimation
                if world_points.shape[0] > 10000:
                    sample_idx = torch.randperm(world_points.shape[0])[:10000]
                    world_points = world_points[sample_idx]
                
                # Store XZ coords for top-down view (X and Z are horizontal, Y is height)
                x_coords_list.append(world_points[:, 0].cpu())
                y_coords_list.append(world_points[:, 2].cpu())
            
            # Clear memory
            del world_points
            torch.cuda.empty_cache()
        
        # Compute bounds
        x_min, x_max = self.runner.bounds_min[0], self.runner.bounds_max[0]
        y_min, y_max = self.runner.bounds_min[2], self.runner.bounds_max[2]  # Using Z axis for horizontal
        
        # Add padding
        padding = 0.5  # meters
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding
        
        # Create grid
        x_bins = np.arange(x_min, x_max + grid_resolution, grid_resolution)
        y_bins = np.arange(y_min, y_max + grid_resolution, grid_resolution)
        
        grid_h = len(y_bins) - 1
        grid_w = len(x_bins) - 1
        
        print(f"Grid size: {grid_w} x {grid_h} (resolution: {grid_resolution}m)")
        print(f"Grid bounds: X[{x_min:.2f}, {x_max:.2f}], Z[{y_min:.2f}, {y_max:.2f}]")
        
        # Initialize grid structures
        if aggregation == 'max':
            bev_map = np.zeros((grid_h, grid_w), dtype=np.float32)
        elif aggregation == 'mean':
            bev_sum = np.zeros((grid_h, grid_w), dtype=np.float32)
            bev_count = np.zeros((grid_h, grid_w), dtype=np.int32)
        elif aggregation == 'median':
            # Store lists of values for each cell
            cell_values = {}
        
        # Second pass: compute features and aggregate into grid
        print(f"\nPass 2: Computing features and aggregating (method: {aggregation})...")
        total_points_processed = 0
        
        for batch_start in range(0, len(indices), batch_size):
            batch_end = min(batch_start + batch_size, len(indices))
            batch_indices = indices[batch_start:batch_end]
            
            print(f"  Batch {batch_start//batch_size + 1}/{(len(indices)-1)//batch_size + 1} "
                  f"(cameras {batch_start}-{batch_end-1})...")
            
            for idx in batch_indices:
                depth = self.runner.gt_depths[idx]
                c2w = self.runner.c2ws[idx]
                intrinsics = self.runner.intrinsics_tuple
                
                # Get 3D world points
                from utils import unprojection
                world_points = unprojection(depth, intrinsics, c2w, self.device)
                
                # Get features from HashGrid
                feature_map = self.hashgrid.get_hashgrid_features(depth, c2w, intrinsics)
                
                # Reshape to (H, W, D) if needed
                if feature_map.dim() == 2:  # If shape is (H*W, D)
                    feature_map = feature_map.view(depth.shape[0], depth.shape[1], -1)
                
                # Create mask and fill invalid positions with zeros
                mask = (depth > 0.1) & (depth < 10.0)
                mask_3d = mask.unsqueeze(-1)  # Shape: (H, W, 1)
                feature_map = feature_map * mask_3d.float()  # Zero out invalid positions
                
                # Compute similarity scores for the full (H, W, D) feature map
                similarity_scores = self.sam_clip.query(feature_map, text_query)
                similarity_scores = similarity_scores.cpu().numpy()
                
                # Flatten and extract only valid points
                mask_flat = mask.reshape(-1).cpu().numpy()
                world_points_flat = world_points.cpu().numpy()
                similarity_scores_flat = similarity_scores.reshape(-1)
                
                # Extract valid points and their similarity scores
                world_points_valid = world_points_flat[mask_flat]
                similarity_scores_valid = similarity_scores_flat[mask_flat]
                
                # Filter by height (Y is up)
                if height_range is not None:
                    min_h, max_h = height_range
                    height_mask = (world_points_valid[:, 1] >= min_h) & (world_points_valid[:, 1] <= max_h)
                    world_points_valid = world_points_valid[height_mask]
                    similarity_scores_valid = similarity_scores_valid[height_mask]
                
                # Downsample if needed
                if downsample_points and len(world_points_valid) > max_points_per_camera:
                    downsample_indices = np.random.permutation(len(world_points_valid))[:max_points_per_camera]
                    world_points_valid = world_points_valid[downsample_indices]
                    similarity_scores_valid = similarity_scores_valid[downsample_indices]
                
                if len(world_points_valid) == 0:
                    continue
                
                # Get XZ coordinates for top-down view (X, Z are horizontal)
                points_xy = world_points_valid[:, [0, 2]]
                
                # Bin points into grid
                x_indices = np.searchsorted(x_bins, points_xy[:, 0]) - 1
                y_indices = np.searchsorted(y_bins, points_xy[:, 1]) - 1
                
                # Clip to valid range
                x_indices = np.clip(x_indices, 0, grid_w - 1)
                y_indices = np.clip(y_indices, 0, grid_h - 1)
                
                # Aggregate into grid
                if aggregation == 'max':
                    for i in range(len(points_xy)):
                        xi, yi = x_indices[i], y_indices[i]
                        bev_map[yi, xi] = max(bev_map[yi, xi], similarity_scores_valid[i])
                
                elif aggregation == 'mean':
                    for i in range(len(points_xy)):
                        xi, yi = x_indices[i], y_indices[i]
                        bev_sum[yi, xi] += similarity_scores_valid[i]
                        bev_count[yi, xi] += 1
                
                elif aggregation == 'median':
                    for i in range(len(points_xy)):
                        xi, yi = x_indices[i], y_indices[i]
                        key = (yi, xi)
                        if key not in cell_values:
                            cell_values[key] = []
                        cell_values[key].append(similarity_scores_valid[i])
                
                total_points_processed += len(points_xy)
                
                # Free memory immediately
                del world_points, feature_map, similarity_scores, points_xy
                torch.cuda.empty_cache()
            
            print(f"    Points processed so far: {total_points_processed}")
        
        # Finalize aggregation
        if aggregation == 'mean':
            print("Finalizing mean aggregation...")
            bev_map = np.zeros((grid_h, grid_w), dtype=np.float32)
            mask = bev_count > 0
            bev_map[mask] = bev_sum[mask] / bev_count[mask]
        
        elif aggregation == 'median':
            print("Finalizing median aggregation...")
            bev_map = np.zeros((grid_h, grid_w), dtype=np.float32)
            for (yi, xi), values in cell_values.items():
                bev_map[yi, xi] = np.median(values)
        
        # Flip vertically for proper visualization (matplotlib origin is top-left)
        bev_map = np.flipud(bev_map)
        
        print(f"\nTotal points processed: {total_points_processed}")
        print(f"Non-zero cells: {np.count_nonzero(bev_map)} / {bev_map.size} ({100*np.count_nonzero(bev_map)/bev_map.size:.1f}%)")

        
        # Store grid metadata (note: y_min/y_max represent Z axis bounds for top-down view)
        grid_info = {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,  # Actually Z axis min
            'y_max': y_max,  # Actually Z axis max
            'resolution': grid_resolution,
            'width': grid_w,
            'height': grid_h,
            'aggregation': aggregation
        }
        
        # Visualize
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create extent for proper axis labels
        extent = [x_min, x_max, y_min, y_max]

        # Percentile-based color scaling to focus on higher values
        # Ignore low values by using percentile clipping
        # Adjust percentiles (e.g., 10, 30, 50 for p_low) to control focus on high values
        non_zero_values = bev_map[bev_map > 0]
        if len(non_zero_values) > 0:
            # Use percentiles to focus on variation in higher scores
            p_low = np.percentile(non_zero_values, 20)  # Ignore bottom 20% (increase to ignore more)
            p_high = np.percentile(non_zero_values, 99)  # Cap at 99th percentile
            
            print(f"Color scaling: {p_low:.4f} (20th percentile) to {p_high:.4f} (99th percentile)")
            
            # Clip and normalize to [0, 1] range
            bev_map_scaled = np.clip(bev_map, p_low, p_high)
            bev_map_scaled = (bev_map_scaled - p_low) / (p_high - p_low + 1e-8)
        else:
            bev_map_scaled = bev_map
        
        # Set zero/background values to NaN for grey background
        bev_map_scaled = bev_map_scaled.copy()
        bev_map_scaled[bev_map == 0] = np.nan
        
        # Create colormap with grey background for NaN values
        cmap = plt.get_cmap(colormap).copy()
        cmap.set_bad(color='lightgray')
        
        im = ax.imshow(bev_map_scaled, cmap=cmap, aspect='equal', extent=extent, 
                      vmin=0, vmax=1, interpolation='nearest')
        
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Z (meters)', fontsize=12)
        ax.set_title(f"Bird's-Eye View Relevancy: '{text_query}' (Top-Down: X-Z plane)\n"
                    f"Resolution: {grid_resolution}m, Aggregation: {aggregation}", 
                    fontsize=14)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Similarity Score', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved bird's-eye view to {save_path}")
        
        plt.show()
        
        print(f"\nBEV Map Statistics:")
        print(f"  Range: [{bev_map.min():.4f}, {bev_map.max():.4f}]")
        print(f"  Mean: {bev_map.mean():.4f}, Std: {bev_map.std():.4f}")
        
        return bev_map, grid_info
    
    def create_multi_query_bev(self, text_queries, grid_resolution=0.1, 
                                height_range=None, save_path=None):
        """
        Create multiple bird's-eye views for different queries side-by-side.
        
        Args:
            text_queries: List of text queries
            grid_resolution: Grid cell size in meters
            height_range: Optional (min_height, max_height) filter
            save_path: Optional path to save figure
            
        Returns:
            bev_maps: List of BEV maps
            grid_info: Shared grid metadata
        """
        num_queries = len(text_queries)
        fig, axes = plt.subplots(1, num_queries, figsize=(6*num_queries, 5))
        
        if num_queries == 1:
            axes = [axes]
        
        bev_maps = []
        
        for idx, query in enumerate(text_queries):
            print(f"\nProcessing query {idx+1}/{num_queries}: '{query}'")
            
            bev_map, grid_info = self.create_birds_eye_view(
                text_query=query,
                grid_resolution=grid_resolution,
                height_range=height_range,
                save_path=None  # Don't save individual maps
            )
            
            bev_maps.append(bev_map)
            
            # Plot on subplot
            extent = [grid_info['x_min'], grid_info['x_max'], 
                     grid_info['y_min'], grid_info['y_max']]
            
            im = axes[idx].imshow(bev_map, cmap='jet', aspect='auto', 
                                 extent=extent, vmin=0, vmax=1)
            axes[idx].set_xlabel('X (meters)')
            axes[idx].set_ylabel('Y (meters)')
            axes[idx].set_title(f"'{query}'")
            axes[idx].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nSaved multi-query BEV to {save_path}")
        
        plt.show()
        plt.close()
        
        return bev_maps, grid_info
    
    def compare_gt_vs_pred(self, img_index, text_query, save_path=None):
        """
        Compare ground truth CLIP features vs predicted HashGrid features.
        
        Args:
            img_index: Camera index to visualize
            text_query: Text query for comparison
            save_path: Optional save path
        """
        # Get data
        depth = self.runner.gt_depths[img_index]
        rgb = self.runner.gt_images[img_index]
        c2w = self.runner.c2ws[img_index]
        intrinsics = self.runner.intrinsics_tuple
        
        # Ground truth features
        rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)
        gt_features = self.sam_clip.extract_dense_features(rgb_np)
        gt_sim = self.sam_clip.query(gt_features, text_query).cpu().numpy()
        
        # Predicted features
        pred_features = self.hashgrid.get_hashgrid_features(depth, c2w, intrinsics)
        pred_sim = self.sam_clip.query(pred_features, text_query).cpu().numpy()
        
        # Compute difference
        diff = np.abs(gt_sim - pred_sim)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Ground truth
        im0 = axes[0, 0].imshow(gt_sim, cmap='jet', vmin=0, vmax=1)
        axes[0, 0].set_title(f"Ground Truth: '{text_query}'")
        axes[0, 0].axis('off')
        plt.colorbar(im0, ax=axes[0, 0])
        
        # Prediction
        im1 = axes[0, 1].imshow(pred_sim, cmap='jet', vmin=0, vmax=1)
        axes[0, 1].set_title(f"HashGrid Prediction: '{text_query}'")
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Difference
        im2 = axes[1, 0].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        axes[1, 0].set_title(f"Absolute Difference")
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0])
        
        # Original RGB
        axes[1, 1].imshow(rgb.cpu().numpy())
        axes[1, 1].set_title("Original RGB")
        axes[1, 1].axis('off')
        
        # Compute metrics
        mask = (depth > 0.1) & (depth < 10.0)
        gt_flat = gt_features[mask]
        pred_flat = pred_features[mask]
        
        gt_norm = gt_flat / (gt_flat.norm(dim=-1, keepdim=True) + 1e-8)
        pred_norm = pred_flat / (pred_flat.norm(dim=-1, keepdim=True) + 1e-8)
        cosine_sim = (gt_norm * pred_norm).sum(dim=-1).cpu().numpy()
        
        # Add statistics text
        stats_text = f"Feature Cosine Similarity:\n"
        stats_text += f"  Mean: {cosine_sim.mean():.4f}\n"
        stats_text += f"  Std: {cosine_sim.std():.4f}\n\n"
        stats_text += f"Similarity Score Difference:\n"
        stats_text += f"  Mean: {diff[mask.cpu().numpy()].mean():.4f}\n"
        stats_text += f"  Max: {diff[mask.cpu().numpy()].max():.4f}"
        
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison to {save_path}")
        
        plt.show()