from sklearn import ensemble
import torch
import numpy as np
import os

class BEVGrid:
    """
    Base class for 2D Bird's Eye View Grids.
    Handles grid initialization, dimensions, and coordinate transforms.
    """
    def __init__(self, config, scene_bounds, device=None):
        self.cfg = config
        self.device = device if device else config.device
    
        # BEV Map
        self.bev_resolution = 0.05
        self.bev_initialized = False
        
        # Dimensions
        self.bev_width = 0
        self.bev_height = 0
        self.bev_min_x = 0.0
        self.bev_max_x = 0.0
        self.bev_min_z = 0.0
        self.bev_max_z = 0.0
        self.initialize_from_bounds(scene_bounds)

    def initialize_from_bounds(self, scene_bounds):
        """Calculates grid dimensions from scene bounds."""
        if isinstance(scene_bounds, torch.Tensor):
            min_b = scene_bounds[0].cpu().numpy()
            max_b = scene_bounds[1].cpu().numpy()
        else:
            min_b = scene_bounds[0]
            max_b = scene_bounds[1]
        
        # Extract bounds (assuming Y is up, X is left-right, Z is forward-back)
        self.bev_min_x = float(min_b[0])
        self.bev_max_x = float(max_b[0])
        self.bev_min_y = float(min_b[1])
        self.bev_max_y = float(max_b[1])
        self.bev_min_z = float(min_b[2])
        self.bev_max_z = float(max_b[2])
        
        # CRITICAL FIX: Calculate grid size based on ACTUAL scene bounds
        x_span = self.bev_max_x - self.bev_min_x
        z_span = self.bev_max_z - self.bev_min_z
        
        self.bev_width = int(np.ceil(x_span / self.bev_resolution))
        self.bev_height = int(np.ceil(z_span / self.bev_resolution))
        
        print(f"Scene Bounds:")
        print(f"  X: [{self.bev_min_x:.2f}, {self.bev_max_x:.2f}] m (span: {x_span:.2f} m)")
        print(f"  Z: [{self.bev_min_z:.2f}, {self.bev_max_z:.2f}] m (span: {z_span:.2f} m)")
        print(f"BEV Grid Initialized: {self.bev_width} x {self.bev_height} cells ({self.bev_resolution}m/cell)")
        self.bev_initialized = True

    def generate_bev_sample_points(self, height_filter=None, height_samples=50):
        """
        Generate 3D sample points for BEV evaluation.
        
        Args:
            height_filter: Tuple of (min_y, max_y) coordinates (height) to sample. 
                          If None, uses scene bounds.
            height_samples: Number of height levels to sample (default: 50)
            
        Returns:
            points: numpy array of shape (N, 3) where N = bev_width * height_samples * bev_height
            grid_shape: Tuple (bev_width, height_samples, bev_height) for reshaping
        """
        if not self.bev_initialized:
            raise RuntimeError("BEV grid not initialized!")
        
        # Use scene bounds if not specified
        if height_filter is None:
            min_y = self.bev_min_y
            max_y = self.bev_max_y
        else:
            min_y, max_y = height_filter
        
        # Generate all grid cell centers
        x_coords = np.linspace(self.bev_min_x + self.bev_resolution/2, self.bev_max_x - self.bev_resolution/2, self.bev_width)
        y_coords = np.linspace(min_y, max_y, height_samples) # Vertical dimension
        z_coords = np.linspace(self.bev_min_z + self.bev_resolution/2, self.bev_max_z - self.bev_resolution/2, self.bev_height)
        
        # Create 3D meshgrid in (Y, Z, X) order to match memory-efficient layouts
        Y, Z, X = np.meshgrid(y_coords, z_coords, x_coords, indexing='ij')
        
        # Stack into (N, 3) array of 3D points
        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)
        grid_shape = (height_samples, self.bev_height, self.bev_width)
        
        return points, grid_shape


class SimilarityGrid(BEVGrid):
    """
    Subclass for Semantic Similarity Mapping.
    Computes 2D BEV map of similarity between ensemble features and text query.
    """
    def __init__(self, config, ensemble, sam_clip, scene_bounds):
        super().__init__(config, scene_bounds, device=config.device)
        self.ensemble = ensemble
        self.sam_clip = sam_clip
        self.bev_sim_map_3d = None

    def compute_similarity_map(self, text_query, height_samples=200, batch_size=100000, occupancy_grid=None):
        if not self.bev_initialized:
            print("BEV grid not initialized!")
            return None

        # 1. Embed Text
        inputs = self.sam_clip.clip_processor(text=[text_query], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_embed = self.sam_clip.clip_model.get_text_features(**inputs)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

        min_y, max_y = self.bev_min_y, self.bev_max_y
        points, grid_shape = self.generate_bev_sample_points(height_filter=(min_y, max_y), height_samples=height_samples)
        total_points = points.shape[0]

        # Filter points by occupancy if grid is provided
        mask = None
        if occupancy_grid is not None and occupancy_grid.occupancy_map is not None:
            x_idx = ((points[:, 0] - self.bev_min_x) / self.bev_resolution).astype(np.int64)
            z_idx = ((points[:, 2] - self.bev_min_z) / self.bev_resolution).astype(np.int64)
            y_idx = ((points[:, 1] - self.bev_min_y) / (self.bev_max_y - self.bev_min_y + 1e-6) * (occupancy_grid.height_samples - 1)).astype(np.int64)
            
            x_idx = np.clip(x_idx, 0, self.bev_width - 1)
            z_idx = np.clip(z_idx, 0, self.bev_height - 1)
            y_idx = np.clip(y_idx, 0, occupancy_grid.height_samples - 1)
            
            x_indices = torch.from_numpy(x_idx).to(self.device)
            z_indices = torch.from_numpy(z_idx).to(self.device)
            y_indices = torch.from_numpy(y_idx).to(self.device)

            occ_map = occupancy_grid.occupancy_map
            mask = (occ_map[y_indices, z_indices, x_indices] == 1)
            query_points = points[mask.cpu().numpy()]
        else:
            query_points = points

        total_query_points = query_points.shape[0]
        if total_query_points == 0:
            return np.zeros((self.bev_height, self.bev_width))
            
        points_tensor = torch.from_numpy(query_points).float().to(self.device)
        
        # 3. Ensemble Forward Pass (Mean Feature)
        all_query_sims = []
        num_batches = int(np.ceil(total_query_points / batch_size))
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, total_query_points)
                batch_points = points_tensor[start_idx:end_idx]
                
                # Get mean feature from each model and average them
                batch_means = []
                for model in self.ensemble:
                    # forward returns (mean, variance). We want normalized mean.
                    mean, _ = model.forward(batch_points, normalize=True)
                    batch_means.append(mean)
                
                # Stack and Average: (Num_Models, B, D) -> (B, D)
                ensemble_mean = torch.stack(batch_means, dim=0).mean(dim=0)
                
                # Re-normalize the ensemble mean for Cosine Similarity
                ensemble_mean = ensemble_mean / (ensemble_mean.norm(dim=-1, keepdim=True) + 1e-8)
                
                # 4. Compute Similarity
                # (B, D) @ (1, D).T -> (B, 1)
                sim = torch.matmul(ensemble_mean, text_embed.T).squeeze(-1)
                
                # Normalize [-1, 1] -> [0, 1]
                sim = (sim + 1.0) / 2.0
                all_query_sims.append(sim)
        
        all_query_sims = torch.cat(all_query_sims, dim=0) # (total_query_points,)

        # 4. Reconstruct full 3D similarity grid
        full_sims = torch.zeros(total_points, device=self.device)
        if mask is not None:
            full_sims[mask] = all_query_sims
        else:
            full_sims = all_query_sims
            
        self.bev_sim_map_3d = full_sims.reshape(grid_shape) # (Y, Z, X)

        return 
    
    def get_2d_map(self):
        """Returns 2D BEV similarity map (max similarity across height)."""
        if self.bev_sim_map_3d is None:
            return None
        # Max over height samples (dim 0) -> (Z, X)
        grid_2d, _ = self.bev_sim_map_3d.max(dim=0)
        return grid_2d
    
    def save(self, step):
        """Save similarity map to disk."""
        sim_2d = self.get_2d_map()
        if sim_2d is not None:
            sim_maps_dir = os.path.join(self.cfg.output_dir, "sim_maps")
            os.makedirs(sim_maps_dir, exist_ok=True)
            path = os.path.join(sim_maps_dir, f"bev_similarity_{step}.npy")
            np.save(path, sim_2d.cpu().numpy())
            print(f"Similarity map saved to {path}")

class OccupancyGrid(BEVGrid):
    """
    Subclass for Binary Occupancy Mapping.
    Records visited areas on the grid.
    """
    def __init__(self, config, scene_bounds, device=None, height_samples=100):
        self.height_samples = height_samples
        self.occupancy_map = None
        super().__init__(config, scene_bounds, device=device)

    def initialize_from_bounds(self, scene_bounds):
        super().initialize_from_bounds(scene_bounds)
        self._init_map()
        
    def _init_map(self):
        # Initialize 3D zero map (0 = free/unknown, 1 = occupied/visited)
        # Shape: (Y/Height_Samples, Z/Height, X/Width)
        if self.bev_width > 0 and self.bev_height > 0:
            self.occupancy_map = torch.zeros((self.height_samples, self.bev_height, self.bev_width), device=self.device, dtype=torch.uint8)

    def update(self, points, min_y=None, max_y=None):
        """Update occupancy map from 3D points within the specified height range."""
        if not self.bev_initialized: return

        # Use provided filter range or fall back to scene bounds
        f_min_y = min_y if min_y is not None else self.bev_min_y
        f_max_y = max_y if max_y is not None else self.bev_max_y

        mask = (points[:, 0] >= self.bev_min_x) & (points[:, 0] < self.bev_max_x) & \
            (points[:, 2] >= self.bev_min_z) & (points[:, 2] < self.bev_max_z) & \
            (points[:, 1] >= f_min_y) & (points[:, 1] < f_max_y)
            
        valid_points = points[mask]
        
        if valid_points.shape[0] == 0: return

        # Map to grid indices
        x_indices = ((valid_points[:, 0] - self.bev_min_x) / self.bev_resolution).long()
        z_indices = ((valid_points[:, 2] - self.bev_min_z) / self.bev_resolution).long()
        # Map world Y coordinate to grid index based on the full vertical range [bev_min_y, bev_max_y]
        y_indices = ((valid_points[:, 1] - self.bev_min_y) / (self.bev_max_y - self.bev_min_y + 1e-6) * (self.height_samples - 1)).long()
        
        # Clamp to be safe
        x_indices = torch.clamp(x_indices, 0, self.bev_width - 1)
        z_indices = torch.clamp(z_indices, 0, self.bev_height - 1)
        y_indices = torch.clamp(y_indices, 0, self.height_samples - 1)
        
        # Set occupancy (1 for occupied)
        self.occupancy_map[y_indices, z_indices, x_indices] = 1

    def get_2d_map(self, min_y=None, max_y=None):
        """Returns 2D BEV occupancy map (1 if any cell in column is occupied)."""
        if self.occupancy_map is None: return None

        # Use provided height slice or default to full scene range
        f_min_y = min_y if min_y is not None else self.bev_min_y
        f_max_y = max_y if max_y is not None else self.bev_max_y

        # Map world height to grid indices based on the grid's Y range [bev_min_y, bev_max_y]
        min_y_idx = int((f_min_y - self.bev_min_y) / (self.bev_max_y - self.bev_min_y + 1e-6) * (self.height_samples - 1))
        max_y_idx = int((f_max_y - self.bev_min_y) / (self.bev_max_y - self.bev_min_y + 1e-6) * (self.height_samples - 1))
        min_y_idx = max(0, min_y_idx)
        max_y_idx = min(self.height_samples - 1, max_y_idx)
        occ_slice = self.occupancy_map[min_y_idx:max_y_idx+1, :, :]
        # Max over height samples
        occ_2d, _ = occ_slice.max(dim=0)
        return occ_2d

    def save(self, step):
        """Save occupancy map to disk."""
        if self.occupancy_map is not None:
            occ_2d = self.get_2d_map()
            occ_maps_dir = os.path.join(self.cfg.output_dir, "occ_maps")
            os.makedirs(occ_maps_dir, exist_ok=True)
            path = os.path.join(occ_maps_dir, f"bev_occupancy_{step}.npy")
            np.save(path, occ_2d.cpu().numpy())
            print(f"Occupancy map saved to {path}")


class UncertaintyGrid(BEVGrid):
    """
    Subclass for Uncertainty Mapping (Epistemic & Aleatoric).
    Maintains a 2D Bird's Eye View Grid representing feature field uncertainty.
    """
    def __init__(self, config, ensemble, scene_bounds):
        super().__init__(config, scene_bounds, device=config.device)
        self.ensemble = ensemble
        
        # Core Tensors
        self.bev_epi_umap = None
        self.bev_ale_umap = None

        # 3D maps
        self.bev_epi_umap_3d = None
        self.bev_ale_umap_3d = None

    def forward_single(self, model, points, batch_size=100000):
        """
        Forward pass points through a single ensemble member with batching.
        
        Args:
            model: Single hashgrid model from the ensemble
            points: Tensor of 3D points (N, 3)
            batch_size: Number of points to process at once
            
        Returns:
            predictions: Tensor of predicted features (N, feature_dim)
            variances: Tensor of aleatoric variances (N, feature_dim)
        """
        total_points = points.shape[0]
        predictions_list = []
        variances_list = []
        num_batches = int(np.ceil(total_points / batch_size))
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, total_points)
                batch_points = points[start_idx:end_idx]
                
                mean, variance = model.forward(batch_points, normalize=True)
                predictions_list.append(mean)
                variances_list.append(variance)
        
        predictions = torch.cat(predictions_list, dim=0)
        variances = torch.cat(variances_list, dim=0)
        
        return predictions, variances
    
    def forward_ensemble(self, points, batch_size=100000):
        """
        Forward pass points through all ensemble members.
        
        Args:
            points: Tensor of 3D points (N, 3) on device
            batch_size: Number of points to process at once per model
            
        Returns:
            ensemble_predictions: Tensor of shape (num_models, N, feature_dim)
            ensemble_variances: Tensor of shape (num_models, N, feature_dim)
        """
        ensemble_predictions = []
        ensemble_variances = []
        
        for idx, model in enumerate(self.ensemble):
            predictions, variances = self.forward_single(model, points, batch_size)
            ensemble_predictions.append(predictions)
            ensemble_variances.append(variances)
        
        # Stack: (num_models, N, feature_dim)
        ensemble_predictions = torch.stack(ensemble_predictions, dim=0)
        ensemble_variances = torch.stack(ensemble_variances, dim=0)
        
        return ensemble_predictions, ensemble_variances
    
    # ========== Uncertainty Computation Functions ==========
    
    def compute_epistemic_uncertainty(self, ensemble_predictions):
        """
        Compute epistemic uncertainty (variance across ensemble predictions).
        
        Args:
            ensemble_predictions: Tensor of shape (num_models, N, feature_dim)
            
        Returns:
            epistemic: Tensor of shape (N,) - mean variance across features
        """
        # Compute variance across ensemble (dim=0), then average across features (dim=-1)
        epistemic = ensemble_predictions.var(dim=0).mean(dim=-1)
        return epistemic
    
    def compute_aleatoric_uncertainty(self, ensemble_variances):
        """
        Compute aleatoric uncertainty (mean of predicted variances).
        
        Args:
            ensemble_variances: Tensor of shape (num_models, N, feature_dim)
            
        Returns:
            aleatoric: Tensor of shape (N,) - mean variance across ensemble and features
        """
        # Average across ensemble (dim=0), then average across features (dim=-1)
        aleatoric = ensemble_variances.mean(dim=0).mean(dim=-1)
        return aleatoric

    
    def aggregate_height_samples(self, uncertainties, grid_shape):
        """
        Reshape and aggregate uncertainties along height dimension.
        
        Args:
            uncertainties: Tensor of shape (N,) on CPU
            grid_shape: Tuple (height_samples, bev_height, bev_width)
            
        Returns:
            bev_2d: Tensor of shape (bev_height, bev_width) - BEV uncertainty map
        """
        # uncertainties are flat, reshape to (Y, Z, X)
        grid_3d = uncertainties.reshape(grid_shape)
        
        # Average along height dimension (axis 0) -> (Z, X)
        bev_2d = grid_3d.mean(dim=0)

        return bev_2d
    
    # ========== High-Level Pipeline Function ==========

    def forward_pass(self, height_filter=None, height_samples=10, batch_size=100000):
        """
        Complete pipeline: Generate points, run ensemble, compute uncertainties, aggregate.
        
        Args:
            height_filter: Tuple of (min_y, max_y) coordinates (height) to sample. 
                          If None, uses scene bounds.
            height_samples: Number of height levels to sample (default: 10)
            batch_size: Number of points to process at once (to avoid OOM)
        
        Returns:
            None (stores results in self.bev_epi_umap and self.bev_ale_umap)
        """
        if not self.bev_initialized:
            print("BEV grid not initialized!")
            return
        
        # Step 1: Generate sampling points
        points, grid_shape = self.generate_bev_sample_points(height_filter, height_samples)
        total_points = points.shape[0]
        print(f"Processing {total_points:,} points ({grid_shape[0]}x{grid_shape[1]}x{grid_shape[2]})...")
        
        # Step 2: Convert to tensor and run ensemble forward pass
        points_tensor = torch.from_numpy(points).float().to(self.device)
        print(f"Running forward pass through {len(self.ensemble)} ensemble members...")
        ensemble_predictions, ensemble_variances = self.forward_ensemble(points_tensor, batch_size)
        
        # Step 3: Compute uncertainties
        epistemic_uncertainty = self.compute_epistemic_uncertainty(ensemble_predictions)
        aleatoric_uncertainty = self.compute_aleatoric_uncertainty(ensemble_variances)
        
        # Step 3b: Store 3D
        self.bev_epi_umap_3d = epistemic_uncertainty
        self.bev_ale_umap_3d = aleatoric_uncertainty

        # Step 4: Move to CPU and aggregate height samples
        epi_2d = self.aggregate_height_samples(epistemic_uncertainty.cpu(), grid_shape)
        ale_2d = self.aggregate_height_samples(aleatoric_uncertainty.cpu(), grid_shape)

        # Step 5: Store results
        self.set_umaps(epi_2d, ale_2d)
        
        return

    # ========== Utility Functions ==========

    def set_umaps(self, epi_map, ale_map):
        self.bev_epi_umap = epi_map
        self.bev_ale_umap = ale_map
    
    def get_umaps(self):
        return self.bev_epi_umap, self.bev_ale_umap
    
    def clear_umaps(self):
        self.bev_epi_umap = None
        self.bev_ale_umap = None
        self.bev_epi_umap_3d = None
        self.bev_ale_umap_3d = None
        torch.cuda.empty_cache()
    
    def visualize_bev_map(self, save_path=None, show=False, height_filter=None):
        """
        Args:
            save_path: Path to save the visualization (optional)
            show: Whether to display the plot (default: False)
        
        Returns:
            fig, ax: matplotlib figure and axis objects
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        

        # Reshape uncertainty map to 2D grid
        uncertainty_grid = self.bev_epi_umap.cpu().reshape(
            self.bev_height, self.bev_width
        ).numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot heatmap with log scale
        extent = [self.bev_min_x, self.bev_max_x, self.bev_min_z, self.bev_max_z]
        im = ax.imshow(
            uncertainty_grid, 
            origin='lower',
            aspect='equal',
            extent=extent,
            cmap='turbo',
            interpolation='nearest',
            norm=LogNorm()
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Epistemic Uncertainty', rotation=270, labelpad=20, fontsize=12)
        
        # Labels and title
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Z Position (m)', fontsize=12)
        ax.set_title(
            f'BEV Uncertainty Map\n'
            f'Resolution: {self.bev_resolution}m/cell, '
            f'Grid: {self.bev_width}x{self.bev_height}',
            fontsize=14,
            fontweight='bold'
        )
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add statistics text
        stats_text = (
            f'Min: {uncertainty_grid.min():.6f}\n'
            f'Max: {uncertainty_grid.max():.6f}\n'
            f'Mean: {uncertainty_grid.mean():.6f}\n'
            f'Std: {uncertainty_grid.std():.6f}'
        )
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"BEV map saved to: {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        
        return fig, ax
    
    def save(self, step, save_dir=None):
        """Save the BEV uncertainty maps as numpy files."""
        umaps_dir = os.path.join(self.cfg.output_dir, "umaps")
        os.makedirs(umaps_dir, exist_ok=True)
        epi_path = os.path.join(umaps_dir, f"bev_epistemic_uncertainty_{step}.npy")
        ale_path = os.path.join(umaps_dir, f"bev_aleatoric_uncertainty_{step}.npy")
        
        epi_array = self.bev_epi_umap.cpu().numpy()
        ale_array = self.bev_ale_umap.cpu().numpy()
        
        np.save(epi_path, epi_array)
        np.save(ale_path, ale_array)
        
        print(f"BEV maps saved (step {step}, shape {epi_array.shape}) to: {umaps_dir}")
