from sklearn import ensemble
import torch
import numpy as np
import os

class BEVGrid:
    """
    Base class for 2D Bird's Eye View Grids.
    Handles grid initialization, dimensions, and coordinate transforms.
    """
    def __init__(self, config, scene_bounds=None, device=None):
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
        
        if scene_bounds is not None:
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


class OccupancyGrid(BEVGrid):
    """
    Subclass for Binary Occupancy Mapping.
    Records visited areas on the grid.
    """
    def __init__(self, config, scene_bounds, device=None):
        super().__init__(config, scene_bounds, device)
        self.occupancy_map = None
        if self.bev_initialized:
            self._init_map()

    def initialize_from_bounds(self, scene_bounds):
        super().initialize_from_bounds(scene_bounds)
        self._init_map()
        
    def _init_map(self):
        # Initialize zero map (0 = free/unknown, 1 = occupied/visited)
        # Shape: (Height/Z, Width/X)
        self.occupancy_map = torch.zeros((self.bev_height, self.bev_width), device=self.device, dtype=torch.uint8)

    def update(self, points):
        """
        Update occupancy map with observed points.
        Args:
             points: (N, 3) tensor of points in world coordinates.
        """
        if not self.bev_initialized: return
        
        # Filter points within X/Z bounds
        mask = (points[:, 0] >= self.bev_min_x) & (points[:, 0] < self.bev_max_x) & \
               (points[:, 2] >= self.bev_min_z) & (points[:, 2] < self.bev_max_z)
        valid_points = points[mask]
        
        if valid_points.shape[0] == 0: return

        # Map to grid indices
        x_indices = ((valid_points[:, 0] - self.bev_min_x) / self.bev_resolution).long()
        z_indices = ((valid_points[:, 2] - self.bev_min_z) / self.bev_resolution).long()
        
        # Clamp to be safe
        x_indices = torch.clamp(x_indices, 0, self.bev_width - 1)
        z_indices = torch.clamp(z_indices, 0, self.bev_height - 1)
        
        # Set occupancy (1 for occupied)
        self.occupancy_map[z_indices, x_indices] = 1

    def save(self, step):
        """Save occupancy map to disk."""
        if self.occupancy_map is not None:
            occ_maps_dir = os.path.join(self.cfg.output_dir, "occ_maps")
            os.makedirs(occ_maps_dir, exist_ok=True)
            path = os.path.join(occ_maps_dir, f"bev_occupancy_{step}.npy")
            np.save(path, self.occupancy_map.cpu().numpy())
            print(f"Occupancy map saved to {path}")


class UncertaintyGrid(BEVGrid):
    """
    Subclass for Uncertainty Mapping (Epistemic & Aleatoric).
    Maintains a 2D Bird's Eye View Grid representing feature field uncertainty.
    """
    def __init__(self, config, ensemble):
        # Extract bounds from the first model in the ensemble
        model = ensemble[0]
        bounds = None
        if hasattr(model, 'scene_bounds'):
            bounds = model.scene_bounds
        else:
            print("Warning: Bounds not found on ensemble. BEV uninitialized.")
        
        super().__init__(config, scene_bounds=bounds, device=config.device)
        self.ensemble = ensemble
        self.iteration_num = 0
        
        # Core Tensors
        self.bev_epi_umap = None
        self.bev_ale_umap = None
    
    def generate_bev_sample_points(self, height_filter=None, height_samples=10):
        """
        Generate 3D sample points for BEV uncertainty evaluation.
        
        Args:
            height_filter: Tuple of (min_y, max_y) coordinates (height) to sample. 
                          If None, uses scene bounds.
            height_samples: Number of height levels to sample (default: 10)
            
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
        x_coords = np.linspace(
            self.bev_min_x + self.bev_resolution/2, 
            self.bev_max_x - self.bev_resolution/2, 
            self.bev_width
        )
        y_coords = np.linspace(min_y, max_y, height_samples)
        z_coords = np.linspace(
            self.bev_min_z + self.bev_resolution/2, 
            self.bev_max_z - self.bev_resolution/2, 
            self.bev_height
        )
        
        # Create 3D meshgrid
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        
        # Stack into (N, 3) array of 3D points
        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)
        grid_shape = (self.bev_width, height_samples, self.bev_height)
        
        return points, grid_shape

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
            grid_shape: Tuple (bev_width, height_samples, bev_height)
            
        Returns:
            bev_2d: Tensor of shape (bev_height, bev_width) - BEV uncertainty map
        """
        bev_width, height_samples, bev_height = grid_shape
        
        # Reshape to 3D grid: (bev_width, height_samples, bev_height)
        grid_3d = uncertainties.reshape(bev_width, height_samples, bev_height)
        
        # Average along height dimension (axis 1) -> (bev_width, bev_height)
        grid_2d = grid_3d.mean(dim=1)
        
        # Transpose to (bev_height, bev_width) for standard BEV representation
        bev_2d = grid_2d.transpose(0, 1)
        
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
        
        # Step 4: Move to CPU and aggregate height samples
        epi_2d = self.aggregate_height_samples(epistemic_uncertainty.cpu(), grid_shape)
        ale_2d = self.aggregate_height_samples(aleatoric_uncertainty.cpu(), grid_shape)
        
        # Step 5: Store results
        self.set_umaps(epi_2d, ale_2d)
        
        print(f"Forward pass complete!")
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
        torch.cuda.empty_cache()
    
    def compute_and_save_uncertainty_snapshot(self, iteration, height_filter=(0.1, 2.0), 
                                             height_samples=10, batch_size=100000):
        """
        Args:
            iteration: Current training iteration/step number
            height_filter: Height range to sample (default: 0.1-2.0m)
            height_samples: Number of height levels to sample
            batch_size: Batch size for forward pass
            
        Returns:
            elapsed_time: Time taken to compute uncertainties (seconds)
        """
        import time
        self.iteration_num = iteration
        
        start_time = time.time()
        self.forward_pass(height_filter=height_filter, 
                         height_samples=height_samples, 
                         batch_size=batch_size)
        elapsed = time.time() - start_time
        
        self.save_bev_maps()
        self.clear_umaps()
        
        return elapsed
    
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
    
    def save_bev_maps(self, save_dir=None):
        """Save the BEV uncertainty maps as numpy files."""
        umaps_dir = os.path.join(self.cfg.output_dir, "umaps")
        os.makedirs(umaps_dir, exist_ok=True)
        epi_path = os.path.join(umaps_dir, f"bev_epistemic_uncertainty_{self.iteration_num}.npy")
        ale_path = os.path.join(umaps_dir, f"bev_aleatoric_uncertainty_{self.iteration_num}.npy")
        
        epi_array = self.bev_epi_umap.cpu().numpy()
        ale_array = self.bev_ale_umap.cpu().numpy()
        
        np.save(epi_path, epi_array)
        np.save(ale_path, ale_array)
        
        print(f"BEV maps saved (step {self.iteration_num}, shape {epi_array.shape}) to: {umaps_dir}")
