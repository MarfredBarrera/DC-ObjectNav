from sklearn import ensemble
import torch
import numpy as np
import os

class BEVGrid:
    """
    Maintains a 2D Bird's Eye View Grid representing the spatial layout of 
    features and model uncertainty, updated via moving average.
    
    Fixed version that correctly sizes the grid based on scene bounds.
    """
    def __init__(self, config, ensemble):
        self.cfg = config
        self.ensemble = ensemble
        self.device = config.device
    
        # BEV Map
        self.bev_resolution = 0.01
        self.bev_initialized = False
        self.bev_update_active = False
        
        # Core Tensors
        self.bev_epi_umap = None
        self.bev_ale_umap = None
        
        # Dimensions
        self.bev_width = 0
        self.bev_height = 0
        self.bev_min_x = 0.0
        self.bev_max_x = 0.0
        self.bev_min_z = 0.0
        self.bev_max_z = 0.0
        

        self.iteration_num = 0

        self.initialize_bev()

    def initialize_bev(self):
        """Extracts scene bounds from the ensemble and calculates grid dimensions."""
        model = self.ensemble[0]
        
        # Robustly extract bounds depending on how your HashGrid stores them
        if hasattr(model, 'scene_bounds'):
            min_b = model.scene_bounds[0].cpu().numpy()
            max_b = model.scene_bounds[1].cpu().numpy()
        else:
            print("Warning: Bounds not found on ensemble. BEV uninitialized.")
            return
        
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
        
        # Initialize single-channel map tensors (count and uncertainty)
        total_cells = self.bev_height * self.bev_width
        self.bev_uncertainty_map = torch.zeros((total_cells,), device=self.device, dtype=torch.float32)
        self.bev_count_map = torch.zeros((total_cells,), device=self.device, dtype=torch.int32)
        
        print(f"Total cells: {total_cells:,}")
        self.bev_initialized = True

    def forward_pass(self, height_filter = None, height_samples=10, batch_size=100000):
        """
        Forward pass all points within the boundary through the ensemble.
        Samples across a range of heights and averages uncertainty along the height dimension.
        Computes uncertainty (variance across ensemble predictions) for each grid cell.
        
        Args:
            height_filter: Tuple of (min_y, max_y) coordinates (height) to sample. If None, uses scene bounds.
            height_samples: Number of height levels to sample (default: 10)
            batch_size: Number of points to process at once (to avoid OOM)
        
        Returns:
            epi_grid: (bev_height, bev_width) numpy array of epistemic uncertainties
            ale_grid: (bev_height, bev_width) numpy array of aleatoric uncertainties
        """
        if not self.bev_initialized:
            print("BEV grid not initialized!")
            return None, None
        
        # Use scene bounds if not specified
        if height_filter is None:
            min_y = self.bev_min_y
            max_y = self.bev_max_y
        else:
            min_y, max_y = height_filter
        
        # print(f"Running forward pass on {self.bev_width}x{self.bev_height} grid...")
        # print(f"Height range: [{min_y:.2f}, {max_y:.2f}]m with {height_samples} samples")
        
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
        # Shape: (bev_width * height_samples * bev_height, 3)
        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)
        total_points = points.shape[0]
        
        print(f"Processing {total_points:,} points ({self.bev_width}x{height_samples}x{self.bev_height})...")
        
        # Convert to tensor
        points_tensor = torch.from_numpy(points).float().to(self.device)
        
        # Process in batches to avoid memory issues
        epi_uncertainties = []
        ale_uncertainties = []
        num_batches = int(np.ceil(total_points / batch_size))
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_points)
            batch_points = points_tensor[start_idx:end_idx]
            
            # Run through all ensemble models
            ensemble_predictions = []
            ensemble_ale_var = []
            with torch.no_grad():
                for model in self.ensemble:
                    mean, variance = model.forward(batch_points, normalize=True)
                    ensemble_predictions.append(mean)
                    ensemble_ale_var.append(variance)
            
            # Stack predictions: (num_models, batch_size, feature_dim)
            ensemble_predictions = torch.stack(ensemble_predictions, dim=0)
            ensemble_ale_var = torch.stack(ensemble_ale_var, dim=0)
            
            # Compute epistemic uncertainty (variance across ensemble)
            # Shape: (batch_size, feature_dim) -> (batch_size,)
            epistemic_uncertainty = ensemble_predictions.var(dim=0).mean(dim=-1)
            aleatoric_uncertainty = ensemble_ale_var.mean(dim=0).mean(dim=-1)
            
            epi_uncertainties.append(epistemic_uncertainty.cpu())
            ale_uncertainties.append(aleatoric_uncertainty.cpu())
            
            # if (i + 1) % 10 == 0 or (i + 1) == num_batches:
            #     print(f"  Batch {i+1}/{num_batches} complete")
        
        # Concatenate all batches
        epi_uncertainties = torch.cat(epi_uncertainties, dim=0)
        ale_uncertainties = torch.cat(ale_uncertainties, dim=0)
        
        # Reshape to 3D grid: (bev_width, height_samples, bev_height)
        epi_3d = epi_uncertainties.reshape(self.bev_width, height_samples, self.bev_height)
        ale_3d = ale_uncertainties.reshape(self.bev_width, height_samples, self.bev_height)
        
        # Average along height dimension (axis 1)
        # Result: (bev_width, bev_height)
        epi_2d = epi_3d.mean(dim=1)
        ale_2d = ale_3d.mean(dim=1)
        
        # Transpose to (bev_height, bev_width) for standard BEV representation
        epi_2d = epi_2d.transpose(0, 1)
        ale_2d = ale_2d.transpose(0, 1)
        
        # Store
        self.bev_epi_umap = epi_2d
        self.bev_ale_umap = ale_2d
        
        # # Convert to numpy arrays for return
        # epi_grid = epi_2d.numpy()
        # ale_grid = ale_2d.numpy()
        
        print(f"Forward pass complete!")
        
        return
    
    def visualize_bev_map(self, save_path=None, show=False, height_filter=None):
        """
        Visualize the BEV uncertainty map as a heatmap.
        
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
        
        np.save(epi_path, self.bev_epi_umap.cpu().numpy())
        np.save(ale_path, self.bev_ale_umap.cpu().numpy())
        
        print(f"BEV maps saved to: {umaps_dir}")
