import torch
import numpy as np
import time
import pickle
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
        self.bev_uncertainty_map = None
        self.bev_count_map = None
        
        # Dimensions
        self.bev_width = 0
        self.bev_height = 0
        self.bev_min_x = 0.0
        self.bev_max_x = 0.0
        self.bev_min_z = 0.0
        self.bev_max_z = 0.0

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