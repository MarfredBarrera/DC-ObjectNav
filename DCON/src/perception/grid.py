import torch
import numpy as np
import os

class VoxelGrid:
    """
    Base class for 3D Voxel Grids.
    Handles grid initialization, dimensions, and coordinate transforms.
    """
    def __init__(self, config, scene_bounds, device=None):
        self.cfg = config
        self.device = device if device else config.device
        self.resolution = config.voxel_resolution
        
        self.initialized = False
        self.num_x = 0
        self.num_y = 0
        self.num_z = 0
        self.min_x, self.min_y, self.min_z = 0.0, 0.0, 0.0
        self.max_x, self.max_y, self.max_z = 0.0, 0.0, 0.0
        
        self.initialize_from_bounds(scene_bounds)

    def initialize_from_bounds(self, scene_bounds):
        """Calculates grid dimensions from scene bounds using unified resolution."""
        if isinstance(scene_bounds, torch.Tensor):
            min_b = scene_bounds[0].cpu().numpy()
            max_b = scene_bounds[1].cpu().numpy()
        else:
            min_b = scene_bounds[0]
            max_b = scene_bounds[1]
        
        # Habitat: Y is up, X is left-right, Z is back-forward
        self.min_x, self.min_y, self.min_z = float(min_b[0]), float(min_b[1]), float(min_b[2])
        self.max_x, self.max_y, self.max_z = float(max_b[0]), float(max_b[1]), float(max_b[2])
        
        # User requested max_height clipping
        if hasattr(self.cfg, "grid_max_height"):
            self.max_y = min(self.max_y, self.grid_max_height if hasattr(self, "grid_max_height") else self.cfg.grid_max_height)
            # Re-ensure self.max_y is set correctly if we used the config value
            self.max_y = min(self.max_y, self.cfg.grid_max_height)
        
        x_span = self.max_x - self.min_x
        y_span = self.max_y - self.min_y
        z_span = self.max_z - self.min_z
        
        self.num_x = int(np.ceil(x_span / self.resolution))
        self.num_y = int(np.ceil(y_span / self.resolution))
        self.num_z = int(np.ceil(z_span / self.resolution))
        
        print(f"Voxel Grid Initialized: {self.num_x}x{self.num_y}x{self.num_z} voxels ({self.resolution}m resolution)")
        self.initialized = True

    def generate_sample_points(self, height_filter=None):
        """
        Generate 3D sample points for voxel evaluation.
        
        Args:
            height_filter: Tuple of (min_y, max_y) coordinates. If None, uses all voxels.
            
        Returns:
            points: numpy array of shape (N, 3)
            grid_shape: Tuple (num_y_subset, num_z, num_x)
        """
        if not self.initialized:
            raise RuntimeError("Grid not initialized!")
        
        if height_filter is None:
            min_y, max_y = self.min_y, self.max_y
            num_y_subset = self.num_y
        else:
            min_y, max_y = height_filter
            num_y_subset = int(np.ceil((max_y - min_y) / self.resolution))
        
        x_coords = np.linspace(self.min_x + self.resolution/2, self.max_x - self.resolution/2, self.num_x)
        y_coords = np.linspace(min_y + self.resolution/2, max_y - self.resolution/2, num_y_subset)
        z_coords = np.linspace(self.min_z + self.resolution/2, self.max_z - self.resolution/2, self.num_z)
        
        # Grid order (Y, Z, X) to match memory-efficient layouts
        Y, Z, X = np.meshgrid(y_coords, z_coords, x_coords, indexing='ij')
        
        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)
        grid_shape = (num_y_subset, self.num_z, self.num_x)
        
        return points, grid_shape

    def get_y_indices(self, min_y, max_y):
        """Helper to get voxel indices for a height range."""
        idx_start = int((min_y - self.min_y) / self.resolution)
        idx_end = int((max_y - self.min_y) / self.resolution)
        return max(0, idx_start), min(self.num_y, idx_end)

class SimilarityGrid(VoxelGrid):
    """3D Similarity Voxel Grid with 2D BEV projection."""
    def __init__(self, config, ensemble, semantics, scene_bounds):
        super().__init__(config, scene_bounds, device=config.device)
        self.ensemble = ensemble
        self.semantics = semantics
        self.voxels = None # 3D Similarity Map (Y, Z, X)

    def compute_similarity_map(self, text_query, batch_size=100000, occupancy_grid=None):
        if not self.initialized: return None

        # Embed Text via MaskCLIP
        with torch.no_grad():
            text_embed = self.semantics.encode_text(text_query)

        # Full grid points
        points, grid_shape = self.generate_sample_points()
        total_points = points.shape[0]

        query_points = points

        if query_points.shape[0] == 0:
            self.voxels = torch.zeros(grid_shape, device=self.device)
            return

        points_tensor = torch.from_numpy(query_points).float().to(self.device)
        all_sims = []
        num_batches = int(np.ceil(query_points.shape[0] / batch_size))
        
        with torch.no_grad():
            for i in range(num_batches):
                start, end = i * batch_size, min((i + 1) * batch_size, query_points.shape[0])
                batch_pts = points_tensor[start:end]
                
                # Ensemble mean
                batch_means = []
                for model in self.ensemble:
                    mean, _ = model.forward(batch_pts, normalize=True)
                    batch_means.append(mean)
                
                ensemble_mean = torch.stack(batch_means, dim=0).mean(dim=0)
                ensemble_mean = ensemble_mean / (ensemble_mean.norm(dim=-1, keepdim=True) + 1e-8)
                
                sim = torch.matmul(ensemble_mean, text_embed.T).squeeze(-1)
                sim = (sim + 1.0) / 2.0
                all_sims.append(sim)
        
        all_sims = torch.cat(all_sims, dim=0)
        self.voxels = all_sims.reshape(grid_shape)

    def get_2d_map(self, min_y=0.0, max_y=1.5):
        if self.voxels is None: return None
        y_start, y_end = self.get_y_indices(min_y, max_y)
        if y_start >= y_end: return torch.zeros((self.num_z, self.num_x), device=self.device)
        
        # Implement Top-K Mean (Top 5%)
        slice = self.voxels[y_start:y_end]
        num_slices = slice.shape[0]
        k = max(1, int(0.05 * num_slices))
        
        # topk along Y dimension (axis 0)
        top_vals, _ = torch.topk(slice, k, dim=0)
        return top_vals.mean(dim=0)

    def save(self, step):
        sim_2d = self.get_2d_map()
        if sim_2d is not None:
            path = os.path.join(self.cfg.output_dir, f"sim_maps/bev_similarity_{step}.npy")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, sim_2d.detach().cpu().numpy())

class OccupancyGrid(VoxelGrid):
    """3D Occupancy Voxel Grid."""
    def __init__(self, config, scene_bounds):
        super().__init__(config, scene_bounds, device=config.device)
        self.unseen_val, self.free_val, self.occupied_val = 0, 1, 2
        self.voxels = torch.full((self.num_y, self.num_z, self.num_x), self.unseen_val, device=self.device, dtype=torch.uint8)

    def update_from_observation(self, depth, c2w, intrinsics, max_dist=None):
        if max_dist is None:
            max_dist = self.cfg.max_sensor_dist

        thickness = self.cfg.voxel_resolution * 1.2
        # thickness = 0.075

        if not self.initialized: return
        fx, fy, cx, cy, H, W = intrinsics
        
        # Get world coords of all cells
        y_c = torch.linspace(self.min_y, self.max_y, self.num_y, device=self.device)
        z_c = torch.linspace(self.min_z + self.resolution/2, self.max_z - self.resolution/2, self.num_z, device=self.device)
        x_c = torch.linspace(self.min_x + self.resolution/2, self.max_x - self.resolution/2, self.num_x, device=self.device)
        
        Y_idx, Z_idx, X_idx = torch.meshgrid(torch.arange(self.num_y, device=self.device),
                                             torch.arange(self.num_z, device=self.device),
                                             torch.arange(self.num_x, device=self.device), indexing='ij')
        
        Y_pts, Z_pts, X_pts = torch.meshgrid(y_c, z_c, x_c, indexing='ij')
        pts_world = torch.stack([X_pts.flatten(), Y_pts.flatten(), Z_pts.flatten(), torch.ones_like(X_pts.flatten())], dim=1)
        
        pts_cam = (torch.inverse(c2w) @ pts_world.T).T
        dist = pts_cam[:, 2]
        frustum_mask = (dist > self.cfg.min_sensor_dist) & (dist < max_dist)
        if not frustum_mask.any(): return
            
        valid_cam = pts_cam[frustum_mask]
        u = (valid_cam[:, 0] * fx / valid_cam[:, 2]) + cx
        v = (valid_cam[:, 1] * fy / valid_cam[:, 2]) + cy
        
        in_view = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        final_mask = frustum_mask.clone()
        final_mask[frustum_mask] = in_view
        if not final_mask.any(): return
            
        v_u, v_v, v_dist = u[in_view].long(), v[in_view].long(), valid_cam[in_view, 2]
        obs_depth = depth[v_v, v_u]
        
        is_free = v_dist < (obs_depth - thickness)
        is_occ = torch.abs(v_dist - obs_depth) < thickness
        update_mask = is_free | is_occ
        
        if not update_mask.any(): return
        
        # Create a mask that filters final_mask down to ONLY the cells we want to update
        update_global_mask = final_mask.clone()
        update_global_mask[final_mask] = update_mask
        
        # Get the new states just for the valid update cells
        new_states = torch.full_like(v_dist[update_mask], self.free_val, dtype=torch.uint8)
        new_states[is_occ[update_mask]] = self.occupied_val
        
        # Overwrite the grid only for unoccluded cells
        self.voxels[Y_idx.flatten()[update_global_mask], Z_idx.flatten()[update_global_mask], X_idx.flatten()[update_global_mask]] = new_states

    def get_2d_map(self, min_y=0.2, max_y=1.5):
        y_start, y_end = self.get_y_indices(min_y, max_y)
        if y_start >= y_end: return torch.zeros((self.num_z, self.num_x), device=self.device)
        return self.voxels[y_start:y_end].max(dim=0)[0]

    def get_2d_map_dilated(self, min_y=0.2, max_y=1.5, radius=1):
        """Return a 2D occupancy BEV with obstacles dilated by `radius` cells.

        Only free cells (val==1) adjacent to obstacles are promoted to
        obstacle; unseen cells (val==0) are never changed, so exploration
        through unseen space is unaffected.
        """
        raw = self.get_2d_map(min_y, max_y)
        if radius <= 0:
            return raw
        # Binary obstacle mask
        obstacle_mask = (raw >= self.occupied_val).float()  # [Z, X]
        # Max-pool dilates the mask by `radius` in every direction
        kernel = 2 * radius + 1
        dilated = torch.nn.functional.max_pool2d(
            obstacle_mask.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel, stride=1, padding=radius)
        dilated = dilated.squeeze(0).squeeze(0)  # back to [Z, X]
        # Promote only free→obstacle; leave unseen (0) untouched
        out = raw.clone()
        promote = (dilated > 0.5) & (raw == self.free_val)
        out[promote] = self.occupied_val
        return out

    def save(self, step):
        occ_2d = self.get_2d_map()
        path = os.path.join(self.cfg.output_dir, f"occ_maps/bev_occupancy_{step}.npy")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, occ_2d.cpu().numpy())

class UncertaintyGrid(VoxelGrid):
    """3D Uncertainty Voxel Grid."""
    def __init__(self, config, ensemble, scene_bounds):
        super().__init__(config, scene_bounds, device=config.device)
        self.ensemble = ensemble
        self.voxels_epi = None
        self.voxels_ale = None

    def forward_pass(self, height_filter=(0.1,2.0), batch_size=100000):
        if not self.initialized: return
        points, grid_shape = self.generate_sample_points(height_filter)
        pts_tensor = torch.from_numpy(points).float().to(self.device)
        
        preds, vars = [], []
        for model in self.ensemble:
            p_batch, v_batch = [], []
            for i in range(int(np.ceil(pts_tensor.shape[0] / batch_size))):
                start, end = i * batch_size, min((i + 1) * batch_size, pts_tensor.shape[0])
                m, v = model.forward(pts_tensor[start:end], normalize=True)
                p_batch.append(m); v_batch.append(v)
            preds.append(torch.cat(p_batch, dim=0)); vars.append(torch.cat(v_batch, dim=0))
        
        preds, vars = torch.stack(preds), torch.stack(vars)
        epi = preds.var(dim=0).mean(dim=-1)
        ale = vars.mean(dim=0).mean(dim=-1)
        
        self.voxels_epi = epi.reshape(grid_shape)
        self.voxels_ale = ale.reshape(grid_shape)

    def get_2d_map(self, min_y=0.1, max_y=1.5, type='epistemic'):
        voxels = self.voxels_epi if type == 'epistemic' else self.voxels_ale
        if voxels is None: return None
        # Bottom-5% pool along Y: a column with any trained surface voxel
        # snaps to that voxel's (low) uncertainty, while pure-air columns
        # stay uniformly high. Decouples "covered vs uncovered" from
        # "uncertain about scene contents." Mirrors the top-k pattern in
        # SimilarityGrid.get_2d_map, inverted because lower epistemic = more
        # confident.
        num_y = voxels.shape[0]
        k = max(1, int(0.15 * num_y))
        bottom_vals, _ = torch.topk(voxels, k, dim=0, largest=False)
        return bottom_vals.mean(dim=0)

    def save(self, step):
        epi_2d = self.get_2d_map(type='epistemic')
        ale_2d = self.get_2d_map(type='aleatoric')
        path_base = os.path.join(self.cfg.output_dir, "umaps")
        os.makedirs(path_base, exist_ok=True)
        if epi_2d is not None: np.save(os.path.join(path_base, f"bev_epistemic_uncertainty_{step}.npy"), epi_2d.detach().cpu().numpy())
        if ale_2d is not None: np.save(os.path.join(path_base, f"bev_aleatoric_uncertainty_{step}.npy"), ale_2d.detach().cpu().numpy())
