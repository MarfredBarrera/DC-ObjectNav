import torch
import numpy as np

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x

def find_nearest_free_cell(grid, pos, obstacle_val=2, max_dist=20):
    """Finds the closest non-obstacle cell using BFS."""
    if grid[pos[0], pos[1]] < obstacle_val:
        return pos
        
    from collections import deque
    Z_dim, X_dim = grid.shape
    q = deque([pos])
    visited = {pos}
    
    while q:
        curr = q.popleft()
        # Euclidean distance check
        if np.sqrt((curr[0]-pos[0])**2 + (curr[1]-pos[1])**2) > max_dist:
            continue
            
        for dz, dx in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            nz, nx = curr[0] + dz, curr[1] + dx
            if 0 <= nz < Z_dim and 0 <= nx < X_dim:
                if (nz, nx) not in visited:
                    if grid[nz, nx] < obstacle_val:
                        return (nz, nx)
                    visited.add((nz, nx))
                    q.append((nz, nx))
    return None

def get_z_score(m, mask=None, ignore_percentile=0):
    """Compute Z-score of a map, optionally with a mask and percentile filter."""
    m = to_numpy(m)
    if mask is not None:
        mask = to_numpy(mask)
        m_vals = m[mask]
    else:
        m_vals = m.flatten()

    if ignore_percentile > 0 and len(m_vals) > 0:
        thresh = np.percentile(m_vals, ignore_percentile)
        m_vals = m_vals[m_vals >= thresh]

    if len(m_vals) == 0:
        return np.zeros_like(m)

    mean, std = np.mean(m_vals), np.std(m_vals)
    if std < 1e-8: return np.zeros_like(m)
    return (m - mean) / std

def compute_fov_ig(trajectory, epi_map, occ_map, cfg, device="cuda", fov_deg=90, max_dist=None, gamma_ig=0.95, intrinsics=None, sensor_height=1.5):
    if max_dist is None:
        max_dist = int(cfg.max_sensor_dist / cfg.voxel_resolution)
    """
    Compute discounted Information Gain (IG) along a trajectory using Torch.
    Each waypoint's newly observed uncertainty is discounted by gamma_ig^t.
    """
    if len(trajectory) < 1:
        return 0.0, torch.zeros_like(epi_map, dtype=torch.bool, device=device)
        
    Z_dim, X_dim = epi_map.shape

    seen_so_far = torch.zeros((Z_dim, X_dim), dtype=torch.bool, device=device)
    total_discounted_ig = 0.0
    
    # 1. Setup Camera Metrics & Dead Zone
    if intrinsics is not None:
        fx, fy, cx, cy, H, W = intrinsics
        num_rays = int(W // 4) 
        pixels = torch.linspace(0, W - 1, num_rays, device=device)
        ray_angles_relative = torch.atan((pixels - cx) / fx)
        
        vfov = 2 * np.arctan(H / (2 * fy))
        min_dist = sensor_height / np.tan(vfov / 2)
    else:
        fov_rad = np.radians(fov_deg)
        num_rays = int(fov_deg)
        ray_angles_relative = torch.linspace(-fov_rad/2, fov_rad/2, num_rays, device=device)
        min_dist = 1.0 
        
    min_dist_cells = max(1.0, min_dist / cfg.voxel_resolution)
    
    for t in range(len(trajectory)):
        pos_z, pos_x = trajectory[t]
        
        if t < len(trajectory) - 1:
            next_z, next_x = trajectory[t+1]
        elif t > 0:
            prev_z, prev_x = trajectory[t-1]
            next_z = pos_z + (pos_z - prev_z)
            next_x = pos_x + (pos_x - prev_x)
        else:
            next_z, next_x = pos_z + 1, pos_x 
            
        heading = np.arctan2(next_z - pos_z, next_x - pos_x)
        ray_angles = ray_angles_relative + heading
        
        all_steps = torch.arange(1, max_dist + 1, device=device).view(-1, 1) 
        ray_angles_vec = ray_angles.view(1, -1) 
        
        ray_z = (pos_z + all_steps * torch.sin(ray_angles_vec)).long() 
        ray_x = (pos_x + all_steps * torch.cos(ray_angles_vec)).long()
        
        valid_idx = (ray_z >= 0) & (ray_z < Z_dim) & (ray_x >= 0) & (ray_x < X_dim)
        
        ray_z_clamp = torch.clamp(ray_z, 0, Z_dim - 1)
        ray_x_clamp = torch.clamp(ray_x, 0, X_dim - 1)
        
        occ_samples = occ_map[ray_z_clamp, ray_x_clamp]
        occ_samples[~valid_idx] = 0
        
        is_occ = (occ_samples >= 2)
        occ_cumsum = torch.cumsum(is_occ.int(), dim=0)
        shift_cumsum = torch.cat([torch.zeros((1, num_rays), dtype=torch.int32, device=device), occ_cumsum[:-1, :]], dim=0)
        
        is_visible = (shift_cumsum == 0) & valid_idx & (all_steps >= min_dist_cells)
        
        vis_z = ray_z[is_visible]
        vis_x = ray_x[is_visible]
        
        waypoint_mask = torch.zeros((Z_dim, X_dim), dtype=torch.bool, device=device)
        if vis_z.numel() > 0:
            waypoint_mask[vis_z, vis_x] = True

        new_info_mask = waypoint_mask & (~seen_so_far)
        ig_t = epi_map[new_info_mask].sum().item()
        
        total_discounted_ig += (gamma_ig ** t) * ig_t
        seen_so_far |= waypoint_mask

    return total_discounted_ig, seen_so_far

def compute_batch_fov_ig(Z_samples, X_samples, Theta_samples, epi_map, occ_map, cfg, device="cuda", fov_deg=90, max_dist=None, gamma_ig=0.95, intrinsics=None, sensor_height=1.5):
    if max_dist is None:
        max_dist = int(cfg.max_sensor_dist / cfg.voxel_resolution)
    """
    Compute discounted Information Gain (IG) along a batch of trajectories.
    Z_samples, X_samples, Theta_samples: [K, H] tensors.
    """
    K, H_length = Z_samples.shape
    if H_length < 1:
        return torch.zeros(K, device=device), torch.zeros((K, epi_map.shape[0], epi_map.shape[1]), dtype=torch.bool, device=device)
        
    Z_dim, X_dim = epi_map.shape

    seen_so_far = torch.zeros((K, Z_dim, X_dim), dtype=torch.bool, device=device)
    total_discounted_ig = torch.zeros(K, device=device)
    
    # 1. Setup Camera Metrics & Dead Zone
    if intrinsics is not None:
        fx, fy, cx, cy, H, W = intrinsics
        num_rays = int(W // 4) 
        pixels = torch.linspace(0, W - 1, num_rays, device=device)
        ray_angles_relative = torch.atan((pixels - cx) / fx)
        
        vfov = 2 * np.arctan(H / (2 * fy))
        min_dist = sensor_height / np.tan(vfov / 2)
    else:
        fov_rad = np.radians(fov_deg)
        num_rays = int(fov_deg)
        ray_angles_relative = torch.linspace(-fov_rad/2, fov_rad/2, num_rays, device=device)
        min_dist = 1.0 
        
    min_dist_cells = max(1.0, min_dist / cfg.voxel_resolution)
    all_steps = torch.arange(1, max_dist + 1, device=device).view(1, 1, -1) 
    
    for t in range(H_length):
        pos_z = Z_samples[:, t] # [K]
        pos_x = X_samples[:, t] # [K]
        heading = Theta_samples[:, t] # [K]
        
        ray_angles = ray_angles_relative.view(1, -1) + heading.view(-1, 1) # [K, num_rays]
        ray_angles_vec = ray_angles.unsqueeze(-1) # [K, num_rays, 1]
        
        pos_z_vec = pos_z.view(-1, 1, 1) # [K, 1, 1]
        pos_x_vec = pos_x.view(-1, 1, 1) # [K, 1, 1]
        
        ray_z = (pos_z_vec + all_steps * torch.sin(ray_angles_vec)).long() # [K, num_rays, max_dist]
        ray_x = (pos_x_vec + all_steps * torch.cos(ray_angles_vec)).long() # [K, num_rays, max_dist]
        
        valid_idx = (ray_z >= 0) & (ray_z < Z_dim) & (ray_x >= 0) & (ray_x < X_dim)
        
        ray_z_clamp = torch.clamp(ray_z, 0, Z_dim - 1)
        ray_x_clamp = torch.clamp(ray_x, 0, X_dim - 1)
        
        occ_samples = occ_map[ray_z_clamp, ray_x_clamp]
        occ_samples[~valid_idx] = 0
        
        is_occ = (occ_samples >= 2)
        occ_cumsum = torch.cumsum(is_occ.int(), dim=2)
        shift_cumsum = torch.cat([torch.zeros((K, num_rays, 1), dtype=torch.int32, device=device), occ_cumsum[:, :, :-1]], dim=2)
        
        is_visible = (shift_cumsum == 0) & valid_idx & (all_steps >= min_dist_cells)
        
        waypoint_mask = torch.zeros((K, Z_dim, X_dim), dtype=torch.bool, device=device)
        k_idx = torch.arange(K, device=device).view(-1, 1, 1).expand(-1, num_rays, max_dist)
        
        vis_k = k_idx[is_visible]
        vis_z = ray_z[is_visible]
        vis_x = ray_x[is_visible]
        
        if vis_z.numel() > 0:
            waypoint_mask[vis_k, vis_z, vis_x] = True

        new_info_mask = waypoint_mask & (~seen_so_far)
        ig_t = (new_info_mask * epi_map.unsqueeze(0)).sum(dim=(1, 2))
        
        total_discounted_ig += (gamma_ig ** t) * ig_t
        seen_so_far |= waypoint_mask

    return total_discounted_ig, seen_so_far
