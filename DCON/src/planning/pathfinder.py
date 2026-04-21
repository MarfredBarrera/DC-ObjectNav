import torch
import numpy as np
import heapq
from sklearn.mixture import GaussianMixture
import time

class PathFinder:
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = device
        
    def _to_numpy(self, x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return x
        
    def find_nearest_free_cell(self, grid, pos, obstacle_val=2, max_dist=20):
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

    def astar(self, grid, start, goal, obstacle_val=2):
        """
        A* path planning on a 2D numpy grid.
        Handles goals embedded in obstacles by finding the nearest free cell.
        """
        Z_dim, X_dim = grid.shape
        
        # 1. Check start and goal bounds
        if not (0 <= start[0] < Z_dim and 0 <= start[1] < X_dim):
            print("Start out of bounds")
            return None
        if not (0 <= goal[0] < Z_dim and 0 <= goal[1] < X_dim):
            print("Goal out of bounds")
            return None
            
        # 2. If goal is an obstacle, find nearest free cell to target
        original_goal = goal
        if grid[goal[0], goal[1]] >= obstacle_val:
            goal = self.find_nearest_free_cell(grid, goal, obstacle_val)
            if goal is None:
                print(f"Goal {original_goal} is too deep in obstacles.")
                return None
            # print(f"Redirecting goal {original_goal} -> {goal}")
            
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        
        def heuristic(a, b):
            return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
            
        f_score = {start: heuristic(start, goal)}
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
                      
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path = path[::-1]
                
                # If we redirected, tack on the original goal as a final "look" target
                # but only if it's adjacent to our reachable goal
                if original_goal != goal:
                    dist_to_orig = np.sqrt((goal[0]-original_goal[0])**2 + (goal[1]-original_goal[1])**2)
                    if dist_to_orig < 2.0:
                        path.append(original_goal)
                return path
                
            for d in directions:
                neighbor = (current[0] + d[0], current[1] + d[1])
                
                if 0 <= neighbor[0] < Z_dim and 0 <= neighbor[1] < X_dim:
                    if grid[neighbor[0], neighbor[1]] >= obstacle_val:
                        if neighbor != goal: # Allow the target cell even if it's an obstacle (handled above)
                            continue
                            
                    move_cost = 1 if (d[0] == 0 or d[1] == 0) else 1.414
                    tg = g_score[current] + move_cost
                    
                    if neighbor not in g_score or tg < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tg
                        f = tg + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f, neighbor))
                        
        return None

    def get_highest_sim_goal(self, sim_map, obstacle_val=2, occ_map=None):
        """
        Identify the single goal candidate with the highest semantic similarity score,
        ensuring it is reachable/valid and in an explored region.
        """
        target_sim = sim_map.copy()
        if occ_map is not None:
            # Avoid unseen space (0)
            target_sim[occ_map == 0] = 0
            
        y, x = np.where(target_sim == target_sim.max())
        if len(y) == 0:
            return None
        
        goal = (int(y[0]), int(x[0]))
        
        # If the best point is an obstacle, redirect to nearest free space
        if occ_map is not None and occ_map[goal[0], goal[1]] >= obstacle_val:
            goal_free = self.find_nearest_free_cell(occ_map, goal, obstacle_val)
            if goal_free is not None:
                return goal_free
                
        return goal

    def get_gmm_goals(self, sim_map, num_modes=3, pct_thresh=95, occ_map=None):
        """
        Identify goal candidates using cluster means from GMM fitted to high semantic similarity.
        Optionally filters out unseen areas if occ_map is provided.
        """
        # Create a copy of sim_map to avoid modifying the original if we mask it
        target_sim = sim_map.copy()
        if occ_map is not None:
            # Mask out unseen regions (value 0) by setting similarity to 0
            target_sim[occ_map == 0] = 0
            
        threshold = np.percentile(target_sim, pct_thresh)
        y, x = np.where(target_sim >= threshold)
        
        if len(y) == 0:
            y, x = np.where(target_sim == target_sim.max())
            
        points = np.column_stack((y, x))
        
        if len(points) < num_modes:
            # If we don't have enough points to fit num_modes, just return the raw points
            return [(int(p[0]), int(p[1])) for p in points]
            
        gmm = GaussianMixture(n_components=num_modes, random_state=42)
        gmm.fit(points)
        
        centers = gmm.means_
        goals = []
        for c in centers:
            z_s, x_s = int(c[0]), int(c[1])
            # Ensure coordinates are within bounds
            z_s = np.clip(z_s, 0, target_sim.shape[0] - 1)
            x_s = np.clip(x_s, 0, target_sim.shape[1] - 1)
            
            # Final check: skip if the cluster center landed in unseen space
            if occ_map is not None and occ_map[z_s, x_s] == 0:
                continue
                
            goals.append((z_s, x_s))
            
        return goals

    def compute_fov_ig(self, trajectory, epi_map, occ_map, fov_deg=90, max_dist=None, gamma_ig=0.95, intrinsics=None, sensor_height=1.5):
        if max_dist is None:
            # Consolidate with max_sensor_dist from config
            max_dist = int(self.cfg.max_sensor_dist / self.cfg.voxel_resolution)
        """
        Compute discounted Information Gain (IG) along a trajectory using Torch.
        Each waypoint's newly observed uncertainty is discounted by gamma_ig^t.
        """
        if len(trajectory) < 1:
            return 0.0, torch.zeros_like(epi_map, dtype=torch.bool, device=self.device)
            
        Z_dim, X_dim = epi_map.shape

        seen_so_far = torch.zeros((Z_dim, X_dim), dtype=torch.bool, device=self.device)
        total_discounted_ig = 0.0
        
        # 1. Setup Camera Metrics & Dead Zone
        if intrinsics is not None:
            fx, fy, cx, cy, H, W = intrinsics
            # Perspective ray sampling (pinhole model)
            # Sample across the image width to get realistic ray distribution
            num_rays = int(W // 4) # Subsample for performance, but maintain distribution
            pixels = torch.linspace(0, W - 1, num_rays, device=self.device)
            ray_angles_relative = torch.atan((pixels - cx) / fx)
            
            # Vertical FOV for dead zone calculation
            vfov = 2 * np.arctan(H / (2 * fy))
            # The agent cannot see the ground closer than min_dist
            min_dist = sensor_height / np.tan(vfov / 2)
        else:
            # Fallback to uniform angular sampling
            fov_rad = np.radians(fov_deg)
            num_rays = int(fov_deg)
            ray_angles_relative = torch.linspace(-fov_rad/2, fov_rad/2, num_rays, device=self.device)
            min_dist = 1.0 # Default fallback
            
        # Convert min_dist (meters) to grid cells
        min_dist_cells = max(1.0, min_dist / self.cfg.voxel_resolution)
        
        for t in range(len(trajectory)):
            pos_z, pos_x = trajectory[t]
            
            if t < len(trajectory) - 1:
                next_z, next_x = trajectory[t+1]
            elif t > 0:
                prev_z, prev_x = trajectory[t-1]
                next_z = pos_z + (pos_z - prev_z)
                next_x = pos_x + (pos_x - prev_x)
            else:
                next_z, next_x = pos_z + 1, pos_x # Dummy heading for single point
                
            heading = np.arctan2(next_z - pos_z, next_x - pos_x)
            
            # Ray angles relative to current heading
            ray_angles = ray_angles_relative + heading
            
            steps = torch.arange(int(min_dist_cells), max_dist + 1, device=self.device).view(-1, 1) # [max_dist - min_dist, 1]
            ray_angles_vec = ray_angles.view(1, -1) # [1, num_rays]
            
            ray_z = (pos_z + steps * torch.sin(ray_angles_vec)).long() # [max_dist, num_rays]
            ray_x = (pos_x + steps * torch.cos(ray_angles_vec)).long()
            
            valid_idx = (ray_z >= 0) & (ray_z < Z_dim) & (ray_x >= 0) & (ray_x < X_dim)
            
            ray_z_clamp = torch.clamp(ray_z, 0, Z_dim - 1)
            ray_x_clamp = torch.clamp(ray_x, 0, X_dim - 1)
            
            occ_samples = occ_map[ray_z_clamp, ray_x_clamp]
            # Ignore invalid out-of-bounds indices
            occ_samples[~valid_idx] = 0
            
            is_occ = (occ_samples >= 2)
            # Find first obstacle by accumulating obstacle counts along the ray
            occ_cumsum = torch.cumsum(is_occ.int(), dim=0)
            # A cell is visible if the accumulated obstacles before it is 0
            shift_cumsum = torch.cat([torch.zeros((1, num_rays), dtype=torch.int32, device=self.device), occ_cumsum[:-1, :]], dim=0)
            is_visible = (shift_cumsum == 0) & valid_idx
            
            vis_z = ray_z[is_visible]
            vis_x = ray_x[is_visible]
            
            waypoint_mask = torch.zeros((Z_dim, X_dim), dtype=torch.bool, device=self.device)
            if vis_z.numel() > 0:
                waypoint_mask[vis_z, vis_x] = True

            # Calculate novelty: cells visible now but not in previous steps
            new_info_mask = waypoint_mask & (~seen_so_far)
            ig_t = epi_map[new_info_mask].sum().item()
            
            total_discounted_ig += (gamma_ig ** t) * ig_t
            
            # Update global mask for the trajectory
            seen_so_far |= waypoint_mask

        return total_discounted_ig, seen_so_far

    def score_trajectories(self, trajectories, sim_map, epi_map, occ_map, alpha=1.0, beta=1.0, gamma=0.1, intrinsics=None, sensor_height=1.5):
        """
        Score a list of trajectory alternatives.
        Score(T_k) = alpha*Semantic(G_k) + beta*IG(T_k) - gamma*Cost(T_k)
        
        Args:
            trajectories: list of path lists
            sim_map: numpy array
            epi_map: numpy array
            occ_map: numpy array
            
        Returns:
            list of dicts containing score breakdown for each trajectory and the optimal trajectory index
        """
        if not trajectories:
            return [], None
            
        epi_torch = torch.from_numpy(epi_map).to(self.device).float()
        occ_torch = torch.from_numpy(occ_map).to(self.device).long()
        sim_torch = torch.from_numpy(sim_map).to(self.device).float()

        sim_mask = (occ_torch == 2)
        z_sim = self.get_z_score(sim_torch, mask=sim_mask, ignore_percentile=75)
        
        scores = []
        
        start_time = time.time()
        for idx, traj in enumerate(trajectories):
            if not traj:
                scores.append({'score': -float('inf')})
                continue
                
            goal = traj[-1]
            semantic_score = z_sim[goal[0], goal[1]]
            
            ig_score, seen_mask = self.compute_fov_ig(traj, epi_torch, occ_torch, gamma_ig=0.95, intrinsics=intrinsics, sensor_height=sensor_height)
            
            cost_score = len(traj) * self.cfg.voxel_resolution
            
            total_score = (alpha * semantic_score + beta * ig_score - gamma * cost_score)
            scores.append({
                'idx': idx,
                'score': total_score,
                'semantic': semantic_score,
                'ig': ig_score,
                'cost': cost_score,
                'traj': traj,
                'seen_mask': seen_mask.cpu().numpy()
            })
        
        print(f"Scored {len(trajectories)} trajectories in {time.time() - start_time:.4f} seconds")
        # Find best
        valid_scores = [s for s in scores if s['score'] != -float('inf')]
        if not valid_scores:
            return scores, None
            
        best = max(valid_scores, key=lambda x: x['score'])

        return scores, best['idx'], best['traj']

    def mppi_optimize_trajectory(self, ref_traj, epi_map, occ_map, num_samples=30, num_iters=3, lambda_weight=0.5, w_ref=0.0, w_ig=10.0, w_occ=100.0, stride=3, max_horizon=25, intrinsics=None, sensor_height=1.5):
        """
        Optimize a nominal A* trajectory using MPPI unicycle kinematic sampling.
        Subsamples the reference trajectory to save computation time on IG scoring.
        """
        if not ref_traj or len(ref_traj) < 2:
            return ref_traj
            
        # # Downsample reference trajectory to reduce horizon H
        # if len(ref_traj) > max_horizon * stride:
        #     ref_traj = ref_traj[:max_horizon * stride]
        
        ref_traj = ref_traj[::stride]
        
        epi_torch = torch.from_numpy(epi_map).to(self.device).float()
        occ_torch = torch.from_numpy(occ_map).to(self.device).long()
        
        Z_dim, X_dim = epi_torch.shape
        horizon = len(ref_traj)
        
        # 1. Infer nominal controls (v, w) from ref_traj
        Z_ref = torch.tensor([p[0] for p in ref_traj], device=self.device, dtype=torch.float32)
        X_ref = torch.tensor([p[1] for p in ref_traj], device=self.device, dtype=torch.float32)
        
        dZ = Z_ref[1:] - Z_ref[:-1]
        dX = X_ref[1:] - X_ref[:-1]
        
        theta_ref = torch.atan2(dZ, dX)
        theta_ref = torch.cat([theta_ref, theta_ref[-1:]], dim=0) # Padding last state
        
        v_nom = torch.sqrt(dZ**2 + dX**2)
        w_nom = theta_ref[1:] - theta_ref[:-1]
        w_nom = torch.atan2(torch.sin(w_nom), torch.cos(w_nom)) # normalize
        
        v_nom = torch.cat([v_nom, v_nom[-1:]], dim=0)
        w_nom = torch.cat([w_nom, torch.zeros_like(w_nom[-1:])], dim=0)
        
        U_nom = torch.stack([v_nom, w_nom], dim=1) # [horizon, 2]
        
        # Control noise covariance [v_noise, w_noise]
        # v noise scales with typical cell size (1.0), w noise allows sufficient turning
        cov_matrix = torch.tensor([[0.5, 0.0], [0.0, np.pi/4]], device=self.device) 
        
        best_traj = ref_traj
        best_mask = None
        best_score = -float('inf')
        
        for it in range(num_iters):
             # Sample control noise
             noise = torch.randn((num_samples, horizon, 2), device=self.device) @ cov_matrix # [K, H, 2]
             U_samples = U_nom.unsqueeze(0) + noise
             
             # Limit backward driving to maintain forward progress
             U_samples[:, :, 0] = torch.clamp(U_samples[:, :, 0], min=0.0)
             
             # Rollout kinematics
             Z_samples = torch.zeros((num_samples, horizon), device=self.device)
             X_samples = torch.zeros((num_samples, horizon), device=self.device)
             Theta_samples = torch.zeros((num_samples, horizon), device=self.device)
             
             Z_samples[:, 0] = Z_ref[0]
             X_samples[:, 0] = X_ref[0]
             Theta_samples[:, 0] = theta_ref[0]
             
             for t in range(horizon - 1):
                 v_t = U_samples[:, t, 0]
                 w_t = U_samples[:, t, 1]
                 
                 Theta_samples[:, t+1] = Theta_samples[:, t] + w_t
                 Z_samples[:, t+1] = Z_samples[:, t] + v_t * torch.sin(Theta_samples[:, t])
                 X_samples[:, t+1] = X_samples[:, t] + v_t * torch.cos(Theta_samples[:, t])
                 
             # Discretize for grid evaluation
             Z_idx = torch.clamp(Z_samples.long(), 0, Z_dim - 1)
             X_idx = torch.clamp(X_samples.long(), 0, X_dim - 1)
             
             scores = torch.zeros(num_samples, device=self.device)
             
             # Cost 1: Distance to Reference Trajectory (L2)
             dist_cost = torch.mean(torch.sqrt((Z_samples - Z_ref.unsqueeze(0))**2 + (X_samples - X_ref.unsqueeze(0))**2), dim=1)
             scores -= w_ref * dist_cost
             
             # Cost 2: Collision Penalty
             occ_vals = occ_torch[Z_idx, X_idx]
             collision_cost = torch.sum((occ_vals >= 2).float(), dim=1)
             scores -= w_occ * collision_cost
             
             # Cost 3: Information Gain
             for k in range(num_samples):
                 # Convert back to list of tuples for compatibility with compute_fov_ig
                 traj_k = [(int(Z_idx[k, t]), int(X_idx[k, t])) for t in range(horizon)]
                 
                 # Only compute FOV IG if no massive collisions to save time
                 if collision_cost[k] < 3:
                     ig_k, _ = self.compute_fov_ig(traj_k, epi_torch, occ_torch, gamma_ig=0.95, intrinsics=intrinsics, sensor_height=sensor_height)
                     scores[k] += w_ig * ig_k
                     
             # Extract best for current iteration
             iter_best_idx = torch.argmax(scores)
             if scores[iter_best_idx] > best_score and collision_cost[iter_best_idx] == 0:
                 best_score = scores[iter_best_idx].item()
                 best_traj = [(int(Z_idx[iter_best_idx, t]), int(X_idx[iter_best_idx, t])) for t in range(horizon)]
                 _, best_mask = self.compute_fov_ig(best_traj, epi_torch, occ_torch, gamma_ig=0.95, intrinsics=intrinsics, sensor_height=sensor_height)
                 
             # MPPI Update Rule
             # Weight generation using Softmax / Exponential weighting
             beta = torch.max(scores)
             weights = torch.exp((scores - beta) / lambda_weight)
             weights = weights / (torch.sum(weights) + 1e-8)
             
             # Update nominal controls
             U_nom = U_nom + torch.sum(weights.view(-1, 1, 1) * noise, dim=0)

        best_mask_np = best_mask.cpu().numpy() if best_mask is not None else None
        return best_traj, best_mask_np

    def get_z_score(self, m, mask=None, ignore_percentile=0):
        """Compute Z-score of a map, optionally with a mask and percentile filter."""
        m = self._to_numpy(m)
        if mask is not None:
            mask = self._to_numpy(mask)
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