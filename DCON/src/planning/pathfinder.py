import torch
import numpy as np
import heapq
from sklearn.mixture import GaussianMixture
import time

class PathFinder:
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = device
        
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

    def get_gmm_goals(self, sim_map, num_modes=3, pct_thresh=95):
        """
        Identify goal candidates by fitting GMM to regions of high semantic similarity.
        """
        threshold = np.percentile(sim_map, pct_thresh)
        y, x = np.where(sim_map >= threshold)
        
        if len(y) == 0:
            y, x = np.where(sim_map == sim_map.max())
            
        points = np.column_stack((y, x))
        
        n_components = min(num_modes, len(points))
        if n_components == 0:
            return []
            
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(points)
        
        centers = gmm.means_
        goals = []
        for center in centers:
            goals.append((int(center[0]), int(center[1])))
            
        return goals

    def compute_fov_ig(self, trajectory, epi_map, occ_map, fov_deg=90, max_dist=40, gamma_ig=0.95):
        """
        Compute discounted Information Gain (IG) along a trajectory using Torch.
        Each waypoint's newly observed uncertainty is discounted by gamma_ig^t.
        """
        if len(trajectory) < 1:
            return 0.0, torch.zeros_like(epi_map, dtype=torch.bool, device=self.device)
            
        Z_dim, X_dim = epi_map.shape
        # if occ_map.shape != epi_map.shape:
        #     print(f"Warning: Shape mismatch! epi_map: {epi_map.shape}, occ_map: {occ_map.shape}. Resizing occ_map.")
        #     # This shouldn't happen if maps are loaded correctly, but let's be safe
        #     occ_map = torch.nn.functional.interpolate(
        #         occ_map.float().unsqueeze(0).unsqueeze(0), 
        #         size=(Z_dim, X_dim), mode='nearest'
        #     ).squeeze().long()

        seen_so_far = torch.zeros((Z_dim, X_dim), dtype=torch.bool, device=self.device)
        total_discounted_ig = 0.0
        
        fov_rad = fov_deg * np.pi / 180.0
        
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
            
            # Simple obstruction mask based on immediate cell values
            num_rays = int(fov_deg)
            ray_angles = torch.linspace(-fov_rad/2, fov_rad/2, num_rays, device=self.device) + heading
            
            steps = torch.arange(1, max_dist + 1, device=self.device).view(-1, 1)
            ray_z = (pos_z + steps * torch.sin(ray_angles)).long()
            ray_x = (pos_x + steps * torch.cos(ray_angles)).long()
            
            valid_idx = (ray_z >= 0) & (ray_z < Z_dim) & (ray_x >= 0) & (ray_x < X_dim)
            
            # Waypoint-specific visibility mask
            waypoint_mask = torch.zeros((Z_dim, X_dim), dtype=torch.bool, device=self.device)
            
            for r in range(num_rays):
                rz = ray_z[:, r]
                rx = ray_x[:, r]
                v = valid_idx[:, r]
                
                if not v.any(): continue
                rz = rz[v]
                rx = rx[v]
                
                occ_vals = occ_map[rz, rx]
                is_occ = (occ_vals >= 2)
                if is_occ.any():
                    # Find first obstacle along the ray
                    idx_occ = is_occ.nonzero()[0].item()
                    visible_z = rz[:idx_occ+1]
                    visible_x = rx[:idx_occ+1]
                else:
                    visible_z = rz
                    visible_x = rx
                    
                # # Double check bounds before assignment to avoid CUDA assert
                # visible_z = torch.clamp(visible_z, 0, Z_dim - 1)
                # visible_x = torch.clamp(visible_x, 0, X_dim - 1)
                waypoint_mask[visible_z, visible_x] = True

            # Calculate novelty: cells visible now but not in previous steps
            new_info_mask = waypoint_mask & (~seen_so_far)
            ig_t = epi_map[new_info_mask].sum().item()
            
            total_discounted_ig += (gamma_ig ** t) * ig_t
            
            # Update global mask for the trajectory
            seen_so_far |= waypoint_mask

        return total_discounted_ig, seen_so_far

    def score_trajectories(self, trajectories, sim_map, epi_map, occ_map, alpha=1.0, beta=1.0, gamma=0.1):
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
        
        scores = []
        
        start_time = time.time()
        for idx, traj in enumerate(trajectories):
            if not traj:
                scores.append({'score': -float('inf')})
                continue
                
            goal = traj[-1]
            semantic_score = sim_map[goal[0], goal[1]]
            
            ig_score, seen_mask = self.compute_fov_ig(traj, epi_torch, occ_torch, gamma_ig=0.95)
            
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
