# import torch
# import numpy as np
# import heapq
# import time
# from sklearn.mixture import GaussianMixture
# from .utils import find_nearest_free_cell, get_z_score, compute_fov_ig

# class AStarPlanner:
#     def __init__(self, cfg, device="cuda"):
#         self.cfg = cfg
#         self.device = device
        
#     def plan(self, grid, start, goal, obstacle_val=2):
#         """
#         A* path planning on a 2D numpy grid.
#         Handles goals embedded in obstacles by finding the nearest free cell.
#         """
#         Z_dim, X_dim = grid.shape
        
#         # 1. Check start and goal bounds
#         if not (0 <= start[0] < Z_dim and 0 <= start[1] < X_dim):
#             print("Start out of bounds")
#             return None
#         if not (0 <= goal[0] < Z_dim and 0 <= goal[1] < X_dim):
#             print("Goal out of bounds")
#             return None
            
#         # 2. If goal is an obstacle, find nearest free cell to target
#         original_goal = goal
#         if grid[goal[0], goal[1]] >= obstacle_val:
#             goal = find_nearest_free_cell(grid, goal, obstacle_val)
#             if goal is None:
#                 print(f"Goal {original_goal} is too deep in obstacles.")
#                 return None
            
#         open_set = []
#         heapq.heappush(open_set, (0, start))
        
#         came_from = {}
#         g_score = {start: 0}
        
#         def heuristic(a, b):
#             return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
            
#         f_score = {start: heuristic(start, goal)}
        
#         directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
#                       (-1, -1), (-1, 1), (1, -1), (1, 1)]
                      
#         while open_set:
#             current = heapq.heappop(open_set)[1]
            
#             if current == goal:
#                 path = []
#                 while current in came_from:
#                     path.append(current)
#                     current = came_from[current]
#                 path.append(start)
#                 path = path[::-1]
                
#                 # If we redirected, tack on the original goal as a final "look" target
#                 # but only if it's adjacent to our reachable goal
#                 if original_goal != goal:
#                     dist_to_orig = np.sqrt((goal[0]-original_goal[0])**2 + (goal[1]-original_goal[1])**2)
#                     if dist_to_orig < 2.0:
#                         path.append(original_goal)
#                 return path
                
#             for d in directions:
#                 neighbor = (current[0] + d[0], current[1] + d[1])
                
#                 if 0 <= neighbor[0] < Z_dim and 0 <= neighbor[1] < X_dim:
#                     if grid[neighbor[0], neighbor[1]] >= obstacle_val:
#                         if neighbor != goal: # Allow the target cell even if it's an obstacle (handled above)
#                             continue
                            
#                     move_cost = 1 if (d[0] == 0 or d[1] == 0) else 1.414
#                     tg = g_score[current] + move_cost
                    
#                     if neighbor not in g_score or tg < g_score[neighbor]:
#                         came_from[neighbor] = current
#                         g_score[neighbor] = tg
#                         f = tg + heuristic(neighbor, goal)
#                         heapq.heappush(open_set, (f, neighbor))
                        
#         return None

#     def get_gmm_goals(self, sim_map, num_modes=3, pct_thresh=95, occ_map=None):
#         """
#         Identify goal candidates using cluster means from GMM fitted to high semantic similarity.
#         Optionally filters out unseen areas if occ_map is provided.
#         """
#         target_sim = sim_map.copy()
#         if occ_map is not None:
#             # Mask out unseen regions (value 0) by setting similarity to 0
#             target_sim[occ_map == 0] = 0
            
#         threshold = np.percentile(target_sim, pct_thresh)
#         y, x = np.where(target_sim >= threshold)
        
#         if len(y) == 0:
#             y, x = np.where(target_sim == target_sim.max())
            
#         points = np.column_stack((y, x))
        
#         if len(points) < num_modes:
#             # If we don't have enough points to fit num_modes, just return the raw points
#             return [(int(p[0]), int(p[1])) for p in points]
            
#         gmm = GaussianMixture(n_components=num_modes, random_state=42)
#         gmm.fit(points)
        
#         centers = gmm.means_
#         goals = []
#         for c in centers:
#             z_s, x_s = int(c[0]), int(c[1])
#             # Ensure coordinates are within bounds
#             z_s = np.clip(z_s, 0, target_sim.shape[0] - 1)
#             x_s = np.clip(x_s, 0, target_sim.shape[1] - 1)
            
#             # Final check: skip if the cluster center landed in unseen space
#             if occ_map is not None and occ_map[z_s, x_s] == 0:
#                 continue
                
#             goals.append((z_s, x_s))
            
#         return goals

#     def score_trajectories(self, trajectories, sim_map, epi_map, occ_map, alpha=1.0, beta=1.0, gamma=0.1, intrinsics=None, sensor_height=1.5):
#         """
#         Score a list of trajectory alternatives.
#         Score(T_k) = alpha*Semantic(G_k) + beta*IG(T_k) - gamma*Cost(T_k)
#         """
#         if not trajectories:
#             return [], None
            
#         epi_torch = torch.from_numpy(epi_map).to(self.device).float()
#         occ_torch = torch.from_numpy(occ_map).to(self.device).long()
#         sim_torch = torch.from_numpy(sim_map).to(self.device).float()

#         sim_mask = (occ_torch == 2)
#         z_sim = get_z_score(sim_torch, mask=sim_mask, ignore_percentile=75)
        
#         scores = []
        
#         start_time = time.time()
#         for idx, traj in enumerate(trajectories):
#             if not traj:
#                 scores.append({'score': -float('inf')})
#                 continue
                
#             goal = traj[-1]
#             semantic_score = z_sim[goal[0], goal[1]]
            
#             ig_score, seen_mask = compute_fov_ig(
#                 traj, epi_torch, occ_torch, self.cfg, self.device, 
#                 gamma_ig=0.95, intrinsics=intrinsics, sensor_height=sensor_height
#             )
            
#             cost_score = len(traj) * self.cfg.voxel_resolution
            
#             total_score = (alpha * semantic_score + beta * ig_score - gamma * cost_score)
#             scores.append({
#                 'idx': idx,
#                 'score': total_score,
#                 'semantic': semantic_score,
#                 'ig': ig_score,
#                 'cost': cost_score,
#                 'traj': traj,
#                 'seen_mask': seen_mask.cpu().numpy()
#             })
        
#         print(f"Scored {len(trajectories)} trajectories in {time.time() - start_time:.4f} seconds")
#         # Find best
#         valid_scores = [s for s in scores if s['score'] != -float('inf')]
#         if not valid_scores:
#             return scores, None
            
#         best = max(valid_scores, key=lambda x: x['score'])

#         return scores, best['idx'], best['traj']
