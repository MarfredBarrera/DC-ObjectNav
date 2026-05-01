import torch
import numpy as np
import time
from .utils import find_nearest_free_cell, compute_fov_ig, compute_batch_fov_ig

class MPPIPlanner:
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = device

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
            goal_free = find_nearest_free_cell(occ_map, goal, obstacle_val)
            if goal_free is not None:
                return goal_free
                
        return goal

    def optimize_trajectory(self, ref_traj, epi_map, occ_map, num_samples=30, num_iters=3, lambda_weight=0.5, w_ref=0.1, w_ig=10.0, w_occ=1000.0, stride=3, max_horizon=25, intrinsics=None, sensor_height=1.5):
        """
        Optimize a nominal A* trajectory using MPPI unicycle kinematic sampling.
        Subsamples the reference trajectory to save computation time on IG scoring.
        """
        if not ref_traj or len(ref_traj) < 2:
            return ref_traj, None
            
        # ref_traj = ref_traj[::stride]
        # if len(ref_traj) > max_horizon:
            # ref_traj = ref_traj[:max_horizon]
        
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
             
             # Rollout kinematics - Vectorized
             Theta_samples = torch.zeros((num_samples, horizon), device=self.device)
             Theta_samples[:, 0] = theta_ref[0]
             Theta_samples[:, 1:] = theta_ref[0] + torch.cumsum(U_samples[:, :-1, 1], dim=1)

             Z_samples = torch.zeros((num_samples, horizon), device=self.device)
             X_samples = torch.zeros((num_samples, horizon), device=self.device)
             Z_samples[:, 0] = Z_ref[0]
             X_samples[:, 0] = X_ref[0]
             
             Z_samples[:, 1:] = Z_ref[0] + torch.cumsum(U_samples[:, :-1, 0] * torch.sin(Theta_samples[:, :-1]), dim=1)
             X_samples[:, 1:] = X_ref[0] + torch.cumsum(U_samples[:, :-1, 0] * torch.cos(Theta_samples[:, :-1]), dim=1)
             
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
             
             # Cost 3: Information Gain (Vectorized)
             valid_k_mask = collision_cost < 3
             if valid_k_mask.any():
                 batch_ig, _ = compute_batch_fov_ig(
                     Z_idx[valid_k_mask],
                     X_idx[valid_k_mask],
                     Theta_samples[valid_k_mask],
                     epi_torch,
                     occ_torch,
                     self.cfg,
                     self.device,
                     gamma_ig=0.95,
                     intrinsics=intrinsics,
                     sensor_height=sensor_height
                 )
                 scores[valid_k_mask] += w_ig * batch_ig
                     
             # Extract best for current iteration
             iter_best_idx = torch.argmax(scores)
             if scores[iter_best_idx] > best_score and collision_cost[iter_best_idx] == 0:
                 best_score = scores[iter_best_idx].item()
                 best_traj = [(int(Z_idx[iter_best_idx, t]), int(X_idx[iter_best_idx, t])) for t in range(horizon)]
                 _, best_mask = compute_fov_ig(best_traj, epi_torch, occ_torch, self.cfg, self.device, gamma_ig=0.95, intrinsics=intrinsics, sensor_height=sensor_height)
                 
             # MPPI Update Rule
             # Weight generation using Softmax / Exponential weighting
             beta = torch.max(scores)
             weights = torch.exp((scores - beta) / lambda_weight)
             weights = weights / (torch.sum(weights) + 1e-8)
             
             # Update nominal controls
             U_nom = U_nom + torch.sum(weights.view(-1, 1, 1) * noise, dim=0)

        best_mask_np = best_mask.cpu().numpy() if best_mask is not None else None
        return best_traj, best_mask_np
