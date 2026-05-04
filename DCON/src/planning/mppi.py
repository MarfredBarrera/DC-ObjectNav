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

    def get_top_k_sim_goals(self, sim_map, occ_map=None, k=10, obstacle_val=2, min_separation=10):
        """
        Return up to k goal candidates ranked by semantic similarity. Each candidate
        is snapped to the nearest free cell if it lands on an obstacle. Candidates
        are spaced at least `min_separation` cells apart so the fallback list
        explores genuinely different goals rather than near-duplicates.
        """
        target_sim = sim_map.copy().astype(np.float32)
        if occ_map is not None:
            target_sim[occ_map == 0] = -np.inf  # exclude unseen

        flat = target_sim.flatten()
        # Argsort descending; -inf entries naturally land at the end
        order = np.argsort(-flat)

        H, W = target_sim.shape
        chosen = []
        for idx in order:
            if not np.isfinite(flat[idx]):
                break
            y, x = int(idx // W), int(idx % W)
            cand = (y, x)
            if occ_map is not None and occ_map[y, x] >= obstacle_val:
                cand = find_nearest_free_cell(occ_map, cand, obstacle_val)
                if cand is None:
                    continue
            # Enforce separation from already-chosen goals
            if any(abs(cand[0] - cy) + abs(cand[1] - cx) < min_separation for cy, cx in chosen):
                continue
            chosen.append(cand)
            if len(chosen) >= k:
                break
        return chosen

    def optimize_trajectory(self, ref_traj, epi_map, occ_map, num_samples=30, num_iters=3, lambda_weight=1.0, w_ref=0.0, w_ig=10.0, w_occ=1e5, intrinsics=None, sensor_height=1.5, initial_heading=None):
        """
        Optimize a nominal A* trajectory using MPPI unicycle kinematic sampling.
        Subsamples the reference trajectory to save computation time on IG scoring.
        """
        if not ref_traj or len(ref_traj) < 2:
            return ref_traj, None, None

        
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

        # Use agent's actual heading as the rollout starting orientation when provided.
        # Without this, MPPI rolls out from the heading inferred from the first ref-path
        # segment, which may not match the agent's real pose at replan time.
        if initial_heading is not None:
            theta_start = torch.tensor(float(initial_heading), device=self.device, dtype=torch.float32)
        else:
            theta_start = theta_ref[0]
        
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
        best_U = None
        best_mask = None
        best_score = -float('inf')
        
        for it in range(num_iters):
             # Sample control noise
             noise = torch.randn((num_samples, horizon, 2), device=self.device) @ cov_matrix # [K, H, 2]
             U_samples = U_nom.unsqueeze(0) + noise
             
             # Calculate limits based on voxel resolution and timestep
             max_v_cells = (getattr(self.cfg, "mppi_max_v_mps", 1.0) * self.cfg.mppi_dt) / self.cfg.voxel_resolution
             min_v_cells = (getattr(self.cfg, "mppi_min_v_mps", 0.0) * self.cfg.mppi_dt) / self.cfg.voxel_resolution
             max_w_rad = getattr(self.cfg, "mppi_max_w_rps", 2.0) * self.cfg.mppi_dt
             
             # Clamp linear velocity bounds
             U_samples[:, :, 0] = torch.clamp(U_samples[:, :, 0], min=min_v_cells, max=max_v_cells)
             # Clamp angular velocity bounds
             U_samples[:, :, 1] = torch.clamp(U_samples[:, :, 1], min=-max_w_rad, max=max_w_rad)
             
             # Rollout kinematics - Vectorized
             Theta_samples = torch.zeros((num_samples, horizon), device=self.device)
             Theta_samples[:, 0] = theta_start
             Theta_samples[:, 1:] = theta_start + torch.cumsum(U_samples[:, :-1, 1], dim=1)

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
             
             # Massive penalty for colliding trajectories to ensure they carry minimal weight in softmax
             scores -= w_occ * collision_cost
             
             # Cost 3: Information Gain (Vectorized)
             # Only calculate and reward IG for completely collision-free trajectories
             valid_k_mask = collision_cost == 0
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
             if scores[iter_best_idx] > best_score:
                 best_score = scores[iter_best_idx].item()
                 best_traj = [(int(Z_idx[iter_best_idx, t]), int(X_idx[iter_best_idx, t])) for t in range(horizon)]
                 best_U = U_samples[iter_best_idx].detach().cpu().numpy()  # [H, 2] (v_cells_per_step, w_rad_per_step)
                 _, best_mask = compute_fov_ig(best_traj, epi_torch, occ_torch, self.cfg, self.device, gamma_ig=0.95, intrinsics=intrinsics, sensor_height=sensor_height)
                 
             # MPPI Update Rule
             # Weight generation using Softmax / Exponential weighting
             beta = torch.max(scores)
             weights = torch.exp((scores - beta) / lambda_weight)
             weights = weights / (torch.sum(weights) + 1e-8)
             
             # Update nominal controls
             U_nom = U_nom + torch.sum(weights.view(-1, 1, 1) * noise, dim=0)

        best_mask_np = best_mask.cpu().numpy() if best_mask is not None else None
        return best_traj, best_U, best_mask_np
