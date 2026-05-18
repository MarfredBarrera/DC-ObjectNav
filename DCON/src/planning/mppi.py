import torch
import numpy as np
import time
from collections import deque
from .utils import find_nearest_free_cell, compute_fov_ig, compute_batch_fov_ig

class MPPIPlanner:
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = device
        # Previous safe control sequence, used to warm-start the next call.
        # None until the first successful optimize_trajectory.
        self.last_U = None

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

    def get_goals_near_highest_sim(self, sim_map, occ_map, max_candidates=20, obstacle_val=2):
        """
        Return free cells ordered by proximity (in BFS hops on the grid) to the
        single highest-similarity point. Used to commit to one similarity peak
        while still falling back to progressively-farther *approach* points if
        the closest free cell turns out to be A*-unreachable.

        Targets like furniture (e.g. a bed) get marked as obstacles in the
        occupancy grid, so the peak similarity often lands on an obstacle cell.
        We don't need to step ON the obstacle — we just need to get close.

        Returns [] if no observed (non-unseen) cells exist.
        """
        if occ_map is None:
            return []
        H, W = sim_map.shape
        target_sim = sim_map.copy().astype(np.float32)
        target_sim[occ_map == 0] = -np.inf  # exclude unseen
        if not np.any(np.isfinite(target_sim)):
            return []
        flat_idx = int(np.argmax(target_sim))
        tz, tx = flat_idx // W, flat_idx % W

        # BFS outward from the peak through *any* cell type (we want geometric
        # proximity to the target, not free-space proximity). Collect free cells
        # in the order encountered — that's monotonically increasing in BFS-hop
        # distance, which under 8-connectivity is a good proxy for Euclidean.
        visited = np.zeros((H, W), dtype=bool)
        visited[tz, tx] = True
        queue = deque([(tz, tx)])
        candidates = []
        while queue and len(candidates) < max_candidates:
            cz, cx = queue.popleft()
            if occ_map[cz, cx] < obstacle_val and occ_map[cz, cx] != 0:
                candidates.append((cz, cx))
            for dz, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)):
                nz, nx = cz + dz, cx + dx
                if 0 <= nz < H and 0 <= nx < W and not visited[nz, nx]:
                    visited[nz, nx] = True
                    queue.append((nz, nx))
        return candidates

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

    def scheduled_params(self, progress):
        """
        Linearly interpolate the exploration→exploitation schedule.
        progress=0.0 → all *_start values; progress=1.0 → all *_end values.

        w_goal uses a delayed schedule: it holds at *_start for the first half
        of the run, then interpolates to *_end over the second half. This lets
        the agent commit to pure IG-driven exploration early before the
        goal-distance pull kicks in.

        Returns dict with lambda_weight, w_ig, w_goal, cov_scale.
        """
        p = float(np.clip(progress, 0.0, 1.0))
        # Delayed half-run ramp for goal pull: 0 for p<0.5, then linear 0→1.
        p_goal = max(0.0, (p - 0.5))
        def lerp(a, b, t=p):
            return a + t * (b - a)
        return {
            "lambda_weight": lerp(self.cfg.mppi_lambda_start, self.cfg.mppi_lambda_end),
            "w_ig":          lerp(self.cfg.mppi_w_ig_start,   self.cfg.mppi_w_ig_end),
            "w_goal":        lerp(self.cfg.mppi_w_goal_start, self.cfg.mppi_w_goal_end, p_goal),
            "cov_scale":     lerp(self.cfg.mppi_cov_scale_start, self.cfg.mppi_cov_scale_end),
        }

    def optimize_trajectory(self, start, goal, epi_map, occ_map, num_samples=100, num_iters=None, horizon=100, lambda_weight=None, w_goal=None, w_ig=None, w_occ=1e6, w_unseen=0, intrinsics=None, sensor_height=1.5, initial_heading=None, progress=0.0, cov_scale=None):
        """
        Optimize a trajectory from `start` to `goal` using MPPI unicycle
        sampling. No A* reference; nominal controls start at zero and MPPI
        refines them via softmax-weighted updates across iterations.

        `start` and `goal` are (z, x) grid-cell tuples. Exploitation cost is
        mean per-step Euclidean distance from rollout waypoints to `goal`.

        Exploration→exploitation schedule: when lambda_weight/w_ig/w_goal/cov_scale
        are left None, they are filled in from cfg.mppi_*_start/end interpolated
        by `progress` (0.0 = pure exploration, 1.0 = pure exploitation). Pass a
        scalar to pin any single parameter and bypass the schedule for it.
        """
        if start is None or goal is None:
            return ([(int(start[0]), int(start[1]))] if start is not None else []), None, None, None

        # Resolve schedule. Caller-supplied scalars take precedence.
        sched = self.scheduled_params(progress)
        if lambda_weight is None: lambda_weight = sched["lambda_weight"]
        if w_ig is None:          w_ig = sched["w_ig"]
        if w_goal is None:        w_goal = sched["w_goal"]
        if cov_scale is None:     cov_scale = sched["cov_scale"]
        if num_iters is None:     num_iters = getattr(self.cfg, "mppi_num_iters", 5)

        epi_torch = torch.from_numpy(epi_map).to(self.device).float()
        occ_torch = torch.from_numpy(occ_map).to(self.device).long()

        Z_dim, X_dim = epi_torch.shape

        start_z, start_x = float(start[0]), float(start[1])
        goal_z,  goal_x  = float(goal[0]),  float(goal[1])

        # Control bounds (cells/step and rad/step).
        max_v_cells = (getattr(self.cfg, "mppi_max_v_mps", 1.0) * self.cfg.mppi_dt) / self.cfg.voxel_resolution
        min_v_cells = (getattr(self.cfg, "mppi_min_v_mps", 0.0) * self.cfg.mppi_dt) / self.cfg.voxel_resolution
        max_w_rad   = getattr(self.cfg, "mppi_max_w_rps", 2.0) * self.cfg.mppi_dt

        # Starting heading: agent's real pose if provided, else point at goal.
        if initial_heading is not None:
            theta_start = torch.tensor(float(initial_heading), device=self.device, dtype=torch.float32)
        else:
            theta_start = torch.tensor(float(np.arctan2(goal_z - start_z, goal_x - start_x)),
                                       device=self.device, dtype=torch.float32)

        # Warm-start: shift the previously committed control sequence left by
        # one (drop the action already executed) and zero-pad the tail. On the
        # first call (or after a manual reset), last_U is None and U_nom = 0.
        U_nom = torch.zeros((horizon, 2), device=self.device, dtype=torch.float32)
        if self.last_U is not None:
            last_U_t = torch.as_tensor(self.last_U, device=self.device, dtype=torch.float32)
            shifted_len = min(last_U_t.shape[0] - 1, horizon)
            if shifted_len > 0:
                U_nom[:shifted_len] = last_U_t[1:1 + shifted_len]
        
        # Control noise covariance [v_noise, w_noise]
        # v noise scales with typical cell size (1.0), w noise allows sufficient turning.
        # cov_scale (schedule-controlled) widens or tightens the sampling envelope.
        cov_matrix = float(cov_scale) * torch.tensor([[0.5, 0.0], [0.0, np.pi/4]], device=self.device)

        # DIAL-MPC action-level annealing: noise variance grows with horizon
        # index, so early steps (which we'll actually commit to) stay near the
        # warm-started controls while tail steps explore widely. Precomputed
        # once: shape [H], values in (0, 1], h=H-1 is full noise.
        beta_action = self.cfg.mppi_anneal_beta_action
        beta_traj   = self.cfg.mppi_anneal_beta_traj
        h_idx = torch.arange(horizon, device=self.device, dtype=torch.float32)
        action_scale = torch.exp(-(horizon - 1 - h_idx) / (beta_action * horizon))  # [H]

        best_traj = [(int(start_z), int(start_x))]
        best_U = None
        best_mask = None
        best_score = -float('inf')
        for it in range(num_iters):
             # Sample control noise
             noise = torch.randn((num_samples, horizon, 2), device=self.device) @ cov_matrix # [K, H, 2]
            #  # DIAL-MPC trajectory-level annealing: iter 0 = full noise (wide
            #  # exploration), iter N-1 = shrunken noise (local refinement).
            #  # Combined with action-level scaling so iter N-1 + h=0 is the
            #  # most concentrated sample (refining the immediately-committed
            #  # action), iter 0 + h=H-1 is the widest (rough global coverage).
            #  traj_scale = float(np.exp(-it / (beta_traj * max(num_iters, 1))))

             traj_scale = 1.0
             noise = noise * (traj_scale * action_scale).view(1, horizon, 1)
             # Pin sample 0 to zero noise so the unmutated warm-start is always
             # evaluated as-is — protects against the case where every noisy
             # variant happens to be worse than the carried-over plan.
             noise[0] = 0.0
             U_samples = U_nom.unsqueeze(0) + noise

             # Clamp linear / angular velocity bounds
             U_samples[:, :, 0] = torch.clamp(U_samples[:, :, 0], min=min_v_cells, max=max_v_cells)
             U_samples[:, :, 1] = torch.clamp(U_samples[:, :, 1], min=-max_w_rad, max=max_w_rad)

             # Rollout kinematics - Vectorized
             Theta_samples = torch.zeros((num_samples, horizon), device=self.device)
             Theta_samples[:, 0] = theta_start
             Theta_samples[:, 1:] = theta_start + torch.cumsum(U_samples[:, :-1, 1], dim=1)

             Z_samples = torch.zeros((num_samples, horizon), device=self.device)
             X_samples = torch.zeros((num_samples, horizon), device=self.device)
             Z_samples[:, 0] = start_z
             X_samples[:, 0] = start_x

             # Turn-first integration: translate using the heading AFTER each
             # step's angular control, so w_0 actually steers the first step.
             Z_samples[:, 1:] = start_z + torch.cumsum(U_samples[:, :-1, 0] * torch.sin(Theta_samples[:, 1:]), dim=1)
             X_samples[:, 1:] = start_x + torch.cumsum(U_samples[:, :-1, 0] * torch.cos(Theta_samples[:, 1:]), dim=1)

             # Discretize for grid evaluation (round, don't truncate)
             Z_idx = torch.clamp(Z_samples.round().long(), 0, Z_dim - 1)
             X_idx = torch.clamp(X_samples.round().long(), 0, X_dim - 1)

             scores = torch.zeros(num_samples, device=self.device)

             # Cost 1: Goal-distance pull. Mean per-step Euclidean distance from
             # rollout waypoints to the single goal cell. Replaces the old
             # along-A*-path L2 cost — no reference trajectory is used.
             dist_cost = torch.mean(torch.sqrt((Z_samples - goal_z)**2 + (X_samples - goal_x)**2), dim=1)
             scores -= w_goal * dist_cost

             # Cost 2: Collision Penalty (subsampled along each segment).
             # A single step can span many cells (max_v_cells), so checking only
             # the waypoints lets thin walls slip between consecutive samples.
             # Interpolate between waypoints in float space and check every cell
             # along the way. Sub-sample 2.5x the max velocity cell count to guarantee 
             # no diagonal tunneling occurs.
             n_sub = max(2, int(np.ceil(max_v_cells * 2.5)))
             # alphas in [0, 1) so each segment contributes its start + intermediates
             # without double-counting the endpoint shared with the next segment.
             alphas = torch.linspace(0.0, 1.0, n_sub + 1, device=self.device)[:-1]
             Z_seg = Z_samples[:, :-1].unsqueeze(-1) + alphas.view(1, 1, -1) * (Z_samples[:, 1:] - Z_samples[:, :-1]).unsqueeze(-1)
             X_seg = X_samples[:, :-1].unsqueeze(-1) + alphas.view(1, 1, -1) * (X_samples[:, 1:] - X_samples[:, :-1]).unsqueeze(-1)
             # Append the final waypoint so the goal cell itself is also checked.
             Z_chk = torch.cat([Z_seg.reshape(num_samples, -1), Z_samples[:, -1:]], dim=1)
             X_chk = torch.cat([X_seg.reshape(num_samples, -1), X_samples[:, -1:]], dim=1)
             Z_chk_idx = torch.clamp(Z_chk.round().long(), 0, Z_dim - 1)
             X_chk_idx = torch.clamp(X_chk.round().long(), 0, X_dim - 1)
             
             occ_vals = occ_torch[Z_chk_idx, X_chk_idx].clone()
             
             # Forgive the starting cell: if the agent is pressed against a wall and
             # technically starts inside an obstacle (especially with dilation), we
             # don't want to penalize rollouts just for being in the start cell. 
             # We only punish transitions into *new* obstacle cells.
             start_z_idx = min(max(int(round(start_z)), 0), Z_dim - 1)
             start_x_idx = min(max(int(round(start_x)), 0), X_dim - 1)
             at_start = (Z_chk_idx == start_z_idx) & (X_chk_idx == start_x_idx)
             occ_vals[at_start] = 1 # Treated as free
             
             collision_cost = torch.sum((occ_vals >= 2).float(), dim=1)
            #  # Unseen-traversal cost: cells with value 0 are not verified free.
            #  # Penalizing these prevents rollouts from slipping through unseen-cell
            #  # gaps in partially-observed walls. Exploration still works because
            #  # IG is earned from observing unseen cells via raycasts launched from
            #  # seen-free positions — the trajectory itself must stay in seen-free.
            #  unseen_cost = torch.sum((occ_vals == 0).float(), dim=1)

            #  # Penalties: obstacle is catastrophic, unseen is strong but smaller
            #  # so "skim a frontier cell" remains strictly preferred to "crash".
            #  scores -= w_occ * collision_cost
            #  scores -= w_unseen * unseen_cost

             # Cost 3: Information Gain (Vectorized)
             # Only reward IG for collision-free rollouts (the FOV raycast
             # already handles unseen cells correctly via line-of-sight).
             safe_mask = (collision_cost == 0)
             if safe_mask.any():
                 batch_ig, _ = compute_batch_fov_ig(
                     Z_idx[safe_mask],
                     X_idx[safe_mask],
                     Theta_samples[safe_mask],
                     epi_torch,
                     occ_torch,
                     self.cfg,
                     self.device,
                     gamma_ig=0.95,
                     intrinsics=intrinsics,
                     sensor_height=sensor_height
                 )
                 scores[safe_mask] += w_ig * batch_ig

             # Hard-exclude colliding rollouts from the final weighting.
             # Without this, when every sample collides (tight spot, thin wall,
             # bad warm-start) argmax just picks the least-bad crash. With it,
             # we either pick a genuinely safe rollout or fall through to the
             # caller's fallback (returns best_U=None below).
             if safe_mask.any():
                 NEG_INF = torch.tensor(float('-inf'), device=scores.device)
                 masked_scores = torch.where(safe_mask, scores, NEG_INF)

                 # iter best: argmax over safe rollouts only
                 iter_best_idx = torch.argmax(masked_scores)
                 if masked_scores[iter_best_idx] > best_score:
                     best_score = masked_scores[iter_best_idx].item()
                     best_traj = [(int(Z_idx[iter_best_idx, t]), int(X_idx[iter_best_idx, t])) for t in range(horizon)]
                     best_U = U_samples[iter_best_idx].detach().cpu().numpy()  # [H, 2] (v_cells_per_step, w_rad_per_step)
                     _, best_mask = compute_fov_ig(best_traj, epi_torch, occ_torch, self.cfg, self.device, gamma_ig=0.95, intrinsics=intrinsics, sensor_height=sensor_height)

                 # Softmax: safe rollouts only. Unsafe entries are -inf, so
                 # exp(...) = 0 and they contribute nothing to U_nom update.
                 beta = masked_scores.max()
                 weights = torch.exp((masked_scores - beta) / lambda_weight)
             else:
                 # No safe rollouts this iter: don't update best_U / best_traj.
                 # Still nudge U_nom using the soft collision penalty so the
                 # next iter has a chance of producing safe samples (gradient
                 # toward less-bad region). Worst case: we exit the loop with
                 # best_U=None and the caller falls back to queued action.

                 print("NO SAFE ROLLOUTS")
                 beta = torch.max(scores)
                 weights = torch.exp((scores - beta) / lambda_weight)
             weights = weights / (torch.sum(weights) + 1e-8)
             
             # Update nominal controls
             U_nom = U_nom + torch.sum(weights.view(-1, 1, 1) * noise, dim=0)

        # Final fallback: no safe rollout was ever found across all iters.
        # Rather than returning None and forcing the caller to idle, reconstruct
        # a trajectory from the shifted previous plan and return that so the
        # agent keeps moving along its last-known-safe controls.
        if best_U is None and self.last_U is not None:
            print("MPPI: No safe rollout found, falling back to shifted previous plan")
            last_U_t = torch.as_tensor(self.last_U, device=self.device, dtype=torch.float32)
            U_fallback = torch.zeros((horizon, 2), device=self.device, dtype=torch.float32)
            shifted_len = min(last_U_t.shape[0] - 1, horizon)
            if shifted_len > 0:
                U_fallback[:shifted_len] = last_U_t[1:1 + shifted_len]
            U_fallback[:, 0] = torch.clamp(U_fallback[:, 0], min=min_v_cells, max=max_v_cells)
            U_fallback[:, 1] = torch.clamp(U_fallback[:, 1], min=-max_w_rad, max=max_w_rad)

            Theta_fb = torch.zeros(horizon, device=self.device)
            Theta_fb[0] = theta_start
            Theta_fb[1:] = theta_start + torch.cumsum(U_fallback[:-1, 1], dim=0)
            Z_fb = torch.zeros(horizon, device=self.device)
            X_fb = torch.zeros(horizon, device=self.device)
            Z_fb[0] = start_z
            X_fb[0] = start_x
            Z_fb[1:] = start_z + torch.cumsum(U_fallback[:-1, 0] * torch.sin(Theta_fb[1:]), dim=0)
            X_fb[1:] = start_x + torch.cumsum(U_fallback[:-1, 0] * torch.cos(Theta_fb[1:]), dim=0)
            Z_fb_idx = torch.clamp(Z_fb.round().long(), 0, Z_dim - 1)
            X_fb_idx = torch.clamp(X_fb.round().long(), 0, X_dim - 1)

            best_U = U_fallback.detach().cpu().numpy()
            best_traj = [(int(Z_fb_idx[t]), int(X_fb_idx[t])) for t in range(horizon)]
            # Don't update self.last_U here — the fallback isn't certified safe.

        # Save the committed plan for next call's warm-start, but only when MPPI
        # actually found a safe rollout this call (not the fallback path).
        if best_U is not None and best_score > -float('inf'):
            self.last_U = best_U.copy()

        best_mask_np = best_mask.cpu().numpy() if best_mask is not None else None
        final_score = best_score if best_score > -float('inf') else None
        return best_traj, best_U, best_mask_np, final_score
