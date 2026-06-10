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
        # Adaptive horizon: shrinks on consecutive "no safe rollout" failures
        # so a stuck agent can find a shorter feasible plan, then snaps back
        # to the default horizon on the first successful plan.
        self.default_horizon = getattr(cfg, "mppi_horizon", 150)
        self.horizon_shrink_step = getattr(cfg, "mppi_horizon_shrink_step", 20)
        self.min_horizon = getattr(cfg, "mppi_min_horizon", 10)
        self.current_horizon = self.default_horizon
        # Latched exploit confidence — hysteresis so a brief detection keeps
        # the goal-pull engaged after the target falls out of view. Updated
        # each call as max(incoming, prev * decay).
        self.exploit_conf = 0.0
        # Wedge counter: consecutive replans where MPPI's optimal first
        # action is essentially (0, 0). Read/incremented by the caller; used
        # to trigger a recovery rotation when the agent is stuck against a
        # wall and every motion sample collides.
        self.stuck_counter = 0

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


    def optimize_trajectory(self, start, goal, ig_map, occ_map,
    num_samples=250, num_iters=None, horizon=None,
    lambda_weight=None, w_goal=1, w_ig=None,
    intrinsics=None, sensor_height=1.5, initial_heading=None,
    progress=0.0, ig_source="unseen", goal_confidence=0.0):
        """
        Optimize a trajectory from `start` to `goal` using MPPI unicycle
        sampling. No A* reference; nominal controls start at zero and MPPI
        refines them via softmax-weighted updates across iterations.

        `start` and `goal` are (z, x) grid-cell tuples. Exploitation cost is
        mean per-step Euclidean distance from rollout waypoints to `goal`.

        Collision handling is hard exclusion: any rollout that touches an
        obstacle in its horizon is scored -inf and dropped from both selection
        and the softmax, so the committed plan is always collision-free. When
        no rollout is safe, returns best_U=None and the caller recovers.

        Cost weights (`mppi_lambda`, `mppi_w_ig`, `mppi_w_goal`) come from
        cfg. Control-noise stddev is anchored to half the actuator limits and
        modulated by DIAL annealing across MPPI iterations (`beta_traj`) and
        horizon steps (`beta_action`).
        """
        if start is None or goal is None:
            return ([(int(start[0]), int(start[1]))] if start is not None else []), None, None, None

        mppi_lambda = self.cfg.mppi_lambda
        w_ig = self.cfg.mppi_w_ig
        w_goal = self.cfg.mppi_w_goal

        # # Exploration→exploitation mixing weight, length-agnostic.
        # # lam(progress=0) = 1.0 → pure IG; lam(progress=1) ≈ 0.368.
        # lam = float(np.exp(-float(np.clip(progress, 0.0, 1.0))/2.0))

        # Detection-confidence weighting with hysteresis.
        #   latch = max(incoming, prev * decay)  → sticks high after a brief detection
        #   below `threshold`: w_conf = 0 (exploitation hard-off, IG full)
        #   above:            w_conf = conf_scale * (a^latch - 1)/(a - 1)
        # With a < 1 the curve is concave/saturating — small post-threshold
        # values already produce near-full goal pull. IG is damped
        # symmetrically by `(1 - w_conf_norm)`.
        a_conf = float(getattr(self.cfg, "mppi_conf_weight_a", 0.1))
        conf_scale = float(getattr(self.cfg, "mppi_conf_weight_scale", 100.0))
        conf_decay = float(getattr(self.cfg, "mppi_conf_decay", 0.9))
        conf_threshold = float(getattr(self.cfg, "mppi_conf_threshold", 0.1))
        incoming_conf = float(np.clip(goal_confidence, 0.0, 1.0))
        self.exploit_conf = max(incoming_conf, self.exploit_conf * conf_decay)
        if self.exploit_conf <= conf_threshold:
            w_conf = 0.0
            w_ig_conf = 1.0
        else:
            w_conf_norm = (a_conf ** self.exploit_conf - 1.0) / (a_conf - 1.0)
            w_conf = conf_scale * w_conf_norm
            w_ig_conf = 1.0 - w_conf_norm
        # Expose for logging / visualization. Latched exploit confidence (post
        # decay+ratchet) and the resulting goal-distance weight.
        self.last_exploit_conf = float(self.exploit_conf)
        self.last_w_conf = float(w_conf)

        if num_iters is None:     num_iters = getattr(self.cfg, "mppi_num_iters", 5)
        # Caller-supplied horizon pins the value (and bypasses adaptive
        # shrinking entirely); otherwise use the planner's current adaptive
        # horizon, which shrinks after each consecutive failed replan.
        horizon_pinned = horizon is not None
        if horizon is None:       horizon = self.current_horizon

        start_z, start_x = float(start[0]), float(start[1])
        goal_z,  goal_x  = float(goal[0]),  float(goal[1])

        occ_torch = torch.from_numpy(occ_map).to(self.device).long()
        # Collision-only view of occupancy: carve a small disk around the goal
        # so the target cell (which is often an obstacle — e.g. the bed, the
        # pillow) is reachable. The original occ_torch is unchanged for IG
        # raycasts, which still see the target as a proper occluder.
        r_carve = getattr(self.cfg, "mppi_goal_carve_radius", 1)
        occ_collision = occ_torch.clone()
        if r_carve > 0:
            gz = int(round(goal_z))
            gx = int(round(goal_x))
            z0, z1 = max(0, gz - r_carve), min(occ_torch.shape[0], gz + r_carve + 1)
            x0, x1 = max(0, gx - r_carve), min(occ_torch.shape[1], gx + r_carve + 1)
            zz, xx = torch.meshgrid(
                torch.arange(z0, z1, device=self.device),
                torch.arange(x0, x1, device=self.device),
                indexing='ij',
            )
            disk = ((zz - gz) ** 2 + (xx - gx) ** 2) <= r_carve ** 2
            patch = occ_collision[z0:z1, x0:x1]
            patch[disk] = 1  # treat as free for collision
            occ_collision[z0:z1, x0:x1] = patch
        # IG signal: either the supplied uncertainty map ("epistemic"), or a
        # binary mask of unseen occupancy cells ("unseen"). The unseen branch
        # ignores ig_map entirely — the caller can pass anything (e.g. None
        # placeholder or the epi map for logging) without affecting the cost.
        if ig_source == "unseen":
            ig_torch = (occ_torch == 0).float()
        else:
            ig_torch = torch.from_numpy(ig_map).to(self.device).float()

        Z_dim, X_dim = ig_torch.shape

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
        had_warmstart = self.last_U is not None
        U_nom = torch.zeros((horizon, 2), device=self.device, dtype=torch.float32)
        if had_warmstart:
            last_U_t = torch.as_tensor(self.last_U, device=self.device, dtype=torch.float32)
            shifted_len = min(last_U_t.shape[0] - 1, horizon)
            if shifted_len > 0:
                U_nom[:shifted_len] = last_U_t[1:1 + shifted_len]

        # Control-noise stddev per channel, anchored to half the actuator
        # limit so the sampling envelope auto-scales when mppi_max_v_mps /
        # mppi_max_w_rps change. No separate cov_scale knob — annealing
        # (below) handles iteration- and horizon-time modulation.
        cov_diag = torch.tensor(
            [0.5 * max_v_cells, 0.5 * max_w_rad],
            device=self.device, dtype=torch.float32,
        )

        w_control = 0.1

        # DIAL-MPC action-level annealing: noise variance grows with horizon
        # index, so early steps (which we'll actually commit to) stay near the
        # warm-started controls while tail steps explore widely. Precomputed
        # once: shape [H], values in (0, 1], h=H-1 is full noise.
        # On cold start (no warm-start to anchor to), annealing would squash
        # h=0 noise to near-zero around U_nom=0, locking the agent into a
        # stationary plan. Use uniform full noise across the horizon instead
        # so every step explores freely.
        beta_action = self.cfg.mppi_anneal_beta_action
        beta_traj   = self.cfg.mppi_anneal_beta_traj
        h_idx = torch.arange(horizon, device=self.device, dtype=torch.float32)
        if had_warmstart:
            action_scale = torch.exp(-(horizon - 1 - h_idx) / (beta_action * horizon))  # [H]
        else:
            action_scale = torch.ones(horizon, device=self.device, dtype=torch.float32)

        best_traj = [(int(start_z), int(start_x))]
        best_U = None
        best_score = -float('inf')
        for it in range(num_iters):
             # Sample control noise at the per-channel stddev, then anneal.
             noise = torch.randn((num_samples, horizon, 2), device=self.device) * cov_diag  # [K, H, 2]
             # DIAL-MPC trajectory-level annealing: iter 0 = full noise (wide
             # exploration), iter N-1 = shrunken noise (local refinement).
             # Combined with action-level scaling so iter N-1 + h=0 is the
             # most concentrated sample (refining the immediately-committed
             # action), iter 0 + h=H-1 is the widest (rough global coverage).
             traj_scale = float(np.exp(-it / (beta_traj * max(num_iters, 1))))
             noise = noise * (traj_scale * action_scale).view(1, horizon, 1)
             # Pin sample 0 to zero noise so the unmutated warm-start is always
             # evaluated as-is — protects against the case where every noisy
             # variant happens to be worse than the carried-over plan. Skip
             # this on cold start, where "U_nom + 0 noise" is just the
             # do-nothing plan and would bias the softmax toward stillness.
             if had_warmstart:
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

             # ── Collision: hard exclusion ──────────────────────────────────
             # Waypoint-level occupancy check. No segment sub-sampling: at the
             # default config a step covers at most ~1 cell (max_v_cells =
             # max_v_mps * dt / resolution), so consecutive waypoints are
             # adjacent cells and there is nothing to tunnel through.
             # A rollout that touches an obstacle ANYWHERE in its horizon is
             # discarded outright (scored -inf below), so only genuinely
             # collision-free plans compete and the argmax commits to a clean
             # escape instead of dwelling on a least-bad path.
             Z_idx = torch.clamp(Z_samples.round().long(), 0, Z_dim - 1)
             X_idx = torch.clamp(X_samples.round().long(), 0, X_dim - 1)
             occ_vals = occ_collision[Z_idx, X_idx]

             # Forgive the starting cell: if the agent is pressed against a wall
             # and technically starts inside an obstacle (especially with
             # dilation), we don't want to penalize rollouts just for being in
             # the start cell. Only entry into *new* obstacle cells counts.
             start_z_idx = min(max(int(round(start_z)), 0), Z_dim - 1)
             start_x_idx = min(max(int(round(start_x)), 0), X_dim - 1)
             at_start = (Z_idx == start_z_idx) & (X_idx == start_x_idx)

             # Arrival radius (cells): a single distance that drives BOTH the
             # obstacle-contact forgiveness below and the EXPLOIT arrival freeze.
             # Anchored to the episode-termination distance in main.py so
             # "arrived" here means the same thing as "done" there — keep these
             # in sync. (`mppi_goal_carve_radius`, the tiny disk opened around
             # the goal cell in `occ_collision`, is a separate, smaller knob.)
             arrival_radius = float(getattr(self.cfg, "stop_distance_m", 1.5)) / self.cfg.voxel_resolution

             # "Has the rollout reached the goal yet?" — monotonic in rollout
             # time via cummax, computed once and reused for both purposes:
             #   1) forgive obstacle contact once inside the radius (the target
             #      object is itself an obstacle, so the final approach must
             #      touch it — without this the planner is left orbiting it);
             #   2) mark the EXPLOIT absorbing region (below).
             sq_dist = (Z_samples - goal_z) ** 2 + (X_samples - goal_x) ** 2     # [K, H]
             reached_goal = torch.cummax(
                 (sq_dist <= arrival_radius ** 2).long(), dim=1).values.bool()   # [K, H]

             collide = (occ_vals >= 2) & ~at_start & ~reached_goal
             safe_mask = ~collide.any(dim=1)                                # [K]
             row = torch.arange(num_samples, device=self.device)

             # EXPLOIT absorbing terminal state: once a rollout enters the
             # arrival radius it is frozen for the rest of the horizon —
             # position pinned to the arrival point, controls zeroed. This makes
             # "reach the target fast, then sit" the highest-scoring behavior:
             # post-arrival motion costs nothing and earlier arrival earns more
             # dwell-step reward. Colliding rollouts are dropped by `safe_mask`
             # in selection, so none can profit by "arriving" through a wall.
             in_exploit = incoming_conf >= 0.99
             if in_exploit:
                 # First-arrival waypoint per rollout (argmax → first True; 0 for
                 # rollouts that never arrive — harmless, since `reached_goal` is
                 # all-False there and the freeze is then a no-op).
                 first_arr_idx = torch.argmax(reached_goal.long(), dim=1)        # [K]
                 ar_z = Z_samples[row, first_arr_idx]                            # [K]
                 ar_x = X_samples[row, first_arr_idx]                            # [K]
                 Z_samples = torch.where(reached_goal, ar_z.unsqueeze(1), Z_samples)
                 X_samples = torch.where(reached_goal, ar_x.unsqueeze(1), X_samples)
                 U_samples = U_samples * (~reached_goal).unsqueeze(-1).float()
                 # Distance pull over the pre-arrival approach only.
                 pre_arr = ~reached_goal
                 n_pre = pre_arr.float().sum(dim=1).clamp(min=1.0)
                 dist_cost = (sq_dist * pre_arr.float()).sum(dim=1) / n_pre
                 post_arr_wp = reached_goal
             else:
                 dist_cost = sq_dist.mean(dim=1)
                 post_arr_wp = torch.zeros((num_samples, horizon), dtype=torch.bool, device=self.device)

             # Re-discretize after the EXPLOIT arrival freeze (round, don't
             # truncate) so IG / best_traj see a rollout that stops at the
             # target rather than continuing through it.
             Z_idx = torch.clamp(Z_samples.round().long(), 0, Z_dim - 1)
             X_idx = torch.clamp(X_samples.round().long(), 0, X_dim - 1)

             scores = torch.zeros(num_samples, device=self.device)
             # Cost 1: Goal-distance pull (squared, to weight far points harder).
             scores -= w_goal * w_conf * dist_cost

             # Graded terminal-arrival reward (EXPLOIT only): proportional to the
             # fraction of the horizon spent absorbed at the target, so a rollout
             # that arrives early (drives straight in) outscores one that arrives
             # late or merely grazes the radius and drifts off.
             arrival_bonus = float(1000.0)
             scores += arrival_bonus * post_arr_wp.float().mean(dim=1)

             control_cost = torch.mean(torch.sqrt(U_samples[:, :, 0]**2 + U_samples[:, :, 1]**2), dim=1)
             scores -= w_control * control_cost

             # Cost 2: Information Gain (vectorized). Only rewarded for
             # collision-free rollouts (the FOV raycast already handles unseen
             # cells via line-of-sight) and skipped entirely when the IG weight
             # is ~0 (EXPLOIT), which saves the most expensive cost term.
             if w_ig_conf * w_ig > 1e-6 and safe_mask.any():
                 batch_ig, _ = compute_batch_fov_ig(
                     Z_idx[safe_mask], X_idx[safe_mask], Theta_samples[safe_mask],
                     ig_torch, occ_torch, self.cfg, self.device,
                     gamma_ig=0.95, intrinsics=intrinsics,
                     sensor_height=sensor_height,
                 )
                 scores[safe_mask] += w_ig_conf * w_ig * batch_ig

             # Hard-exclude colliding rollouts from selection and weighting.
             # When at least one safe rollout exists, argmax + softmax run over
             # the safe set only, so the committed plan is guaranteed collision-
             # free. When none survive (truly wedged), fall through to the soft
             # branch so U_nom still drifts toward a less-bad region; if best_U
             # stays None the caller spins to scan for a way out.
             if safe_mask.any():
                 NEG_INF = torch.tensor(float('-inf'), device=scores.device)
                 masked_scores = torch.where(safe_mask, scores, NEG_INF)
                 iter_best_idx = torch.argmax(masked_scores)
                 if masked_scores[iter_best_idx] > best_score:
                     best_score = masked_scores[iter_best_idx].item()
                     best_traj = [(int(Z_idx[iter_best_idx, t]), int(X_idx[iter_best_idx, t])) for t in range(horizon)]
                     best_U = U_samples[iter_best_idx].detach().cpu().numpy()  # [H, 2] (v_cells_per_step, w_rad_per_step)
                 beta = masked_scores.max()
                 weights = torch.exp((masked_scores - beta) / mppi_lambda)
             else:
                 # No safe rollout this iter. Don't weight by `scores`: its
                 # goal-distance term is actively misleading here — in a corner
                 # the goal sits behind the wall, so "closest to goal" means
                 # "drove straight into it", and U_nom would be nudged deeper
                 # into the obstacle. Instead steer the next iter's samples
                 # toward the controls that survived LONGEST before colliding
                 # (delaying contact ≈ turning out of the pocket). Across the
                 # iter loop this is a CE-style refit toward feasibility.
                 print("NO SAFE ROLLOUTS")
                 # First-collision step per rollout. `collide` is False at the
                 # forgiven start cell, so every rollout here collides at step
                 # >= 1 and argmax returns that first step (higher = better).
                 survival = torch.argmax(collide.long(), dim=1).float()         # [K]
                 # Goal proximity as a sub-step tiebreak among equal-survival
                 # rollouts, normalized to [0, 1) so it can never outweigh one
                 # extra surviving step.
                 d_span = dist_cost.max() - dist_cost.min()
                 goal_tiebreak = (dist_cost.max() - dist_cost) / (d_span + 1e-8)
                 soft = survival + 0.5 * goal_tiebreak
                 beta = soft.max()
                 weights = torch.exp((soft - beta) / mppi_lambda)
             weights = weights / (torch.sum(weights) + 1e-8)
             
             # Update nominal controls
             U_nom = U_nom + torch.sum(weights.view(-1, 1, 1) * noise, dim=0)

        # No safe rollout found this call: clear the warm-start so the next
        # replan samples from zero instead of biasing back toward the plan
        # that led us into the dead-end. The caller handles best_U=None by
        # spinning in place to scan for new options.
        if best_U is None:
            self.last_U = None
            # # Shrink horizon for the next call until either a safe plan is
            # # found or we hit min_horizon. Skip when the caller pinned the
            # # horizon explicitly — they own the schedule in that case.
            # if not horizon_pinned:
            #     self.current_horizon = max(
            #         self.min_horizon,
            #         self.current_horizon - self.horizon_shrink_step,
            #     )
        else:
            self.last_U = best_U.copy()
            if not horizon_pinned:
                self.current_horizon = self.default_horizon

        return best_traj, best_U
