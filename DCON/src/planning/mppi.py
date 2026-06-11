import torch
import numpy as np
from collections import deque
from types import SimpleNamespace
from .utils import find_nearest_free_cell, compute_batch_fov_ig


class MPPIPlanner:
    # Fixed cost constants (not exposed as config knobs).
    CONTROL_COST_WEIGHT = 0.1   # penalty on per-step control magnitude
    ARRIVAL_BONUS = 1000.0      # EXPLOIT reward per horizon-fraction dwelt at goal
    IG_DISCOUNT = 0.95          # per-step discount on information gain
    EXPLOIT_CONF = 0.99         # goal_confidence >= this ⇒ EXPLOIT (arrival freeze)

    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = device
        # Previous safe control sequence, warm-started into the next call.
        # None until the first successful optimize_trajectory.
        self.last_U = None
        # Latched detection confidence (hysteresis so a brief sighting keeps the
        # goal-pull engaged after the target leaves view). `last_w_conf` is the
        # resulting goal weight, read by the caller for logging.
        self.exploit_conf = 0.0
        self.last_w_conf = 0.0

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

    # def get_top_k_sim_goals(self, sim_map, occ_map=None, k=10, obstacle_val=2, min_separation=10):
    #     """
    #     Return up to k goal candidates ranked by semantic similarity. Each candidate
    #     is snapped to the nearest free cell if it lands on an obstacle. Candidates
    #     are spaced at least `min_separation` cells apart so the fallback list
    #     explores genuinely different goals rather than near-duplicates.
    #     """
    #     target_sim = sim_map.copy().astype(np.float32)
    #     if occ_map is not None:
    #         target_sim[occ_map == 0] = -np.inf  # exclude unseen

    #     flat = target_sim.flatten()
    #     # Argsort descending; -inf entries naturally land at the end
    #     order = np.argsort(-flat)

    #     H, W = target_sim.shape
    #     chosen = []
    #     for idx in order:
    #         if not np.isfinite(flat[idx]):
    #             break
    #         y, x = int(idx // W), int(idx % W)
    #         cand = (y, x)
    #         if occ_map is not None and occ_map[y, x] >= obstacle_val:
    #             cand = find_nearest_free_cell(occ_map, cand, obstacle_val)
    #             if cand is None:
    #                 continue
    #         # Enforce separation from already-chosen goals
    #         if any(abs(cand[0] - cy) + abs(cand[1] - cx) < min_separation for cy, cx in chosen):
    #             continue
    #         chosen.append(cand)
    #         if len(chosen) >= k:
    #             break
    #     return chosen

    # ─────────────────────────── public API ───────────────────────────

    def optimize_trajectory(self, start, goal, ig_map, occ_map,
                            num_samples=250, num_iters=None, horizon=None,
                            intrinsics=None, sensor_height=1.5,
                            initial_heading=None, ig_source="unseen",
                            goal_confidence=0.0):
        """MPPI unicycle trajectory optimization from `start` to `goal` (both
        (z, x) grid cells). Returns (best_traj, best_U):
          best_traj : list of (z, x) cells — the highest-scoring rollout.
          best_U    : [H, 2] committed controls (v cells/step, w rad/step), or
                      None when no collision-free rollout was found.

        Each iteration samples DIAL-annealed control noise around the warm-
        started nominal, rolls out the unicycle, scores the rollouts, and
        refines the nominal by a softmax-weighted update. Colliding rollouts
        are hard-excluded (see `_score_rollouts` / `_select_and_weight`).
        """
        if start is None or goal is None:
            return ([(int(start[0]), int(start[1]))] if start is not None else []), None

        cfg = self.cfg
        num_iters = cfg.mppi_num_iters if num_iters is None else num_iters
        horizon = cfg.mppi_horizon if horizon is None else horizon
        dt, res = cfg.mppi_dt, cfg.voxel_resolution

        w_conf, w_ig_conf, in_exploit = self._confidence_weights(goal_confidence)

        occ_torch = torch.from_numpy(occ_map).to(self.device).long()
        # IG signal: a binary mask of unseen cells ("unseen"), or the supplied
        # uncertainty map ("epistemic"). The unseen branch ignores ig_map.
        if ig_source == "unseen":
            ig_torch = (occ_torch == 0).float()
        else:
            ig_torch = torch.from_numpy(ig_map).to(self.device).float()
        Z_dim, X_dim = ig_torch.shape

        # Per-call constants shared by the rollout/scoring helpers.
        p = SimpleNamespace(
            num_samples=num_samples, horizon=horizon,
            start_z=float(start[0]), start_x=float(start[1]),
            goal_z=float(goal[0]), goal_x=float(goal[1]),
            Z_dim=Z_dim, X_dim=X_dim,
            w_goal=cfg.mppi_w_goal, w_ig=cfg.mppi_w_ig, mppi_lambda=cfg.mppi_lambda,
            w_conf=w_conf, w_ig_conf=w_ig_conf, in_exploit=in_exploit,
            arrival_radius=cfg.stop_distance_m / res,
            v_min=(cfg.mppi_min_v_mps * dt) / res,
            v_max=(cfg.mppi_max_v_mps * dt) / res,
            w_max=cfg.mppi_max_w_rps * dt,
            intrinsics=intrinsics, sensor_height=sensor_height,
            occ_torch=occ_torch, ig_torch=ig_torch,
            occ_collision=self._carved_occupancy(occ_torch, float(goal[0]), float(goal[1])),
        )
        # Heading: real pose if given, else aim straight at the goal.
        p.theta_start = (float(initial_heading) if initial_heading is not None
                         else float(np.arctan2(p.goal_z - p.start_z, p.goal_x - p.start_x)))

        U_nom, had_warmstart = self._warm_start(horizon)
        cov_diag, action_scale = self._noise_envelope(p, had_warmstart)

        best = SimpleNamespace(traj=[(int(p.start_z), int(p.start_x))], U=None,
                               score=-float('inf'))
        for it in range(num_iters):
            noise = self._sample_noise(cov_diag, action_scale, it, num_iters, had_warmstart, p)
            U_samples, Z, X, Theta = self._rollout(U_nom, noise, p)
            ev = self._score_rollouts(U_samples, Z, X, Theta, p)
            weights = self._select_and_weight(ev, p, best)
            U_nom = U_nom + (weights.view(-1, 1, 1) * noise).sum(dim=0)

        # On failure, clear the warm-start so the next replan samples from zero
        # rather than biasing back toward the plan that led into the dead-end.
        self.last_U = best.U.copy() if best.U is not None else None
        return best.traj, best.U

    # ──────────────────────────── setup ────────────────────────────

    def _confidence_weights(self, goal_confidence):
        """Latched detection confidence → (goal-pull weight, IG weight, exploit
        flag). Hysteresis: exploit_conf = max(incoming, prev * decay). Below
        threshold the goal pull is off (pure IG); above it the concave curve
        w = scale * (a^conf - 1)/(a - 1) saturates quickly (a < 1), and IG is
        damped symmetrically by (1 - w_conf_norm)."""
        cfg = self.cfg
        incoming = float(np.clip(goal_confidence, 0.0, 1.0))
        self.exploit_conf = max(incoming, self.exploit_conf * cfg.mppi_conf_decay)
        if self.exploit_conf <= cfg.mppi_conf_threshold:
            w_conf, w_ig_conf = 0.0, 1.0
        else:
            a = cfg.mppi_conf_weight_a
            norm = (a ** self.exploit_conf - 1.0) / (a - 1.0)
            w_conf, w_ig_conf = cfg.mppi_conf_weight_scale * norm, 1.0 - norm
        self.last_w_conf = float(w_conf)
        return w_conf, w_ig_conf, incoming >= self.EXPLOIT_CONF

    def _carved_occupancy(self, occ_torch, goal_z, goal_x):
        """Collision-map clone with a small free disk (`mppi_goal_carve_radius`)
        carved around the goal cell, so the target — often itself an obstacle
        (a bed, a pillow) — is reachable. IG raycasts keep the un-carved map."""
        r = self.cfg.mppi_goal_carve_radius
        occ = occ_torch.clone()
        if r <= 0:
            return occ
        gz, gx = int(round(goal_z)), int(round(goal_x))
        z0, z1 = max(0, gz - r), min(occ.shape[0], gz + r + 1)
        x0, x1 = max(0, gx - r), min(occ.shape[1], gx + r + 1)
        zz, xx = torch.meshgrid(
            torch.arange(z0, z1, device=self.device),
            torch.arange(x0, x1, device=self.device), indexing='ij')
        disk = ((zz - gz) ** 2 + (xx - gx) ** 2) <= r ** 2
        patch = occ[z0:z1, x0:x1]
        patch[disk] = 1  # treat as free for collision
        occ[z0:z1, x0:x1] = patch
        return occ

    def _warm_start(self, horizon):
        """Nominal seed: the previous control sequence shifted left by one (the
        executed action dropped) and zero-padded. Cold start → zeros."""
        U_nom = torch.zeros((horizon, 2), device=self.device, dtype=torch.float32)
        had = self.last_U is not None
        if had:
            last = torch.as_tensor(self.last_U, device=self.device, dtype=torch.float32)
            n = min(last.shape[0] - 1, horizon)
            if n > 0:
                U_nom[:n] = last[1:1 + n]
        return U_nom, had

    def _noise_envelope(self, p, had_warmstart):
        """Per-channel noise stddev and the DIAL action-level scale [H]. stddev
        is half the actuator limit (auto-scales with the bounds). On cold start
        the action scale is flat so every horizon step explores; with a warm-
        start it grows toward the tail, keeping committed early steps near the
        carried-over plan."""
        cov_diag = torch.tensor([0.5 * p.v_max, 0.5 * p.w_max],
                                device=self.device, dtype=torch.float32)
        h_idx = torch.arange(p.horizon, device=self.device, dtype=torch.float32)
        if had_warmstart:
            beta_a = self.cfg.mppi_anneal_beta_action
            action_scale = torch.exp(-(p.horizon - 1 - h_idx) / (beta_a * p.horizon))
        else:
            action_scale = torch.ones(p.horizon, device=self.device, dtype=torch.float32)
        return cov_diag, action_scale

    # ──────────────────────── per-iteration ────────────────────────

    def _sample_noise(self, cov_diag, action_scale, it, num_iters, had_warmstart, p):
        """Annealed control noise [K, H, 2]. Trajectory-level annealing shrinks
        the variance across iterations (iter 0 widest, iter N-1 narrowest);
        combined with the action-level scale, iter N-1 / step 0 is the most
        concentrated sample. Sample 0 is pinned to the unmutated warm-start when
        one exists, so the carried-over plan is always evaluated as-is."""
        noise = torch.randn((p.num_samples, p.horizon, 2), device=self.device) * cov_diag
        traj_scale = float(np.exp(-it / (self.cfg.mppi_anneal_beta_traj * max(num_iters, 1))))
        noise = noise * (traj_scale * action_scale).view(1, p.horizon, 1)
        if had_warmstart:
            noise[0] = 0.0
        return noise

    def _rollout(self, U_nom, noise, p):
        """Turn-first unicycle rollout of U_nom + noise (clamped to the control
        bounds). Returns (U_samples, Z, X, Theta), each [K, H]."""
        U = U_nom.unsqueeze(0) + noise
        U[:, :, 0].clamp_(p.v_min, p.v_max)
        U[:, :, 1].clamp_(-p.w_max, p.w_max)

        Theta = torch.zeros((p.num_samples, p.horizon), device=self.device)
        Theta[:, 0] = p.theta_start
        Theta[:, 1:] = p.theta_start + torch.cumsum(U[:, :-1, 1], dim=1)

        Z = torch.zeros((p.num_samples, p.horizon), device=self.device)
        X = torch.zeros((p.num_samples, p.horizon), device=self.device)
        Z[:, 0], X[:, 0] = p.start_z, p.start_x
        # Translate using the heading AFTER each step's turn, so w_0 steers step 0.
        Z[:, 1:] = p.start_z + torch.cumsum(U[:, :-1, 0] * torch.sin(Theta[:, 1:]), dim=1)
        X[:, 1:] = p.start_x + torch.cumsum(U[:, :-1, 0] * torch.cos(Theta[:, 1:]), dim=1)
        return U, Z, X, Theta

    def _score_rollouts(self, U_samples, Z, X, Theta, p):
        """Score every rollout. Hard-exclude any that hits an obstacle (the
        start cell always forgiven; the goal-arrival region forgiven in
        EXPLOIT only), apply the EXPLOIT arrival freeze, then combine
        goal-distance, arrival, control, and IG terms. Returns a namespace with the per-rollout `scores`, the
        `safe_mask`/`collide` collision info, `dist_cost`, and the post-freeze
        `U`/`Z_idx`/`X_idx` used to commit the winner."""
        K, H = p.num_samples, p.horizon
        Z_idx = torch.clamp(Z.round().long(), 0, p.Z_dim - 1)
        X_idx = torch.clamp(X.round().long(), 0, p.X_dim - 1)

        # Forgive the start cell (the agent may legitimately start dilated into
        # a wall). `reached_goal` is monotonic in rollout time (cummax) and
        # drives the EXPLOIT freeze below.
        #
        # Arrival forgiveness is EXPLOIT-only: there the goal is a confirmed
        # detection and itself an obstacle, so the final approach must be
        # allowed to touch it. In SEARCH the goal is just a similarity peak —
        # often ON or BEHIND a wall — and `sq_dist` knows nothing about
        # geometry, so forgiving near-goal contact licenses rollouts to
        # tunnel through the wall for the last `arrival_radius` cells.
        # (SEARCH still gets the small `mppi_goal_carve_radius` disk carved
        # in `occ_collision`, which is the intended, much tighter escape.)
        start_z_idx = min(max(int(round(p.start_z)), 0), p.Z_dim - 1)
        start_x_idx = min(max(int(round(p.start_x)), 0), p.X_dim - 1)
        at_start = (Z_idx == start_z_idx) & (X_idx == start_x_idx)
        sq_dist = (Z - p.goal_z) ** 2 + (X - p.goal_x) ** 2
        reached_goal = torch.cummax(
            (sq_dist <= p.arrival_radius ** 2).long(), dim=1).values.bool()
        forgive_arrival = (reached_goal if p.in_exploit
                           else torch.zeros_like(reached_goal))
        collide = (p.occ_collision[Z_idx, X_idx] >= 2) & ~at_start & ~forgive_arrival

        # Subsample each waypoint→waypoint segment. The agent can travel >1
        # cell per horizon step, so a waypoint-only check tunnels through thin
        # walls that fall between two consecutive waypoints. We interpolate
        # `S` interior points per segment, discretize, and fold any hit into
        # the *arrival* waypoint (t+1) so `collide`'s shape and the wedged-
        # branch survival time stay correct.
        S = int(self.cfg.mppi_collision_substeps)
        if S > 0 and H > 1:
            fracs = (torch.arange(1, S + 1, device=self.device, dtype=Z.dtype)
                     / (S + 1)).view(1, 1, S)                 # interior fractions
            Z_mid = Z[:, :-1, None] + (Z[:, 1:] - Z[:, :-1])[..., None] * fracs
            X_mid = X[:, :-1, None] + (X[:, 1:] - X[:, :-1])[..., None] * fracs
            Zi = torch.clamp(Z_mid.round().long(), 0, p.Z_dim - 1)  # [K, H-1, S]
            Xi = torch.clamp(X_mid.round().long(), 0, p.X_dim - 1)
            mid_occ = p.occ_collision[Zi, Xi] >= 2
            mid_at_start = (Zi == start_z_idx) & (Xi == start_x_idx)
            # EXPLOIT-only, same as the waypoint check above.
            seg_forgive = forgive_arrival[:, 1:, None]  # arrived by segment end (monotone)
            mid_collide = (mid_occ & ~mid_at_start & ~seg_forgive).any(dim=2)  # [K, H-1]
            collide[:, 1:] = collide[:, 1:] | mid_collide

        safe_mask = ~collide.any(dim=1)

        # EXPLOIT absorbing terminal state: freeze each rollout at its first
        # arrival waypoint (position pinned, controls zeroed) so "reach the
        # target fast, then sit" is the highest-scoring behavior. SEARCH skips
        # the freeze. Colliding rollouts are dropped by `safe_mask` in
        # selection, so none can profit by "arriving" through a wall.
        if p.in_exploit:
            row = torch.arange(K, device=self.device)
            first_arr = torch.argmax(reached_goal.long(), dim=1)
            Z = torch.where(reached_goal, Z[row, first_arr].unsqueeze(1), Z)
            X = torch.where(reached_goal, X[row, first_arr].unsqueeze(1), X)
            U_samples = U_samples * (~reached_goal).unsqueeze(-1).float()
            pre_arr = (~reached_goal).float()
            dist_cost = (sq_dist * pre_arr).sum(dim=1) / pre_arr.sum(dim=1).clamp(min=1.0)
            post_arr = reached_goal
            # Re-discretize so IG / best_traj see a rollout that stops at the goal.
            Z_idx = torch.clamp(Z.round().long(), 0, p.Z_dim - 1)
            X_idx = torch.clamp(X.round().long(), 0, p.X_dim - 1)
        else:
            dist_cost = sq_dist.mean(dim=1)
            post_arr = torch.zeros((K, H), dtype=torch.bool, device=self.device)

        # Goal-distance pull (squared), graded arrival reward, control penalty.
        scores = -p.w_goal * p.w_conf * dist_cost
        scores += self.ARRIVAL_BONUS * post_arr.float().mean(dim=1)
        control_cost = torch.sqrt(U_samples[:, :, 0] ** 2 + U_samples[:, :, 1] ** 2).mean(dim=1)
        scores -= self.CONTROL_COST_WEIGHT * control_cost

        # Information gain (collision-free rollouts only; skipped when its
        # weight is ~0, i.e. EXPLOIT — saves the most expensive cost term).
        if p.w_ig_conf * p.w_ig > 1e-6 and safe_mask.any():
            batch_ig, _ = compute_batch_fov_ig(
                Z_idx[safe_mask], X_idx[safe_mask], Theta[safe_mask],
                p.ig_torch, p.occ_torch, self.cfg, self.device,
                gamma_ig=self.IG_DISCOUNT, intrinsics=p.intrinsics,
                sensor_height=p.sensor_height)
            scores[safe_mask] += p.w_ig_conf * p.w_ig * batch_ig

        return SimpleNamespace(scores=scores, safe_mask=safe_mask, collide=collide,
                               dist_cost=dist_cost, U=U_samples, Z_idx=Z_idx, X_idx=X_idx)

    def _select_and_weight(self, ev, p, best):
        """Update `best` with this iteration's top collision-free rollout and
        return the softmax weights for the U_nom update.

        With safe rollouts the argmax + softmax run over them only, so the
        committed plan is guaranteed collision-free. With NONE safe (wedged),
        weight instead by survival time — steps before first collision — so the
        next iter's samples steer toward delaying contact (turning out of the
        pocket) rather than toward the goal-closest crash. Goal proximity is a
        normalized sub-step tiebreak. If `best.U` stays None the caller recovers."""
        if ev.safe_mask.any():
            neg_inf = torch.tensor(float('-inf'), device=self.device)
            logits = torch.where(ev.safe_mask, ev.scores, neg_inf)
            idx = torch.argmax(logits)
            if logits[idx] > best.score:
                best.score = logits[idx].item()
                best.traj = [(int(ev.Z_idx[idx, t]), int(ev.X_idx[idx, t]))
                             for t in range(p.horizon)]
                best.U = ev.U[idx].detach().cpu().numpy()  # [H, 2]
        else:
            print("NO SAFE ROLLOUTS")
            # `collide` is False at the forgiven start cell, so every rollout
            # here collides at step >= 1 and argmax gives that first step.
            survival = torch.argmax(ev.collide.long(), dim=1).float()
            span = ev.dist_cost.max() - ev.dist_cost.min()
            goal_tiebreak = (ev.dist_cost.max() - ev.dist_cost) / (span + 1e-8)
            logits = survival + 0.5 * goal_tiebreak

        weights = torch.exp((logits - logits.max()) / p.mppi_lambda)
        return weights / (weights.sum() + 1e-8)
