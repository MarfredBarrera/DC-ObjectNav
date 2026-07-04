import torch
import numpy as np
from types import SimpleNamespace
from .utils import compute_batch_fov_ig, goal_distance_field, reachable_min


class MPPIPlanner:
    # Fixed cost constants (not exposed as config knobs).
    CONTROL_COST_WEIGHT = 0.0   # penalty on per-step control magnitude
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
        # Obstacle-aware start→goal distance (m) from the last plan's distance
        # field; None until a plan runs. Read by the caller's stop check.
        self.last_goal_dist_m = None

    # ─────────────────────────── public API ───────────────────────────

    def optimize_trajectory(self, start, goal, ig_map, occ_map,
                            num_samples=250, num_iters=None, horizon=None,
                            intrinsics=None, sensor_height=1.5,
                            initial_heading=None,
                            goal_confidence=0.0):
        """MPPI unicycle trajectory optimization from `start` to `goal` (both
        (z, x) grid cells). `ig_map` is the epistemic-uncertainty BEV map used
        as the information-gain signal. Returns (best_traj, best_U):
          best_traj : list of (z, x) cells — the highest-scoring rollout.
          best_U    : [H, 2] committed controls (v cells/step, w rad/step), or
                      None when no collision-free rollout was found.

        Each iteration samples DIAL-annealed control noise around the warm-
        started nominal, rolls out the unicycle, scores the rollouts, and
        refines the nominal by a softmax-weighted update. Colliding rollouts
        are hard-excluded (see `_score_rollouts` / `_select_and_weight`).
        """
        self.last_goal_dist_m = None
        if start is None or goal is None:
            return ([(int(start[0]), int(start[1]))] if start is not None else []), None

        cfg = self.cfg
        num_iters = cfg.mppi_num_iters if num_iters is None else num_iters
        horizon = cfg.mppi_horizon if horizon is None else horizon
        dt, res = cfg.mppi_dt, cfg.voxel_resolution

        w_conf, w_ig_conf, in_exploit = self._confidence_weights(goal_confidence)

        occ_torch = torch.from_numpy(occ_map).to(self.device).long()
        ig_torch = torch.from_numpy(ig_map).to(self.device).float()
        Z_dim, X_dim = ig_torch.shape

        # Obstacle-aware distance-to-go (cells): Dijkstra wavefront from the
        # goal cell over the BEV, with occupied cells traversable at
        # `mppi_occupied_cell_cost` per cell (see goal_distance_field). The
        # field is anchored at the cheapest cell the AGENT can actually reach
        # (min over the start cell's 8-connected non-occupied component —
        # typically the free floor where the goal-object's surface meets open
        # space), so values are ~"cells of free-space travel to the closest
        # reachable spot by the goal": the goal-pull gradient wraps around
        # walls instead of pointing through them (no Euclidean local minima),
        # and the arrival test can't leak across a wall (the crossing penalty
        # pushes far-side cells well past arrival_radius). Restricting the
        # anchor to the reachable component matters: an enclosed observed-free
        # pocket inside the goal object's blob (floor seen under/behind
        # furniture) would otherwise win the global min and inflate the
        # anchored distance at every reachable cell past the arrival radius —
        # the agent then orbits the object forever, unable to arrive or stop.
        # In EXPLOIT, unseen cells additionally cost `mppi_unseen_cell_cost`
        # each, so the committed approach prefers observed-free routes over
        # optimistic shortcuts through unexplored space (which may hide a wall
        # and force an SPL-burning backtrack). SEARCH keeps unseen at cost 1 —
        # exploration is supposed to enter unseen space.
        unseen_cost = cfg.mppi_unseen_cell_cost if in_exploit else 1.0
        raw_field = goal_distance_field(occ_map, goal, cfg.mppi_occupied_cell_cost,
                                        unseen_cell_cost=unseen_cost)
        anchor = reachable_min(raw_field, occ_map, start)
        dist_field = torch.from_numpy(raw_field).to(self.device)
        dist_field = (dist_field - float(anchor)).clamp_(min=0.0)
        # Start-cell distance in meters, read by the caller's stop check (an
        # obstacle-aware stand-in for geodesic distance, so the agent can't
        # declare TARGET REACHED through a wall).
        sz = min(max(int(round(float(start[0]))), 0), Z_dim - 1)
        sx = min(max(int(round(float(start[1]))), 0), X_dim - 1)
        self.last_goal_dist_m = float(dist_field[sz, sx]) * res

        # Per-call constants shared by the rollout/scoring helpers.
        p = SimpleNamespace(
            num_samples=num_samples, horizon=horizon,
            start_z=float(start[0]), start_x=float(start[1]),
            goal_z=float(goal[0]), goal_x=float(goal[1]),
            Z_dim=Z_dim, X_dim=X_dim,
            w_goal=cfg.mppi_w_goal, w_ig=cfg.mppi_w_ig, mppi_lambda=cfg.mppi_lambda,
            w_conf=w_conf, w_ig_conf=w_ig_conf, in_exploit=in_exploit,
            arrival_radius=cfg.stop_distance_m / res,
            dist_field=dist_field,
            v_min=(cfg.mppi_min_v_mps * dt) / res,
            v_max=(cfg.mppi_max_v_mps * dt) / res,
            w_max=cfg.mppi_max_w_rps * dt,
            intrinsics=intrinsics, sensor_height=sensor_height,
            occ_torch=occ_torch, ig_torch=ig_torch,
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
        """Score every rollout. Apply the EXPLOIT arrival freeze, then
        hard-exclude any rollout whose executed (post-freeze) trajectory hits
        an obstacle (only the start cell is forgiven), and combine
        goal-distance, arrival, control, and IG terms. Returns a namespace with the per-rollout `scores`, the
        `safe_mask`/`collide` collision info, `dist_cost`, and the post-freeze
        `U`/`Z_idx`/`X_idx` used to commit the winner."""
        K, H = p.num_samples, p.horizon
        # Obstacle-aware distance-to-go (cells) at every waypoint, looked up
        # in the anchored goal distance field (see optimize_trajectory). It
        # replaces the old Euclidean distance in both the cost and the arrival
        # test: the gradient routes around walls (no corner local minima) and
        # cells across a wall carry the crossing penalty, so they can never
        # register as "arrived".
        goal_dist = p.dist_field[
            torch.clamp(Z.round().long(), 0, p.Z_dim - 1),
            torch.clamp(X.round().long(), 0, p.X_dim - 1)]
        sq_dist = goal_dist ** 2
        # Monotonic in rollout time (cummax); drives the EXPLOIT freeze.
        reached_goal = torch.cummax(
            (goal_dist <= p.arrival_radius).long(), dim=1).values.bool()

        # EXPLOIT absorbing terminal state: freeze each rollout at its first
        # arrival waypoint (position pinned, controls zeroed) so "reach the
        # target fast, then sit" is the highest-scoring behavior. SEARCH skips
        # the freeze. The freeze runs BEFORE the collision check so collisions
        # are judged on the executed trajectory: the never-executed
        # post-arrival tail can't produce phantom hits, and the pinned arrival
        # cell itself must be free. There is NO arrival forgiveness — the goal
        # cell is often on the object surface (an obstacle), so rollouts must
        # stabilize on a free cell within `arrival_radius` of it, never inside
        # it; the freeze + arrival reward provide the incentive to sit there.
        if p.in_exploit:
            row = torch.arange(K, device=self.device)
            first_arr = torch.argmax(reached_goal.long(), dim=1)
            Z = torch.where(reached_goal, Z[row, first_arr].unsqueeze(1), Z)
            X = torch.where(reached_goal, X[row, first_arr].unsqueeze(1), X)
            U_samples = U_samples * (~reached_goal).unsqueeze(-1).float()
            pre_arr = (~reached_goal).float()
            dist_cost = (sq_dist * pre_arr).sum(dim=1) / pre_arr.sum(dim=1).clamp(min=1.0)
            post_arr = reached_goal
        else:
            dist_cost = sq_dist.mean(dim=1)
            post_arr = torch.zeros((K, H), dtype=torch.bool, device=self.device)

        Z_idx = torch.clamp(Z.round().long(), 0, p.Z_dim - 1)
        X_idx = torch.clamp(X.round().long(), 0, p.X_dim - 1)

        # Forgive the start cell only (the agent may legitimately start dilated
        # into a wall).
        start_z_idx = min(max(int(round(p.start_z)), 0), p.Z_dim - 1)
        start_x_idx = min(max(int(round(p.start_x)), 0), p.X_dim - 1)
        at_start = (Z_idx == start_z_idx) & (X_idx == start_x_idx)
        collide = (p.occ_torch[Z_idx, X_idx] >= 2) & ~at_start

        # Subsample each waypoint→waypoint segment. The agent can travel >1
        # cell per horizon step, so a waypoint-only check tunnels through thin
        # walls that fall between two consecutive waypoints. We interpolate
        # `S` interior points per segment, discretize, and fold any hit into
        # the *arrival* waypoint (t+1) so `collide`'s shape and the wedged-
        # branch survival time stay correct. Post-arrival segments are
        # degenerate (both endpoints pinned to the arrival cell), so they just
        # re-check that cell.
        S = int(self.cfg.mppi_collision_substeps)
        if S > 0 and H > 1:
            fracs = (torch.arange(1, S + 1, device=self.device, dtype=Z.dtype)
                     / (S + 1)).view(1, 1, S)                 # interior fractions
            Z_mid = Z[:, :-1, None] + (Z[:, 1:] - Z[:, :-1])[..., None] * fracs
            X_mid = X[:, :-1, None] + (X[:, 1:] - X[:, :-1])[..., None] * fracs
            Zi = torch.clamp(Z_mid.round().long(), 0, p.Z_dim - 1)  # [K, H-1, S]
            Xi = torch.clamp(X_mid.round().long(), 0, p.X_dim - 1)
            mid_occ = p.occ_torch[Zi, Xi] >= 2
            mid_at_start = (Zi == start_z_idx) & (Xi == start_x_idx)
            mid_collide = (mid_occ & ~mid_at_start).any(dim=2)  # [K, H-1]
            collide[:, 1:] = collide[:, 1:] | mid_collide

        safe_mask = ~collide.any(dim=1)

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
