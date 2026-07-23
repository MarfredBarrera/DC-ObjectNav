import math
from typing import Tuple

import numpy as np
import torch
import magnum as mn

import habitat_sim
import habitat_sim.utils.common as utils
import habitat_sim.physics as physics

from src.habitat.habitat_utils import get_camera_matrix, HABITAT_TO_CV


class SimInterface:
    """Thin wrapper around Habitat-Sim exposing robot control and sensor reads as plain tensors."""

    def __init__(self, cfg, sim, agent):
        self.cfg = cfg
        self.sim = sim
        self.agent = agent
        self.frames_processed: int = 0

        self._vel_control = physics.VelocityControl()
        self._vel_control.controlling_lin_vel = True
        self._vel_control.controlling_ang_vel = True
        self._vel_control.lin_vel_is_local = True   # apply v in agent frame so heading matters
        self._vel_control.ang_vel_is_local = True

        self.H = cfg.img_height
        self.W = cfg.img_width
        fov_x = math.radians(cfg.fov)
        self.fx = 0.5 * self.W / math.tan(0.5 * fov_x)
        self.fy = self.fx
        self.cx = self.W / 2.0
        self.cy = self.H / 2.0
        self.intrinsics: tuple = (self.fx, self.fy, self.cx, self.cy, self.H, self.W)

    def step(self, u: list, dt: float = 0.1) -> None:
        """Kinematic step. u = [forward_velocity (m/s), yaw_angular_velocity (rad/s)]."""
        lin_vel, ang_vel = float(u[0]), float(u[1])

        # Limit inputs
        max_w = getattr(self.cfg, "mppi_max_w_rps", 2.0)
        min_v = getattr(self.cfg, "mppi_min_v_mps", 0.0)
        max_v = getattr(self.cfg, "mppi_max_v_mps", 1.0)
        
        lin_vel = np.clip(lin_vel, min_v, max_v)
        ang_vel = np.clip(ang_vel, -max_w, max_w)

        agent_state = self.agent.get_state()
        pos = mn.Vector3(agent_state.position)
        rot = utils.quat_to_magnum(agent_state.rotation)

        # Angular step (yaw around local Y axis)
        dq = mn.Quaternion.rotation(mn.Rad(float(ang_vel * dt)), mn.Vector3(0.0, 1.0, 0.0))
        new_rot = rot * dq
        
        # Linear step (Habitat forward is local -Z)
        forward_world = rot.transform_vector(mn.Vector3(0.0, 0.0, -1.0))
        new_pos = pos + forward_world * float(lin_vel * dt)

        # # Enforce NavMesh constraints in Habitat
        # if self.sim.pathfinder.is_loaded:
        #     # If the new position isn't freely navigable, try to snap it to the navmesh to slide
        #     if not self.sim.pathfinder.is_navigable(new_pos):
        #         snapped_pos = self.sim.pathfinder.snap_point(new_pos)
        #         if not np.isnan(snapped_pos[0]):
        #             new_pos = snapped_pos
        #         else:
        #             new_pos = pos  # If snapping fails entirely, reject the linear translation
        # Enforce NavMesh constraints in Habitat

        # if self.sim.pathfinder.is_loaded:
        # try_step handles collisions and sliding, preventing "snapping" through walls
        new_pos = self.sim.pathfinder.try_step(pos, new_pos)

        agent_state.position = np.array(new_pos)
        agent_state.rotation = utils.quat_from_magnum(new_rot)
        self.agent.set_state(agent_state)

    def step_discrete(self, action: str, native: bool = False) -> None:
        """Execute one Habitat ObjectNav discrete primitive kinematically.

        action ∈ {"move_forward", "turn_left", "turn_right"}. Magnitudes come
        from cfg.discrete_forward_m / cfg.discrete_turn_deg. Forward translation
        reuses the NavMesh `try_step` so collisions slide exactly as in
        continuous mode.

        Two callers, two turn-sign conventions:
          - `native=False` (default, the MPPI tracking controller
            `discrete_action_from_plan`): turn names are in the MPPI grid-
            heading convention — "turn_left" increases grid θ — and are mapped
            onto Habitat's yaw via cfg.mppi_w_sign (the same sign the
            continuous `step` applies to ω), so a turn rotates the agent the
            same physical way the planner intended.
          - `native=True` (DD-PPO's EXPLOIT actions): DD-PPO's action strings
            are Habitat's own native action space already — not routed through
            any grid-θ abstraction — so mppi_w_sign must NOT be applied; that
            config knob only exists to correct the grid<->habitat handedness
            mismatch for the MPPI path. Applying it here inverted DD-PPO's
            turns: confirmed empirically — after commanding "turn_left" with
            w_sign applied, the bit-exact-verified bearing-to-goal (phi) grew
            by +discrete_turn_deg every step instead of shrinking, i.e. the
            agent turned away from the goal and could only ever spin in place.
        """
        forward_m = float(self.cfg.discrete_forward_m)
        turn_rad = math.radians(float(self.cfg.discrete_turn_deg))
        sign = 1.0 if native else float(self.cfg.mppi_w_sign)

        agent_state = self.agent.get_state()
        pos = mn.Vector3(agent_state.position)
        rot = utils.quat_to_magnum(agent_state.rotation)

        if action == "move_forward":
            forward_world = rot.transform_vector(mn.Vector3(0.0, 0.0, -1.0))
            new_pos = self.sim.pathfinder.try_step(pos, pos + forward_world * forward_m)
            agent_state.position = np.array(new_pos)
        elif action in ("turn_left", "turn_right"):
            yaw = sign * (turn_rad if action == "turn_left" else -turn_rad)
            dq = mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3(0.0, 1.0, 0.0))
            agent_state.rotation = utils.quat_from_magnum(rot * dq)
        else:
            raise ValueError(f"unknown discrete action: {action!r}")

        self.agent.set_state(agent_state)

    def get_observations(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (rgb [H,W,3] float32 0-1, depth [H,W] float32 metres, c2w [4,4] OpenCV frame)."""
        obs = self.sim.get_sensor_observations()

        rgb = torch.from_numpy(obs["rgb"][..., :3]).float() / 255.0
        depth = torch.from_numpy(obs["depth"]).float()

        c2w_hab = get_camera_matrix(self.agent)
        c2w = torch.from_numpy(c2w_hab @ HABITAT_TO_CV).float()

        self.frames_processed += 1
        return rgb, depth, c2w

    @property
    def agent_position(self) -> np.ndarray:
        return np.array(self.agent.get_state().position)

    @property
    def agent_rotation(self):
        return self.agent.get_state().rotation
