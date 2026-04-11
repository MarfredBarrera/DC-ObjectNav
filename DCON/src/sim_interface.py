import math
from typing import Tuple

import numpy as np
import torch
import magnum as mn

import habitat_sim
import habitat_sim.utils.common as utils
import habitat_sim.physics as physics

from src.habitat_utils import get_camera_matrix, HABITAT_TO_CV


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

        # Habitat forward is -Z
        self._vel_control.linear_velocity = np.array([0.0, 0.0, -lin_vel])
        self._vel_control.angular_velocity = np.array([0.0, ang_vel, 0.0])

        agent_state = self.agent.get_state()
        rigid_state = habitat_sim.RigidState(
            utils.quat_to_magnum(agent_state.rotation),
            mn.Vector3(agent_state.position),
        )
        new_state = self._vel_control.integrate_transform(dt, rigid_state)

        agent_state.position = np.array(new_state.translation)
        agent_state.rotation = utils.quat_from_magnum(new_state.rotation)
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
