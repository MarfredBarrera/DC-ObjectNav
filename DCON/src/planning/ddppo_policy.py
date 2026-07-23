"""Standalone DD-PPO PointNav policy (Wijmans et al., 2020) for EXPLOIT.

Replaces the deterministic A* + waypoint controller with the actual
pretrained PointGoal navigation policy from the DD-PPO paper: a
depth-only, recurrent (LSTM) RL policy trained on ~2.5 billion simulated
frames, which learns collision-avoidance implicitly from raw depth rather
than from any explicit occupancy grid. Point-goal navigation is exactly
the sub-problem this model was built to solve, and it sidesteps the whole
class of BEV-occupancy-vs-Habitat-collision mismatches the A* controller
had to work around.

This module is a minimal, dependency-free reimplementation of
`habitat_baselines.rl.ddppo.policy.resnet_policy.PointNavResNetPolicy`
(vendored from facebookresearch/habitat-lab, tag v0.1.7) trimmed to
exactly the configuration our checkpoint uses (confirmed by inspecting
its `state_dict` and `model_args`):

    backbone=se_resneXt101, rnn_type=LSTM, hidden_size=1024,
    num_recurrent_layers=2, resnet_baseplanes=32, sensors=DEPTH_SENSOR
    (no RGB, no running_mean_and_var input normalization)

Module names below match the checkpoint's state_dict keys exactly
(including the upstream `tgt_embeding` typo) so `load_state_dict` works
with no remapping. Only inference is implemented (no training path).
"""

from typing import List, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ─────────────────────── vendored: resnet.py (backbone) ───────────────────────

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=groups)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SE(nn.Module):
    def __init__(self, planes, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(planes, int(planes / r)), nn.ReLU(True),
            nn.Linear(int(planes / r), planes), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.squeeze(x).view(b, c)
        x = self.excite(x)
        return x.view(b, c, 1, 1)


def _build_bottleneck_branch(inplanes, planes, ngroups, stride, expansion, groups=1):
    return nn.Sequential(
        conv1x1(inplanes, planes), nn.GroupNorm(ngroups, planes), nn.ReLU(True),
        conv3x3(planes, planes, stride, groups=groups), nn.GroupNorm(ngroups, planes), nn.ReLU(True),
        conv1x1(planes, planes * expansion), nn.GroupNorm(ngroups, planes * expansion),
    )


class Bottleneck(nn.Module):
    expansion = 4
    resneXt = False

    def __init__(self, inplanes, planes, ngroups, stride=1, downsample=None, cardinality=1):
        super().__init__()
        self.convs = _build_bottleneck_branch(inplanes, planes, ngroups, stride, self.expansion, groups=cardinality)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def _impl(self, x):
        identity = x
        out = self.convs(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

    def forward(self, x):
        return self._impl(x)


class SEBottleneck(Bottleneck):
    def __init__(self, inplanes, planes, ngroups, stride=1, downsample=None, cardinality=1):
        super().__init__(inplanes, planes, ngroups, stride, downsample, cardinality)
        self.se = SE(planes * self.expansion)

    def _impl(self, x):
        identity = x
        out = self.convs(x)
        out = self.se(out) * out
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class SEResNeXtBottleneck(SEBottleneck):
    expansion = 2
    resneXt = True


Block = Union[Type[Bottleneck], Type[SEResNeXtBottleneck]]


class ResNet(nn.Module):
    def __init__(self, in_channels, base_planes, ngroups, block: Block, layers: List[int], cardinality=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_planes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(ngroups, base_planes), nn.ReLU(True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.cardinality = cardinality
        self.inplanes = base_planes
        if block.resneXt:
            base_planes *= 2
        self.layer1 = self._make_layer(block, ngroups, base_planes, layers[0])
        self.layer2 = self._make_layer(block, ngroups, base_planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, ngroups, base_planes * 2 * 2, layers[2], stride=2)
        self.layer4 = self._make_layer(block, ngroups, base_planes * 2 * 2 * 2, layers[3], stride=2)
        self.final_channels = self.inplanes
        self.final_spatial_compress = 1.0 / (2 ** 5)

    def _make_layer(self, block, ngroups, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.GroupNorm(ngroups, planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, ngroups, stride, downsample, cardinality=self.cardinality)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, ngroups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def se_resneXt101(in_channels, base_planes, ngroups):
    return ResNet(in_channels, base_planes, ngroups, SEResNeXtBottleneck, [3, 4, 23, 3],
                 cardinality=int(base_planes / 2))


# ────────────────── vendored: rnn_state_encoder.py (LSTM only) ──────────────────

class LSTMStateEncoder(nn.Module):
    """Single-step LSTM wrapper matching habitat_baselines' RNNStateEncoder
    for the batch-size-1, one-step-at-a-time inference case (no packed
    sequences needed — this codebase always calls with T=1)."""

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.num_recurrent_layers = num_layers * 2  # h and c stacked
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    def pack_hidden(self, hidden_states):
        return torch.cat(hidden_states, 0)

    def unpack_hidden(self, hidden_states):
        h, c = torch.chunk(hidden_states, 2, 0)
        return (h, c)

    def forward(self, x, hidden_states, masks):
        # hidden_states: [1, num_recurrent_layers, hidden]:contentReference[oaicite:0]{index=0} -> permute to [num_recurrent_layers, 1, hidden]
        hidden_states = hidden_states.permute(1, 0, 2)
        hidden_states = torch.where(masks.view(1, -1, 1), hidden_states, hidden_states.new_zeros(()))
        x, hidden_states = self.rnn(x.unsqueeze(0), self.unpack_hidden(hidden_states))
        hidden_states = self.pack_hidden(hidden_states)
        x = x.squeeze(0)
        hidden_states = hidden_states.permute(1, 0, 2)
        return x, hidden_states


# ────────────────── vendored: utils/common.py (action head) ──────────────────

class CustomFixedCategorical(torch.distributions.Categorical):
    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class CategoricalNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        return CustomFixedCategorical(logits=self.linear(x))


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)


# ───────────────── trimmed PointNavResNetNet (depth + pointgoal only) ─────────────────

class ResNetEncoder(nn.Module):
    """Depth-only visual encoder (no RGB, no input normalization — this
    checkpoint's `model_args.sensors == 'DEPTH_SENSOR'` and its state_dict
    carries no `running_mean_and_var.*` keys, confirming
    `normalize_visual_inputs=False`)."""

    def __init__(self, depth_hw, baseplanes=32, ngroups=16, make_backbone=se_resneXt101):
        super().__init__()
        spatial_size = depth_hw // 2
        self.running_mean_and_var = nn.Sequential()  # identity: normalize_visual_inputs=False
        self.backbone = make_backbone(1, baseplanes, ngroups)
        final_spatial = int(spatial_size * self.backbone.final_spatial_compress)
        after_compression_flat_size = 2048
        num_compression_channels = int(round(after_compression_flat_size / (final_spatial ** 2)))
        self.compression = nn.Sequential(
            nn.Conv2d(self.backbone.final_channels, num_compression_channels,
                     kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, num_compression_channels), nn.ReLU(True),
        )
        self.output_shape = (num_compression_channels, final_spatial, final_spatial)

    def forward(self, depth):
        # depth: [B, H, W, 1] float32 in [0, 1] (Habitat's NORMALIZE_DEPTH convention)
        x = depth.permute(0, 3, 1, 2)
        x = F.avg_pool2d(x, 2)
        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x


class PointNavResNetNet(nn.Module):
    """Visual + pointgoal-with-GPS-compass + previous-action -> LSTM.

    Module names (`prev_action_embedding`, `tgt_embeding` [sic, matches the
    upstream typo], `visual_encoder`, `visual_fc`, `state_encoder`) match the
    checkpoint's state_dict exactly.
    """

    def __init__(self, depth_hw=256, hidden_size=1024, num_recurrent_layers=2,
                resnet_baseplanes=32, num_actions=4):
        super().__init__()
        self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
        self.tgt_embeding = nn.Linear(3, 32)  # [rho, cos(-phi), sin(-phi)]
        self.visual_encoder = ResNetEncoder(depth_hw, baseplanes=resnet_baseplanes,
                                            ngroups=resnet_baseplanes // 2)
        self.visual_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(np.prod(self.visual_encoder.output_shape)), hidden_size),
            nn.ReLU(True),
        )
        self._hidden_size = hidden_size
        rnn_input_size = hidden_size + 32 + 32  # visual + goal + prev_action
        self.state_encoder = LSTMStateEncoder(rnn_input_size, hidden_size, num_recurrent_layers)

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, depth, pointgoal, prev_actions, masks, rnn_hidden_states):
        visual_feats = self.visual_fc(self.visual_encoder(depth))

        # Same 2D polar -> [rho, cos(-phi), sin(-phi)] transform as upstream.
        goal_3d = torch.stack(
            [pointgoal[:, 0], torch.cos(-pointgoal[:, 1]), torch.sin(-pointgoal[:, 1])], -1)
        goal_feats = self.tgt_embeding(goal_3d)

        prev_actions = prev_actions.squeeze(-1)
        start_token = torch.zeros_like(prev_actions)
        prev_action_feats = self.prev_action_embedding(
            torch.where(masks.view(-1), prev_actions + 1, start_token))

        x = torch.cat([visual_feats, goal_feats, prev_action_feats], dim=1)
        out, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
        return out, rnn_hidden_states


class PointNavResNetPolicy(nn.Module):
    def __init__(self, depth_hw=256, hidden_size=1024, num_recurrent_layers=2,
                resnet_baseplanes=32, num_actions=4):
        super().__init__()
        self.net = PointNavResNetNet(depth_hw, hidden_size, num_recurrent_layers,
                                     resnet_baseplanes, num_actions)
        self.action_distribution = CategoricalNet(hidden_size, num_actions)
        self.critic = CriticHead(hidden_size)  # unused at inference; kept so state_dict loads cleanly

    @torch.no_grad()
    def act(self, depth, pointgoal, rnn_hidden_states, prev_actions, masks, deterministic=True):
        features, rnn_hidden_states = self.net(depth, pointgoal, prev_actions, masks, rnn_hidden_states)
        distribution = self.action_distribution(features)
        action = distribution.mode() if deterministic else distribution.sample()
        return action, rnn_hidden_states


# ───────────────────────────── goal-sensor math ─────────────────────────────

def compute_pointgoal_polar(agent_position, agent_rotation, goal_position) -> Tuple[float, float]:
    """Exact replica of habitat-lab's IntegratedPointGoalGPSAndCompassSensor
    (POLAR, 2D): rotates the world-frame direction-to-goal into the agent's
    local frame (forward = -Z) via the inverse rotation quaternion, then
    returns (rho, -phi) with phi = atan2(local_x, -local_z). Sign/convention
    must match training exactly since the policy's `tgt_embeding` layer was
    fit to this specific parameterization.
    """
    direction_vector = np.asarray(goal_position, dtype=np.float64) - np.asarray(agent_position, dtype=np.float64)
    inv_rot = agent_rotation.inverse()
    vq = np.quaternion(0.0, *direction_vector)
    direction_vector_agent = (inv_rot * vq * inv_rot.inverse()).imag
    rho = float(np.hypot(-direction_vector_agent[2], direction_vector_agent[0]))
    phi = float(np.arctan2(direction_vector_agent[0], -direction_vector_agent[2]))
    return rho, -phi


# ────────────────────────────── loading + wrapper ──────────────────────────────

def load_ddppo_policy(checkpoint_path: str, device: str = "cuda") -> PointNavResNetPolicy:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["model_args"]
    if args.backbone != "se_resneXt101" or args.rnn_type != "LSTM":
        raise ValueError(
            f"ddppo_policy.py is vendored for backbone=se_resneXt101/rnn_type=LSTM only; "
            f"checkpoint has backbone={args.backbone!r}, rnn_type={args.rnn_type!r}")
    policy = PointNavResNetPolicy(
        hidden_size=args.hidden_size, num_recurrent_layers=args.num_recurrent_layers,
        resnet_baseplanes=args.resnet_baseplanes)
    state_dict = {k[len("actor_critic."):]: v for k, v in ckpt["state_dict"].items()
                 if k.startswith("actor_critic.")}
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    return policy


class DDPPONavigator:
    """Stateful driver: manages the LSTM hidden state / previous-action /
    not-done mask across replans and exposes one `act()` call per replan,
    mirroring `PPOAgent` in habitat_baselines' challenge submission wrapper.

    Actions: 0=STOP, 1=MOVE_FORWARD, 2=TURN_LEFT, 3=TURN_RIGHT (Habitat's
    fixed `HabitatSimActionsSingleton` default ordering).
    """
    ACTION_NAMES = ["stop", "move_forward", "turn_left", "turn_right"]

    def __init__(self, policy: PointNavResNetPolicy, device: str = "cuda", depth_hw: int = 256):
        self.policy = policy
        self.device = device
        self.depth_hw = depth_hw
        self.hidden_states = None
        self.prev_actions = None
        self.not_done_masks = None
        self.reset()

    def reset(self):
        self.hidden_states = torch.zeros(
            1, self.policy.net.num_recurrent_layers, self.policy.net._hidden_size, device=self.device)
        self.not_done_masks = torch.zeros(1, 1, device=self.device, dtype=torch.bool)
        self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)

    def act(self, depth_raw, min_depth_m, max_depth_m, agent_position, agent_rotation,
           goal_position, deterministic: bool = False) -> str:
        """`depth_raw` is a [H, W] (or [H, W, 1]) depth tensor/array in
        METERS at any resolution; resized + clipped/normalized to Habitat's
        training convention (256x256, [min_depth_m, max_depth_m] -> [0, 1])
        here. Returns one of ACTION_NAMES; also advances internal state.

        `deterministic=False` (sample from the categorical, not argmax)
        matches Habitat's own reference evaluation wrapper
        (habitat_baselines/agents/ppo_agents.py: `actor_critic.act(...,
        deterministic=False)`) — confirmed load-bearing in practice: greedy
        argmax got stuck in a stable turn_left/turn_right 2-cycle (the
        previous-action embedding feeds back into the LSTM, so a
        deterministic policy can lock into "correcting" its own last turn
        forever with zero randomness to escape it).
        """
        depth = torch.as_tensor(depth_raw, dtype=torch.float32, device=self.device)
        if depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        else:
            depth = depth.permute(2, 0, 1).unsqueeze(0)  # [1,1,H,W]
        # Nearest, not bilinear: our sim renders at a higher native resolution
        # (512x512) than training's native 256x256, so this resize is our own
        # addition with no upstream equivalent. Bilinear blends near/far depth
        # across object edges (a doorframe, a chair leg against open floor)
        # into a fabricated intermediate depth — exactly the kind of
        # phantom-obstacle reading a depth-only collision-avoidance policy
        # would react to. Nearest-exact preserves the hard discontinuity.
        depth = F.interpolate(depth, size=(self.depth_hw, self.depth_hw),
                              mode="nearest-exact")
        # A depth-sensor miss (no ray intersection — a hole in the scanned
        # Gibson mesh, a window, etc.) reads back as exactly 0.0, not a small
        # positive distance. Normalizing that literally would tell DD-PPO
        # "solid obstacle at point-blank range" for what is actually unknown/
        # open space — confirmed empirically: every observed stall showed a
        # large contiguous block of exact-zero pixels centered in frame.
        # Treat a miss as far/clear (matches how the rest of this codebase
        # already excludes depth <= min_sensor_dist as invalid rather than
        # "near").
        depth = torch.where(depth <= min_depth_m, torch.full_like(depth, max_depth_m), depth)
        depth = ((depth - min_depth_m) / (max_depth_m - min_depth_m)).clamp(0.0, 1.0)
        depth = depth.permute(0, 2, 3, 1)  # [1, H, W, 1]

        rho, phi = compute_pointgoal_polar(agent_position, agent_rotation, goal_position)
        pointgoal = torch.tensor([[rho, phi]], dtype=torch.float32, device=self.device)

        action, self.hidden_states = self.policy.act(
            depth, pointgoal, self.hidden_states, self.prev_actions, self.not_done_masks,
            deterministic=deterministic)

        self.not_done_masks.fill_(True)
        self.prev_actions.copy_(action)
        return self.ACTION_NAMES[int(action.item())]
