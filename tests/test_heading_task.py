from __future__ import annotations

import importlib.util
import math
import torch
from pathlib import Path

_MATH_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/heading_math.py"
)
_SPEC = importlib.util.spec_from_file_location("heading_math", _MATH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
compute_heading_reward = _MODULE.compute_heading_reward


def test_heading_reward_prefers_matched_velocity_and_facing() -> None:
    heading = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    facing = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    velocity = torch.tensor([[1.2, 0.0], [-1.2, 0.0]])
    robot_facing = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])

    reward = compute_heading_reward(heading, facing, velocity, robot_facing, target_speed=1.2)

    assert reward[0] > reward[1]
    assert torch.isclose(reward[0], torch.tensor(1.0), atol=1e-6)


def test_heading_reward_uses_paper_constants() -> None:
    heading = torch.tensor([[0.0, 1.0]])
    facing = torch.tensor([[0.0, 1.0]])
    velocity = torch.tensor([[0.0, 0.0]])
    robot_facing = torch.tensor([[0.0, 1.0]])

    reward = compute_heading_reward(heading, facing, velocity, robot_facing, target_speed=1.2)
    expected = 0.7 * math.exp(-0.25 * 1.2**2) + 0.3

    assert torch.isclose(reward[0], torch.tensor(expected), atol=1e-6)


def test_motion_reset_heading_prefers_velocity_over_yaw() -> None:
    # Keep this test independent of IsaacLab imports by mirroring the small
    # provider math: above the threshold, heading follows root xy velocity.
    root_vel = torch.tensor([[0.0, 2.0, 0.0], [0.01, 0.0, 0.0]])
    speed = torch.norm(root_vel[:, :2], dim=-1, keepdim=True)
    vel_dir = root_vel[:, :2] / torch.clamp(speed, min=1.0e-6)
    yaw_dir = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    heading = torch.where((speed[:, 0] >= 0.2).unsqueeze(-1), vel_dir, yaw_dir)

    assert torch.allclose(heading[0], torch.tensor([0.0, 1.0]))
    assert torch.allclose(heading[1], torch.tensor([0.0, 1.0]))
