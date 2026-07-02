from __future__ import annotations

import importlib.util
import sys
import torch
import types
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "terminations.py"
)

isaaclab_module = types.ModuleType("isaaclab")
isaaclab_utils_module = types.ModuleType("isaaclab.utils")
isaaclab_math_module = types.ModuleType("isaaclab.utils.math")
assets_module = types.ModuleType("isaaclab.assets")
managers_module = types.ModuleType("isaaclab.managers")
sensors_module = types.ModuleType("isaaclab.sensors")
commands_module = types.ModuleType("whole_body_tracking.tasks.tracking.mdp.commands")
heading_math_module = types.ModuleType("whole_body_tracking.tasks.tracking.mdp.heading_math")
rewards_module = types.ModuleType("whole_body_tracking.tasks.tracking.mdp.rewards")


class DummyTerminationManager:
    pass


class DummySceneEntityCfg:
    def __init__(self, name=None, **kwargs):
        self.name = name
        for key, value in kwargs.items():
            setattr(self, key, value)


class DummyAsset:
    pass


def _quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    del quat
    return vec


def _compute_height_filtered_contact_mask(*args, **kwargs) -> torch.Tensor:
    del kwargs
    return torch.zeros_like(args[0], dtype=torch.bool)


def _get_body_indexes(command, body_names):
    del body_names
    return list(range(len(command.cfg.body_names)))


isaaclab_math_module.quat_apply_inverse = _quat_apply_inverse
assets_module.Articulation = DummyAsset
assets_module.RigidObject = DummyAsset
managers_module.SceneEntityCfg = DummySceneEntityCfg
managers_module.TerminationManager = DummyTerminationManager
sensors_module.ContactSensor = DummyAsset
commands_module.MotionCommand = DummyAsset
heading_math_module.compute_height_filtered_contact_mask = _compute_height_filtered_contact_mask
rewards_module._get_body_indexes = _get_body_indexes
sys.modules.setdefault("isaaclab", isaaclab_module)
sys.modules.setdefault("isaaclab.utils", isaaclab_utils_module)
sys.modules.setdefault("isaaclab.utils.math", isaaclab_math_module)
sys.modules.setdefault("isaaclab.assets", assets_module)
sys.modules.setdefault("isaaclab.managers", managers_module)
sys.modules.setdefault("isaaclab.sensors", sensors_module)
sys.modules.setdefault("whole_body_tracking.tasks.tracking.mdp.commands", commands_module)
sys.modules.setdefault("whole_body_tracking.tasks.tracking.mdp.heading_math", heading_math_module)
sys.modules.setdefault("whole_body_tracking.tasks.tracking.mdp.rewards", rewards_module)

SPEC = importlib.util.spec_from_file_location("terminations", MODULE_PATH)
terminations = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(terminations)

compute_parkour_reach_timeout = terminations._compute_parkour_reach_timeout


def _compute(
    *,
    local_time_steps: list[int],
    robot_anchor_pos_w: list[list[float]],
    end_margin_steps: int = 1,
) -> torch.Tensor:
    return compute_parkour_reach_timeout(
        motion_ids=torch.tensor([0], dtype=torch.long),
        local_time_steps=torch.tensor(local_time_steps, dtype=torch.long),
        frame_end_per_env=torch.tensor([100], dtype=torch.long),
        max_future_step=10,
        motion_anchor_pos_w=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        time_step_end_idx=torch.tensor([2], dtype=torch.long),
        env_origins=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        robot_anchor_pos_w=torch.tensor(robot_anchor_pos_w, dtype=torch.float32),
        distance_threshold=0.30,
        end_margin_steps=end_margin_steps,
    )


def test_parkour_reach_timeout_stays_false_before_motion_end() -> None:
    timeout = _compute(local_time_steps=[88], robot_anchor_pos_w=[[3.0, 0.0, 0.0]])

    assert timeout.tolist() == [False]


def test_parkour_reach_timeout_stays_false_when_far_from_target() -> None:
    timeout = _compute(local_time_steps=[89], robot_anchor_pos_w=[[3.5, 0.0, 0.0]])

    assert timeout.tolist() == [False]


def test_parkour_reach_timeout_triggers_near_motion_end_and_target() -> None:
    timeout = _compute(local_time_steps=[89], robot_anchor_pos_w=[[3.2, 0.0, 0.0]])

    assert timeout.tolist() == [True]


def test_parkour_reach_timeout_margin_covers_last_executable_step() -> None:
    without_margin = _compute(local_time_steps=[89], robot_anchor_pos_w=[[3.0, 0.0, 0.0]], end_margin_steps=0)
    with_margin = _compute(local_time_steps=[89], robot_anchor_pos_w=[[3.0, 0.0, 0.0]], end_margin_steps=1)

    assert without_margin.tolist() == [False]
    assert with_margin.tolist() == [True]
