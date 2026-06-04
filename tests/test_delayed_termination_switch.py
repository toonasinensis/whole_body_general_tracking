from __future__ import annotations

import importlib.util
import sys
import torch
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MDP_PATH = ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "tasks" / "tracking" / "mdp"


class _TerminationManagerStub:
    def reset(self, env_ids=None):
        return {}

    def compute(self):
        self._terminated_buf[:] = self._next_terminated
        self._truncated_buf[:] = self._next_truncated
        return self._truncated_buf | self._terminated_buf


def _install_import_stubs() -> None:
    isaaclab = types.ModuleType("isaaclab")
    managers = types.ModuleType("isaaclab.managers")
    sensors = types.ModuleType("isaaclab.sensors")
    utils = types.ModuleType("isaaclab.utils")
    math = types.ModuleType("isaaclab.utils.math")
    assets = types.ModuleType("isaaclab.assets")

    managers.SceneEntityCfg = type("SceneEntityCfg", (), {})
    managers.TerminationManager = _TerminationManagerStub
    sensors.ContactSensor = type("ContactSensor", (), {})
    math.quat_error_magnitude = lambda a, b: torch.zeros(a.shape[:-1], device=a.device)
    math.quat_apply_inverse = lambda quat, vec: vec.expand(quat.shape[0], -1)
    assets.Articulation = type("Articulation", (), {})
    assets.RigidObject = type("RigidObject", (), {})

    sys.modules.setdefault("isaaclab", isaaclab)
    sys.modules.setdefault("isaaclab.managers", managers)
    sys.modules.setdefault("isaaclab.sensors", sensors)
    sys.modules.setdefault("isaaclab.utils", utils)
    sys.modules.setdefault("isaaclab.utils.math", math)
    sys.modules.setdefault("isaaclab.assets", assets)

    for name in (
        "whole_body_tracking",
        "whole_body_tracking.tasks",
        "whole_body_tracking.tasks.tracking",
        "whole_body_tracking.tasks.tracking.mdp",
    ):
        module = sys.modules.setdefault(name, types.ModuleType(name))
        module.__path__ = []

    commands = types.ModuleType("whole_body_tracking.tasks.tracking.mdp.commands")
    commands.MotionCommand = type("MotionCommand", (), {})
    sys.modules.setdefault("whole_body_tracking.tasks.tracking.mdp.commands", commands)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_install_import_stubs()
rewards = _load_module("whole_body_tracking.tasks.tracking.mdp.rewards", MDP_PATH / "rewards.py")
terminations = _load_module("whole_body_tracking.tasks.tracking.mdp.terminations", MDP_PATH / "terminations.py")


class _CommandManagerStub:
    def __init__(self, command):
        self.command = command

    def get_term(self, name: str):
        assert name == "motion"
        return self.command


class _EnvStub:
    def __init__(self, command, termination_manager):
        self.command_manager = _CommandManagerStub(command)
        self.termination_manager = termination_manager


class _ManagerWithMasks:
    def __init__(self, *, delayed_env_mask=None, delayed_active_mask=None):
        if delayed_env_mask is not None:
            self.delayed_termination_env_mask = torch.tensor(delayed_env_mask, dtype=torch.bool)
        if delayed_active_mask is not None:
            self.delayed_termination_active_mask = torch.tensor(delayed_active_mask, dtype=torch.bool)


def _make_motion_command(num_envs: int = 4):
    command = types.SimpleNamespace()
    command.cfg = types.SimpleNamespace(body_names=["left_foot", "right_foot"])

    command.body_pos_relative_w = torch.zeros(num_envs, 2, 3)
    command.robot_body_pos_w = torch.zeros(num_envs, 2, 3)
    command.body_pos_relative_w[:, 0, 2] = 1.0

    command.anchor_pos_w = torch.zeros(num_envs, 3)
    command.robot_anchor_pos_w = torch.zeros(num_envs, 3)
    command.anchor_quat_w = torch.zeros(num_envs, 4)
    command.robot_anchor_quat_w = torch.zeros(num_envs, 4)
    return command


def test_delayed_manager_active_mask_only_tracks_suppressed_termination_window() -> None:
    base = _TerminationManagerStub()
    base._terminated_buf = torch.zeros(3, dtype=torch.bool)
    base._truncated_buf = torch.zeros(3, dtype=torch.bool)
    base._next_terminated = torch.tensor([True, False, True])
    base._next_truncated = torch.zeros(3, dtype=torch.bool)

    manager = terminations.DelayedTerminationManager(
        base=base,
        delay_env_mask=torch.tensor([True, True, False]),
        max_delay_steps=3,
    )

    dones = manager.compute()

    assert dones.tolist() == [False, False, True]
    assert manager.delayed_termination_active_mask.tolist() == [True, False, False]
    assert manager._delay_counters.tolist() == [1, 0, 0]

    manager._next_terminated = torch.zeros(3, dtype=torch.bool)
    dones = manager.compute()

    assert dones.tolist() == [False, False, False]
    assert manager.delayed_termination_active_mask.tolist() == [False, False, False]
    assert manager._delay_counters.tolist() == [0, 0, 0]


def test_ee_body_termination_is_disabled_only_for_delayed_env_subset() -> None:
    command = _make_motion_command()
    manager = _ManagerWithMasks(delayed_env_mask=[True, False, True, False])
    env = _EnvStub(command, manager)

    terminated = terminations.bad_motion_body_pos_z_only(
        env,
        command_name="motion",
        threshold=0.4,
        body_names=["left_foot"],
        disable_on_delayed_termination_envs=True,
    )

    assert terminated.tolist() == [False, True, False, True]


def test_ee_body_termination_unchanged_when_delayed_subset_masking_is_off() -> None:
    command = _make_motion_command()
    manager = _ManagerWithMasks(delayed_env_mask=[True, False, True, False])
    env = _EnvStub(command, manager)

    terminated = terminations.bad_motion_body_pos_z_only(
        env,
        command_name="motion",
        threshold=0.4,
        body_names=["left_foot"],
    )

    assert terminated.tolist() == [True, True, True, True]


def test_full_tracking_reward_is_zeroed_only_during_delayed_active_window() -> None:
    command = _make_motion_command()
    manager = _ManagerWithMasks(delayed_active_mask=[True, False, True, False])
    env = _EnvStub(command, manager)

    reward = rewards.motion_global_anchor_position_error_exp(
        env,
        command_name="motion",
        std=0.3,
        disable_on_delayed_termination=True,
    )

    assert reward.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_root_z_reward_stays_enabled_during_delayed_active_window() -> None:
    command = _make_motion_command()
    manager = _ManagerWithMasks(delayed_active_mask=[True, False, True, False])
    env = _EnvStub(command, manager)

    reward = rewards.motion_global_anchor_position_z_error_exp(env, command_name="motion", std=0.3)

    assert reward.tolist() == [1.0, 1.0, 1.0, 1.0]
