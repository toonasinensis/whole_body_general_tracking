#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

WBT_ROOT = Path(__file__).resolve().parents[2]
RSL_RL_ROOT = WBT_ROOT.parent / "rsl_rl"
if RSL_RL_ROOT.is_dir():
    sys.path.insert(0, str(RSL_RL_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sweep zero-action G1 standing poses for HeadingWalkAMP-G1.")
parser.add_argument("--task", type=str, default="HeadingWalkAMP-G1")
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import whole_body_tracking.tasks  # noqa: F401

CANDIDATES = [
    ("default_z078", 0.78, -0.312, 0.669, -0.363),
    ("default_z072", 0.72, -0.312, 0.669, -0.363),
    ("default_z068", 0.68, -0.312, 0.669, -0.363),
    ("home_z080", 0.80, -0.100, 0.300, -0.200),
    ("home_z074", 0.74, -0.100, 0.300, -0.200),
    ("deep_z074", 0.74, -0.450, 0.900, -0.450),
    ("deep_z070", 0.70, -0.450, 0.900, -0.450),
    ("mid_z074", 0.74, -0.350, 0.750, -0.400),
    ("mid_z070", 0.70, -0.350, 0.750, -0.400),
    ("tall_z082", 0.82, -0.200, 0.450, -0.250),
    ("tall_z076", 0.76, -0.200, 0.450, -0.250),
    ("amp_test_z078", 0.78, -0.312, 0.669, -0.363),
]


def _set_matching_joints(
    joint_pos: torch.Tensor, joint_names: list[str], env_id: int, pattern: str, value: float
) -> None:
    for joint_id, name in enumerate(joint_names):
        if pattern in name:
            joint_pos[env_id, joint_id] = value


def _write_candidate_states(base_env) -> None:
    robot = base_env.scene["robot"]
    num_envs = min(base_env.num_envs, len(CANDIDATES))
    env_ids = torch.arange(num_envs, device=base_env.device, dtype=torch.long)

    root_pose = robot.data.default_root_state[env_ids, :7].clone()
    root_pose[:, :3] = base_env.scene.env_origins[env_ids]
    root_pose[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=base_env.device).repeat(num_envs, 1)

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    for env_id, (_, root_z, hip_pitch, knee, ankle_pitch) in enumerate(CANDIDATES[:num_envs]):
        root_pose[env_id, 2] += root_z
        _set_matching_joints(joint_pos, robot.joint_names, env_id, "hip_pitch_joint", hip_pitch)
        _set_matching_joints(joint_pos, robot.joint_names, env_id, "knee_joint", knee)
        _set_matching_joints(joint_pos, robot.joint_names, env_id, "ankle_pitch_joint", ankle_pitch)
        _set_matching_joints(joint_pos, robot.joint_names, env_id, "hip_roll_joint", 0.0)
        _set_matching_joints(joint_pos, robot.joint_names, env_id, "hip_yaw_joint", 0.0)
        _set_matching_joints(joint_pos, robot.joint_names, env_id, "ankle_roll_joint", 0.0)

    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=base_env.device), env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    action_term = base_env.action_manager.get_term("joint_pos")
    if isinstance(action_term._offset, torch.Tensor):
        action_term._offset[env_ids] = joint_pos
    action_term._raw_actions[env_ids] = 0.0
    action_term._processed_actions[env_ids] = joint_pos
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)


def _print_summary(base_env, step: int) -> None:
    robot = base_env.scene["robot"]
    root_z = robot.data.root_pos_w[:, 2]
    tilt = torch.norm(robot.data.projected_gravity_b[:, :2], dim=1)
    print(f"\n[step {step}]")
    for env_id, candidate in enumerate(CANDIDATES[: base_env.num_envs]):
        name, root_z0, hip, knee, ankle = candidate
        print(
            f"  env={env_id:02d} {name:<14s} init_z={root_z0:.2f} "
            f"hip={hip:+.3f} knee={knee:+.3f} ankle={ankle:+.3f} "
            f"root_z={root_z[env_id].item():.3f} tilt_xy={tilt[env_id].item():.3f}"
        )


def main() -> None:
    torch.manual_seed(args_cli.seed)

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = len(CANDIDATES)
    env_cfg.seed = args_cli.seed
    env_cfg.terminations.time_out = None
    env_cfg.terminations.bad_orientation = None
    env_cfg.terminations.illegal_contact = None
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    base_env = env.unwrapped

    env.reset()
    _write_candidate_states(base_env)
    _print_summary(base_env, 0)

    action_dim = sum(base_env.action_manager.action_term_dim)
    actions = torch.zeros((base_env.num_envs, action_dim), device=base_env.device)
    for step in range(1, args_cli.steps + 1):
        env.step(actions)
        if step in {1, 10, 25, 50, args_cli.steps}:
            _print_summary(base_env, step)
        if not simulation_app.is_running():
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
