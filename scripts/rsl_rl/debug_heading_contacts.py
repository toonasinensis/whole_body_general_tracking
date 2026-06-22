#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

WBT_ROOT = Path(__file__).resolve().parents[2]
RSL_RL_ROOT = WBT_ROOT.parent / "rsl_rl"
if RSL_RL_ROOT.is_dir():
    sys.path.insert(0, str(RSL_RL_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug first-step contacts for HeadingWalkAMP-G1.")
parser.add_argument("--task", type=str, default="HeadingWalkAMP-G1")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--action_mode", choices=["zero", "random"], default="zero")
parser.add_argument("--random_scale", type=float, default=1.0)
parser.add_argument("--threshold", type=float, default=5.0)
parser.add_argument("--topk", type=int, default=8)
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

ALLOWED_CONTACT_BODIES = {
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
}


def _illegal_body_ids(body_names: list[str]) -> list[int]:
    pattern = re.compile(
        r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)" r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
    )
    return [idx for idx, name in enumerate(body_names) if pattern.match(name)]


def _print_contact_summary(base_env, step: int, threshold: float, topk: int) -> None:
    robot = base_env.scene["robot"]
    sensor = base_env.scene.sensors["contact_forces"]
    forces = sensor.data.net_forces_w_history
    if forces is None:
        print("[debug_heading_contacts] contact_forces has no net_forces_w_history")
        return

    force_norm = torch.norm(forces, dim=-1).amax(dim=1)
    illegal_ids = torch.tensor(_illegal_body_ids(robot.body_names), device=force_norm.device, dtype=torch.long)
    illegal_force = force_norm[:, illegal_ids]
    illegal_body_z = robot.data.body_pos_w[:, illegal_ids, 2]
    height_filtered = (illegal_force > threshold) & (illegal_body_z < 0.35)
    per_env_max, per_env_arg = illegal_force.max(dim=1)
    illegal_envs = torch.nonzero(per_env_max > threshold, as_tuple=False).flatten()
    filtered_envs = torch.nonzero(height_filtered.any(dim=1), as_tuple=False).flatten()

    root_z = robot.data.root_pos_w[:, 2]
    gravity_xy = torch.norm(robot.data.projected_gravity_b[:, :2], dim=1)
    print(
        f"[step {step:03d}] illegal_envs={illegal_envs.numel()}/{base_env.num_envs} "
        f"height_filtered={filtered_envs.numel()}/{base_env.num_envs} "
        f"max_illegal_force={per_env_max.max().item():.3f} "
        f"root_z_mean={root_z.mean().item():.3f} root_z_min={root_z.min().item():.3f} "
        f"tilt_xy_mean={gravity_xy.mean().item():.3f}"
    )

    show_envs = illegal_envs[: min(topk, illegal_envs.numel())]
    if show_envs.numel() == 0:
        show_envs = torch.arange(min(topk, base_env.num_envs), device=force_norm.device)

    for env_id_t in show_envs:
        env_id = int(env_id_t.item())
        body_local = int(per_env_arg[env_id].item())
        body_id = int(illegal_ids[body_local].item())
        body_name = robot.body_names[body_id]
        body_z = robot.data.body_pos_w[env_id, body_id, 2].item()
        print(
            f"  env={env_id:03d} body={body_name:<28s} "
            f"force={per_env_max[env_id].item():8.3f} body_z={body_z:7.3f} root_z={root_z[env_id].item():7.3f}"
        )

    body_max = force_norm.max(dim=0).values
    top_vals, top_ids = torch.topk(body_max, k=min(topk, body_max.numel()))
    print("  top bodies:")
    for value, body_id_t in zip(top_vals, top_ids):
        body_id = int(body_id_t.item())
        marker = "allowed" if robot.body_names[body_id] in ALLOWED_CONTACT_BODIES else "illegal"
        print(f"    {robot.body_names[body_id]:<28s} {value.item():8.3f} {marker}")


def main() -> None:
    torch.manual_seed(args_cli.seed)

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    base_env = env.unwrapped

    obs, _ = env.reset()
    del obs
    action_dim = sum(base_env.action_manager.action_term_dim)
    print(
        f"[debug_heading_contacts] task={args_cli.task} num_envs={base_env.num_envs} "
        f"action_dim={action_dim} action_mode={args_cli.action_mode}"
    )
    print("[debug_heading_contacts] bodies:")
    print("  " + ", ".join(base_env.scene["robot"].body_names))

    _print_contact_summary(base_env, step=0, threshold=args_cli.threshold, topk=args_cli.topk)

    for step in range(1, args_cli.steps + 1):
        if args_cli.action_mode == "zero":
            actions = torch.zeros((base_env.num_envs, action_dim), device=base_env.device)
        else:
            actions = args_cli.random_scale * torch.randn((base_env.num_envs, action_dim), device=base_env.device)
        env.step(actions)
        _print_contact_summary(base_env, step=step, threshold=args_cli.threshold, topk=args_cli.topk)
        if not simulation_app.is_running():
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
