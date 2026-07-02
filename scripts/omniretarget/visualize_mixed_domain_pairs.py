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

parser = argparse.ArgumentParser(description="Visualize mixed flat/mesh motion domains.")
parser.add_argument("--task", type=str, default="TerrainPairMixed-G1")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--pair_limit", type=int, default=8)
parser.add_argument("--pairs_jsonl", type=str, default="data/omniretarget/g1_terrain/pairs_z_scale_1.0.jsonl")
parser.add_argument("--flat_dataset_txt", type=str, default="data/tracking_npz_data/lafan_named.txt")
parser.add_argument("--flat_env_ratio", type=float, default=0.25)
parser.add_argument("--domain_separator_cell_count", type=int, default=32)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--print_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, _agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    if hasattr(env_cfg, "pairs_jsonl"):
        env_cfg.pairs_jsonl = args_cli.pairs_jsonl
    if hasattr(env_cfg, "pair_limit"):
        env_cfg.pair_limit = args_cli.pair_limit
    if hasattr(env_cfg, "flat_dataset_txt"):
        env_cfg.flat_dataset_txt = args_cli.flat_dataset_txt
    if hasattr(env_cfg, "flat_env_ratio"):
        env_cfg.flat_env_ratio = args_cli.flat_env_ratio
    if hasattr(env_cfg, "domain_separator_cell_count"):
        env_cfg.domain_separator_cell_count = args_cli.domain_separator_cell_count
    if hasattr(env_cfg, "configure_domains"):
        env_cfg.configure_domains()

    env_cfg.episode_length_s = 9999.0
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    base_env = env.unwrapped
    obs, _ = env.reset()
    del obs

    motion_cmd = base_env.command_manager.get_term("motion")
    rows = motion_cmd.get_debug_mapping_rows(args_cli.print_envs)
    print("[visualize_mixed_domain_pairs] env/domain/terrain/motion:")
    for row in rows:
        print(
            f"  env={row['env']:03d} domain={row['domain']} pair={row['pair_id']} "
            f"terrain={row['terrain']} motion={row['motion']}"
        )

    action_dim = sum(base_env.action_manager.action_term_dim)
    actions = torch.zeros((base_env.num_envs, action_dim), device=base_env.device)
    for _ in range(args_cli.steps):
        env.step(actions)
        if not simulation_app.is_running():
            break
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
