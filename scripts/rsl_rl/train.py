# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--registry_name", type=str, default="test1", help="The name of the wand registry.")
parser.add_argument("--resume_path", type=str, default=None, help="Path to the model file.")
parser.add_argument("--resume_primitive_path", type=str, default=None, help="Path to the frozen primitive model file.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--dataset_txt", type=str, default=None, help="Path to the motion dataset_txt.")
parser.add_argument("--smpl_file_path", type=str, default=None, help="Path to the SMPL file.")
parser.add_argument("--pairs_jsonl", type=str, default=None, help="Path to paired terrain-motion manifest.")
parser.add_argument(
    "--flat_dataset_txt", type=str, default=None, help="Path to flat motion dataset_txt for mixed tasks."
)
parser.add_argument("--flat_env_ratio", type=float, default=None, help="Flat-domain env ratio for mixed tasks.")
parser.add_argument("--flat_wbc_env_ratio", type=float, default=None, help="Flat WBC env ratio for mixed tasks.")
parser.add_argument(
    "--flat_velocity_env_ratio",
    type=float,
    default=None,
    help="Flat velocity-command env ratio for mixed tasks.",
)
parser.add_argument(
    "--velocity_terrain_env_ratio",
    type=float,
    default=None,
    help="Procedural velocity-terrain env ratio for mixed tasks.",
)
parser.add_argument(
    "--velocity_terrain_cell_count",
    type=int,
    default=None,
    help="Number of procedural velocity-terrain cells for mixed tasks.",
)
parser.add_argument(
    "--velocity_terrain_profile",
    type=str,
    default=None,
    help="Procedural velocity-terrain profile for mixed tasks.",
)
parser.add_argument("--mesh_env_ratio", type=float, default=None, help="Mesh WBC env ratio for mixed tasks.")
parser.add_argument(
    "--domain_separator_cell_count",
    type=int,
    default=None,
    help="Ground-only terrain cells inserted between mixed domains.",
)
parser.add_argument(
    "--terrain_border_width",
    type=float,
    default=None,
    help="Flat border width around the whole mixed terrain grid.",
)

parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.pairs_jsonl is not None:
    os.environ["WBT_PAIRS_JSONL"] = args_cli.pairs_jsonl
if args_cli.flat_dataset_txt is not None:
    os.environ["WBT_FLAT_DATASET_TXT"] = args_cli.flat_dataset_txt
if args_cli.flat_env_ratio is not None:
    os.environ["WBT_FLAT_ENV_RATIO"] = str(args_cli.flat_env_ratio)
if args_cli.flat_wbc_env_ratio is not None:
    os.environ["WBT_FLAT_WBC_ENV_RATIO"] = str(args_cli.flat_wbc_env_ratio)
if args_cli.flat_velocity_env_ratio is not None:
    os.environ["WBT_FLAT_VELOCITY_ENV_RATIO"] = str(args_cli.flat_velocity_env_ratio)
if args_cli.velocity_terrain_env_ratio is not None:
    os.environ["WBT_VELOCITY_TERRAIN_ENV_RATIO"] = str(args_cli.velocity_terrain_env_ratio)
if args_cli.velocity_terrain_cell_count is not None:
    os.environ["WBT_VELOCITY_TERRAIN_CELL_COUNT"] = str(args_cli.velocity_terrain_cell_count)
if args_cli.velocity_terrain_profile is not None:
    os.environ["WBT_VELOCITY_TERRAIN_PROFILE"] = str(args_cli.velocity_terrain_profile)
if args_cli.mesh_env_ratio is not None:
    os.environ["WBT_MESH_ENV_RATIO"] = str(args_cli.mesh_env_ratio)
if args_cli.domain_separator_cell_count is not None:
    os.environ["WBT_DOMAIN_SEPARATOR_CELL_COUNT"] = str(args_cli.domain_separator_cell_count)
if args_cli.terrain_border_width is not None:
    os.environ["WBT_TERRAIN_BORDER_WIDTH"] = str(args_cli.terrain_border_width)

# For torchrun multi-process launch, bind each process to its own GPU early
# so AppLauncher and downstream tensors share the same device.
if args_cli.distributed:
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    args_cli.device = f"cuda:{local_rank}"

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    motion_cfg = getattr(getattr(env_cfg, "commands", None), "motion", None)
    if args_cli.resume_primitive_path is not None:
        if not hasattr(agent_cfg.algorithm, "primitive_checkpoint_path"):
            raise ValueError("--resume_primitive_path is only supported by algorithms with primitive_checkpoint_path.")
        print(f"[INFO]: Using primitive checkpoint from CLI: {args_cli.resume_primitive_path}")
        agent_cfg.algorithm.primitive_checkpoint_path = args_cli.resume_primitive_path
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    if args_cli.distributed:
        env_cfg.sim.device = args_cli.device

        agent_cfg.device = args_cli.device
        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed
    else:
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed

    # load the motion file from the wandb registry
    registry_name = args_cli.registry_name
    if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
        registry_name += ":latest"

    # import wandb
    # api = wandb.Api()
    # artifact = api.artifact(registry_name)
    if args_cli.motion_file:
        if motion_cfg is None:
            raise ValueError("--motion_file is only supported by env configs with a motion command.")
        motion_cfg.motion_file = args_cli.motion_file
    if args_cli.dataset_txt:
        if motion_cfg is None:
            raise ValueError("--dataset_txt is only supported by env configs with a motion command.")
        print(f"[INFO]: Using motion file filter from CLI: {args_cli.dataset_txt}")
        motion_cfg.dataset_txt = args_cli.dataset_txt
        print(f"[INFO]: Overriding motion file filter in the environment config with: {motion_cfg.dataset_txt}")
    if args_cli.pairs_jsonl:
        if not hasattr(env_cfg, "pairs_jsonl"):
            raise ValueError("--pairs_jsonl is only supported by env configs with a pairs_jsonl field.")
        print(f"[INFO]: Using paired terrain manifest from CLI: {args_cli.pairs_jsonl}")
        env_cfg.pairs_jsonl = args_cli.pairs_jsonl
        if hasattr(env_cfg, "configure_pairs"):
            env_cfg.configure_pairs()
        elif motion_cfg is not None and hasattr(motion_cfg, "pairs_jsonl"):
            motion_cfg.pairs_jsonl = args_cli.pairs_jsonl
    if args_cli.flat_dataset_txt:
        if not hasattr(env_cfg, "flat_dataset_txt"):
            raise ValueError("--flat_dataset_txt is only supported by mixed terrain env configs.")
        print(f"[INFO]: Using flat dataset from CLI: {args_cli.flat_dataset_txt}")
        env_cfg.flat_dataset_txt = args_cli.flat_dataset_txt
    if args_cli.flat_env_ratio is not None:
        if not hasattr(env_cfg, "flat_env_ratio"):
            if not hasattr(env_cfg, "flat_wbc_env_ratio") or not hasattr(env_cfg, "flat_velocity_env_ratio"):
                raise ValueError("--flat_env_ratio is only supported by mixed terrain env configs.")
        print(f"[INFO]: Using flat env ratio from CLI: {args_cli.flat_env_ratio}")
        if hasattr(env_cfg, "flat_env_ratio"):
            env_cfg.flat_env_ratio = args_cli.flat_env_ratio
        else:
            env_cfg.flat_wbc_env_ratio = args_cli.flat_env_ratio * 0.5
            env_cfg.flat_velocity_env_ratio = args_cli.flat_env_ratio * 0.5
            env_cfg.mesh_env_ratio = max(0.0, 1.0 - args_cli.flat_env_ratio)
    for arg_name, cfg_name in (
        ("flat_wbc_env_ratio", "flat_wbc_env_ratio"),
        ("flat_velocity_env_ratio", "flat_velocity_env_ratio"),
        ("mesh_env_ratio", "mesh_env_ratio"),
    ):
        value = getattr(args_cli, arg_name)
        if value is None:
            continue
        if not hasattr(env_cfg, cfg_name):
            raise ValueError(f"--{arg_name} is only supported by mixed terrain env configs.")
        print(f"[INFO]: Using {cfg_name} from CLI: {value}")
        setattr(env_cfg, cfg_name, value)
    if args_cli.domain_separator_cell_count is not None:
        if not hasattr(env_cfg, "domain_separator_cell_count"):
            raise ValueError("--domain_separator_cell_count is only supported by mixed terrain env configs.")
        print(f"[INFO]: Using mixed domain separator cells from CLI: {args_cli.domain_separator_cell_count}")
        env_cfg.domain_separator_cell_count = args_cli.domain_separator_cell_count
    if args_cli.terrain_border_width is not None:
        if not hasattr(env_cfg, "terrain_border_width"):
            raise ValueError("--terrain_border_width is only supported by mixed terrain env configs.")
        print(f"[INFO]: Using mixed terrain border width from CLI: {args_cli.terrain_border_width}")
        env_cfg.terrain_border_width = args_cli.terrain_border_width
    if hasattr(env_cfg, "configure_domains"):
        env_cfg.configure_domains()
    if args_cli.distributed and motion_cfg is not None:
        motion_cfg.distributed = True
        motion_cfg.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        motion_cfg.total_rank = int(os.getenv("WORLD_SIZE", "1"))
    if args_cli.smpl_file_path is not None:
        if motion_cfg is None:
            raise ValueError("--smpl_file_path is only supported by env configs with a motion command.")
        print(f"[INFO]: Using SMPL file from CLI: {args_cli.smpl_file_path}")
        motion_cfg.smpl_file_path = args_cli.smpl_file_path
        print(f"[INFO]: Overriding SMPL file in the environment config with: {motion_cfg.smpl_file_path}")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    if motion_cfg is not None:
        motion_cfg.log_run_name = log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=registry_name
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if args_cli.resume:
        # get path to previous checkpoint
        resume_path = args_cli.resume_path
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
    # dump the configuration into log-directory
    # dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    # dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    # dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    # dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
