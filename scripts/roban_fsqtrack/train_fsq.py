from __future__ import annotations
import argparse
import os
import sys

# NOTE ##############################################################
# Use project's custom_rsl_rl (ActorCriticFSQVAE, FSQVAE_PPO, etc.) #
# instead of pip-installed rsl_rl.                                  ###############
# this will set all the rsl_rl imports to use the rsl_rl in custom_rsl_rl package #
# NOTE ############################################################################
_script_dir = os.path.dirname(os.path.abspath(__file__))
_custom_rsl_rl = os.path.abspath(os.path.join(_script_dir, "..", "..", "third_party", "custom_rsl_rl"))
if os.path.isdir(_custom_rsl_rl) and _custom_rsl_rl not in sys.path: sys.path.insert(0, _custom_rsl_rl)

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train FSQ-Track (FSQ-VAE) RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Fsqtrack-Flat-Roban-v0",
    help="Name of the FSQ-Track task (Gym ID), e.g. Fsqtrack-Flat-Roban-v0.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--motion_path", type=str, default=None, help="Path to local motion data (overrides registry).")
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wandb registry (if not using --motion_path).")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video: args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

### launch isaac sim ###
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from datetime import datetime
from isaaclab.envs import (
    DirectMARLEnv,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
# Register Roban FSQ-Track env (uses fsqtrack_env_cfg_roban + config/roban/flat_env_cfg).
import whole_body_tracking.tasks.fsqtrack.config.roban_s22  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    # Ensure Weights & Biases has the required config key.
    # custom_rsl_rl's WandbSummaryWriter expects cfg["wandb_project"] to exist when logger=="wandb".
    if getattr(agent_cfg, "logger", None) is not None and str(agent_cfg.logger).lower() == "wandb":
        if getattr(agent_cfg, "wandb_project", None) in (None, ""):
            # Prefer CLI project name; fall back to env var; otherwise a safe default.
            agent_cfg.wandb_project = (
                args_cli.log_project_name
                or os.environ.get("WANDB_PROJECT")
                or "fsqtrack_roban_s22"
            )
    agent_cfg.max_iterations = (args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    ####################
    # load motion data #
    ####################
    if args_cli.motion_path is not None:
        registry_name = "none"
        motion_path = os.path.abspath(args_cli.motion_path)
        # Support both single-motion (motion_file) and multi-motion (dataset_dir) commands.
        motion_cfg = env_cfg.commands.motion
        if hasattr(motion_cfg, "motion_file"):
            # Single NPZ file
            motion_cfg.motion_file = motion_path
        elif hasattr(motion_cfg, "dataset_dir"):
            # MultiMotionCommand (SON): treat path as dataset directory
            motion_cfg.dataset_dir = motion_path
        else:
            raise ValueError(
                "commands.motion has neither 'motion_file' nor 'dataset_dir'. "
                "Please check your command configuration."
            )
    else:
        if args_cli.registry_name is None:
            raise ValueError("Please provide either --motion_path or --registry_name")
        registry_name = args_cli.registry_name
        if ":" not in registry_name:
            registry_name += ":latest"
        import pathlib
        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        artifact_root = pathlib.Path(artifact.download())
        motion_cfg = env_cfg.commands.motion
        if hasattr(motion_cfg, "motion_file"):
            motion_cfg.motion_file = str(artifact_root / "motion.npz")
        elif hasattr(motion_cfg, "dataset_dir"):
            motion_cfg.dataset_dir = str(artifact_root)
        else:
            raise ValueError(
                "commands.motion has neither 'motion_file' nor 'dataset_dir'. "
                "Please check your command configuration."
            )

    ############
    # set logs #
    ############
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name: log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    
    ############################
    # create isaac environment #
    ############################
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    ### video recording ###
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

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    ### set environment and runner ###
    env = RslRlVecEnvWrapper(env)
    print(f"[INFO] Agent configuration: {agent_cfg.to_dict()}")

    runner_cfg_dict = agent_cfg.to_dict()
    # Mirror the enforced project into the dict so the runner logger sees it.
    if getattr(agent_cfg, "logger", None) is not None and str(agent_cfg.logger).lower() == "wandb":
        runner_cfg_dict.setdefault("wandb_project", getattr(agent_cfg, "wandb_project", None))

    runner = OnPolicyRunner(
        env, 
        runner_cfg_dict,
        log_dir=log_dir, 
        device=agent_cfg.device, 
        registry_name=registry_name
    )
    runner.add_git_repo_to_log(__file__)
    
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
    simulation_app.close()
