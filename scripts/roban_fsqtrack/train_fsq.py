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

# If launched via torchrun, each process must use cuda:{LOCAL_RANK}. AppLauncher defaults to
# --device cuda:0, which would otherwise apply to every rank and break custom_rsl_rl's check.
_world_size = int(os.environ.get("WORLD_SIZE", "1"))
_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
if _world_size > 1:
    _expected_dev = f"cuda:{_local_rank}"
    if args_cli.device is not None and str(args_cli.device) != _expected_dev:
        print(
            f"[WARN] torchrun (WORLD_SIZE={_world_size}): overriding --device {args_cli.device!r} "
            f"with {_expected_dev!r} for LOCAL_RANK={_local_rank} (required by RSL multi-GPU)."
        )
    args_cli.device = _expected_dev
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
    # RSL runner checks device == f"cuda:{LOCAL_RANK}" when WORLD_SIZE>1 (see custom OnPolicyRunner).
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    # Distributed training: shard motion bins per-rank so each process doesn't load the full dataset.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
        motion_cfg = env_cfg.commands.motion
        if hasattr(motion_cfg, "distributed_world_size"):
            motion_cfg.distributed_world_size = world_size
        if hasattr(motion_cfg, "distributed_rank"):
            motion_cfg.distributed_rank = rank
        if hasattr(motion_cfg, "distributed_data_split"):
            motion_cfg.distributed_data_split = world_size > 1

    # URDF import (Roban) writes a generated USD under a temp directory. The default directory name
    # uses second-resolution time + random.randint(0, 9999), so several torchrun ranks starting
    # together can collide, corrupt the USD, and then activate_contact_sensors sees no rigid bodies.
    if world_size > 1 and hasattr(env_cfg, "scene"):
        import tempfile

        _robot = getattr(env_cfg.scene, "robot", None)
        _spawn = getattr(_robot, "spawn", None) if _robot is not None else None
        if _spawn is not None and getattr(_spawn, "usd_dir", None) is None:
            _lr = int(os.environ.get("LOCAL_RANK", "0"))
            _usd_dir = os.path.join(
                tempfile.gettempdir(),
                "IsaacLab",
                f"urdf_conv_ws{world_size}_rank{_lr}_pid{os.getpid()}",
            )
            env_cfg.scene.robot = _robot.replace(spawn=_spawn.replace(usd_dir=_usd_dir))
            print(f"[INFO] Multi-GPU: per-rank URDF USD output (avoid import races): {_usd_dir}")

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
