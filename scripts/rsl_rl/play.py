"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--dataset_txt", type=str, default=None, help="Path to the motion dataset_txt.")

parser.add_argument("--resume_path", type=str, default=None, help="Path to the model file.")
parser.add_argument(
    "--tb_log_dir",
    type=str,
    default=None,
    help="TensorBoard output directory used when --logger=tensorboard. Defaults to <checkpoint_dir>/tensorboard/play.",
)
parser.add_argument(
    "--tb_log_interval",
    type=int,
    default=1,
    help="Log command metrics to TensorBoard every N env steps when --logger=tensorboard.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = False  # True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

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
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env_cfg.terminations.ee_body_pos = None
    env_cfg.terminations.anchor_ori = None
    env_cfg.terminations.anchor_pos = None
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        pass
    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        # resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        resume_path = args_cli.resume_path

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file
            print(
                f"[INFO]: Overriding motion file in the environment config with: {env_cfg.commands.motion.motion_file}"
            )

        if args_cli.dataset_txt is not None:
            print(f"[INFO]: Using motion file filter from CLI: {args_cli.dataset_txt}")
            env_cfg.commands.motion.dataset_txt = args_cli.dataset_txt
            print(
                "[INFO]: Overriding motion file filter in the environment config with:"
                f" {env_cfg.commands.motion.dataset_txt}"
            )

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    env_cfg.episode_length_s = 9999
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
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

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    base_env = env.unwrapped
    motion_cmd = None
    if hasattr(base_env, "command_manager"):
        try:
            motion_cmd = base_env.command_manager.get_term("motion")
        except Exception:
            motion_cmd = None

    tb_writer = None
    tb_log_interval = max(1, args_cli.tb_log_interval)
    if args_cli.logger == "tensorboard":
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as err:
            raise ImportError("TensorBoard logging requested, but 'tensorboard' is not installed.") from err

        tb_log_dir = args_cli.tb_log_dir or os.path.join(log_dir, "tensorboard", "play")
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"[INFO] TensorBoard logging enabled. log_dir: {tb_log_dir}")
        if motion_cmd is None:
            print("[WARN] Could not find command term 'motion'; command metrics will not be logged.")

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")  # noqa: F841

    # export_motion_policy_as_onnx(
    #     env.unwrapped,
    #     ppo_runner.alg.policy,
    #     normalizer=ppo_runner.obs_normalizer,
    #     path=export_model_dir,
    #     filename="policy.onnx",
    # )
    # attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)
    # reset environment
    # import ipdb; ipdb.set_trace()
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            if isinstance(actions, dict) and "actions" in actions:
                actions = actions["actions"]
            obs, _, _, _ = env.step(actions)

        timestep += 1
        if tb_writer is not None and motion_cmd is not None and timestep % tb_log_interval == 0:
            for key, value in motion_cmd.metrics.items():
                if torch.is_tensor(value):
                    scalar = float(value.mean().item())
                else:
                    scalar = float(value)
                tb_writer.add_scalar(f"command/{key}", scalar, global_step=timestep)

        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
