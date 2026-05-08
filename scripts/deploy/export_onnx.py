"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
# key arguments
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--resume_path", type=str, default=None, help="Path to the model file.")
parser.add_argument(
    "--onnx_dir",
    type=str,
    default=None,
    help="Output directory for exported ONNX (default: <checkpoint_dir>/exported).",
)
parser.add_argument(
    "--onnx_filename",
    type=str,
    default="policy.onnx",
    help="ONNX filename (default: policy.onnx).",
)
# used to construct the environment
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument(
    "--motion_file_txt",
    type=str,
    default=None,
    help="Optional txt file listing relative .npz paths under --motion_file (maps to commands.motion.dataset_txt).",
)


# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from whole_body_tracking.utils.exporter import export_motion_policy_as_onnx


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # create isaac environment
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    resume_path = args_cli.resume_path
    if resume_path is None:
        raise ValueError("Missing required argument: --resume_path (path to checkpoint .pt)")
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if args_cli.motion_file is None or args_cli.motion_file_txt is None:
        raise ValueError("Missing required argument: --motion_file or --motion_file_txt")
    # used as placeholder
    env_cfg.commands.motion.motion_file = args_cli.motion_file
    env_cfg.commands.motion.dataset_txt = args_cli.motion_file_txt

    # export policy to onnx
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    ppo_runner.eval_mode()
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    # TODO add metadata to onnx file
    env.close()


if __name__ == "__main__":
    # run the main function
    main()  # pyright: ignore[reportCallIssue]
    # close sim app
    simulation_app.close()
