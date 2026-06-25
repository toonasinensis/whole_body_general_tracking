"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# -----------------------------------------------------------------------------
# Debug-friendly CLI presets
#
# When launching via Cursor's debugger, it's common to start the script without
# any CLI args. To make debugging easy, we inject a default arg list only when:
# - a debugger is attached (sys.gettrace() is not None), and
# - the user didn't pass any CLI args (len(sys.argv) <= 1).
#
# You can override presets without editing code by setting:
#   PLAY_PT_ARGS="--task ... --resume_path ... --num_envs 1"
# -----------------------------------------------------------------------------
def _maybe_inject_debug_cli_args() -> None:
    # Respect real CLI usage (any args provided).
    if len(sys.argv) > 1:
        return
    # Only inject when a debugger is attached.
    if sys.gettrace() is None:
        return

    env_args = os.environ.get("PLAY_PT_ARGS")
    if env_args:
        sys.argv.extend(env_args.split())
        return

    # Fallback defaults for debugger sessions. Adjust these two paths once,
    # or prefer the PLAY_PT_ARGS env var in your debug configuration.
    sys.argv.extend(
        [
            "--task",
            "Tracking-Flat-RobanS22-v0",
            "--num_envs",
            "1",
            "--motion_file",
            "data/roban_motions_quick_test",
            "--resume_path",
            "logs/rsl_rl/roban_flat/2026-04-15_22-34-18/model_3500.pt",
            "--export_onnx",
        ]
    )


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
parser.add_argument("--resume_path", type=str, default=None, help="Path to the model file.")

# onnx export (optional)
parser.add_argument("--export_onnx", action="store_true", default=False, help="Export policy to ONNX and exit.")
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

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

import os

_maybe_inject_debug_cli_args()
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
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.terminations.ee_body_pos = None
    env_cfg.terminations.anchor_ori = None
    env_cfg.terminations.anchor_pos = None

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = args_cli.resume_path
    if resume_path is None:
        raise ValueError("Missing required argument: --resume_path (path to checkpoint .pt)")
    if args_cli.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
        env_cfg.commands.motion.motion_file = args_cli.motion_file
        print(f"[INFO]: Overriding motion file in the environment config with: {env_cfg.commands.motion.motion_file}")
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

    # export policy to onnx (optional)
    if args_cli.export_onnx:
        export_model_dir = args_cli.onnx_dir or os.path.join(os.path.dirname(resume_path), "exported")
        export_motion_policy_as_onnx(
            env.unwrapped,
            ppo_runner.alg.get_policy(), # MLPModel defined in Isaaclab
            path=export_model_dir,
            filename=args_cli.onnx_filename,
        )

        attach_onnx_metadata(env.unwrapped, args_cli.wandb_path or "none", export_model_dir, args_cli.onnx_filename)

        env.close()
        return

    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()  # pyright: ignore[reportCallIssue]
    # close sim app
    simulation_app.close()
