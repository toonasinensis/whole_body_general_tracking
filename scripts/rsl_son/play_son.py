"""Script to play a trained SON (multi-motion) RL agent checkpoint."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import pathlib
import sys

import torch
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a SON RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record video during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the SON task (Gym ID).")
parser.add_argument(
    "--motion_file",
    type=str,
    default=None,
    help="Path to a single NPZ motion file for play (arbitrary length). Overrides --motion_path when set.",
)
parser.add_argument(
    "--motion_path",
    type=str,
    default=None,
    help=(
        "Path to motion data: "
        "for MultiMotionCommand use a dataset directory (with info.yaml); "
        "overrides wandb artifact when set."
    ),
)
parser.add_argument("--model_path", type=str, default=None, help="Direct path to model checkpoint file (.pt)")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import tempfile

import gymnasium as gym

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

import whole_body_tracking.tasks  # noqa: F401
import whole_body_tracking.tasks.son.config.kuavo_s52  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with trained SON RSL-RL agent."""
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path
        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        files = [f.name for f in wandb_run.files() if "model" in f.name]
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)
        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        motion_path_to_set, _ = _motion_path_for_play(args_cli.motion_file, args_cli.motion_path)
        if motion_path_to_set is not None:
            print(f"[INFO]: Using motion from CLI: {motion_path_to_set}")
            _set_motion_path(env_cfg, motion_path_to_set)
        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is not None and motion_path_to_set is None:
            _set_motion_path(env_cfg, str(pathlib.Path(art.download())))
        elif art is None and motion_path_to_set is None:
            print("[WARN] No motion artifact found in the run.")
    else:
        motion_path_to_set, _ = _motion_path_for_play(args_cli.motion_file, args_cli.motion_path)
        if motion_path_to_set is not None:
            _set_motion_path(env_cfg, motion_path_to_set)

        if args_cli.model_path is not None:
            resume_path = args_cli.model_path
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        else:
            print(f"[INFO] Loading experiment from directory: {log_root_path}")
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # For play mode: disable episode timeout and external disturbances so the policy
    # can run continuously without scripted perturbations.
    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
        # Disable time-based episode termination; state-based terms (e.g., falls) remain active.
        env_cfg.terminations.time_out.time_out = False

    if hasattr(env_cfg, "events"):
        # Effectively disable push_robot by using an extremely large interval.
        if hasattr(env_cfg.events, "push_robot"):
            env_cfg.events.push_robot.interval_range_s = (1.0e9, 1.0e9)
        # Disable stochastic external force/torque by setting probability to zero.
        if hasattr(env_cfg.events, "base_external_force_torque"):
            if isinstance(env_cfg.events.base_external_force_torque.params, dict):
                env_cfg.events.base_external_force_torque.params["probability"] = 0.0

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    log_dir = os.path.dirname(resume_path)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)

    obs, _ = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

    env.close()


def _set_motion_path(env_cfg, path: str):
    """Set motion data path on env_cfg.commands.motion (single file or dataset dir)."""
    motion_cfg = env_cfg.commands.motion
    if hasattr(motion_cfg, "motion_file"):
        motion_cfg.motion_file = path
    elif hasattr(motion_cfg, "dataset_dir"):
        # MultiMotionCommand (SON): single dataset directory
        motion_cfg.dataset_dir = path
    elif hasattr(motion_cfg, "dataset_dirs"):
        motion_cfg.dataset_dirs = [path]
    else:
        raise ValueError("commands.motion has neither 'motion_file', 'dataset_dir', nor 'dataset_dirs'.")


def _motion_path_for_play(motion_file: str | None, motion_path: str | None) -> tuple[str | None, str | None]:
    """Resolve motion input for play: single NPZ file or dataset directory.

    If motion_file is set, creates a temporary dataset directory containing that single
    NPZ (with a minimal info.yaml) so MultiMotionCommand can load it. Returns
    (path_to_set, temp_dir_to_cleanup) where temp_dir_to_cleanup is None when using
    motion_path or when the caller should not remove the temp dir.

    Returns:
        (path, temp_dir): path to pass to _set_motion_path; temp_dir is not None only
        when a temp dir was created (caller may optionally clean it up).
    """
    if motion_file is not None:
        path = os.path.abspath(motion_file)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Motion file not found: {path}")
        temp_dir = tempfile.mkdtemp(prefix="son_play_motion_")
        # Motion_Dataset expects info.yaml with dataset name and split -> motion_name: quantity
        # and looks for <temp_dir>/<robot_name>/<motion_name>.npz or <temp_dir>/<motion_name>.npz
        # robot_name is "s52"; if s52 subdir doesn't exist it uses temp_dir. Use flat layout.
        info_yaml = os.path.join(temp_dir, "info.yaml")
        with open(info_yaml, "w") as f:
            f.write("dataset: play\n")
            f.write("train:\n")
            f.write("  motion: 1\n")
        # Symlink the user's NPZ as motion.npz so the single clip is named "motion"
        link_path = os.path.join(temp_dir, "motion.npz")
        try:
            os.symlink(path, link_path)
        except OSError:
            import shutil
            shutil.copy2(path, link_path)
        return temp_dir, temp_dir
    if motion_path is not None:
        return os.path.abspath(motion_path), None
    return None, None


if __name__ == "__main__":
    main()
    simulation_app.close()
