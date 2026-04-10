"""Script to play a trained FSQ-Track (FSQ-VAE) RL agent checkpoint."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import pathlib
import sys
import tempfile

# Use project's custom_rsl_rl (ActorCriticFSQVAE, FSQVAE_PPO, etc.) instead of pip-installed rsl_rl.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_custom_rsl_rl = os.path.abspath(os.path.join(_script_dir, "..", "..", "third_party", "custom_rsl_rl"))
if os.path.isdir(_custom_rsl_rl) and _custom_rsl_rl not in sys.path:
    sys.path.insert(0, _custom_rsl_rl)

import torch
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play FSQ-Track (FSQ-VAE) RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record video during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Fsqtrack-Flat-Roban-v0",
    help="Name of the FSQ-Track task (Gym ID), e.g. Fsqtrack-Flat-Roban-v0.",
)
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
parser.add_argument(
    "--no_disturbance",
    action="store_true",
    default=True,
    help="Disable external disturbances (push, force/torque) during play. Default: True.",
)
parser.add_argument(
    "--disturbance",
    action="store_true",
    default=False,
    help="Enable external disturbances during play (overrides --no_disturbance).",
)
parser.add_argument("--model_path", type=str, default=None, help="Direct path to model checkpoint file (.pt)")
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Maximum number of simulation steps to run (default: unlimited). Stops play loop after this many steps.",
)
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=None,
    help="Max episode duration in seconds. Overrides env_cfg.episode_length_s. Requires time_out termination enabled.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import sys

import gymnasium as gym

from isaaclab.envs import (
    DirectMARLEnv,
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
import whole_body_tracking.tasks.fsqtrack.config.roban  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with trained FSQ-Track RSL-RL agent."""
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # Disable external disturbances during play unless --disturbance is set
    if args_cli.no_disturbance and not args_cli.disturbance:
        _disable_disturbances(env_cfg)

    # Optionally set max episode length (seconds); enables time_out if needed
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
        if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out.time_out = True

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
        timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break
        if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
            print(f"[INFO] Reached max_steps={args_cli.max_steps}, stopping.")
            break

    env.close()


def _motion_path_for_play(motion_file: str | None, motion_path: str | None) -> tuple[str | None, str | None]:
    """Resolve motion input for play: single NPZ file or dataset directory.

    If motion_file is set, returns its absolute path (file). If motion_path is set,
    returns its absolute path (file or dir). _set_motion_path handles creating a
    temp dataset dir when a single .npz is passed and config expects dataset_dir/dataset_dirs.

    Returns:
        (path, temp_dir): path to pass to _set_motion_path; temp_dir is always None
        (no temp dir created here; _set_motion_path may create one internally).
    """
    if motion_file is not None:
        path = os.path.abspath(motion_file)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Motion file not found: {path}")
        return path, None
    if motion_path is not None:
        return os.path.abspath(motion_path), None
    return None, None


def _set_motion_path(env_cfg, path: str):
    """Set motion data path on env_cfg.commands.motion (single file or dataset dir)."""
    path = os.path.abspath(path)
    motion_cfg = env_cfg.commands.motion
    if hasattr(motion_cfg, "motion_file"):
        if not os.path.isfile(path):
            raise ValueError(f"motion_file config expects a file path, got directory: {path}")
        motion_cfg.motion_file = path
    elif hasattr(motion_cfg, "dataset_dir"):
        # MultiMotionCommand (FSQ): single dataset directory
        if os.path.isfile(path) and path.lower().endswith(".npz"):
            path = _make_single_npz_dataset_dir(path)
        motion_cfg.dataset_dir = path
    elif hasattr(motion_cfg, "dataset_dirs"):
        if os.path.isfile(path) and path.lower().endswith(".npz"):
            path = _make_single_npz_dataset_dir(path)
        motion_cfg.dataset_dirs = [path]
    else:
        raise ValueError("commands.motion has neither 'motion_file', 'dataset_dir', nor 'dataset_dirs'.")


def _make_single_npz_dataset_dir(npz_path: str) -> str:
    """Create a temporary dataset dir with info.yaml and a symlink to the single .npz.

    Motion_Dataset requires each dataset_dir to have info.yaml listing motions and splits.
    For a single .npz we create a temp dir, write a minimal info.yaml, and symlink the npz.
    """
    import yaml

    npz_path = pathlib.Path(npz_path).resolve()
    if not npz_path.is_file() or npz_path.suffix.lower() != ".npz":
        raise FileNotFoundError(f"Not a .npz file: {npz_path}")
    motion_name = npz_path.stem
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="fsqtrack_play_"))
    info = {"dataset": "play_single", "train": {motion_name: 1}}
    with open(tmp_dir / "info.yaml", "w") as f:
        yaml.safe_dump(info, f, default_flow_style=False)
    (tmp_dir / f"{motion_name}.npz").symlink_to(npz_path)
    return str(tmp_dir)


def _disable_disturbances(env_cfg):
    """Remove interval-based disturbance events (push, external force/torque) for clean play."""
    events = getattr(env_cfg, "events", None)
    if events is None:
        return
    for name in ("push_robot", "base_external_force_torque"):
        if hasattr(events, name):
            delattr(events, name)


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
    simulation_app.close()
