"""Export checkpoint to ONNX, then play ONNX in one Isaac Sim session."""

import argparse
import os
import sys

import numpy as np
import onnxruntime as ort
import torch

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


def _maybe_inject_debug_cli_args() -> None:
    if len(sys.argv) > 1:
        return
    if sys.gettrace() is None:
        return

    env_args = os.environ.get("EXPORT_PLAY_ONNX_ARGS")
    if env_args:
        sys.argv.extend(env_args.split())
        return

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
            "--video",
        ]
    )


parser = argparse.ArgumentParser(description="Export checkpoint to ONNX and play it without restarting Isaac Sim.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos while playing ONNX policy.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--env_spacing",
    type=float,
    default=None,
    help="Spacing distance between environments (overrides env_cfg.scene.env_spacing).",
)
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument(
    "--motion_file_txt",
    type=str,
    default=None,
    help="Optional txt file listing relative .npz paths under --motion_file (maps to commands.motion.dataset_txt).",
)
parser.add_argument("--resume_path", type=str, required=True, help="Path to the model checkpoint (.pt).")
parser.add_argument(
    "--onnx_dir",
    type=str,
    default=None,
    help="Output directory for exported ONNX (default: <checkpoint_dir>/exported).",
)
parser.add_argument("--onnx_filename", type=str, default="policy.onnx", help="ONNX filename (default: policy.onnx).")

# append RSL-RL args (also provides optional wandb_path used in metadata)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher args
AppLauncher.add_app_launcher_args(parser)

_maybe_inject_debug_cli_args()
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app exactly once for both export and play
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


def _to_numpy_policy_obs(obs_data) -> np.ndarray:
    policy_obs = obs_data.get("policy", obs_data) if isinstance(obs_data, dict) else obs_data
    if torch.is_tensor(policy_obs):
        return policy_obs.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(policy_obs, dtype=np.float32)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.env_spacing is not None:
        env_cfg.scene.env_spacing = args_cli.env_spacing
        print(f"[INFO] Using env spacing: {args_cli.env_spacing}")
    env_cfg.episode_length_s = 9999
    for term in ["ee_body_pos", "anchor_ori", "anchor_pos"]:
        if hasattr(env_cfg.terminations, term):
            setattr(env_cfg.terminations, term, None)

    if args_cli.motion_file is not None:
        env_cfg.commands.motion.motion_file = args_cli.motion_file
        print(f"[INFO] Using motion file: {args_cli.motion_file}")
    if args_cli.motion_file_txt is not None and hasattr(env_cfg.commands.motion, "dataset_txt"):
        env_cfg.commands.motion.dataset_txt = args_cli.motion_file_txt
        print(f"[INFO] Using motion file txt: {args_cli.motion_file_txt}")

    if not os.path.isfile(args_cli.resume_path):
        raise FileNotFoundError(f"Checkpoint not found: {args_cli.resume_path}")

    export_model_dir = args_cli.onnx_dir or os.path.join(os.path.dirname(args_cli.resume_path), "exported")
    onnx_path = os.path.join(export_model_dir, args_cli.onnx_filename)

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(export_model_dir, "videos", "onnx_pipeline"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during ONNX play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    vec_env = RslRlVecEnvWrapper(env)
    ppo_runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(args_cli.resume_path)

    print(f"[INFO] Exporting ONNX to: {onnx_path}")
    export_motion_policy_as_onnx(
        vec_env.unwrapped,
        ppo_runner.alg.get_policy(),
        path=export_model_dir,
        filename=args_cli.onnx_filename,
    )
    attach_onnx_metadata(vec_env.unwrapped, args_cli.wandb_path or "none", export_model_dir, args_cli.onnx_filename)

    print(f"[INFO] Loading ONNX runtime session from: {onnx_path}")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    print(f"[INFO] ONNX input: '{input_name}', output: '{output_name}'")

    obs_dict, _ = env.reset()
    obs_np = _to_numpy_policy_obs(obs_dict)
    timestep = 0
    print("[INFO] Starting ONNX play loop...")

    while simulation_app.is_running():
        actions_np = ort_session.run([output_name], {input_name: obs_np})[0]
        actions = torch.from_numpy(actions_np).to(vec_env.unwrapped.device)
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        obs_np = _to_numpy_policy_obs(obs_dict)
        timestep += 1

        if args_cli.video and timestep >= args_cli.video_length:
            print(f"[INFO] Video recording completed after {timestep} steps.")
            break

        terminated_any = bool(torch.any(terminated).item()) if torch.is_tensor(terminated) else bool(np.any(terminated))
        truncated_any = bool(torch.any(truncated).item()) if torch.is_tensor(truncated) else bool(np.any(truncated))
        if terminated_any or truncated_any:
            obs_dict, _ = env.reset()
            obs_np = _to_numpy_policy_obs(obs_dict)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
