import argparse
import os
import sys
import torch
import numpy as np
import onnxruntime as ort

import gymnasium as gym

# === MUST come before any isaaclab imports that use pxr/Usd ===
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# -----------------------------------------------------------------------------
# Debug-friendly CLI presets
# -----------------------------------------------------------------------------
def _maybe_inject_debug_cli_args() -> None:
    if len(sys.argv) > 1:
        return
    if sys.gettrace() is None:
        return

    env_args = os.environ.get("PLAY_PT_ARGS")
    if env_args:
        sys.argv.extend(env_args.split())
        return

    sys.argv.extend(
        [
            "--task", "Tracking-Flat-RobanS22-v0",
            "--num_envs", "1",
            "--motion_file", "data/roban_motions_quick_test",
            "--onnx_path", "logs/rsl_rl/roban_flat/2026-04-15_22-34-18/exported/policy.onnx",
        ]
    )


# -----------------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Test exported ONNX motion policy")

parser.add_argument("--task", type=str, default=None, help="Name of the task")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--motion_file", type=str, default=None, help="Path to motion file")
parser.add_argument("--onnx_path", type=str, required=True, help="Path to the exported .onnx file")
parser.add_argument("--video", action="store_true", default=False, help="Record video")
parser.add_argument("--video_length", type=int, default=500, help="Video length in steps")
parser.add_argument("--disable_fabric", action="store_true", default=False)

AppLauncher.add_app_launcher_args(parser)

_maybe_inject_debug_cli_args()
args_cli, hydra_args = parser.parse_known_args()

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch the simulator (critical step)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Safe imports after SimulationApp is running
# -----------------------------------------------------------------------------
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils.hydra import hydra_task_config

# Register your tasks
import whole_body_tracking.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    """Test the exported ONNX policy."""
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = 9999

    # Disable problematic terminations for long play
    for term in ["ee_body_pos", "anchor_ori", "anchor_pos"]:
        if hasattr(env_cfg.terminations, term):
            setattr(env_cfg.terminations, term, None)

    if args_cli.motion_file is not None:
        env_cfg.commands.motion.motion_file = args_cli.motion_file
        print(f"[INFO] Using motion file: {args_cli.motion_file}")

    # Create environment
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    # === Convert MARL → single agent if needed (this was the bug) ===
    if isinstance(env.unwrapped, DirectMARLEnv):
        print("[INFO] Converting DirectMARLEnv to single-agent using multi_agent_to_single_agent")
        env = multi_agent_to_single_agent(env)

    # === Load ONNX ===
    print(f"[INFO] Loading ONNX model from: {args_cli.onnx_path}")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    ort_session = ort.InferenceSession(
        args_cli.onnx_path,
        sess_options=sess_options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    print(f"[INFO] ONNX input: '{input_name}' → output: '{output_name}'")

    # Video wrapper
    if args_cli.video:
        video_dir = os.path.join(os.path.dirname(args_cli.onnx_path), "videos", "onnx_test")
        os.makedirs(video_dir, exist_ok=True)
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording video to: {video_dir}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Reset
    obs_dict, _ = env.reset()
    obs = obs_dict.get("policy", obs_dict)
    obs = obs.cpu().numpy() if torch.is_tensor(obs) else np.asarray(obs, dtype=np.float32)

    timestep = 0
    print("[INFO] Starting inference loop... (Ctrl+C to stop)")

    try:
        while simulation_app.is_running():
            # ONNX forward pass
            actions_np = ort_session.run([output_name], {input_name: obs})[0]
            actions = torch.from_numpy(actions_np).to(env.unwrapped.device)

            # Step the environment
            obs_dict, _, terminated, truncated, _ = env.step(actions)

            # Prepare next obs
            obs = obs_dict.get("policy", obs_dict)
            obs = obs.cpu().numpy() if torch.is_tensor(obs) else np.asarray(obs, dtype=np.float32)

            timestep += 1

            if args_cli.video and timestep >= args_cli.video_length:
                print(f"[INFO] Video recording completed after {timestep} steps.")
                break

            if terminated.any() or truncated.any():
                print(f"[INFO] Episode ended at step {timestep}. Resetting...")
                obs_dict, _ = env.reset()
                obs = obs_dict.get("policy", obs_dict)
                obs = obs.cpu().numpy() if torch.is_tensor(obs) else np.asarray(obs, dtype=np.float32)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
