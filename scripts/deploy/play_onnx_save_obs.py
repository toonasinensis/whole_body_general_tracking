"""
Used for DEBUG

Play ONNX policy in IsaacLab and save policy observations to CSV.

This script loads an ONNX policy, runs it in IsaacLab simulation, and saves the
policy observations (the 'policy' obs group) to a CSV file. The observations
include the motion_* terms defined in tracking_env_cfg.py.

The first X dimensions of each observation row correspond to the motion command
observations (motion_joint_pos, motion_joint_vel, motion_anchor_lin_vel_b,
motion_anchor_ang_vel_b, motion_anchor_project_gravity, motion_anchor_pos_z).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

import gymnasium as gym

# === MUST come before any isaaclab imports that use pxr/Usd ===
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


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
            "--task",
            "Tracking-Flat-RobanS22-v0",
            "--num_envs",
            "1",
            "--motion_file",
            "data/roban_motions",
            "--motion_file_txt",
            "data/roban_motions_list/debug_motion.txt",
            "--onnx_path",
            "logs/rsl_rl/roban_flat/2026-04-15_22-34-18/exported/policy.onnx",
            "--output_csv",
            "logs/policy_observations.csv",
        ]
    )


# -----------------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play ONNX policy and save observations to CSV")

parser.add_argument("--task", type=str, default=None, help="Name of the task")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--motion_file", type=str, default=None, help="Path to motion file (directory with .npz files)")
parser.add_argument(
    "--motion_file_txt",
    type=str,
    default=None,
    help="Optional txt file listing relative .npz paths under --motion_file",
)
parser.add_argument("--onnx_path", type=str, required=True, help="Path to the exported .onnx file")
parser.add_argument(
    "--output_csv",
    type=str,
    required=True,
    help="Output CSV path (used as a prefix). Script will write one CSV per observation term. "
    "If --output_dir is not provided, term CSVs are written into <output_csv>_terms.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Optional output directory. If provided, term CSVs are written here. "
    "Otherwise defaults to <output_csv>_terms.",
)
parser.add_argument(
    "--output_anchor_csv",
    type=str,
    default=None,
    help="Optional anchor pose CSV path. If provided, dumps motion anchor position/orientation in world frame "
    "(anchor_pos_w, anchor_quat_w) each step. If not provided, defaults to <output_dir>/anchor_pose_w.csv.",
)
parser.add_argument("--max_steps", type=int, default=None, help="Maximum steps to record (default: run until motion ends)")
parser.add_argument(
    "--history_index",
    type=int,
    default=0,
    help="Which history slice to save per term when observations are history-stacked. "
    "0 selects the first slice in each term block (matches indices like 0, 210, 420, ...).",
)
parser.add_argument("--video", action="store_true", default=False, help="Record video")
parser.add_argument("--video_length", type=int, default=500, help="Video length in steps")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument(
    "--output_obs_dump_csv",
    type=str,
    default=None,
    help="Optional path to write full policy observation vector each step as a single CSV "
    "(columns obs_0..obs_{N-1}). If not provided, defaults to <output_dir>/observations_dump.csv. "
    "Note: time_s/step columns are intentionally omitted.",
)
parser.add_argument(
    "--dump_env_id",
    type=int,
    default=0,
    help="Which env_id to dump for per-step observation CSVs when num_envs > 1 (default: 0).",
)
parser.add_argument(
    "--zero_unfilled_history",
    action="store_true",
    default=True,
    help="When dumping observations, zero out not-yet-filled history slots after reset "
    "(early history frames are zeros; newest frame is hist{H-1}).",
)
parser.add_argument(
    "--no_zero_unfilled_history",
    action="store_false",
    dest="zero_unfilled_history",
    help="Disable zeroing of unfilled history slots in dumped observations.",
)

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


def _to_numpy_policy_obs(obs_data) -> np.ndarray:
    """Extract policy observations as numpy array."""
    policy_obs = obs_data.get("policy", obs_data) if isinstance(obs_data, dict) else obs_data
    if torch.is_tensor(policy_obs):
        return policy_obs.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(policy_obs, dtype=np.float32)


def _select_env_row(obs: np.ndarray, env_id: int) -> np.ndarray:
    """Return a (D,) vector for a single env from (E,D) or already-flat input."""
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        eid = int(env_id)
        if eid < 0 or eid >= arr.shape[0]:
            raise ValueError(f"dump_env_id={eid} out of range for obs shape {arr.shape}")
        return arr[eid]
    return arr.reshape(-1)


def _zero_unfilled_history_inplace(
    flat_obs: np.ndarray,
    term_dims: list[tuple[str, int]],
    history_length: int,
    steps_since_reset: int,
) -> None:
    """
    Observation layout (term-major, history concatenated first):
      [term0_hist0..hist{H-1}, term1_hist0..hist{H-1}, ...]

    Assume the newest frame is hist{H-1}. After reset:
      steps_since_reset=0 -> only hist{H-1} valid; hist0..hist{H-2} should be zero.
      steps_since_reset=1 -> hist{H-2},hist{H-1} valid; hist0..hist{H-3} should be zero.
    """
    H = int(history_length)
    if H <= 1:
        return
    k = max(0, int(steps_since_reset))
    num_valid = min(H, k + 1)
    first_valid = H - num_valid  # history indices [0, first_valid) are unfilled

    cursor = 0
    for _term_name, dim in term_dims:
        dim = int(dim)
        block = dim * H
        for h in range(first_valid):
            s = cursor + h * dim
            e = s + dim
            flat_obs[s:e] = 0.0
        cursor += block


def _process_obs_for_policy(
    obs: np.ndarray,
    term_dims: list[tuple[str, int]],
    history_length: int,
    steps_since_reset: int,
    *,
    zero_unfilled_history: bool,
) -> np.ndarray:
    """
    Apply any preprocessing to observations before feeding the ONNX policy.
    Currently: optionally zero unfilled history slots (term-major layout).
    """
    if not zero_unfilled_history:
        return obs
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 1:
        out = arr.copy()
        _zero_unfilled_history_inplace(out, term_dims, history_length, steps_since_reset)
        return out
    if arr.ndim == 2:
        out = arr.copy()
        for i in range(out.shape[0]):
            _zero_unfilled_history_inplace(out[i], term_dims, history_length, steps_since_reset)
        return out
    out = arr.reshape(-1).copy()
    _zero_unfilled_history_inplace(out, term_dims, history_length, steps_since_reset)
    return out

def _get_policy_term_dims(env) -> list[tuple[str, int]]:
    """
    Return policy term base dims in order.
    Returns list of (term_name, dim_per_step).
    """
    unwrapped = env.unwrapped
    obs_manager = unwrapped.observation_manager
    active_terms = obs_manager.active_terms.get("policy", [])

    # The dimensions from tracking_env_cfg.py PolicyCfg
    motion_term_dims = {
        "motion_joint_pos": 21,
        "motion_joint_vel": 21,
        "motion_anchor_lin_vel_b": 3,
        "motion_anchor_ang_vel_b": 3,
        "motion_anchor_project_gravity": 3,
        "motion_anchor_pos_z": 1,
    }

    result: list[tuple[str, int]] = []
    for term_name in active_terms:
        dim = None
        if term_name == "motion_joint_pos":
            dim = motion_term_dims["motion_joint_pos"]
        elif term_name == "motion_joint_vel":
            dim = motion_term_dims["motion_joint_vel"]
        elif term_name == "motion_anchor_lin_vel_b":
            dim = motion_term_dims["motion_anchor_lin_vel_b"]
        elif term_name == "motion_anchor_ang_vel_b":
            dim = motion_term_dims["motion_anchor_ang_vel_b"]
        elif term_name == "motion_anchor_project_gravity":
            dim = motion_term_dims["motion_anchor_project_gravity"]
        elif term_name == "motion_anchor_pos_z":
            dim = motion_term_dims["motion_anchor_pos_z"]
        elif term_name == "motion_anchor_ori_b":
            dim = 6  # 6D rotation representation
        elif term_name == "projected_gravity":
            dim = 3
        elif term_name == "base_ang_vel":
            dim = 3
        elif term_name == "joint_pos":
            dim = 21
        elif term_name == "joint_vel":
            dim = 21
        elif term_name == "actions":
            dim = 21
        else:
            # Unknown term - try to infer from observation manager
            try:
                obs_term = obs_manager._group_obs_terms["policy"].get(term_name)
                if obs_term is not None:
                    # Sample observation to get shape
                    sample = obs_term.func(env)
                    if torch.is_tensor(sample):
                        dim = int(sample.numel())
                    else:
                        dim = int(np.prod(np.asarray(sample).shape))
                else:
                    dim = 0
            except Exception:
                dim = 0

        if dim:
            result.append((term_name, int(dim)))

    return result


def _infer_history_length(obs_dim: int, term_dims: list[tuple[str, int]]) -> int:
    base = int(sum(dim for _, dim in term_dims))
    if base <= 0:
        return 1
    if obs_dim % base != 0:
        # Fall back: treat as no history (best effort).
        return 1
    return int(obs_dim // base)


def _build_history_slices(
    term_dims: list[tuple[str, int]],
    history_length: int,
    history_index: int,
) -> list[tuple[str, int, int]]:
    """Map each term to (start,end) slice selecting one history frame within that term block."""
    if history_length <= 0:
        history_length = 1
    if history_index < 0 or history_index >= history_length:
        raise ValueError(f"--history_index={history_index} out of range for history_length={history_length}")

    slices: list[tuple[str, int, int]] = []
    cursor = 0
    for term_name, dim in term_dims:
        block = int(dim) * int(history_length)
        start = cursor + int(history_index) * int(dim)
        end = start + int(dim)
        slices.append((term_name, start, end))
        cursor += block
    return slices


def _extract_history_slice(obs: np.ndarray, slices: list[tuple[str, int, int]]) -> np.ndarray:
    flat = obs.reshape(-1)
    parts = [flat[start:end] for _, start, end in slices]
    return np.concatenate(parts, axis=0) if parts else flat[:0]


def _sanitize_filename(name: str) -> str:
    return "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in name).strip("_") or "term"


def _resolve_output_dir(output_csv: str, output_dir: str | None) -> Path:
    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        out_dir = Path(str(output_csv) + "_terms")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _extract_term(obs: np.ndarray, start: int, end: int) -> np.ndarray:
    flat = obs.reshape(-1)
    return flat[start:end]


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    """Play ONNX policy and save observations to CSV."""
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = 9999

    # Disable problematic terminations for long play
    for term in ["ee_body_pos", "anchor_ori", "anchor_pos"]:
        if hasattr(env_cfg.terminations, term):
            setattr(env_cfg.terminations, term, None)

    # Ensure deterministic evaluation behavior: always start motion from first frame.
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
        setattr(env_cfg.commands.motion, "eval_mode", True)

    # Configure motion file
    if args_cli.motion_file is not None:
        env_cfg.commands.motion.motion_file = args_cli.motion_file
        print(f"[INFO] Using motion file: {args_cli.motion_file}")
    if args_cli.motion_file_txt is not None and hasattr(env_cfg.commands.motion, "dataset_txt"):
        env_cfg.commands.motion.dataset_txt = args_cli.motion_file_txt
        print(f"[INFO] Using motion file txt: {args_cli.motion_file_txt}")

    # Disable observation corruption/noise (Unoise, etc.) during saving.
    if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False

    # Create environment
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    # Convert MARL → single agent if needed
    if isinstance(env.unwrapped, DirectMARLEnv):
        print("[INFO] Converting DirectMARLEnv to single-agent using multi_agent_to_single_agent")
        env = multi_agent_to_single_agent(env)

    term_dims = _get_policy_term_dims(env)
    print("[INFO] Policy terms (base dims):")
    for term_name, dim in term_dims:
        print(f"  {term_name}: dim={dim}")

    # Load ONNX
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
        video_dir = os.path.join(os.path.dirname(args_cli.onnx_path), "videos", "onnx_obs_recorder")
        os.makedirs(video_dir, exist_ok=True)
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording video to: {video_dir}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    out_dir = _resolve_output_dir(args_cli.output_csv, args_cli.output_dir)

    # Dump full policy observation vector each step (obs_0..obs_{N-1}) for a single env.
    obs_dump_csv_path = (
        Path(args_cli.output_obs_dump_csv)
        if args_cli.output_obs_dump_csv is not None
        else (out_dir / "observations_dump.csv")
    )
    obs_dump_fh = open(obs_dump_csv_path, "w", newline="")
    obs_dump_writer = csv.writer(obs_dump_fh)
    print(f"[INFO] Dumping full observations to: {obs_dump_csv_path} (env_id={args_cli.dump_env_id})")

    # Dump motion anchor pose in world frame alongside obs (enabled by default).
    anchor_csv_path = (
        Path(args_cli.output_anchor_csv)
        if args_cli.output_anchor_csv is not None
        else (out_dir / "anchor_pose_w.csv")
    )
    anchor_fh = open(anchor_csv_path, "w", newline="")
    anchor_writer = csv.writer(anchor_fh)
    # Write as plain text to avoid CSV quoting.
    anchor_fh.write("# command_term=motion fields=anchor_pos_w(xyz),anchor_quat_w(wxyz)\n")
    anchor_writer.writerow(["timestep", "env_id", "pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z"])
    print(f"[INFO] Dumping anchor pose to: {anchor_csv_path}")

    # Reset and get first observation
    obs_dict, _ = env.reset()
    obs = _to_numpy_policy_obs(obs_dict)
    obs_row = _select_env_row(obs, args_cli.dump_env_id)
    obs_dim = obs.shape[-1]

    history_length = _infer_history_length(obs_dim, term_dims)
    history_slices = _build_history_slices(term_dims, history_length, args_cli.history_index)
    obs_saved = _extract_history_slice(obs_row, history_slices)
    obs_saved_dim = int(obs_saved.shape[0])

    print(f"[INFO] Observation shape (full): {obs.shape}, dim={obs_dim}")
    print(f"[INFO] Inferred history_length={history_length}, saving history_index={args_cli.history_index}")
    print("[INFO] Saved slices (within full obs):")
    for term_name, start, end in history_slices:
        print(f"  {term_name}: [{start}:{end}] (dim={end-start})")
    print(f"[INFO] Saved obs dim: {obs_saved_dim}")
    print(f"[INFO] Saving per-term CSVs under: {out_dir}")

    timestep = 0
    steps_since_reset = 0
    max_steps = float("inf") if args_cli.max_steps is None else int(args_cli.max_steps)
    print(f"[INFO] Starting inference loop... (max_steps={max_steps if args_cli.max_steps is not None else 'unlimited'})")

    try:
        # Preprocess initial obs for the policy input (t=0 after reset).
        obs = _process_obs_for_policy(
            obs,
            term_dims=term_dims,
            history_length=history_length,
            steps_since_reset=steps_since_reset,
            zero_unfilled_history=bool(args_cli.zero_unfilled_history),
        )
        # Write header for full dump CSV.
        obs_dump_writer.writerow([f"obs_{i}" for i in range(int(obs_dim))])
        term_handles: dict[str, tuple[Path, csv.writer, object]] = {}
        for term_name, start, end in history_slices:
            safe = _sanitize_filename(term_name)
            term_path = out_dir / f"{safe}.csv"
            fh = open(term_path, "w", newline="")
            writer = csv.writer(fh)
            writer.writerow(
                [
                    f"# term={term_name} full_slice=[{start}:{end}] dim={end-start} "
                    f"history_index={args_cli.history_index} history_length={history_length}"
                ]
            )
            writer.writerow([f"{term_name}_{i:03d}" for i in range(end - start)])
            term_handles[term_name] = (term_path, writer, fh)

        while simulation_app.is_running() and timestep < max_steps:
            # ONNX forward pass
            actions_np = ort_session.run([output_name], {input_name: obs})[0]
            # IsaacLab env exposes device on the unwrapped instance, but type checkers don't know it.
            actions = torch.from_numpy(actions_np).to(getattr(env.unwrapped, "device", "cpu"))
            # Step the environment
            obs_dict, _, terminated, truncated, _ = env.step(actions)
            # Get observation
            obs_raw = _to_numpy_policy_obs(obs_dict)
            # Preprocess for next policy input (and for dumping).
            obs = _process_obs_for_policy(
                obs_raw,
                term_dims=term_dims,
                history_length=history_length,
                steps_since_reset=steps_since_reset,
                zero_unfilled_history=bool(args_cli.zero_unfilled_history),
            )
            obs_row_dump = _select_env_row(obs, args_cli.dump_env_id).reshape(-1)

            # Dump anchor pos/orientation (world).
            # Motion command term name is "motion" per tracking_env_cfg.py
            cmd = getattr(env.unwrapped, "command_manager").get_term("motion")
            pos_w = cmd.anchor_pos_w
            quat_w = cmd.anchor_quat_w
            if torch.is_tensor(pos_w):
                pos_w = pos_w.detach().cpu().numpy()
            if torch.is_tensor(quat_w):
                quat_w = quat_w.detach().cpu().numpy()
            pos_w = np.asarray(pos_w, dtype=np.float32)
            quat_w = np.asarray(quat_w, dtype=np.float32)
            for env_id in range(int(pos_w.shape[0])):
                p = pos_w[env_id].reshape(-1).tolist()
                q = quat_w[env_id].reshape(-1).tolist()
                anchor_writer.writerow(
                    [
                        int(timestep),
                        int(env_id),
                        f"{p[0]:.8f}",
                        f"{p[1]:.8f}",
                        f"{p[2]:.8f}",
                        f"{q[0]:.8f}", # w
                        f"{q[1]:.8f}", # x
                        f"{q[2]:.8f}", # y
                        f"{q[3]:.8f}", # z
                    ]
                )

            # Write each term separately (only the selected history slice).
            for term_name, start, end in history_slices:
                _, writer, _fh = term_handles[term_name]
                vals = _extract_term(obs_row_dump, start, end)
                writer.writerow([f"{x:.8f}" for x in vals.tolist()])

            # Write full obs vector (selected env only).
            obs_dump_writer.writerow([f"{x:.8f}" for x in obs_row_dump.tolist()])

            timestep += 1
            steps_since_reset += 1

            if args_cli.video and timestep >= args_cli.video_length:
                print(f"[INFO] Video recording completed after {timestep} steps.")
                break

            terminated_any = bool(torch.any(terminated).item()) if torch.is_tensor(terminated) else bool(np.any(terminated))
            truncated_any = bool(torch.any(truncated).item()) if torch.is_tensor(truncated) else bool(np.any(truncated))
            if terminated_any or truncated_any:
                print(f"[INFO] Episode ended at step {timestep}. Resetting...")
                obs_dict, _ = env.reset()
                steps_since_reset = 0
                obs = _to_numpy_policy_obs(obs_dict)
                obs = _process_obs_for_policy(
                    obs,
                    term_dims=term_dims,
                    history_length=history_length,
                    steps_since_reset=steps_since_reset,
                    zero_unfilled_history=bool(args_cli.zero_unfilled_history),
                )

        if args_cli.max_steps is not None and timestep >= max_steps:
            print(f"[INFO] Reached max_steps={max_steps}. Terminating simulation.")

    except KeyboardInterrupt:
        print(f"\n[INFO] Stopped by user at step {timestep}.")
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            for _term, (_path, _writer, fh) in list(term_handles.items()):
                try:
                    getattr(fh, "close")()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if anchor_fh is not None:
                anchor_fh.close()
        except Exception:
            pass
        try:
            if obs_dump_fh is not None:
                obs_dump_fh.close()
        except Exception:
            pass
        print(f"[INFO] Recorded {timestep} observations (per-term CSVs) under {out_dir}")
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
