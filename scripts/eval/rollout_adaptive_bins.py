"""Evaluate adaptive sampling and dump motion-bin probabilities.

This script runs the tracking environment with the trained policy, with
`MotionCommandCfg.adaptive_sample=True` (same as training). As episodes terminate,
`MotionCommand.bin_failed_count` is updated and used to bias sampling.

After running for N steps, the script computes the *same* sampling probabilities
as `MotionCommand._adaptive_sampling()` and dumps a sorted list of bins:
  - bin_index
  - sampling_probability
  - global_frame_start / global_frame_end (approx)
  - mapped motion file + local frame range (by bin start timestamp)

Note: In the current implementation of `MotionCommand`, bins are defined over the
*concatenated global timeline* of all loaded motions, not per-motion.

python scripts/eval/eval_adaptive_bins.py \
  --task Tracking-Flat-RobanS22-v0 \
  --resume_path logs/rsl_rl/roban_flat/2026-04-17_21-27-35/model_25600.pt \
  --motion_file data/roban_motions \
  --motion_file_txt data/roban_motions_list/motions_main_kept_500.txt \
  --max_motion_num 5000 \
  --steps 10000 \
  --out data/eval_results/adaptive_bins_roban.json \
  --device cuda:0 \
  --headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


def _unwrap_policy_obs(obs):
    if isinstance(obs, tuple) and len(obs) > 0:
        obs = obs[0]
    if isinstance(obs, dict):
        for key in ("policy", "obs", "observations"):
            if key in obs:
                return obs[key]
        return next(iter(obs.values()))
    return obs


@dataclass
class RunningStats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return (self.m2 / (self.n - 1)) ** 0.5


# ---------------- CLI ----------------
parser = argparse.ArgumentParser(description="Evaluate adaptive sampling bins and dump probabilities.")
parser.add_argument("--task", type=str, required=True, help="IsaacLab task name (Hydra registry key).")
parser.add_argument("--resume_path", type=str, required=True, help="Path to the trained model checkpoint.")
parser.add_argument(
    "--motion_file",
    type=str,
    required=True,
    help="Motion dataset root (directory of .npz or layout expected by MotionLoader); same as train.py --motion_file.",
)
parser.add_argument(
    "--motion_file_txt",
    type=str,
    default=None,
    help="Optional txt listing relative .npz paths under --motion_file (maps to commands.motion.dataset_txt), same as train.py.",
)
parser.add_argument("--steps", type=int, default=20000, help="Number of env steps to run before dumping bins.")
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=500,
    help="Steps to skip before accumulating metrics (default: 500).",
)
parser.add_argument("--seed", type=int, default=42, help="Seed (passed to agent config).")
parser.add_argument(
    "--max_motion_num",
    type=int,
    default=-1,
    help="Cap motions loaded into memory (commands.motion.max_motion_num). Use -1 for all; same semantics as train.py.",
)
parser.add_argument(
    "--out",
    type=str,
    default="assets/eval_results/adaptive_bins.json",
    help="Output JSON path.",
)
parser.add_argument(
    "--out_txt",
    type=str,
    default=None,
    help=(
        "Optional output TXT path (quick_test.txt format: one motion file per line). "
        "If omitted, writes next to --out as '<stem>_motions_sorted.txt'."
    ),
)
parser.add_argument(
    "--disable_noise",
    action="store_true",
    default=False,
    help="Optionally disable pose/velocity/joint noise during eval.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ---------------- imports after app launch ----------------
import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401


def _compute_sampling_probabilities(motion_cmd) -> torch.Tensor:
    """Match MotionCommand._adaptive_sampling() probability computation."""
    # clipped failure history
    clipped = torch.clamp(
        motion_cmd.bin_failed_count, max=motion_cmd.cfg.failure_cap_beta * motion_cmd.bin_failed_count.mean()
    )
    probs = torch.nn.functional.pad(
        clipped.unsqueeze(0).unsqueeze(0),
        (0, motion_cmd.cfg.adaptive_kernel_size - 1),
        mode="replicate",
    )
    probs = torch.nn.functional.conv1d(probs, motion_cmd.kernel.view(1, 1, -1)).view(-1)
    probs = probs / (probs.sum() + 1e-8)
    probs = (1 - motion_cmd.cfg.adaptive_uniform_ratio) * probs + (motion_cmd.cfg.adaptive_uniform_ratio) / float(
        motion_cmd.bin_count
    )
    return probs


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg.seed = args_cli.seed

    motion_root = os.path.abspath(os.path.expanduser(args_cli.motion_file))
    env_cfg.commands.motion.motion_file = motion_root
    # Optional dataset list (same as scripts/qq_rsl_rl/train.py).
    if args_cli.motion_file_txt is not None and hasattr(env_cfg.commands.motion, "dataset_txt"):
        env_cfg.commands.motion.dataset_txt = os.path.abspath(os.path.expanduser(args_cli.motion_file_txt))
    # Cap loaded motions (-1 = all in MotionLoader._find_npz_files).
    env_cfg.commands.motion.max_motion_num = int(args_cli.max_motion_num)

    # Keep adaptive sampling enabled (same as training).
    env_cfg.commands.motion.adaptive_sample = True

    if args_cli.disable_noise:
        env_cfg.commands.motion.pose_range = {k: (0.0, 0.0) for k in ["x", "y", "z", "roll", "pitch", "yaw"]}
        env_cfg.commands.motion.velocity_range = {k: (0.0, 0.0) for k in ["x", "y", "z", "roll", "pitch", "yaw"]}
        env_cfg.commands.motion.joint_position_range = (0.0, 0.0)

    resume_path = os.path.abspath(os.path.expanduser(args_cli.resume_path))
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"--resume_path not found: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    vec_env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)

    base_env = vec_env.unwrapped
    motion_cmd = base_env.command_manager.get_term("motion")

    obs = _unwrap_policy_obs(vec_env.reset())
    steps = int(args_cli.steps)
    warmup = int(args_cli.warmup_steps)

    print(f"[INFO] Running adaptive sampling eval: steps={steps}, num_envs={base_env.num_envs}")
    print(f"[INFO] motion_file: {motion_root}")
    if args_cli.motion_file_txt is not None:
        print(f"[INFO] motion_file_txt -> dataset_txt: {getattr(env_cfg.commands.motion, 'dataset_txt', None)}")
    print(f"[INFO] max_motion_num: {env_cfg.commands.motion.max_motion_num}")
    print(f"[INFO] Checkpoint: {resume_path}")

    # Metrics to average (match commands.py _update_metrics)
    metric_keys = [
        "error_anchor_pos",
        "error_anchor_rot",
        "error_anchor_lin_vel",
        "error_anchor_ang_vel",
        "error_body_pos",
        "error_body_rot",
        "error_body_lin_vel",
        "error_body_ang_vel",
        "error_joint_pos",
        "error_joint_vel",
    ]
    stats = {k: RunningStats() for k in metric_keys}

    for t in range(steps):
        if not simulation_app.is_running():
            break
        with torch.no_grad():
            actions = policy(_unwrap_policy_obs(obs))
            obs = vec_env.step(actions)[0]

        if t < warmup:
            continue

        # average across envs for a stable scalar per step
        for k in metric_keys:
            if k in motion_cmd.metrics:
                v = motion_cmd.metrics[k].mean()
                stats[k].update(float(v.item()))
        # compute missing body-velocity metrics explicitly if not present in dict
        if "error_body_lin_vel" in metric_keys and "error_body_lin_vel" not in motion_cmd.metrics:
            v_err = torch.norm(motion_cmd.body_lin_vel_w - motion_cmd.robot_body_lin_vel_w, dim=-1).mean()
            stats["error_body_lin_vel"].update(float(v_err.item()))
        if "error_body_ang_vel" in metric_keys and "error_body_ang_vel" not in motion_cmd.metrics:
            w_err = torch.norm(motion_cmd.body_ang_vel_w - motion_cmd.robot_body_ang_vel_w, dim=-1).mean()
            stats["error_body_ang_vel"].update(float(w_err.item()))

    # compute probabilities
    probs = _compute_sampling_probabilities(motion_cmd).detach().cpu()
    bin_failed = motion_cmd.bin_failed_count.detach().cpu()
    bin_count = int(motion_cmd.bin_count)
    total_frames = int(motion_cmd.motion.time_step_total)
    fps = float(motion_cmd.motion.fps)

    # Map each bin to a global frame interval (approx) and to a motion segment by bin start.
    # `time_steps` are global indices into concatenated tensors.
    file_names = list(getattr(motion_cmd.motion, "file_names", []))
    if len(file_names) == 0:
        file_names = [f"motion_{i:05d}.npz" for i in range(int(motion_cmd.motion.motion_num))]

    # motion boundaries
    start_idx = motion_cmd.motion.time_step_start_idx.detach().cpu().tolist()
    end_idx = motion_cmd.motion.time_step_end_idx.detach().cpu().tolist()

    def motion_id_for_ts(ts: int) -> int:
        # linear scan is fine for a few hundred motions; keep simple
        for i, e in enumerate(end_idx):
            if ts < e:
                return i
        return len(end_idx) - 1

    bins: list[dict] = []
    for b in range(bin_count):
        # follow the same definition used in commands.py:
        # current_bin_index = (time_steps * bin_count) // time_step_total
        # => approximate global start/end:
        g0 = int((b * (total_frames - 1)) // bin_count)
        g1 = int(((b + 1) * (total_frames - 1)) // bin_count)
        mid = g0
        mid = max(0, min(mid, total_frames - 1))
        mid_motion = motion_id_for_ts(mid)

        m0 = start_idx[mid_motion]
        m1 = end_idx[mid_motion]
        local0 = int(max(0, mid - m0))
        # report bin end clipped to this motion
        local1 = int(max(local0, min(g1 - m0, m1 - m0)))

        bins.append(
            {
                "bin_index": b,
                "sampling_probability": float(probs[b].item()),
                "bin_failed_count": float(bin_failed[b].item()),
                "global_frame_start": g0,
                "global_frame_end": g1,
                "global_time_s_start": g0 / fps,
                "motion_id_at_bin_start": int(mid_motion),
                "motion_file": file_names[mid_motion],
                "motion_local_frame_start": local0,
                "motion_local_frame_end": local1,
            }
        )

    bins_sorted = sorted(bins, key=lambda x: x["sampling_probability"], reverse=True)
    out_path = Path(args_cli.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_summary = {
        k: {"mean": s.mean, "std": s.std, "n": s.n} for k, s in stats.items() if s.n > 0
    }
    payload = {
        "meta": {
            "motion_file": motion_root,
            "motion_file_txt": getattr(env_cfg.commands.motion, "dataset_txt", None),
            "max_motion_num": int(env_cfg.commands.motion.max_motion_num),
            "checkpoint": resume_path,
            "steps": steps,
            "warmup_steps": warmup,
            "num_envs": int(base_env.num_envs),
        },
        "avg_metrics": metrics_summary,
        "bins": bins_sorted,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote: {out_path}")

    # Also write a quick_test.txt-style file list sorted by probability mass per motion.
    motion_scores: dict[str, float] = {}
    for b in bins_sorted:
        mf = str(b["motion_file"])
        motion_scores[mf] = motion_scores.get(mf, 0.0) + float(b["sampling_probability"])
    motions_sorted = sorted(motion_scores.items(), key=lambda kv: kv[1], reverse=True)
    out_txt = args_cli.out_txt
    if out_txt is None:
        out_txt_path = out_path.with_name(out_path.stem + "_motions_sorted.txt")
    else:
        out_txt_path = Path(out_txt).expanduser().resolve()
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    out_txt_path.write_text("".join(f"{mf}\n" for mf, _ in motions_sorted), encoding="utf-8")
    print(f"[INFO] Wrote: {out_txt_path}")

    vec_env.close()


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
    simulation_app.close()

