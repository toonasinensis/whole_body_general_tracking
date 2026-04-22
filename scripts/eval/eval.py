"""Policy evaluation script for motion-tracking tasks.

Pipeline:
1) Load motions from --motion_file_txt (relative paths under --motion_file).
2) For each motion, roll out the policy in IsaacLab.
3) Motions may have different lengths; the motion command restarts at frame 0
   when the motion finishes (eval_mode starts clips from first frame).
4) No early termination is activated (terminations disabled).
5) Metrics are the same as training: motion-tracking errors from MotionCommand,
   plus mean reward from the env step.
6) Save evaluation results to --output_path after all motions finish.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RSL-RL policy on a list of motions.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs for evaluation (per motion).")
parser.add_argument("--motion_file", type=str, required=True, help="Directory that contains motion npz files.")
parser.add_argument(
    "--motion_file_txt",
    type=str,
    required=True,
    help="Txt file listing relative .npz paths under --motion_file (one per line).",
)
parser.add_argument("--max_motion_num", type=int, default=1, help="Max number of motions to load (per eval motion).")
parser.add_argument("--resume_path", type=str, required=True, help="Path to the model checkpoint (.pt).")
parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="Where to save evaluation results (JSON). A CSV with the same prefix is also written.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--compute_success_rate",
    action="store_true",
    default=False,
    help="Compute tracking success rate using the task termination checks (without actually terminating).",
)

# append RSL-RL cli arguments (for logger/device defaults)
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

import gymnasium as gym  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401, E402
from whole_body_tracking.tasks.tracking import mdp as tracking_mdp  # noqa: E402


@dataclass
class MotionEvalResult:
    motion_relpath: str
    num_steps: int
    mean_reward: float
    metrics_mean: dict[str, float]
    would_terminate: bool = False


def _disable_terminations(env_cfg: Any) -> None:
    """Best-effort disabling of early termination terms used in tracking."""
    if not hasattr(env_cfg, "terminations") or env_cfg.terminations is None:
        return
    for name in ("ee_body_pos", "anchor_ori", "anchor_pos"):
        if hasattr(env_cfg.terminations, name):
            setattr(env_cfg.terminations, name, None)


def _read_motion_list(txt_path: str) -> list[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def _aggregate_step_metrics(step_sums: dict[str, torch.Tensor], step_count: int) -> dict[str, float]:
    out: dict[str, float] = {}
    if step_count <= 0:
        return out
    for k, v in step_sums.items():
        out[k] = (v / float(step_count)).item()
    return out


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, _agent_cfg: RslRlOnPolicyRunnerCfg):
    # Resolve policy cfg from registry + CLI overrides (same as play.py).
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    motion_relpaths = _read_motion_list(args_cli.motion_file_txt)
    if len(motion_relpaths) == 0:
        raise ValueError(f"No motions found in --motion_file_txt: {args_cli.motion_file_txt}")

    # Apply evaluation config.
    # One env per motion.
    env_cfg.scene.num_envs = len(motion_relpaths)
    env_cfg.commands.motion.motion_file = args_cli.motion_file
    env_cfg.commands.motion.eval_mode = True
    env_cfg.commands.motion.fixed_eval_motion_ids = True
    env_cfg.commands.motion.save_adaptive_bins = False
    # In fixed-eval mode we want to load all motions from motion_file_txt.
    env_cfg.commands.motion.max_motion_num = -1
    env_cfg.episode_length_s = 9999
    # Keep rollouts running; optionally still *compute* termination-like failures.
    compute_success_rate = bool(getattr(args_cli, "compute_success_rate", False))
    if not compute_success_rate:
        _disable_terminations(env_cfg)

    # Create env that loads the full motion list once.
    if hasattr(env_cfg.commands.motion, "dataset_txt"):
        env_cfg.commands.motion.dataset_txt = args_cli.motion_file_txt
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        motion_cmd = env.unwrapped.command_manager.get_term("motion")

        # Determine motion file names from MotionLoader if available (preferred).
        file_names = list(getattr(motion_cmd.motion, "file_names", []))
        if len(file_names) == len(motion_relpaths):
            motion_names = file_names
        else:
            motion_names = motion_relpaths

        num_envs = env.unwrapped.num_envs
        if num_envs != len(motion_relpaths):
            raise ValueError(f"Expected num_envs == num_motions, got num_envs={num_envs} num_motions={len(motion_relpaths)}")

        # Reset.
        obs = env.get_observations()

        # Per-env accumulators (only until each env completes 1 cycle).
        step_counts = torch.zeros(num_envs, dtype=torch.long, device=env.unwrapped.device)
        reward_sums = torch.zeros(num_envs, dtype=torch.float32, device=env.unwrapped.device)
        metric_sums: dict[str, torch.Tensor] = {}
        # Success metric: mark if a motion would have early-terminated at least once.
        would_terminate = torch.zeros(num_envs, dtype=torch.bool, device=env.unwrapped.device)

        # Rollout until every env completes 1 cycle.
        while True:
            prev_cycles = motion_cmd.eval_cycle_count.clone()
            active = prev_cycles < 1
            if bool(torch.all(~active).item()):
                break

            with torch.inference_mode():
                actions = policy(obs)
                obs, rewards, _, _ = env.step(actions)

            # Accumulate reward per env.
            reward_sums[active] += rewards[active]
            step_counts[active] += 1

            if compute_success_rate:
                # Use the same termination checks as `config/roban/flat_env_cfg.py`.
                # We do NOT terminate; we only mark failures.
                term_anchor_pos = tracking_mdp.bad_anchor_pos_z_only(env.unwrapped, command_name="motion", threshold=0.4)
                term_anchor_ori = tracking_mdp.bad_anchor_ori(
                    env.unwrapped,
                    asset_cfg=tracking_mdp.SceneEntityCfg("robot"),
                    command_name="motion",
                    threshold=1.0,
                )
                term_ee_body_pos = tracking_mdp.bad_motion_body_pos_z_only(
                    env.unwrapped,
                    command_name="motion",
                    threshold=0.4,
                    body_names=["leg_l6_link", "leg_r6_link", "zarm_l5_link", "zarm_r5_link"],
                )
                term_any = term_anchor_pos | term_anchor_ori | term_ee_body_pos
                would_terminate[active] |= term_any[active]

            # Accumulate command metrics per env.
            for k, v in motion_cmd.metrics.items():
                if not torch.is_tensor(v):
                    continue
                if v.ndim != 1 or v.shape[0] != num_envs:
                    continue
                if k not in metric_sums:
                    metric_sums[k] = torch.zeros(num_envs, device=v.device, dtype=torch.float32)
                metric_sums[k][active] += v[active].float()

        # Build per-motion results.
        results: list[MotionEvalResult] = []
        for env_id in range(num_envs):
            n = int(step_counts[env_id].item())
            mean_reward = (reward_sums[env_id] / max(n, 1)).item()
            metrics_mean: dict[str, float] = {}
            for k, s in metric_sums.items():
                metrics_mean[k] = (s[env_id] / max(n, 1)).item()
            results.append(
                MotionEvalResult(
                    motion_relpath=str(motion_names[env_id]),
                    num_steps=n,
                    mean_reward=float(mean_reward),
                    metrics_mean=metrics_mean,
                    would_terminate=bool(would_terminate[env_id].item()),
                )
            )
    finally:
        env.close()

    # Summaries.
    overall: dict[str, Any] = {
        "task": args_cli.task,
        "resume_path": args_cli.resume_path,
        "motion_file": args_cli.motion_file,
        "motion_file_txt": args_cli.motion_file_txt,
        "num_motions": len(results),
        "num_envs": len(results),
    }

    if results:
        overall["mean_reward"] = sum(r.mean_reward for r in results) / len(results)
        if compute_success_rate:
            early_term_ratio = sum(1 for r in results if r.would_terminate) / len(results)
            overall["early_termination_ratio"] = float(early_term_ratio)
            overall["tracking_success_rate"] = float(1.0 - early_term_ratio)
        # Metric-wise average over motions.
        all_keys = sorted({k for r in results for k in r.metrics_mean.keys()})
        overall_metrics: dict[str, float] = {}
        for k in all_keys:
            vals = [r.metrics_mean[k] for r in results if k in r.metrics_mean]
            overall_metrics[k] = float(sum(vals) / len(vals)) if vals else float("nan")
        overall["metrics_mean"] = overall_metrics

    out_path = Path(args_cli.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "overall": overall,
        "per_motion": [asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Also write a flat CSV for quick inspection.
    csv_path = out_path.with_suffix(".csv")
    metric_keys = sorted({k for r in results for k in r.metrics_mean.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["motion_relpath", "num_steps", "mean_reward", *metric_keys],
        )
        writer.writeheader()
        for r in results:
            row = {"motion_relpath": r.motion_relpath, "num_steps": r.num_steps, "mean_reward": r.mean_reward}
            for k in metric_keys:
                row[k] = r.metrics_mean.get(k, "")
            writer.writerow(row)

    print(f"[INFO] Wrote evaluation JSON: {out_path}")
    print(f"[INFO] Wrote evaluation CSV:  {csv_path}")


if __name__ == "__main__":
    try:
        main()  # pyright: ignore[reportCallIssue]
    finally:
        simulation_app.close()

