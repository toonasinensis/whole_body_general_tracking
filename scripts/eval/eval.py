"""Policy evaluation script for motion-tracking tasks.

Pipeline:
1) Load motions from --motion_file_txt (relative paths under --motion_file).
2) For each motion, roll out the policy in IsaacLab.
3) Motions may have different lengths; the motion command restarts at frame 0
   when the motion finishes (eval_mode starts clips from first frame).
4) No early termination is activated (terminations disabled).
5) Metrics are the same as training: motion-tracking errors from MotionCommand,
   plus mean reward from the env step.
6) When the motion list is large, motions are evaluated in sequential folds
   sized by --num_envs to avoid creating too many env instances at once.
7) Save evaluation results to --output_path after all motions finish.
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
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RSL-RL policy on a list of motions.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of parallel envs for evaluation (fold size). Motions are evaluated sequentially in folds of this size.",
)
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
    for name in ("ee_body_pos", "anchor_ori", "anchor_pos", "anchor_lin_vel"):
        if hasattr(env_cfg.terminations, name):
            setattr(env_cfg.terminations, name, None)


def _read_motion_list(txt_path: str) -> list[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def _default_output_dir_from_resume(resume_path: str) -> Path:
    """Build `eval_results/<run_name>/<checkpoint_stem>/` from a checkpoint path.

    Example:
        logs/.../0422_all_kept/model_69600.pt -> eval_results/0422_all_kept/model_69600/
    """
    ckpt = Path(resume_path)
    run_name = ckpt.parent.name if ckpt.parent.name else "resume_info"
    ckpt_stem = ckpt.stem if ckpt.stem else "checkpoint"
    return Path("eval_results") / run_name / ckpt_stem


def _iter_folds(items: list[str], fold_size: int) -> list[list[str]]:
    if fold_size <= 0:
        raise ValueError(f"fold_size must be > 0, got {fold_size}")
    return [items[i : i + fold_size] for i in range(0, len(items), fold_size)]


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, _agent_cfg: RslRlOnPolicyRunnerCfg):
    # Resolve policy cfg from registry + CLI overrides (same as play.py).
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    all_motion_relpaths = _read_motion_list(args_cli.motion_file_txt)
    if len(all_motion_relpaths) == 0:
        raise ValueError(f"No motions found in --motion_file_txt: {args_cli.motion_file_txt}")

    requested_num_envs = int(getattr(args_cli, "num_envs", 1) or 1)
    if requested_num_envs <= 0:
        raise ValueError(f"--num_envs must be > 0, got {requested_num_envs}")

    # If the motion list is smaller than assigned envs, reduce env count to avoid padding
    # duplicates and creating extra env instances.
    fold_size = min(requested_num_envs, len(all_motion_relpaths))
    folds = _iter_folds(all_motion_relpaths, fold_size=fold_size)
    # import ipdb; ipdb.set_trace()
    
    # Apply evaluation config.
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

    results: list[MotionEvalResult] = []

    # Create a single env/runner once and reuse it. Switching motions between folds is done by
    # updating `dataset_txt` and calling `motion_cmd.resample_motion_files(env)`.
    env_cfg.scene.num_envs = fold_size

    with tempfile.TemporaryDirectory(prefix="eval_folds_") as tmp_dir:
        tmp_dir_p = Path(tmp_dir)

        # Initialize env with the first fold (padded if needed).
        init_fold = folds[0] if len(folds) > 0 else []

        if len(init_fold) == 0:
            raise ValueError("No motions to evaluate after folding.")
        init_padded = list(init_fold)

        if len(init_padded) < fold_size:
            init_padded = init_padded + [init_padded[-1]] * (fold_size - len(init_padded))

        # import ipdb; ipdb.set_trace()
        init_txt = tmp_dir_p / "fold_00000.txt"
        init_txt.write_text("\n".join([str(x).strip() for x in init_padded]) + "\n", encoding="utf-8")
        if hasattr(env_cfg.commands.motion, "dataset_txt"):
            env_cfg.commands.motion.dataset_txt = str(init_txt)

        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env)

        try:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(args_cli.resume_path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)

            motion_cmd = env.unwrapped.command_manager.get_term("motion")

            # Prepare a fold iterator (optional progress bar).
            fold_iter: Any = enumerate(folds)
            if tqdm is not None:
                fold_iter = tqdm(fold_iter, total=len(folds), desc="folds", unit="fold")

            for fold_idx, motion_relpaths in fold_iter:
                if len(motion_relpaths) == 0:
                    continue
                try:
                    valid_n = len(motion_relpaths)
                    padded = list(motion_relpaths)
                    if len(padded) < fold_size:
                        padded = padded + [padded[-1]] * (fold_size - len(padded))

                    fold_txt = tmp_dir_p / f"fold_{fold_idx:05d}.txt"
                    fold_txt.write_text("\n".join([str(x).strip() for x in padded]) + "\n", encoding="utf-8")

                    # Update dataset and resample motions in-place (no simulator restart).
                    if hasattr(motion_cmd.cfg, "dataset_txt"):
                        motion_cmd.cfg.dataset_txt = str(fold_txt)
                    if hasattr(env.unwrapped.cfg.commands.motion, "dataset_txt"):
                        env.unwrapped.cfg.commands.motion.dataset_txt = str(fold_txt)
                    motion_cmd.resample_motion_files(env.unwrapped)
                    # Ensure fixed mapping env_id -> motion_id and per-env lengths are refreshed.
                    if hasattr(motion_cmd, "_setup_fixed_eval_motion_assignment"):
                        motion_cmd._setup_fixed_eval_motion_assignment()

                    # import ipdb; ipdb.set_trace()
                    # Reset env state for this fold.
                    try:
                        env.reset()
                    except Exception:
                        pass
                    obs = env.get_observations()

                    # Always prefer the motion list we asked MotionCommand to load for this fold.
                    # Some MotionLoader implementations keep stale `file_names` across resamples.
                    motion_names = padded

                    num_envs = int(env.unwrapped.num_envs)
                    if num_envs != fold_size:
                        raise ValueError(
                            f"Expected num_envs == fold_size, got num_envs={num_envs} fold_size={fold_size}"
                        )

                    # Per-env motion lengths (control steps == motion frames for MotionCommand).
                    motion_num_steps = motion_cmd.frame_end_per_env.clone().long()
                    max_motion_steps = int(torch.max(motion_num_steps).item()) if motion_num_steps.numel() > 0 else 0
                    if max_motion_steps <= 0:
                        raise RuntimeError("All motions in this fold have non-positive length; cannot run evaluation.")

                    # Only the first `valid_n` envs are real; the rest are padding.
                    valid_mask = (
                        torch.arange(num_envs, device=env.unwrapped.device, dtype=torch.long) < int(valid_n)
                    )

                    step_counts = torch.zeros(num_envs, dtype=torch.long, device=env.unwrapped.device)
                    reward_sums = torch.zeros(num_envs, dtype=torch.float32, device=env.unwrapped.device)
                    metric_sums: dict[str, torch.Tensor] = {}
                    would_terminate = torch.zeros(num_envs, dtype=torch.bool, device=env.unwrapped.device)

                    step_iter = range(max_motion_steps)
                    if tqdm is not None:
                        step_iter = tqdm(
                            step_iter,
                            total=max_motion_steps,
                            desc=f"eval fold {fold_idx + 1}/{len(folds)}",
                            unit="step",
                            leave=False,
                        )

                    for _ in step_iter:
                        active = valid_mask & (step_counts < motion_num_steps)
                        if bool(torch.all(~active).item()):
                            break
                        with torch.inference_mode():
                            actions = policy(obs)
                            obs, rewards, _, _ = env.step(actions)

                        reward_sums[active] += rewards[active]
                        step_counts[active] += 1

                        if compute_success_rate:
                            term_anchor_pos = tracking_mdp.bad_anchor_pos_z_only(
                                env.unwrapped, command_name="motion", threshold=0.4
                            )
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

                        for k, v in motion_cmd.metrics.items():
                            if not torch.is_tensor(v):
                                continue
                            if v.ndim != 1 or v.shape[0] != num_envs:
                                continue
                            if k not in metric_sums:
                                metric_sums[k] = torch.zeros(num_envs, device=v.device, dtype=torch.float32)
                            metric_sums[k][active] += v[active].float()

                    for env_id in range(valid_n):
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
                except Exception as e:
                    # Continue evaluating other folds even if one fold fails to load/run.
                    # This prevents partial CSV outputs that only contain the first fold.
                    print(f"[WARN] fold {fold_idx} failed ({len(motion_relpaths)} motions). Error: {e}")
                    continue
        finally:
            env.close()

    # Summaries.
    overall: dict[str, Any] = {
        "task": args_cli.task,
        "resume_path": args_cli.resume_path,
        "motion_file": args_cli.motion_file,
        "motion_file_txt": args_cli.motion_file_txt,
        "num_motions": len(results),
        "num_envs": int(fold_size),
        "requested_num_envs": int(getattr(args_cli, "num_envs", 1) or 1),
        "num_folds": len(folds),
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

    # Save outputs under an auto-derived folder based on resume checkpoint.
    # This keeps JSON/CSV/failed-list grouped and avoids overwriting across checkpoints.
    out_dir = _default_output_dir_from_resume(args_cli.resume_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(args_cli.output_path).name

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

    # Export failed motions (those that would have early-terminated).
    # Format matches common motion list files: one relative npz path per line.
    failed_list_path = out_path.with_suffix(".failed.txt")
    failed_relpaths = [r.motion_relpath for r in results if r.would_terminate]
    failed_list_path.write_text("\n".join(failed_relpaths) + ("\n" if failed_relpaths else ""), encoding="utf-8")

    print(f"[INFO] Wrote evaluation JSON:        {out_path}")
    print(f"[INFO] Wrote evaluation CSV:         {csv_path}")
    print(f"[INFO] Wrote failed-motion list TXT: {failed_list_path}")


if __name__ == "__main__":
    try:
        main()  # pyright: ignore[reportCallIssue]
    finally:
        simulation_app.close()

