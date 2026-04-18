"""Replay reference motion bins sorted by adaptive sampling probabilities.

Given an `adaptive_bins_*.json` produced by `eval_adaptive_bins.py`, this script
replays the **top-K bins in parallel** (one Isaac env per bin) and **loops** each
bin's local frame range forever until you close the app.

It uses the tracking environment only for robot + scene setup:
- Set robot root/joint state from the reference motion tensors at each timestep
- Render (no policy actions; no RL stepping)

Notes:
- Motion loading uses `meta` from the JSON (written by `eval_adaptive_bins.py`):
  `motion_file`, optional `motion_file_txt`, and `max_motion_num`.
- Each bin entry includes `motion_file` and `motion_local_frame_start/end`, mapped
  into MotionCommand's concatenated timeline via `time_step_start_idx`.

python scripts/eval/replay_adaptive_bins.py \
  --task Tracking-Flat-RobanS22-v0 \
  --bins_json data/eval_results/adaptive_bins_roban.json \
  --top_k 16 \
  --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Replay motion bins from adaptive_bins JSON.")
parser.add_argument("--task", type=str, required=True, help="IsaacLab task name.")
parser.add_argument("--bins_json", type=str, required=True, help="Path to adaptive_bins JSON.")
parser.add_argument("--top_k", type=int, default=16, help="Number of top bins to show in parallel (default: 16).")
parser.add_argument("--start_rank", type=int, default=0, help="Start from this bin rank in the sorted list.")
parser.add_argument(
    "--env_spacing",
    type=float,
    default=2.5,
    help="Spacing between parallel envs (default: 2.5).",
)
parser.add_argument("--seed", type=int, default=42)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
_bins_path = Path(args_cli.bins_json).expanduser()
if not _bins_path.is_file():
    parser.error(f"--bins_json not found: {_bins_path.resolve()}")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401


def _apply_motion_meta_from_payload(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, meta: dict) -> None:
    """Configure motion loader from eval JSON meta (same keys as eval_adaptive_bins writes)."""
    motion_root = meta.get("motion_file") or meta.get("motions_dir")
    if not motion_root:
        raise ValueError(
            "JSON meta is missing 'motion_file' (or legacy 'motions_dir'). "
            "Re-run eval_adaptive_bins.py so meta is written, or patch the JSON."
        )
    env_cfg.commands.motion.motion_file = os.path.abspath(os.path.expanduser(str(motion_root)))
    txt = meta.get("motion_file_txt")
    if txt and hasattr(env_cfg.commands.motion, "dataset_txt"):
        env_cfg.commands.motion.dataset_txt = os.path.abspath(os.path.expanduser(str(txt)))
    env_cfg.commands.motion.max_motion_num = int(meta.get("max_motion_num", -1))
    print(f"[INFO] motion_file={env_cfg.commands.motion.motion_file}")
    if getattr(env_cfg.commands.motion, "dataset_txt", None):
        print(f"[INFO] dataset_txt={env_cfg.commands.motion.dataset_txt}")
    print(f"[INFO] max_motion_num={env_cfg.commands.motion.max_motion_num}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    # Agent cfg is still loaded by Hydra, but we do not use the policy here.
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg.seed = args_cli.seed

    bins_path = Path(args_cli.bins_json).expanduser().resolve()
    payload = json.loads(bins_path.read_text(encoding="utf-8"))
    meta = payload.get("meta") or {}
    _apply_motion_meta_from_payload(env_cfg, meta)

    all_bins: list[dict] = payload["bins"]
    start = int(args_cli.start_rank)
    end = min(len(all_bins), start + int(args_cli.top_k))
    bins = all_bins[start:end]
    if len(bins) == 0:
        raise ValueError(f"No bins to replay for start_rank={start}, top_k={args_cli.top_k} (list has {len(all_bins)} bins).")

    # One env per bin: parallel visualization; each bin segment loops in the sim loop below.
    env_cfg.scene.num_envs = len(bins)
    env_cfg.scene.env_spacing = float(args_cli.env_spacing)
    env_cfg.episode_length_s = 1e9
    # avoid adaptive resampling; we will set timestamps manually
    env_cfg.commands.motion.adaptive_sample = True
    env_cfg.commands.motion.disable_resample_on_end = True
    # prevent random noise affecting visuals (optional but usually desired)
    env_cfg.commands.motion.pose_range = {k: (0.0, 0.0) for k in ["x", "y", "z", "roll", "pitch", "yaw"]}
    env_cfg.commands.motion.velocity_range = {k: (0.0, 0.0) for k in ["x", "y", "z", "roll", "pitch", "yaw"]}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)

    # Use RGB render mode so viewer stays responsive.
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    base_env = env.unwrapped
    motion_cmd = base_env.command_manager.get_term("motion")

    # Build name->motion_id mapping
    file_names: list[str] = list(getattr(motion_cmd.motion, "file_names", []))
    # file_names are relative to motion_file root; compare by basename for bin lookup.
    base_to_id = {Path(n).name: i for i, n in enumerate(file_names)}
    start_idx = motion_cmd.motion.time_step_start_idx
    end_idx = motion_cmd.motion.time_step_end_idx

    # Reset once to initialize buffers.
    env.reset()

    fps = float(motion_cmd.motion.fps)
    print(f"[INFO] fps={fps}, parallel replay of {len(bins)} bins (ranks {start}..{start + len(bins) - 1}), loop per bin")

    def _set_ts_for_bin(bin_item: dict) -> tuple[int, int]:
        motion_file = Path(bin_item["motion_file"]).name
        mid = base_to_id.get(motion_file, None)
        if mid is None:
            raise KeyError(f"motion_file '{motion_file}' not found in loaded MotionCommand file_names.")
        local0 = int(bin_item["motion_local_frame_start"])
        local1 = int(bin_item["motion_local_frame_end"])
        g0 = int(start_idx[mid].item()) + local0
        g1 = int(start_idx[mid].item()) + max(local1, local0 + 1)
        # clamp within this motion
        g0 = max(g0, int(start_idx[mid].item()))
        g1 = min(g1, int(end_idx[mid].item()))
        return g0, g1

    robot = base_env.scene["robot"]
    sim_dt = float(getattr(base_env, "step_dt", 0.02))

    n = len(bins)
    g0_list: list[int] = []
    g1_list: list[int] = []
    for i, b in enumerate(bins):
        g0, g1 = _set_ts_for_bin(b)
        g0_list.append(g0)
        g1_list.append(max(g1, g0 + 1))
        print(
            f"[ENV {i:03d}] p={b['sampling_probability']:.6f} file={Path(b['motion_file']).name} "
            f"local=[{b['motion_local_frame_start']},{b['motion_local_frame_end']}) "
            f"global=[{g0},{g1}) (loops)"
        )

    device = base_env.device
    env_ids = torch.arange(n, device=device, dtype=torch.long)
    g0_t = torch.tensor(g0_list, device=device, dtype=torch.long)
    g1_t = torch.tensor(g1_list, device=device, dtype=torch.long)
    ts = g0_t.clone()
    # MotionCommand debug markers (e.g. green goal velocity arrow) read anchor_pos_w / anchor_lin_vel_w,
    # which are indexed by motion_cmd.time_steps — not by this script's manual ``ts``. Keep them in sync
    # so arrows match the motion frame used for write_root_state_to_sim.
    motion_cmd.time_steps.copy_(ts)

    while simulation_app.is_running():
        motion_cmd.time_steps.copy_(ts)
        # write state for all envs at once (each env may use a different global timestep)
        root_pos = motion_cmd.motion.body_pos_w[ts, motion_cmd.motion_anchor_body_index] + base_env.scene.env_origins
        root_quat = motion_cmd.motion.body_quat_w[ts, motion_cmd.motion_anchor_body_index]
        root_lin = motion_cmd.motion.body_lin_vel_w[ts, motion_cmd.motion_anchor_body_index]
        root_ang = motion_cmd.motion.body_ang_vel_w[ts, motion_cmd.motion_anchor_body_index]
        joint_pos = motion_cmd.motion.joint_pos[ts]
        joint_vel = motion_cmd.motion.joint_vel[ts]

        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        robot.write_root_state_to_sim(torch.cat([root_pos, root_quat, root_lin, root_ang], dim=-1), env_ids=env_ids)

        base_env.sim.render()
        base_env.scene.update(sim_dt)

        # advance each env's timeline and wrap so each bin loops forever
        ts = ts + 1
        ts = torch.where(ts >= g1_t, g0_t, ts)

    env.close()


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
    simulation_app.close()

