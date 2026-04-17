"""Replay reference motion bins sorted by adaptive sampling probabilities.

Given an `adaptive_bins_*.json` produced by `eval_adaptive_bins.py`, this script
replays bins sequentially so you can visually inspect the "hard" bins first.

It uses the tracking environment only for robot + scene setup. For each bin:
- Reset env
- Set robot root/joint state directly from the reference motion tensors
- Render (no policy actions; no environment stepping)

Notes:
- The `adaptive_bins` JSON stores `motion_file` and `motion_local_frame_start/end`.
  We map that into MotionCommand's concatenated timeline using its per-motion
  `time_step_start_idx`.

python scripts/eval/replay_adaptive_bins.py \
  --task Tracking-Flat-G1-v0 \
  --motions data/g1_eval_motions/main \
  --bins_json data/eval_results/adaptive_bins_main.json \
  --top_k 16 \
  --multi \
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


def _unwrap_policy_obs(obs):
    if isinstance(obs, tuple) and len(obs) > 0:
        obs = obs[0]
    if isinstance(obs, dict):
        for key in ("policy", "obs", "observations"):
            if key in obs:
                return obs[key]
        return next(iter(obs.values()))
    return obs


parser = argparse.ArgumentParser(description="Replay motion bins from adaptive_bins JSON.")
parser.add_argument("--task", type=str, required=True, help="IsaacLab task name.")
parser.add_argument("--motions", type=str, required=True, help="Directory of motion npz files (same used in bins eval).")
parser.add_argument("--bins_json", type=str, required=True, help="Path to adaptive_bins JSON.")
parser.add_argument("--top_k", type=int, default=50, help="Replay only top-K bins (default: 50).")
parser.add_argument("--start_rank", type=int, default=0, help="Start from this bin rank in the sorted list.")
parser.add_argument(
    "--multi",
    action="store_true",
    default=False,
    help="Replay top-K bins simultaneously across K envs in a loop (instead of one-by-one).",
)
parser.add_argument(
    "--env_spacing",
    type=float,
    default=2.5,
    help="Environment spacing when using --multi (default: 2.5).",
)
parser.add_argument(
    "--follow_camera",
    action="store_true",
    default=False,
    help="If set, override the viewer camera to follow env[0]. If not set, camera remains user-adjustable.",
)
parser.add_argument("--seed", type=int, default=42)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    # Agent cfg is still loaded by Hydra, but we do not use the policy here.
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg.seed = args_cli.seed

    motions_dir = os.path.abspath(os.path.expanduser(args_cli.motions))
    env_cfg.commands.motion.motion_file = motions_dir
    # env count for replay
    if args_cli.multi:
        env_cfg.scene.num_envs = int(args_cli.top_k)
        env_cfg.scene.env_spacing = float(args_cli.env_spacing)
    else:
        env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 1e9
    # avoid adaptive resampling; we will set timestamps manually
    env_cfg.commands.motion.adaptive_sample = True
    env_cfg.commands.motion.disable_resample_on_end = True
    # prevent random noise affecting visuals (optional but usually desired)
    env_cfg.commands.motion.pose_range = {k: (0.0, 0.0) for k in ["x", "y", "z", "roll", "pitch", "yaw"]}
    env_cfg.commands.motion.velocity_range = {k: (0.0, 0.0) for k in ["x", "y", "z", "roll", "pitch", "yaw"]}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)

    bins_path = Path(args_cli.bins_json).expanduser().resolve()
    bins = json.loads(bins_path.read_text(encoding="utf-8"))["bins"]
    start = int(args_cli.start_rank)
    end = min(len(bins), start + int(args_cli.top_k))
    bins = bins[start:end]

    # Use RGB render mode so viewer stays responsive.
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    base_env = env.unwrapped
    motion_cmd = base_env.command_manager.get_term("motion")

    # Build name->motion_id mapping
    file_names: list[str] = list(getattr(motion_cmd.motion, "file_names", []))
    # file_names are relative to motions_dir, so compare by basename.
    base_to_id = {Path(n).name: i for i, n in enumerate(file_names)}
    start_idx = motion_cmd.motion.time_step_start_idx
    end_idx = motion_cmd.motion.time_step_end_idx

    # Reset once to initialize buffers.
    env.reset()

    fps = float(motion_cmd.motion.fps)
    print(f"[INFO] fps={fps}, replaying {len(bins)} bins from rank {start}")

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

    if not args_cli.multi:
        # one-by-one replay
        for i, b in enumerate(bins):
            if not simulation_app.is_running():
                break

            g0, g1 = _set_ts_for_bin(b)
            steps = max(1, g1 - g0)
            print(
                f"[BIN {start+i:04d}] p={b['sampling_probability']:.6f} "
                f"file={Path(b['motion_file']).name} local=[{b['motion_local_frame_start']},{b['motion_local_frame_end']}) "
                f"global=[{g0},{g1}) t=[{g0/fps:.2f}s,{g1/fps:.2f}s]"
            )

            # reset env (keeps viewer stable; clears contacts)
            try:
                base_env.reset_idx(torch.tensor([0], device=base_env.device, dtype=torch.long))
            except Exception:
                env.reset()

            env_ids0 = torch.tensor([0], device=base_env.device, dtype=torch.long)
            for ts in range(g0, g1):
                if not simulation_app.is_running():
                    break
                root_pos = (
                    motion_cmd.motion.body_pos_w[ts : ts + 1, motion_cmd.motion_anchor_body_index]
                    + base_env.scene.env_origins[:1]
                )
                root_quat = motion_cmd.motion.body_quat_w[ts : ts + 1, motion_cmd.motion_anchor_body_index]
                root_lin = motion_cmd.motion.body_lin_vel_w[ts : ts + 1, motion_cmd.motion_anchor_body_index]
                root_ang = motion_cmd.motion.body_ang_vel_w[ts : ts + 1, motion_cmd.motion_anchor_body_index]
                joint_pos = motion_cmd.motion.joint_pos[ts : ts + 1]
                joint_vel = motion_cmd.motion.joint_vel[ts : ts + 1]

                robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids0)
                robot.write_root_state_to_sim(
                    torch.cat([root_pos, root_quat, root_lin, root_ang], dim=-1), env_ids=env_ids0
                )
                base_env.sim.render()
                base_env.scene.update(sim_dt)

                if args_cli.follow_camera:
                    pos_lookat = robot.data.body_pos_w[0, 0].detach().cpu().numpy()
                    base_env.sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.8]), pos_lookat)
    else:
        # simultaneous replay: one env per bin, loop within each bin range
        n = int(env_cfg.scene.num_envs)
        if n != len(bins):
            raise RuntimeError(f"--multi expects num_envs==top_k==len(bins). Got num_envs={n}, len(bins)={len(bins)}")

        g0_list: list[int] = []
        g1_list: list[int] = []
        for i, b in enumerate(bins):
            g0, g1 = _set_ts_for_bin(b)
            g0_list.append(g0)
            g1_list.append(max(g1, g0 + 1))
            print(
                f"[ENV {i:03d}] p={b['sampling_probability']:.6f} file={Path(b['motion_file']).name} "
                f"local=[{b['motion_local_frame_start']},{b['motion_local_frame_end']}) "
                f"global=[{g0},{g1})"
            )

        device = base_env.device
        env_ids = torch.arange(n, device=device, dtype=torch.long)
        g0_t = torch.tensor(g0_list, device=device, dtype=torch.long)
        g1_t = torch.tensor(g1_list, device=device, dtype=torch.long)
        ts = g0_t.clone()

        while simulation_app.is_running():
            # write state for all envs at once
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

            if args_cli.follow_camera:
                pos_lookat = robot.data.body_pos_w[0, 0].detach().cpu().numpy()
                base_env.sim.set_camera_view(pos_lookat + np.array([6.0, 6.0, 2.0]), pos_lookat)

            # advance timestamps and wrap to g0 at end
            ts = ts + 1
            ts = torch.where(ts >= g1_t, g0_t, ts)

    env.close()


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
    simulation_app.close()

