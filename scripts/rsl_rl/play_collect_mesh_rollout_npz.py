#!/usr/bin/env python3
"""Collect actual policy rollouts on selected flat/mesh domains into tracking-npz format."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

WBT_ROOT = Path(__file__).resolve().parents[2]
RSL_RL_ROOT = WBT_ROOT.parent / "rsl_rl"
if RSL_RL_ROOT.is_dir():
    sys.path.insert(0, str(RSL_RL_ROOT))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


DEFAULT_PAIRS_JSONL = WBT_ROOT / "data/omniretarget/g1_terrain/cropped_pairs/pairs_cropped.jsonl"
DEFAULT_FLAT_MOTION_FILE = "data"
DEFAULT_FLAT_DATASET_TXT = "data/tracking_npz_data/lafan_named.txt"
DEFAULT_OUTPUT_ROOT = WBT_ROOT / "logs/play_collect_mesh_rollout"
HEIGHT_SCAN_GRID_SIZE = (1.6, 1.0)
HEIGHT_SCAN_GRID_RESOLUTION = 0.1
HEIGHT_SCAN_OFFSET_XY = (0.80, 0.0)
HEIGHT_SCAN_OBS_OFFSET = 0.5


def _safe_stem(text: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    stem = re.sub(r"_+", "_", stem).strip("._")
    return stem or "rollout"


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = WBT_ROOT / candidate
    return candidate.resolve()


def _resolve_manifest_path(path: str | Path, manifest_path: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    return candidate.resolve()


def _read_pair_records(pairs_jsonl: str | Path) -> list[dict[str, object]]:
    manifest_path = _resolve_repo_path(pairs_jsonl)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"pairs_jsonl not found: {manifest_path}")

    records: list[dict[str, object]] = []
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_num, line in enumerate(stream, start=1):
            text = line.strip()
            if not text:
                continue
            raw = json.loads(text)
            motion_path = _resolve_manifest_path(raw["tracking_motion"], manifest_path)
            terrain_path = _resolve_manifest_path(raw["terrain_stl"], manifest_path)
            name = str(raw.get("name") or motion_path.stem)
            if not motion_path.is_file():
                raise FileNotFoundError(f"{manifest_path}:{line_num}: missing motion npz: {motion_path}")
            if not terrain_path.is_file():
                raise FileNotFoundError(f"{manifest_path}:{line_num}: missing terrain stl: {terrain_path}")
            records.append(
                {
                    "name": name,
                    "tracking_motion": str(motion_path),
                    "terrain_stl": str(terrain_path),
                    "source_pairs_jsonl": str(manifest_path),
                    "source_pair_index": len(records),
                }
            )

    if not records:
        raise ValueError(f"{manifest_path}: no pair records found")
    return records


def _read_flat_motion_records(motion_file: str | Path, dataset_txt: str | Path) -> list[dict[str, object]]:
    motion_root = _resolve_repo_path(motion_file)
    dataset_path = _resolve_repo_path(dataset_txt)
    if not motion_root.exists():
        raise FileNotFoundError(f"flat_motion_file not found: {motion_root}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"flat_dataset_txt not found: {dataset_path}")

    records: list[dict[str, object]] = []
    with dataset_path.open("r", encoding="utf-8") as stream:
        for line_num, line in enumerate(stream, start=1):
            entry = line.strip()
            if not entry:
                continue
            candidate = Path(entry).expanduser()
            if not candidate.is_absolute():
                candidate = motion_root / candidate
            candidate = candidate.resolve()
            if candidate.suffix.lower() != ".npz":
                continue
            if not candidate.is_file():
                raise FileNotFoundError(f"{dataset_path}:{line_num}: missing flat motion npz: {candidate}")
            records.append(
                {
                    "name": candidate.stem,
                    "tracking_motion": str(candidate),
                    "terrain_stl": None,
                    "source_dataset_txt": str(dataset_path),
                    "source_motion_index": len(records),
                }
            )

    if not records:
        raise ValueError(f"{dataset_path}: no flat motion npz records found")
    return records


def _parse_index_selector(selector: str | int | None, count: int, *, label: str) -> list[int]:
    text = "0" if selector is None else str(selector).strip()
    if not text:
        text = "0"
    if text.lower() in {"all", "*"}:
        return list(range(count))

    indices: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            indices.extend(range(start, end + step, step))
        else:
            indices.append(int(item))

    seen: set[int] = set()
    unique = []
    for index in indices:
        if not 0 <= index < count:
            raise IndexError(f"{label} index {index} is out of range for {count} records")
        if index in seen:
            continue
        seen.add(index)
        unique.append(index)
    if not unique:
        raise ValueError(f"No {label} indices selected")
    return unique


def _parse_mesh_indices(args_cli: argparse.Namespace, pair_count: int) -> list[int]:
    if args_cli.mesh_pair_indices is not None:
        selector = args_cli.mesh_pair_indices
    elif args_cli.all_pairs:
        selector = "all"
    elif args_cli.pair_indices:
        selector = args_cli.pair_indices
    else:
        selector = args_cli.pair_index
    return _parse_index_selector(selector, pair_count, label="mesh pair")


def _parse_domain_selection(text: str) -> set[str]:
    selected: set[str] = set()
    for raw_token in str(text).split(","):
        token = raw_token.strip().lower().replace("-", "_")
        if not token:
            continue
        if token in {"mesh", "terrain"}:
            selected.add("mesh")
        elif token in {"flat", "flat_wbc", "wbc"}:
            selected.add("flat_wbc")
        elif token in {"flat_velocity", "velocity"}:
            selected.add("flat_velocity")
        elif token in {"flat_all", "flat_both"}:
            selected.update({"flat_wbc", "flat_velocity"})
        elif token == "all":
            selected.update({"flat_wbc", "flat_velocity", "mesh"})
        else:
            raise ValueError(
                f"Unsupported collect domain {raw_token!r}. Use mesh, flat, flat_wbc, flat_velocity, flat_all, or all."
            )
    if not selected:
        raise ValueError("--collect_domains selected no domains")
    return selected


def _with_domain(record: dict[str, object], domain: str) -> dict[str, object]:
    out = dict(record)
    out["domain"] = domain
    if domain.startswith("flat"):
        out["name"] = f"{domain}__{record['name']}"
    return out


def _write_selected_mesh_manifest(output_root: Path, selected_mesh_records: list[dict[str, object]]) -> Path:
    selected_manifest = output_root / "selected_pairs.jsonl"
    lines = []
    for pair in selected_mesh_records:
        selected_record = {
            "name": pair["name"],
            "tracking_motion": pair["tracking_motion"],
            "terrain_stl": pair["terrain_stl"],
        }
        lines.append(json.dumps(selected_record, ensure_ascii=False))
    selected_manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return selected_manifest


def _write_selected_flat_dataset(output_root: Path, selected_flat_records: list[dict[str, object]]) -> Path:
    selected_dataset = output_root / "selected_flat_dataset.txt"
    lines = [str(record["tracking_motion"]) for record in selected_flat_records]
    selected_dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return selected_dataset


def _prepare_collection(
    args_cli: argparse.Namespace,
) -> tuple[Path, Path, Path | None, list[dict[str, object]], str, dict[str, int]]:
    selected_domains = _parse_domain_selection(args_cli.collect_domains)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args_cli.run_name or f"{timestamp}_{_safe_stem(args_cli.collect_domains)}"
    output_root = (
        Path(args_cli.output_dir).expanduser().resolve() if args_cli.output_dir else DEFAULT_OUTPUT_ROOT / run_name
    )
    output_root.mkdir(parents=True, exist_ok=True)

    selected_records: list[dict[str, object]] = []
    domain_counts = {"flat_wbc": 0, "flat_velocity": 0, "mesh": 0}

    selected_flat_source_records: list[dict[str, object]] = []
    selected_flat_dataset: Path | None = None
    if selected_domains & {"flat_wbc", "flat_velocity"}:
        flat_records = _read_flat_motion_records(args_cli.flat_motion_file, args_cli.flat_dataset_txt)
        flat_indices = _parse_index_selector(args_cli.flat_motion_indices, len(flat_records), label="flat motion")
        selected_flat_source_records = [flat_records[index] for index in flat_indices]
        selected_flat_dataset = _write_selected_flat_dataset(output_root, selected_flat_source_records)

        for domain in ("flat_wbc", "flat_velocity"):
            if domain not in selected_domains:
                continue
            for record in selected_flat_source_records:
                selected_records.append(_with_domain(record, domain))
            domain_counts[domain] = len(selected_flat_source_records)

    pairs_jsonl = _resolve_repo_path(args_cli.pairs_jsonl)
    env_pairs_jsonl = pairs_jsonl
    if "mesh" in selected_domains:
        mesh_records = _read_pair_records(args_cli.pairs_jsonl)
        mesh_indices = _parse_mesh_indices(args_cli, len(mesh_records))
        selected_mesh_records = [_with_domain(mesh_records[index], "mesh") for index in mesh_indices]
        env_pairs_jsonl = _write_selected_mesh_manifest(output_root, selected_mesh_records)
        selected_records.extend(selected_mesh_records)
        domain_counts["mesh"] = len(selected_mesh_records)

    if not selected_records:
        raise ValueError("No rollout records selected")

    return output_root, env_pairs_jsonl, selected_flat_dataset, selected_records, run_name, domain_counts


parser = argparse.ArgumentParser(description="Collect mesh-terrain play rollouts into tracking npz format.")
parser.add_argument("--task", type=str, default="TerrainPairMixed-G1")
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Deprecated for this collector. The script creates one env per selected rollout record.",
)
parser.add_argument("--resume_path", type=str, required=True, help="Path to the RSL-RL checkpoint.")
parser.add_argument("--pairs_jsonl", type=str, default=str(DEFAULT_PAIRS_JSONL), help="Source terrain-motion manifest.")
parser.add_argument(
    "--collect_domains",
    type=str,
    default="mesh",
    help="Comma-separated domains to collect: mesh, flat/flat_wbc, flat_velocity, flat_all, or all.",
)
parser.add_argument(
    "--mesh_pair_indices",
    type=str,
    default=None,
    help="Mesh pair selector: 'all', a single index, or comma/range syntax like '0,3,5-8'.",
)
parser.add_argument("--pair_index", type=int, default=0, help="Source pair index to collect.")
parser.add_argument(
    "--pair_indices",
    type=str,
    default=None,
    help="Comma-separated pair indices/ranges to collect, e.g. '0,3,5-8'. Overrides --pair_index.",
)
parser.add_argument("--all_pairs", action="store_true", help="Collect every pair in --pairs_jsonl.")
parser.add_argument("--flat_motion_file", type=str, default=DEFAULT_FLAT_MOTION_FILE, help="Flat motion root.")
parser.add_argument("--flat_dataset_txt", type=str, default=DEFAULT_FLAT_DATASET_TXT, help="Flat motion dataset txt.")
parser.add_argument(
    "--flat_motion_indices",
    type=str,
    default="0",
    help="Flat motion selector: 'all', a single index, or comma/range syntax like '0,3,5-8'.",
)
parser.add_argument(
    "--output_dir", type=str, default=None, help="Output root. Defaults to logs/play_collect_mesh_rollout/<run>."
)
parser.add_argument("--max_steps", type=int, default=0, help="Maximum frames to collect. 0 means full motion length.")
parser.add_argument(
    "--encoder_mode",
    type=str,
    default="robot",
    choices=["split", "robot", "latent", "g1", "smpl", "encoder_g1", "encoder_smpl"],
    help="Compatibility option; rollout collection always forces robot encoder mode.",
)
parser.add_argument(
    "--legacy_task_obs",
    action="store_true",
    help="Add legacy 3D 'task' observation used by older paired-terrain checkpoints.",
)
parser.add_argument(
    "--domain_separator_cell_count",
    type=int,
    default=0,
    help="Separator cells are disabled by default for compact collection layouts.",
)
parser.add_argument(
    "--motion_reset_z_offset",
    type=float,
    default=0.10,
    help="Raise the reset robot root above the reference motion by this many meters.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.num_envs is not None and int(args_cli.num_envs) < 1:
    raise ValueError("--num_envs must be >= 1 when provided")
if int(args_cli.max_steps) < 0:
    raise ValueError("--max_steps must be >= 0")
if args_cli.encoder_mode != "robot":
    print(f"[INFO] Ignoring --encoder_mode={args_cli.encoder_mode}; rollout collection always uses robot.")
args_cli.encoder_mode = "robot"
MOTION_RESET_Z_OFFSET = float(args_cli.motion_reset_z_offset)

OUTPUT_ROOT, SELECTED_PAIRS_JSONL, SELECTED_FLAT_DATASET_TXT, SELECTED_RECORDS, RUN_NAME, DOMAIN_COUNTS = (
    _prepare_collection(args_cli)
)
SELECTED_ENV_COUNT = len(SELECTED_RECORDS)
os.environ["WBT_PAIRS_JSONL"] = str(SELECTED_PAIRS_JSONL)
if SELECTED_FLAT_DATASET_TXT is not None:
    os.environ["WBT_FLAT_DATASET_TXT"] = str(SELECTED_FLAT_DATASET_TXT)
os.environ["WBT_FLAT_WBC_ENV_RATIO"] = str(DOMAIN_COUNTS["flat_wbc"])
os.environ["WBT_FLAT_VELOCITY_ENV_RATIO"] = str(DOMAIN_COUNTS["flat_velocity"])
os.environ["WBT_MESH_ENV_RATIO"] = str(DOMAIN_COUNTS["mesh"])
os.environ["WBT_DOMAIN_SEPARATOR_CELL_COUNT"] = str(args_cli.domain_separator_cell_count)

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.tasks.tracking import mdp as tracking_mdp


class LegacyTaskObsVecEnvWrapper:
    """Expose the old paired-terrain 'task' obs group without changing the environment config."""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _with_task_obs(self, obs):
        if "task" not in obs:
            obs["task"] = tracking_mdp.motion_velocity_command_yaw_b(self.unwrapped, "motion")
        return obs

    def reset(self):
        obs, extras = self.env.reset()
        return self._with_task_obs(obs), extras

    def get_observations(self):
        return self._with_task_obs(self.env.get_observations())

    def step(self, actions):
        obs, rewards, dones, extras = self.env.step(actions)
        return self._with_task_obs(obs), rewards, dones, extras


def _set_backbone_encoder_mode(policy, encoder_mode: str) -> None:
    backbone = getattr(policy, "backbone", policy)
    if not hasattr(backbone, "set_encoder_mode"):
        print(f"[WARN] Policy backbone {type(backbone).__name__} does not support --encoder_mode; keeping default.")
        return
    backbone.set_encoder_mode(encoder_mode)
    active = backbone.get_encoder_mode() if hasattr(backbone, "get_encoder_mode") else encoder_mode
    print(f"[INFO] MyModel encoder mode set to: {active}")


def _disable_non_timeout_terminations(env_cfg) -> None:
    terminations = getattr(env_cfg, "terminations", None)
    if terminations is None:
        return
    seen: set[str] = set()
    for source in (vars(terminations), vars(type(terminations))):
        for name, value in source.items():
            if name in seen or name.startswith("_"):
                continue
            seen.add(name)
            if isinstance(value, DoneTerm) and "time_out" not in name:
                setattr(terminations, name, None)


def _force_motion_reset_cfg(env_cfg) -> None:
    motion_cfg = getattr(getattr(env_cfg, "commands", None), "motion", None)
    if motion_cfg is None:
        return
    motion_cfg.eval_mode = True
    motion_cfg.motion_sampling_start_frame = 0
    motion_cfg.pose_range_env_ratio = 0.0
    motion_cfg.velocity_range = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    motion_cfg.joint_position_range = (0.0, 0.0)


def _as_np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)


def _env_group(obs: dict, group_name: str, env_id: int) -> np.ndarray:
    if group_name not in obs:
        raise KeyError(f"Missing observation group {group_name!r}. Available groups: {list(obs.keys())}")
    value = obs[group_name]
    if not torch.is_tensor(value):
        raise TypeError(f"Observation group {group_name!r} is not a tensor: {type(value).__name__}")
    return _as_np(value[int(env_id)])


def _force_motion_frame_zero(base_env, motion_cmd, env_count: int) -> tuple[torch.Tensor, int]:
    env_ids = torch.arange(int(env_count), dtype=torch.long, device=base_env.device)
    motion_ids = env_ids.clone()
    motion_cmd.motion_ids[env_ids] = motion_ids
    motion_cmd.local_time_steps[env_ids] = 0
    max_future_step = int(getattr(motion_cmd.cfg, "max_future_step", 0))
    motion_lengths = (
        motion_cmd.motion.time_step_end_idx[motion_ids] - motion_cmd.motion.time_step_start_idx[motion_ids]
    ).long()
    # CommandTerm resamples before the last frames to protect future-reference obs.
    # This collector owns the stopping condition, so extend the guard and stop at the real final frame.
    motion_cmd.frame_end_per_env[env_ids] = motion_lengths + max_future_step + 1
    if hasattr(motion_cmd, "_sync_env_origins_to_domains"):
        motion_cmd._sync_env_origins_to_domains(env_ids)

    robot = base_env.scene["robot"]
    root_state = torch.cat(
        [
            motion_cmd.body_pos_w[env_ids, 0],
            motion_cmd.body_quat_w[env_ids, 0],
            motion_cmd.body_lin_vel_w[env_ids, 0],
            motion_cmd.body_ang_vel_w[env_ids, 0],
        ],
        dim=-1,
    )
    root_state[:, 2] += float(MOTION_RESET_Z_OFFSET)
    robot.write_joint_state_to_sim(motion_cmd.joint_pos[env_ids], motion_cmd.joint_vel[env_ids], env_ids=env_ids)
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    return motion_lengths.detach().cpu(), max_future_step


def _reset_env(vec_env) -> None:
    result = vec_env.reset()
    if isinstance(result, tuple):
        return


def _make_buffers() -> dict[str, list[np.ndarray]]:
    return {
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
        "height_scan": [],
        "height_scan_raw": [],
        "height_scan_points": [],
    }


def _append_frame(buffers: dict[str, list[np.ndarray]], base_env, obs: dict, env_id: int) -> None:
    robot = base_env.scene["robot"]
    env_id = int(env_id)
    env_origin = base_env.scene.env_origins[env_id]
    buffers["joint_pos"].append(_as_np(robot.data.joint_pos[env_id]))
    buffers["joint_vel"].append(_as_np(robot.data.joint_vel[env_id]))
    buffers["body_pos_w"].append(_as_np(robot.data.body_pos_w[env_id] - env_origin[None, :]))
    buffers["body_quat_w"].append(_as_np(robot.data.body_quat_w[env_id]))
    buffers["body_lin_vel_w"].append(_as_np(robot.data.body_lin_vel_w[env_id]))
    buffers["body_ang_vel_w"].append(_as_np(robot.data.body_ang_vel_w[env_id]))
    buffers["height_scan"].append(_env_group(obs, "terrain", env_id))
    height_scanner = base_env.scene.sensors.get("height_scanner")
    if height_scanner is not None:
        raw = (
            height_scanner.data.pos_w[env_id, 2] - height_scanner.data.ray_hits_w[env_id, :, 2] - HEIGHT_SCAN_OBS_OFFSET
        )
        points = height_scanner.data.ray_hits_w[env_id] - env_origin[None, :]
        buffers["height_scan_raw"].append(_as_np(raw))
        buffers["height_scan_points"].append(_as_np(points))


def _stack_buffers(buffers: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    out = {}
    for key, values in buffers.items():
        if not values:
            raise RuntimeError(f"No frames collected for {key}")
        out[key] = np.stack(values, axis=0).astype(np.float32, copy=False)
    return out


def _write_rollout_npz(
    arrays: dict[str, np.ndarray],
    base_env,
    record_info: dict[str, object],
    motion_length: int,
    max_future_step: int,
) -> tuple[dict[str, object], Path, Path]:
    pair_name = _safe_stem(str(record_info["name"]))
    rollout_name = f"{pair_name}__rollout"
    motion_dir = OUTPUT_ROOT / "tracking_npz_data"
    terrain_dir = OUTPUT_ROOT / "terrains" / "stl"
    motion_dir.mkdir(parents=True, exist_ok=True)
    terrain_dir.mkdir(parents=True, exist_ok=True)

    motion_out = motion_dir / f"{rollout_name}.npz"
    terrain_out = terrain_dir / f"{rollout_name}.stl"
    source_terrain = str(record_info.get("terrain_stl") or "")
    if source_terrain:
        shutil.copy2(source_terrain, terrain_out)
    else:
        _write_flat_ground_stl(terrain_out)

    fps = 1.0 / (float(base_env.cfg.decimation) * float(base_env.cfg.sim.dt))
    frame_count = int(arrays["joint_pos"].shape[0])
    robot = base_env.scene["robot"]
    np.savez_compressed(
        motion_out,
        fps=np.asarray([fps], dtype=np.float32),
        joint_names=np.asarray(list(robot.data.joint_names)),
        body_names=np.asarray(list(robot.data.body_names)),
        joint_pos=arrays["joint_pos"],
        joint_vel=arrays["joint_vel"],
        body_pos_w=arrays["body_pos_w"],
        body_pos_frame=np.asarray(["terrain_local"]),
        body_quat_w=arrays["body_quat_w"],
        body_lin_vel_w=arrays["body_lin_vel_w"],
        body_ang_vel_w=arrays["body_ang_vel_w"],
        height_scan=arrays["height_scan"],
        height_scan_raw=arrays["height_scan_raw"],
        height_scan_points=arrays["height_scan_points"],
        height_scan_shape=np.asarray(arrays["height_scan"].shape[1:], dtype=np.int32),
        height_scan_source=np.asarray(["obs.terrain.height_scan"]),
        height_scan_raw_source=np.asarray(["height_scanner.data.pos_w.z - ray_hits_w.z - 0.5"]),
        height_scan_points_source=np.asarray(["height_scanner.data.ray_hits_w - env_origin"]),
        height_scan_grid_size=np.asarray(HEIGHT_SCAN_GRID_SIZE, dtype=np.float32),
        height_scan_grid_resolution=np.asarray([HEIGHT_SCAN_GRID_RESOLUTION], dtype=np.float32),
        height_scan_offset_xy=np.asarray(HEIGHT_SCAN_OFFSET_XY, dtype=np.float32),
        height_scan_obs_offset=np.asarray([HEIGHT_SCAN_OBS_OFFSET], dtype=np.float32),
        motion_reset_z_offset=np.asarray([float(MOTION_RESET_Z_OFFSET)], dtype=np.float32),
        frames=np.asarray([frame_count], dtype=np.int32),
        duration_s=np.asarray([frame_count / max(fps, 1.0e-6)], dtype=np.float32),
        source_domain=np.asarray([str(record_info["domain"])]),
        source_pairs_jsonl=np.asarray([str(record_info.get("source_pairs_jsonl") or "")]),
        source_dataset_txt=np.asarray([str(record_info.get("source_dataset_txt") or "")]),
        source_pair_index=np.asarray([int(record_info.get("source_pair_index", -1))], dtype=np.int32),
        source_motion_index=np.asarray([int(record_info.get("source_motion_index", -1))], dtype=np.int32),
        source_motion=np.asarray([str(record_info["tracking_motion"])]),
        source_terrain=np.asarray([source_terrain]),
        checkpoint_path=np.asarray([str(Path(args_cli.resume_path).expanduser().resolve())]),
        rollout_motion_length=np.asarray([motion_length], dtype=np.int32),
        rollout_max_future_step=np.asarray([max_future_step], dtype=np.int32),
    )

    record = {
        "name": rollout_name,
        "tracking_motion": str(motion_out.relative_to(OUTPUT_ROOT)),
        "terrain_stl": str(terrain_out.relative_to(OUTPUT_ROOT)),
        "source_domain": str(record_info["domain"]),
        "source_name": str(record_info["name"]),
        "source_motion": str(record_info["tracking_motion"]),
        "source_terrain": source_terrain,
        "source_pairs_jsonl": str(record_info.get("source_pairs_jsonl") or ""),
        "source_dataset_txt": str(record_info.get("source_dataset_txt") or ""),
        "source_pair_index": int(record_info.get("source_pair_index", -1)),
        "source_motion_index": int(record_info.get("source_motion_index", -1)),
        "frames": frame_count,
        "duration_s": frame_count / max(fps, 1.0e-6),
        "motion_reset_z_offset": float(MOTION_RESET_Z_OFFSET),
        "checkpoint_path": str(Path(args_cli.resume_path).expanduser().resolve()),
    }
    return record, motion_out, terrain_out


def _write_flat_ground_stl(path: Path) -> None:
    import trimesh

    ground = trimesh.creation.box(extents=(16.0, 16.0, 0.01))
    ground.apply_translation((0.0, 0.0, -0.005))
    ground.export(path)


def _write_outputs(
    env_buffers: list[dict[str, list[np.ndarray]]],
    base_env,
    motion_lengths: torch.Tensor,
    max_future_step: int,
) -> tuple[list[Path], list[Path], Path]:
    records = []
    motion_paths = []
    terrain_paths = []
    for env_id, record_info in enumerate(SELECTED_RECORDS):
        arrays = _stack_buffers(env_buffers[env_id])
        record, motion_out, terrain_out = _write_rollout_npz(
            arrays,
            base_env,
            record_info,
            motion_length=int(motion_lengths[env_id].item()),
            max_future_step=int(max_future_step),
        )
        records.append(record)
        motion_paths.append(motion_out)
        terrain_paths.append(terrain_out)

    manifest_out = OUTPUT_ROOT / "pairs_rollout.jsonl"
    manifest_out.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return motion_paths, terrain_paths, manifest_out


def _target_frame_counts(motion_lengths: torch.Tensor) -> list[int]:
    counts = [max(1, int(length.item())) for length in motion_lengths]
    if int(args_cli.max_steps) <= 0:
        return counts
    max_steps = int(args_cli.max_steps)
    return [min(count, max_steps) for count in counts]


def _pin_finished_envs_before_step(motion_cmd, active: list[bool], motion_lengths: torch.Tensor) -> None:
    """Keep finished envs on their last valid reference frame while other envs continue."""
    inactive_ids = [env_id for env_id, is_active in enumerate(active) if not is_active]
    if not inactive_ids:
        return
    env_ids = torch.as_tensor(inactive_ids, dtype=torch.long, device=motion_cmd.device)
    lengths = motion_lengths.to(device=motion_cmd.device, dtype=torch.long)[env_ids]
    # MotionCommand increments local_time_steps before indexing reference tensors.
    # Setting length - 2 here makes the update read length - 1, the last valid frame.
    motion_cmd.local_time_steps[env_ids] = lengths - 2


def _apply_legacy_task_obs_cfg(agent_cfg) -> None:
    backbone = getattr(getattr(agent_cfg, "actor", None), "backbone", None)
    if isinstance(backbone, dict):
        encoder_cfg = backbone.get("encoder", {})
        if isinstance(encoder_cfg, dict) and "encoder_smpl" in encoder_cfg:
            encoder_cfg["encoder_smpl"]["encoder_groups"] = ["task"]
        backbone["encoder_mask_group"] = None
        backbone["aux_loss_mask_group"] = None
    agent_cfg.obs_groups = {
        "actor": ["prop", "rbt_cmd_mf", "smpl_cmd_mf", "terrain", "task"],
        "critic": ["critic", "terrain"],
    }
    print("[INFO] Enabled legacy paired-terrain task obs compatibility.")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    resume_path = Path(args_cli.resume_path).expanduser().resolve()
    if not resume_path.is_file():
        raise FileNotFoundError(f"resume_path not found: {resume_path}")
    if args_cli.legacy_task_obs:
        _apply_legacy_task_obs_cfg(agent_cfg)

    if args_cli.num_envs is not None and int(args_cli.num_envs) != SELECTED_ENV_COUNT:
        print(
            "[WARN] Ignoring --num_envs="
            f"{args_cli.num_envs}; collector creates one env per selected rollout record ({SELECTED_ENV_COUNT})."
        )

    env_cfg.scene.num_envs = SELECTED_ENV_COUNT
    if hasattr(env_cfg, "pairs_jsonl"):
        env_cfg.pairs_jsonl = str(SELECTED_PAIRS_JSONL)
    if hasattr(env_cfg, "pair_limit"):
        env_cfg.pair_limit = None
    if SELECTED_FLAT_DATASET_TXT is not None and hasattr(env_cfg, "flat_dataset_txt"):
        env_cfg.flat_dataset_txt = str(SELECTED_FLAT_DATASET_TXT)
    if hasattr(env_cfg, "flat_wbc_env_ratio"):
        env_cfg.flat_wbc_env_ratio = float(DOMAIN_COUNTS["flat_wbc"])
    if hasattr(env_cfg, "flat_velocity_env_ratio"):
        env_cfg.flat_velocity_env_ratio = float(DOMAIN_COUNTS["flat_velocity"])
    if hasattr(env_cfg, "mesh_env_ratio"):
        env_cfg.mesh_env_ratio = float(DOMAIN_COUNTS["mesh"])
    if hasattr(env_cfg, "domain_separator_cell_count"):
        env_cfg.domain_separator_cell_count = int(args_cli.domain_separator_cell_count)
    if hasattr(env_cfg, "configure_domains"):
        env_cfg.configure_domains()
    env_cfg.episode_length_s = 9999.0
    _disable_non_timeout_terminations(env_cfg)
    _force_motion_reset_cfg(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)
    if args_cli.legacy_task_obs:
        env = LegacyTaskObsVecEnvWrapper(env)

    runner_device = getattr(agent_cfg, "device", None) or getattr(args_cli, "device", None) or env.unwrapped.device
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=runner_device)
    ppo_runner.load(str(resume_path))
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    if args_cli.encoder_mode is not None:
        _set_backbone_encoder_mode(policy, args_cli.encoder_mode)

    base_env = env.unwrapped
    _reset_env(env)
    motion_cmd = base_env.command_manager.get_term("motion")
    motion_lengths, max_future_step = _force_motion_frame_zero(base_env, motion_cmd, SELECTED_ENV_COUNT)
    target_frames = _target_frame_counts(motion_lengths)
    obs = env.get_observations()
    env_buffers = [_make_buffers() for _ in SELECTED_RECORDS]
    active = [True] * SELECTED_ENV_COUNT
    print(f"[INFO] Collecting {SELECTED_ENV_COUNT} rollout(s): domains={DOMAIN_COUNTS}")
    for env_id, record_info in enumerate(SELECTED_RECORDS):
        print(
            "[INFO]   "
            f"env={env_id} domain={record_info['domain']} name={record_info['name']} "
            f"motion_length={int(motion_lengths[env_id].item())} target_frames={target_frames[env_id]}"
        )
        loaded_name = Path(getattr(motion_cmd.motion, "file_names", [])[int(motion_cmd.motion_ids[env_id].item())]).stem
        source_name = Path(str(record_info["tracking_motion"])).stem
        if loaded_name != source_name:
            raise RuntimeError(
                f"Selected record/motion loader mismatch for env {env_id}: selected={source_name}, loaded={loaded_name}"
            )

    with torch.inference_mode():
        while simulation_app.is_running():
            for env_id, is_active in enumerate(active):
                if not is_active:
                    continue
                _append_frame(env_buffers[env_id], base_env, obs, env_id)
                if len(env_buffers[env_id]["joint_pos"]) >= target_frames[env_id]:
                    active[env_id] = False
                    print(
                        "[INFO]   finished "
                        f"env={env_id} domain={SELECTED_RECORDS[env_id]['domain']} "
                        f"name={SELECTED_RECORDS[env_id]['name']} "
                        f"frames={len(env_buffers[env_id]['joint_pos'])}"
                    )
            if not any(active):
                break
            actions = policy(obs)
            if isinstance(actions, dict) and "actions" in actions:
                actions = actions["actions"]
            _pin_finished_envs_before_step(motion_cmd, active, motion_lengths)
            obs, _, _, _ = env.step(actions)

    motion_paths, terrain_paths, manifest_out = _write_outputs(env_buffers, base_env, motion_lengths, max_future_step)
    print(f"[INFO] Saved {len(motion_paths)} rollout motion npz file(s) under: {OUTPUT_ROOT / 'tracking_npz_data'}")
    print(f"[INFO] Saved {len(terrain_paths)} rollout terrain stl file(s) under: {OUTPUT_ROOT / 'terrains' / 'stl'}")
    print(f"[INFO] Saved rollout manifest: {manifest_out}")
    print(
        "[INFO] Visualize with:\n"
        "  python scripts/omniretarget/visualize_g1_terrain_pairs_viser.py "
        f"--pairs_jsonl {manifest_out} --show_height_scan"
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
