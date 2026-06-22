#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from isaaclab.app import AppLauncher

WBT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = WBT_ROOT / "data" / "omniretarget" / "g1_terrain"

G1_URDF_QPOS_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert prepared OmniRetarget G1 qpos motions into whole_body_tracking NPZ format."
    )
    parser.add_argument(
        "--data_root", default=str(DEFAULT_DATA_ROOT), help="Output root from prepare_g1_terrain_dataset.py"
    )
    parser.add_argument("--input_dir", default=None, help="Raw qpos npz directory. Defaults to data_root/raw_qpos.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Tracking npz output directory. Defaults to data_root/tracking_npz_data.",
    )
    parser.add_argument("--dataset_txt", default=None, help="Dataset txt output path.")
    parser.add_argument("--summary_path", default=None, help="Conversion summary JSON output path.")
    parser.add_argument("--include_regex", default=None, help="Optional regex passed to pathlib-style stem filtering.")
    parser.add_argument("--max_files", type=int, default=None, help="Convert at most this many files.")
    parser.add_argument("--target_fps", type=float, default=50.0, help="FPS written to the output npz files.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing converted npz files.")
    parser.add_argument(
        "--articulation_root_prim_path",
        default="/pelvis/pelvis",
        help="Relative articulation root path inside the spawned G1 prim.",
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Disable per-frame render calls while collecting articulation body states.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if getattr(args, "experience", "") == "":
        args.experience = str(Path(__file__).with_name("kit_app") / "isaaclab.core.headless.5_1.kit")
    return args


args_cli = _parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp


@configclass
class ConversionSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(intensity=2500.0))
    robot: ArticulationCfg = None


def _load_g1_cylinder_cfg() -> ArticulationCfg:
    wbt_source = Path(__file__).resolve().parents[2] / "source" / "whole_body_tracking" / "whole_body_tracking"
    assets_dir = wbt_source / "assets"
    g1_path = wbt_source / "robots" / "g1.py"
    package = types.ModuleType("whole_body_tracking")
    package.__path__ = [str(wbt_source)]
    assets = types.ModuleType("whole_body_tracking.assets")
    assets.ASSET_DIR = str(assets_dir)
    sys.modules.setdefault("whole_body_tracking", package)
    sys.modules["whole_body_tracking.assets"] = assets

    spec = importlib.util.spec_from_file_location("_omniretarget_g1_cfg", g1_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    cfg = module.G1_CYLINDER_CFG
    cfg.articulation_root_prim_path = args_cli.articulation_root_prim_path
    cfg.spawn = sim_utils.MjcfFileCfg(
        asset_path=str(assets_dir / "unitree_description" / "mjcf" / "g1.xml"),
        fix_base=False,
        import_sites=False,
        self_collision=True,
        force_usd_conversion=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    )
    return cfg


def _find_inputs(input_dir: Path, include_regex: str | None, max_files: int | None) -> list[Path]:
    files = sorted(input_dir.rglob("*.npz"))
    if include_regex is not None:
        import re

        compiled = re.compile(include_regex)
        files = [path for path in files if compiled.search(path.stem)]
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No raw qpos npz files found under {input_dir}")
    return files


def _read_scalar(raw: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in raw.files:
        return None
    value = np.asarray(raw[key])
    if value.shape == ():
        return str(value.item())
    return str(value.reshape(-1)[0])


def _robot_joint_names(robot) -> list[str]:
    return list(getattr(robot.data, "joint_names", getattr(robot, "joint_names", [])))


def _resample_qpos(qpos: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    if qpos.shape[0] <= 1 or abs(source_fps - target_fps) < 1e-6:
        return qpos.astype(np.float32, copy=False)
    duration = (qpos.shape[0] - 1) / source_fps
    times_out = np.arange(0.0, duration + 1e-9, 1.0 / target_fps, dtype=np.float64)
    source_positions = times_out * source_fps
    idx0 = np.floor(source_positions).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, qpos.shape[0] - 1)
    blend = (source_positions - idx0).astype(np.float32)

    out = np.empty((len(times_out), qpos.shape[1]), dtype=np.float32)
    out[:, 4:] = (1.0 - blend[:, None]) * qpos[idx0, 4:] + blend[:, None] * qpos[idx1, 4:]
    quat0 = torch.from_numpy(qpos[idx0, 0:4].astype(np.float32))
    quat1 = torch.from_numpy(qpos[idx1, 0:4].astype(np.float32))
    blend_t = torch.from_numpy(blend)
    out[:, 0:4] = torch.stack([quat_slerp(a, b, t) for a, b, t in zip(quat0, quat1, blend_t)]).numpy()
    out[:, 0:4] /= np.linalg.norm(out[:, 0:4], axis=1, keepdims=True).clip(min=1e-8)
    return out


def _compute_root_velocity(
    root_pos: torch.Tensor, root_quat: torch.Tensor, dt: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if root_pos.shape[0] == 1:
        return torch.zeros_like(root_pos), torch.zeros((1, 3), dtype=root_pos.dtype, device=root_pos.device)
    root_lin_vel = torch.gradient(root_pos, spacing=dt, dim=0)[0]
    if root_quat.shape[0] < 3:
        q_rel = quat_mul(root_quat[1:], quat_conjugate(root_quat[:-1]))
        omega = axis_angle_from_quat(q_rel) / dt
        root_ang_vel = torch.cat([omega[:1], omega[-1:]], dim=0)
    else:
        q_rel = quat_mul(root_quat[2:], quat_conjugate(root_quat[:-2]))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        root_ang_vel = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
    return root_lin_vel, root_ang_vel


def _motion_to_output(
    input_path: Path,
    output_path: Path,
    summary_root: Path,
    scene: InteractiveScene,
    sim: SimulationContext,
    qpos_joint_indexes: torch.Tensor,
    target_fps: float,
    no_render: bool,
) -> dict:
    robot = scene["robot"]
    with np.load(input_path, allow_pickle=False) as raw:
        qpos = np.asarray(raw["qpos"], dtype=np.float32)
        source_fps = float(np.asarray(raw["fps"], dtype=np.float32).reshape(-1)[0])
        terrain_id = _read_scalar(raw, "terrain_id")
        z_scale = _read_scalar(raw, "z_scale")
        terrain_stl = _read_scalar(raw, "terrain_stl")
        source_member = _read_scalar(raw, "source_member")

    terrain_stl_out = terrain_stl or ""
    if terrain_stl:
        terrain_path = Path(terrain_stl).expanduser()
        if not terrain_path.is_absolute():
            terrain_path = input_path.parent / terrain_path
        if terrain_path.exists():
            terrain_stl_out = os.path.relpath(terrain_path.resolve(), output_path.parent)

    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"{input_path}: expected qpos shape [T, 36], got {qpos.shape}")
    qpos = _resample_qpos(qpos, source_fps=source_fps, target_fps=target_fps)
    dt = 1.0 / target_fps
    root_pos = torch.as_tensor(qpos[:, 4:7], dtype=torch.float32, device=sim.device)
    root_quat = torch.as_tensor(qpos[:, 0:4], dtype=torch.float32, device=sim.device)
    root_quat = root_quat / torch.linalg.norm(root_quat, dim=-1, keepdim=True).clamp_min(1e-8)
    root_lin_vel, root_ang_vel = _compute_root_velocity(root_pos, root_quat, dt)
    qpos_joint_pos = torch.as_tensor(qpos[:, 7:], dtype=torch.float32, device=sim.device)
    qpos_joint_vel = (
        torch.gradient(qpos_joint_pos, spacing=dt, dim=0)[0]
        if qpos_joint_pos.shape[0] > 1
        else torch.zeros_like(qpos_joint_pos)
    )

    log = {
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    for frame in range(qpos.shape[0]):
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] = root_pos[frame]
        root_state[:, 3:7] = root_quat[frame]
        root_state[:, 7:10] = root_lin_vel[frame]
        root_state[:, 10:13] = root_ang_vel[frame]

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, qpos_joint_indexes] = qpos_joint_pos[frame]
        joint_vel[:, qpos_joint_indexes] = qpos_joint_vel[frame]

        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        scene.write_data_to_sim()
        if not no_render:
            sim.render()
        scene.update(dt)

        log["joint_pos"].append(robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32).copy())
        log["joint_vel"].append(robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float32).copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0].detach().cpu().numpy().astype(np.float32).copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0].detach().cpu().numpy().astype(np.float32).copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].detach().cpu().numpy().astype(np.float32).copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].detach().cpu().numpy().astype(np.float32).copy())

    output = {
        "fps": np.asarray([target_fps], dtype=np.float32),
        "source_fps": np.asarray([source_fps], dtype=np.float32),
        "source_qpos": qpos.astype(np.float32),
        "source_member": np.asarray(source_member or ""),
        "terrain_id": np.asarray(terrain_id or ""),
        "z_scale": np.asarray(z_scale or ""),
        "terrain_stl": np.asarray(terrain_stl_out),
        "joint_names": np.asarray(_robot_joint_names(robot)),
        "body_names": np.asarray(list(robot.body_names)),
        "qpos_joint_names": np.asarray(G1_URDF_QPOS_JOINT_NAMES),
    }
    output.update({key: np.stack(values, axis=0) for key, values in log.items()})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output)
    return {
        "input": os.path.relpath(input_path, summary_root),
        "output": os.path.relpath(output_path, summary_root),
        "source_fps": source_fps,
        "target_fps": target_fps,
        "frames": int(output["joint_pos"].shape[0]),
        "duration_s": (
            float((output["joint_pos"].shape[0] - 1) / target_fps) if output["joint_pos"].shape[0] > 1 else 0.0
        ),
        "terrain_stl": terrain_stl_out,
    }


def main() -> None:
    data_root = Path(args_cli.data_root).expanduser().resolve()
    input_dir = Path(args_cli.input_dir).expanduser().resolve() if args_cli.input_dir else data_root / "raw_qpos"
    output_dir = (
        Path(args_cli.output_dir).expanduser().resolve() if args_cli.output_dir else data_root / "tracking_npz_data"
    )
    input_files = _find_inputs(input_dir, args_cli.include_regex, args_cli.max_files)

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / float(args_cli.target_fps)
    sim = SimulationContext(sim_cfg)
    scene_cfg = ConversionSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = _load_g1_cylinder_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.update(sim.get_physics_dt())

    robot = scene["robot"]
    qpos_joint_indexes = torch.tensor(
        robot.find_joints(G1_URDF_QPOS_JOINT_NAMES, preserve_order=True)[0],
        dtype=torch.long,
        device=sim.device,
    )
    print("=== G1 joint order ===", flush=True)
    print("qpos order:", G1_URDF_QPOS_JOINT_NAMES, flush=True)
    print("isaac order:", _robot_joint_names(robot), flush=True)
    print("qpos->isaac indexes:", qpos_joint_indexes.detach().cpu().tolist(), flush=True)

    records = []
    for i, input_path in enumerate(input_files):
        output_path = output_dir / f"{input_path.stem}.npz"
        if output_path.exists() and not args_cli.force:
            print(f"[{i + 1}/{len(input_files)}] skip existing {output_path}", flush=True)
            continue
        print(f"[{i + 1}/{len(input_files)}] convert {input_path.name}", flush=True)
        records.append(
            _motion_to_output(
                input_path=input_path,
                output_path=output_path,
                summary_root=data_root,
                scene=scene,
                sim=sim,
                qpos_joint_indexes=qpos_joint_indexes,
                target_fps=float(args_cli.target_fps),
                no_render=bool(args_cli.no_render),
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if args_cli.dataset_txt:
        dataset_txt = Path(args_cli.dataset_txt).expanduser().resolve()
    elif args_cli.output_dir:
        dataset_txt = output_dir / "tracking_dataset.txt"
    else:
        dataset_txt = data_root / "tracking_dataset.txt"
    converted_files = sorted(output_dir.rglob("*.npz"))
    dataset_txt.parent.mkdir(parents=True, exist_ok=True)
    dataset_txt.write_text(
        "\n".join(os.path.relpath(path, output_dir) for path in converted_files) + ("\n" if converted_files else ""),
        encoding="utf-8",
    )
    if args_cli.summary_path:
        summary_path = Path(args_cli.summary_path).expanduser().resolve()
    elif args_cli.output_dir:
        summary_path = output_dir / "conversion_summary.json"
    else:
        summary_path = data_root / "conversion_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Converted {len(records)} motions into {output_dir}", flush=True)
    print(f"Dataset txt: {dataset_txt}", flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
