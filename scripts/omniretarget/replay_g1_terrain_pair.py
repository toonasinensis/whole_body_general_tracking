#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import trimesh

from isaaclab.app import AppLauncher

WBT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = WBT_ROOT / "data" / "omniretarget" / "g1_terrain"


def _load_paired_manifest_module():
    module_name = "_wbt_paired_manifest_for_replay"
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = (
        Path(__file__).resolve().parents[2]
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "terrains"
        / "paired_manifest.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load paired manifest module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


TerrainMotionPairManifest = _load_paired_manifest_module().TerrainMotionPairManifest


@dataclass(frozen=True)
class ResolvedPair:
    pair_index: int | None
    motion_file: Path
    terrain_file: Path
    name: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay paired OmniRetarget G1 terrain motions in IsaacLab.")
    parser.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--pairs_jsonl", default=None, help="Defaults to data_root/pairs.jsonl.")
    parser.add_argument("--pair_index", type=int, default=0, help="Index in pairs.jsonl to replay.")
    parser.add_argument(
        "--pair_indices",
        default=None,
        help="Batch selection, e.g. '0:16', '0,3,7', '0:32:2', or 'all'. Overrides --pair_index.",
    )
    parser.add_argument("--motion_file", default=None, help="Override motion npz path.")
    parser.add_argument("--terrain_file", default=None, help="Override terrain STL path.")
    parser.add_argument("--loop", action="store_true", help="Loop the selected motion.")
    parser.add_argument("--playback_speed", type=float, default=1.0)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None, help="Stop after this many replay frames.")
    parser.add_argument("--env_spacing", type=float, default=3.0, help="Spacing for batch visualization envs.")
    parser.add_argument("--terrain_size", type=float, nargs=2, default=(6.0, 6.0), help="Ground patch size per pair.")
    parser.add_argument("--ground_thickness", type=float, default=0.05, help="Ground patch thickness per pair.")
    parser.add_argument("--ground_z", type=float, default=0.0, help="Ground patch top surface z.")
    parser.add_argument("--no_ground", action="store_true", help="Disable the extra ground patch under each STL.")
    parser.add_argument(
        "--follow_env",
        type=int,
        default=None,
        help="Camera follows this env id. Default is overview for batch and env 0 for single pair.",
    )
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Only validate and summarize the selected mesh-motion pair without launching Isaac Sim.",
    )
    parser.add_argument(
        "--allow_stem_mismatch",
        action="store_true",
        help="Allow motion/STL/name stem mismatch when using manual overrides.",
    )
    parser.add_argument(
        "--articulation_root_prim_path",
        default=None,
        help="Optional relative articulation root path inside the spawned G1 prim.",
    )
    parser.add_argument(
        "--terrain_collision",
        action="store_true",
        help="Attach collision to the terrain mesh. Default is visual-only for robust dataset inspection.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _parse_pair_indices(spec: str, pair_count: int) -> list[int]:
    text = spec.strip().lower()
    if text == "all":
        return list(range(pair_count))

    indices: list[int] = []
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = part.split(":")
            if len(pieces) not in (2, 3):
                raise ValueError(f"Invalid pair range {part!r}; expected start:stop or start:stop:step")
            start = int(pieces[0]) if pieces[0] else 0
            stop = int(pieces[1]) if pieces[1] else pair_count
            step = int(pieces[2]) if len(pieces) == 3 and pieces[2] else 1
            if step <= 0:
                raise ValueError(f"Invalid pair range {part!r}; step must be positive")
            indices.extend(range(start, stop, step))
        else:
            indices.append(int(part))

    if not indices:
        raise ValueError("--pair_indices selected no pairs")
    bad = [index for index in indices if index < 0 or index >= pair_count]
    if bad:
        raise IndexError(f"pair indices out of range for {pair_count} pairs: {bad[:8]}")
    return indices


def _decode_names(raw: np.lib.npyio.NpzFile, key: str) -> list[str] | None:
    if key not in raw.files:
        return None
    values = np.asarray(raw[key]).reshape(-1).tolist()
    result = []
    for value in values:
        if isinstance(value, bytes):
            result.append(value.decode("utf-8"))
        else:
            result.append(str(value))
    return result


def _resolve_pairs(args: argparse.Namespace) -> list[ResolvedPair]:
    if args.motion_file or args.terrain_file:
        if not (args.motion_file and args.terrain_file):
            raise ValueError("--motion_file and --terrain_file must be provided together")
        if args.pair_indices:
            raise ValueError("--pair_indices cannot be combined with manual --motion_file/--terrain_file overrides")
        motion_file = Path(args.motion_file).expanduser().resolve()
        terrain_file = Path(args.terrain_file).expanduser().resolve()
        return [
            ResolvedPair(pair_index=None, motion_file=motion_file, terrain_file=terrain_file, name=motion_file.stem)
        ]

    pairs_jsonl = (
        Path(args.pairs_jsonl).expanduser().resolve()
        if args.pairs_jsonl
        else Path(args.data_root).expanduser().resolve() / "pairs.jsonl"
    )
    manifest = TerrainMotionPairManifest.from_jsonl(pairs_jsonl)
    if args.pair_indices:
        pair_indices = _parse_pair_indices(args.pair_indices, manifest.count)
    else:
        if not 0 <= args.pair_index < manifest.count:
            raise IndexError(f"--pair_index {args.pair_index} is out of range for {manifest.count} pairs")
        pair_indices = [args.pair_index]

    resolved_pairs = []
    for pair_index in pair_indices:
        pair = manifest.pairs[pair_index]
        resolved_pairs.append(
            ResolvedPair(
                pair_index=pair_index,
                motion_file=Path(pair.motion_path),
                terrain_file=Path(pair.terrain_path),
                name=pair.name,
            )
        )
    return resolved_pairs


def _resolve_pair(args: argparse.Namespace) -> tuple[Path, Path, str]:
    pair = _resolve_pairs(args)[0]
    return pair.motion_file, pair.terrain_file, pair.name


def _validate_resolved_pair(
    motion_file: Path, terrain_file: Path, name: str, *, allow_stem_mismatch: bool = False
) -> None:
    if not motion_file.is_file():
        raise FileNotFoundError(motion_file)
    if not terrain_file.is_file():
        raise FileNotFoundError(terrain_file)
    if allow_stem_mismatch:
        return
    if motion_file.stem != terrain_file.stem or name != motion_file.stem:
        raise ValueError(
            "Mesh-motion pair stem mismatch: "
            f"name={name!r}, motion={motion_file.stem!r}, terrain={terrain_file.stem!r}. "
            "Use --allow_stem_mismatch only for manual debugging."
        )


def _load_motion(motion_file: Path) -> dict:
    if not motion_file.is_file():
        raise FileNotFoundError(motion_file)
    with np.load(motion_file, allow_pickle=False) as raw:
        data = {
            "fps": float(np.asarray(raw["fps"], dtype=np.float32).reshape(-1)[0]),
            "joint_pos": np.asarray(raw["joint_pos"], dtype=np.float32),
            "joint_vel": np.asarray(raw["joint_vel"], dtype=np.float32),
            "body_pos_w": np.asarray(raw["body_pos_w"], dtype=np.float32),
            "body_quat_w": np.asarray(raw["body_quat_w"], dtype=np.float32),
            "body_lin_vel_w": np.asarray(raw["body_lin_vel_w"], dtype=np.float32),
            "body_ang_vel_w": np.asarray(raw["body_ang_vel_w"], dtype=np.float32),
            "joint_names": _decode_names(raw, "joint_names"),
            "body_names": _decode_names(raw, "body_names"),
        }
    return data


def _summarize_pair(
    motion_file: Path,
    terrain_file: Path,
    name: str,
    *,
    allow_stem_mismatch: bool = False,
    label: str | None = None,
) -> None:
    _validate_resolved_pair(motion_file, terrain_file, name, allow_stem_mismatch=allow_stem_mismatch)
    motion = _load_motion(motion_file)
    mesh = trimesh.load(terrain_file, force="mesh", process=False)
    if mesh.is_empty:
        raise ValueError(f"{terrain_file}: terrain mesh is empty")
    stem_match = motion_file.stem == terrain_file.stem == name
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    print(f"Pair summary{f' ({label})' if label else ''}", flush=True)
    print(f"  name: {name}", flush=True)
    print(f"  stem_match: {stem_match}", flush=True)
    print(f"  motion: {motion_file}", flush=True)
    print(f"  terrain: {terrain_file}", flush=True)
    print(
        f"  motion_frames: {motion['joint_pos'].shape[0]}, fps: {motion['fps']}, "
        f"joints: {motion['joint_pos'].shape[1]}, bodies: {motion['body_pos_w'].shape[1]}",
        flush=True,
    )
    print(
        f"  terrain_vertices: {len(mesh.vertices)}, faces: {len(mesh.faces)}, "
        f"bounds_min: {bounds[0].tolist()}, bounds_max: {bounds[1].tolist()}",
        flush=True,
    )


args_cli = _parse_args()
if args_cli.check_only:
    selected_pairs = _resolve_pairs(args_cli)
    print(f"Resolved {len(selected_pairs)} pair(s)", flush=True)
    for env_id, pair in enumerate(selected_pairs):
        pair_label = f"env={env_id:03d}"
        if pair.pair_index is not None:
            pair_label += f", pair={pair.pair_index:03d}"
        _summarize_pair(
            pair.motion_file,
            pair.terrain_file,
            pair.name,
            allow_stem_mismatch=bool(args_cli.allow_stem_mismatch),
            label=pair_label,
        )
    sys.exit(0)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.terrains.utils import create_prim_from_mesh
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class ReplayPairSceneCfg(InteractiveSceneCfg):
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(intensity=2500.0))
    robot: ArticulationCfg = None


@dataclass
class ReplayState:
    pair: ResolvedPair
    motion: dict
    root_body_index: int
    joint_indexes: torch.Tensor

    @property
    def frames(self) -> int:
        return int(self.motion["joint_pos"].shape[0])

    @property
    def fps(self) -> float:
        return float(self.motion["fps"])


def _load_g1_cylinder_cfg() -> ArticulationCfg:
    from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

    if args_cli.articulation_root_prim_path:
        return G1_CYLINDER_CFG.replace(articulation_root_prim_path=args_cli.articulation_root_prim_path)
    return G1_CYLINDER_CFG


def _spawn_visual_mesh(prim_path: str, mesh: trimesh.Trimesh, color: tuple[float, float, float]) -> None:
    prim = sim_utils.create_prim(
        f"{prim_path}/mesh",
        "Mesh",
        attributes={
            "points": mesh.vertices,
            "faceVertexIndices": mesh.faces.flatten(),
            "faceVertexCounts": np.asarray([3] * len(mesh.faces)),
            "subdivisionScheme": "bilinear",
        },
    )
    visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
    visual_material.func(f"{prim_path}/visualMaterial", visual_material)
    sim_utils.bind_visual_material(prim.GetPrimPath(), f"{prim_path}/visualMaterial")


def _spawn_terrain(
    terrain_file: Path,
    *,
    prim_path: str,
    enable_collision: bool,
    translation: np.ndarray | None = None,
    color: tuple[float, float, float] = (0.42, 0.46, 0.40),
) -> None:
    mesh = trimesh.load(terrain_file, force="mesh", process=False)
    if mesh.is_empty:
        raise ValueError(f"{terrain_file}: terrain mesh is empty")
    if not args_cli.no_ground:
        ground_size = tuple(float(value) for value in args_cli.terrain_size)
        ground_thickness = float(args_cli.ground_thickness)
        if ground_size[0] <= 0.0 or ground_size[1] <= 0.0:
            raise ValueError(f"--terrain_size must be positive, got {ground_size}")
        if ground_thickness <= 0.0:
            raise ValueError(f"--ground_thickness must be positive, got {ground_thickness}")
        ground = trimesh.creation.box(extents=(ground_size[0], ground_size[1], ground_thickness))
        ground.apply_translation((0.0, 0.0, float(args_cli.ground_z) - ground_thickness * 0.5))
        mesh = trimesh.util.concatenate([ground, mesh])
    if translation is not None:
        mesh = mesh.copy()
        mesh.apply_translation(np.asarray(translation, dtype=np.float64))
    print(
        f"Spawning terrain at {prim_path}: {terrain_file} "
        f"(vertices={len(mesh.vertices)}, faces={len(mesh.faces)}, collision={enable_collision})",
        flush=True,
    )
    if enable_collision:
        create_prim_from_mesh(
            prim_path,
            mesh,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
    else:
        sim_utils.create_prim(prim_path, "Xform")
        _spawn_visual_mesh(prim_path, mesh, color)


def _robot_joint_names(robot: Articulation) -> list[str]:
    return list(getattr(robot.data, "joint_names", getattr(robot, "joint_names", [])))


def _prepare_replay_state(sim: SimulationContext, robot: Articulation, pair: ResolvedPair) -> ReplayState:
    motion = _load_motion(pair.motion_file)
    body_names = motion["body_names"] or list(robot.body_names)
    if "pelvis" not in body_names:
        raise ValueError(f"{pair.motion_file}: body_names must include pelvis for root replay")
    root_body_index = body_names.index("pelvis")
    joint_names = motion["joint_names"] or _robot_joint_names(robot)
    joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]
    if len(joint_indexes) != len(joint_names):
        raise ValueError(
            f"{pair.motion_file}: only resolved {len(joint_indexes)} / {len(joint_names)} motion joints on robot"
        )
    joint_indexes_t = torch.as_tensor(joint_indexes, dtype=torch.long, device=sim.device)
    return ReplayState(pair=pair, motion=motion, root_body_index=root_body_index, joint_indexes=joint_indexes_t)


def _terrain_color(env_id: int) -> tuple[float, float, float]:
    palette = (
        (0.42, 0.46, 0.40),
        (0.48, 0.43, 0.36),
        (0.36, 0.45, 0.50),
        (0.44, 0.40, 0.50),
        (0.38, 0.50, 0.42),
    )
    return palette[env_id % len(palette)]


def _set_overview_camera(sim: SimulationContext, env_origins: torch.Tensor) -> None:
    origins = env_origins.detach().cpu().numpy()
    center = origins.mean(axis=0)
    span = np.ptp(origins, axis=0)
    radius = float(max(span[0], span[1], 2.0))
    eye = center + np.asarray([radius * 0.75 + 2.0, -radius * 0.95 - 2.0, max(2.5, radius * 0.55 + 1.8)])
    lookat = center + np.asarray([0.0, 0.0, 0.6])
    sim.set_camera_view(eye, lookat)


def _spawn_pair_terrains(scene: InteractiveScene, pairs: list[ResolvedPair], enable_collision: bool) -> None:
    if len(pairs) > 1:
        sim_utils.create_prim("/World/paired_terrains", "Xform")
    for env_id, pair in enumerate(pairs):
        env_origin = scene.env_origins[env_id].detach().cpu().numpy()
        prim_path = "/World/paired_terrain" if len(pairs) == 1 else f"/World/paired_terrains/env_{env_id:03d}"
        _spawn_terrain(
            pair.terrain_file,
            prim_path=prim_path,
            enable_collision=enable_collision,
            translation=env_origin,
            color=_terrain_color(env_id),
        )
        pair_text = f"pair={pair.pair_index:03d}" if pair.pair_index is not None else "pair=manual"
        print(f"  env={env_id:03d} {pair_text} name={pair.name}", flush=True)


def run_replays(sim: SimulationContext, scene: InteractiveScene, pairs: list[ResolvedPair]) -> None:
    robot: Articulation = scene["robot"]
    replays = [_prepare_replay_state(sim, robot, pair) for pair in pairs]

    frame_stride = max(1, int(args_cli.frame_stride))
    sim_dt = 1.0 / max(max(replay.fps for replay in replays) * float(args_cli.playback_speed), 1e-6)
    follow_env = args_cli.follow_env
    if follow_env is None and len(replays) == 1:
        follow_env = 0
    if follow_env is not None and not 0 <= follow_env < len(replays):
        raise IndexError(f"--follow_env {follow_env} is out of range for {len(replays)} envs")
    if follow_env is None:
        _set_overview_camera(sim, scene.env_origins)

    print(f"Replaying {len(replays)} pair(s)")
    for env_id, replay in enumerate(replays):
        pair = replay.pair
        pair_text = f"pair={pair.pair_index:03d}" if pair.pair_index is not None else "pair=manual"
        print(
            f"  env={env_id:03d} {pair_text} name={pair.name} "
            f"frames={replay.frames}, fps={replay.fps}, stride={frame_stride}"
        )

    frame_ids = [0 for _ in replays]
    played_frames = 0
    while simulation_app.is_running():
        root_state = robot.data.default_root_state.clone()
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        for env_id, replay in enumerate(replays):
            motion = replay.motion
            frame = min(frame_ids[env_id], replay.frames - 1)
            root_state[env_id, :3] = (
                torch.as_tensor(motion["body_pos_w"][frame, replay.root_body_index], device=sim.device)
                + scene.env_origins[env_id]
            )
            root_state[env_id, 3:7] = torch.as_tensor(
                motion["body_quat_w"][frame, replay.root_body_index], device=sim.device
            )
            root_state[env_id, 7:10] = torch.as_tensor(
                motion["body_lin_vel_w"][frame, replay.root_body_index], device=sim.device
            )
            root_state[env_id, 10:13] = torch.as_tensor(
                motion["body_ang_vel_w"][frame, replay.root_body_index], device=sim.device
            )
            joint_pos[env_id, replay.joint_indexes] = torch.as_tensor(motion["joint_pos"][frame], device=sim.device)
            joint_vel[env_id, replay.joint_indexes] = torch.as_tensor(motion["joint_vel"][frame], device=sim.device)

        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)

        if follow_env is not None:
            lookat = root_state[follow_env, :3].detach().cpu().numpy()
            sim.set_camera_view(lookat + np.asarray([2.0, -2.0, 0.8]), lookat)

        any_active = False
        for env_id, replay in enumerate(replays):
            next_frame = frame_ids[env_id] + frame_stride
            if next_frame >= replay.frames:
                if args_cli.loop:
                    frame_ids[env_id] = next_frame % replay.frames
                    any_active = True
                else:
                    frame_ids[env_id] = replay.frames - 1
            else:
                frame_ids[env_id] = next_frame
                any_active = True
        played_frames += 1
        if args_cli.max_frames is not None and played_frames >= args_cli.max_frames:
            break
        if not args_cli.loop and not any_active:
            break


def main() -> None:
    pairs = _resolve_pairs(args_cli)
    print(f"Resolved {len(pairs)} pair(s)", flush=True)
    for env_id, pair in enumerate(pairs):
        pair_label = f"env={env_id:03d}"
        if pair.pair_index is not None:
            pair_label += f", pair={pair.pair_index:03d}"
        _summarize_pair(
            pair.motion_file,
            pair.terrain_file,
            pair.name,
            allow_stem_mismatch=bool(args_cli.allow_stem_mismatch),
            label=pair_label,
        )
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    print("Creating simulation context", flush=True)
    sim = SimulationContext(sim_cfg)
    scene_cfg = ReplayPairSceneCfg(num_envs=len(pairs), env_spacing=float(args_cli.env_spacing))
    scene_cfg.robot = _load_g1_cylinder_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
    print("Creating interactive scene", flush=True)
    scene = InteractiveScene(scene_cfg)
    _spawn_pair_terrains(scene, pairs, enable_collision=bool(args_cli.terrain_collision))
    print("Resetting simulation", flush=True)
    sim.reset()
    scene.update(sim.get_physics_dt())
    print("Starting replay", flush=True)
    run_replays(sim, scene, pairs)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
