#!/usr/bin/env python3
"""Viser viewer for paired G1 terrain/action data.

This is intentionally independent of IsaacLab so pair inspection is fast:
it reads pairs.jsonl, displays the selected STL terrain, and replays the
matching tracking npz on the G1 URDF.

Example:
    python scripts/omniretarget/visualize_g1_terrain_pairs_viser.py \
        --pairs_jsonl data/omniretarget/g1_terrain/pairs_z_scale_1.0.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import numpy as np
import re
import shutil
import sys
import time
import trimesh
from dataclasses import dataclass
from pathlib import Path

try:
    import viser
    import viser.extras
    import yourdfpy
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "viser, trimesh, and yourdfpy are required.\n"
        "Install in your viewer env, e.g. python -m pip install viser yourdfpy trimesh"
    ) from exc


WBT_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = WBT_ROOT / "source/whole_body_tracking/whole_body_tracking/assets"
G1_URDF_PATH = ASSET_DIR / "unitree_description/urdf/g1/main.urdf"
DEFAULT_PAIRS_JSONL = "data/omniretarget/g1_terrain/pairs_z_scale_1.0.jsonl"
FALLBACK_PAIRS_JSONL = "data/omniretarget/g1_terrain/pairs.jsonl"

G1_SKELETON_EDGES = [
    ("pelvis", "torso_link"),
    ("pelvis", "left_hip_roll_link"),
    ("left_hip_roll_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_roll_link"),
    ("pelvis", "right_hip_roll_link"),
    ("right_hip_roll_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_roll_link"),
    ("torso_link", "left_shoulder_roll_link"),
    ("left_shoulder_roll_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_yaw_link"),
    ("torso_link", "right_shoulder_roll_link"),
    ("right_shoulder_roll_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_yaw_link"),
]


def _load_paired_manifest_module():
    module_name = "_wbt_paired_manifest_for_viser"
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = WBT_ROOT / "source/whole_body_tracking/whole_body_tracking/terrains/paired_manifest.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load paired manifest module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_paired_manifest = _load_paired_manifest_module()
TerrainMotionPairManifest = _paired_manifest.TerrainMotionPairManifest
resolve_repo_path = _paired_manifest.resolve_repo_path


@dataclass(frozen=True)
class PairInfo:
    index: int
    name: str
    motion_file: Path
    terrain_file: Path


@dataclass
class MotionData:
    npz: np.lib.npyio.NpzFile
    fps: float
    joint_pos: np.ndarray
    body_pos: np.ndarray
    body_quat: np.ndarray
    height_scan: np.ndarray | None
    height_scan_points: np.ndarray | None
    height_scan_grid_size: tuple[float, float]
    height_scan_grid_resolution: float
    height_scan_offset_xy: tuple[float, float]
    height_scan_obs_offset: float
    joint_names: list[str]
    body_names: list[str]
    root_body_index: int
    joint_to_urdf_perm: np.ndarray | None


@dataclass
class SceneHandles:
    terrain: object | None = None
    terrain_wire: object | None = None
    robot_frame: object | None = None
    robot_urdf: object | None = None
    skeleton: object | None = None
    body_points: object | None = None
    height_scan_points: object | None = None
    root_frame: object | None = None
    label: object | None = None


@dataclass
class LoadedScene:
    pair: PairInfo
    motion: MotionData
    mesh_bounds: np.ndarray
    world_offset: np.ndarray
    handles: SceneHandles
    edge_indices: np.ndarray | None


def parse_args() -> argparse.Namespace:
    default_pairs = DEFAULT_PAIRS_JSONL
    if not resolve_repo_path(default_pairs).is_file():
        default_pairs = FALLBACK_PAIRS_JSONL

    parser = argparse.ArgumentParser(description="Inspect G1 terrain-motion pairs with viser.")
    parser.add_argument("--pairs_jsonl", type=str, default=default_pairs)
    parser.add_argument("--pair_index", type=int, default=0, help="Initial pair index.")
    parser.add_argument("--pair_limit", type=int, default=None)
    parser.add_argument("--urdf_path", type=str, default=str(G1_URDF_PATH))
    parser.add_argument("--skeleton", action="store_true", help="Use skeleton only instead of URDF mesh.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--playback_speed", type=float, default=1.0)
    parser.add_argument("--start_paused", action="store_true")
    parser.add_argument("--target_fps", type=float, default=30.0)
    parser.add_argument("--terrain_opacity", type=float, default=0.82)
    parser.add_argument("--terrain_wireframe", action="store_true")
    parser.add_argument("--show_height_scan", action="store_true", help="Display saved obs.terrain.height_scan points.")
    parser.add_argument("--no_ground", action="store_true")
    parser.add_argument("--ground_size", type=float, nargs=2, default=(6.0, 6.0))
    parser.add_argument("--ground_thickness", type=float, default=0.04)
    parser.add_argument("--ground_z", type=float, default=0.0)
    parser.add_argument("--no_center", action="store_true", help="Do not recenter terrain/motion around the grid.")
    parser.add_argument(
        "--crop_output_dir",
        type=str,
        default=None,
        help="Directory for cropped motion-terrain pairs. Defaults to <pairs_jsonl_dir>/cropped_pairs.",
    )
    parser.add_argument("--crop_manifest_name", type=str, default="pairs_cropped.jsonl")
    parser.add_argument("--crop_name_suffix", type=str, default="crop")
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Validate pairs and print summaries without starting the viser server.",
    )
    return parser.parse_args()


def _decode_string_array(raw: np.lib.npyio.NpzFile, key: str) -> list[str]:
    if key not in raw.files:
        return []
    values = np.asarray(raw[key]).reshape(-1).tolist()
    result = []
    for value in values:
        if isinstance(value, bytes):
            result.append(value.decode("utf-8"))
        else:
            result.append(str(value))
    return result


def _pair_options(pairs: list[PairInfo]) -> list[str]:
    return [f"{pair.index:03d} {pair.name}" for pair in pairs]


def _safe_stem(text: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    stem = re.sub(r"_+", "_", stem).strip("._")
    return stem or "pair"


def _collect_pairs(pairs_jsonl: str, pair_limit: int | None) -> list[PairInfo]:
    manifest = TerrainMotionPairManifest.from_jsonl(pairs_jsonl, limit=pair_limit)
    return [
        PairInfo(
            index=i,
            name=pair.name,
            motion_file=Path(pair.motion_path),
            terrain_file=Path(pair.terrain_path),
        )
        for i, pair in enumerate(manifest.pairs)
    ]


def _load_motion(pair: PairInfo, urdf_joint_names: list[str] | None) -> MotionData:
    raw = np.load(pair.motion_file, allow_pickle=True, mmap_mode="r")
    for key in ("fps", "joint_pos", "body_pos_w", "body_quat_w"):
        if key not in raw.files:
            raw.close()
            raise KeyError(f"{pair.motion_file}: missing {key}")

    joint_names = _decode_string_array(raw, "joint_names")
    body_names = _decode_string_array(raw, "body_names")
    if not body_names:
        raw.close()
        raise KeyError(f"{pair.motion_file}: body_names is required for terrain-pair visualization")
    if not joint_names:
        raw.close()
        raise KeyError(f"{pair.motion_file}: joint_names is required for URDF joint mapping")

    root_body_name = "pelvis" if "pelvis" in body_names else body_names[0]
    root_body_index = body_names.index(root_body_name)

    joint_to_urdf_perm = None
    if urdf_joint_names is not None:
        npz_joint_index = {name: i for i, name in enumerate(joint_names)}
        missing = [name for name in urdf_joint_names if name not in npz_joint_index]
        if missing:
            raw.close()
            raise ValueError(f"{pair.motion_file}: URDF joints missing from npz joint_names: {missing}")
        joint_to_urdf_perm = np.asarray([npz_joint_index[name] for name in urdf_joint_names], dtype=np.int64)

    height_scan = None
    if "height_scan" in raw.files:
        height_scan = np.asarray(raw["height_scan"], dtype=np.float32)
        if height_scan.ndim > 2:
            height_scan = height_scan.reshape(height_scan.shape[0], -1)
    height_scan_points = None
    if "height_scan_points" in raw.files:
        height_scan_points = np.asarray(raw["height_scan_points"], dtype=np.float32)
    grid_size = (1.6, 1.0)
    if "height_scan_grid_size" in raw.files:
        values = np.asarray(raw["height_scan_grid_size"], dtype=np.float32).reshape(-1)
        if values.size >= 2:
            grid_size = (float(values[0]), float(values[1]))
    grid_resolution = 0.1
    if "height_scan_grid_resolution" in raw.files:
        values = np.asarray(raw["height_scan_grid_resolution"], dtype=np.float32).reshape(-1)
        if values.size >= 1:
            grid_resolution = float(values[0])
    grid_offset = (0.80, 0.0)
    if "height_scan_offset_xy" in raw.files:
        values = np.asarray(raw["height_scan_offset_xy"], dtype=np.float32).reshape(-1)
        if values.size >= 2:
            grid_offset = (float(values[0]), float(values[1]))
    obs_offset = 0.5
    if "height_scan_obs_offset" in raw.files:
        values = np.asarray(raw["height_scan_obs_offset"], dtype=np.float32).reshape(-1)
        if values.size >= 1:
            obs_offset = float(values[0])

    return MotionData(
        npz=raw,
        fps=float(np.asarray(raw["fps"], dtype=np.float32).reshape(-1)[0]),
        joint_pos=np.asarray(raw["joint_pos"], dtype=np.float32),
        body_pos=np.asarray(raw["body_pos_w"], dtype=np.float32),
        body_quat=np.asarray(raw["body_quat_w"], dtype=np.float32),
        height_scan=height_scan,
        height_scan_points=height_scan_points,
        height_scan_grid_size=grid_size,
        height_scan_grid_resolution=grid_resolution,
        height_scan_offset_xy=grid_offset,
        height_scan_obs_offset=obs_offset,
        joint_names=joint_names,
        body_names=body_names,
        root_body_index=root_body_index,
        joint_to_urdf_perm=joint_to_urdf_perm,
    )


def _load_terrain_mesh(pair: PairInfo, args: argparse.Namespace) -> trimesh.Trimesh:
    mesh = trimesh.load(pair.terrain_file, force="mesh", process=False)
    if mesh.is_empty:
        raise ValueError(f"{pair.terrain_file}: terrain mesh is empty")

    if not args.no_ground:
        size_x, size_y = (float(args.ground_size[0]), float(args.ground_size[1]))
        thickness = float(args.ground_thickness)
        ground = trimesh.creation.box(extents=(size_x, size_y, thickness))
        ground.apply_translation((0.0, 0.0, float(args.ground_z) - thickness * 0.5))
        mesh = trimesh.util.concatenate([ground, mesh])
    return mesh


def _build_edge_indices(body_names: list[str]) -> np.ndarray | None:
    name_to_index = {name: i for i, name in enumerate(body_names)}
    edges = [
        (name_to_index[a], name_to_index[b]) for a, b in G1_SKELETON_EDGES if a in name_to_index and b in name_to_index
    ]
    if not edges:
        return None
    return np.asarray(edges, dtype=np.int64)


def _quat_wxyz_to_yaw(quat: np.ndarray) -> float:
    w, x, y, z = (float(v) for v in quat)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _height_scan_local_xy(motion: MotionData) -> np.ndarray | None:
    if motion.height_scan is None:
        return None
    count = int(motion.height_scan.shape[1])
    if count <= 0:
        return None
    size_x, size_y = motion.height_scan_grid_size
    resolution = max(float(motion.height_scan_grid_resolution), 1.0e-6)
    num_x = int(round(float(size_x) / resolution)) + 1
    num_y = int(round(float(size_y) / resolution)) + 1
    if num_x * num_y != count:
        num_x = int(round(np.sqrt(count)))
        while num_x > 1 and count % num_x != 0:
            num_x -= 1
        num_y = max(1, count // max(num_x, 1))
    xs = (np.arange(num_x, dtype=np.float32) - (num_x - 1) * 0.5) * resolution + motion.height_scan_offset_xy[0]
    ys = (np.arange(num_y, dtype=np.float32) - (num_y - 1) * 0.5) * resolution + motion.height_scan_offset_xy[1]
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    xy = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)
    return xy[:count]


def _height_scan_points_for_frame(motion: MotionData, frame: int, world_offset: np.ndarray) -> np.ndarray | None:
    if motion.height_scan_points is not None:
        return motion.height_scan_points[frame].astype(np.float32, copy=False) + world_offset[None, :]
    local_xy = _height_scan_local_xy(motion)
    if motion.height_scan is None or local_xy is None:
        return None
    try:
        scanner_body_index = motion.body_names.index("torso_link")
    except ValueError:
        scanner_body_index = motion.root_body_index
    scanner_body = motion.body_pos[frame, scanner_body_index]
    yaw = _quat_wxyz_to_yaw(motion.body_quat[frame, scanner_body_index])
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    xy = scanner_body[:2][None, :] + local_xy @ rot.T
    z = scanner_body[2] - motion.height_scan[frame, : local_xy.shape[0]] - float(motion.height_scan_obs_offset)
    return np.concatenate([xy, z[:, None]], axis=-1).astype(np.float32) + world_offset[None, :]


def _mesh_color(pair_index: int) -> tuple[int, int, int]:
    palette = (
        (120, 152, 136),
        (150, 132, 110),
        (118, 142, 170),
        (142, 124, 164),
        (116, 158, 116),
    )
    return palette[pair_index % len(palette)]


def _terrain_status(pair: PairInfo, motion: MotionData, mesh: trimesh.Trimesh) -> str:
    frames = int(motion.joint_pos.shape[0])
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    return (
        f"pair={pair.index:03d}\n"
        f"name={pair.name}\n"
        f"frames={frames}, fps={motion.fps:.1f}\n"
        f"motion={pair.motion_file.name}\n"
        f"terrain={pair.terrain_file.name}\n"
        f"bounds_min={np.round(bounds[0], 3).tolist()}\n"
        f"bounds_max={np.round(bounds[1], 3).tolist()}"
    )


def _crop_pair(
    pair: PairInfo,
    *,
    start_frame: int,
    end_frame: int,
    output_dir: Path,
    manifest_name: str,
    name_suffix: str,
) -> tuple[Path, Path, Path, str]:
    with np.load(pair.motion_file, allow_pickle=True) as raw:
        frame_count = int(np.asarray(raw["joint_pos"]).shape[0])
        start = max(0, min(int(start_frame), frame_count - 1))
        end = max(start + 1, min(int(end_frame), frame_count))
        if end <= start:
            raise ValueError(f"Invalid crop range [{start}, {end}) for {pair.name} with {frame_count} frames")

        suffix = _safe_stem(name_suffix)
        crop_name = _safe_stem(f"{pair.name}_{suffix}_{start:05d}_{end:05d}")
        motion_dir = output_dir / "tracking_npz_data"
        terrain_dir = output_dir / "terrains" / "stl"
        motion_dir.mkdir(parents=True, exist_ok=True)
        terrain_dir.mkdir(parents=True, exist_ok=True)

        motion_out = motion_dir / f"{crop_name}.npz"
        terrain_out = terrain_dir / f"{crop_name}{pair.terrain_file.suffix.lower() or '.stl'}"

        arrays = {}
        for key in raw.files:
            value = raw[key]
            if value.ndim > 0 and value.shape[0] == frame_count:
                arrays[key] = np.asarray(value[start:end])
            else:
                arrays[key] = np.asarray(value)
        arrays["frames"] = np.asarray([end - start], dtype=np.int32)
        if "fps" in arrays:
            fps = float(np.asarray(arrays["fps"], dtype=np.float32).reshape(-1)[0])
        else:
            fps = 30.0
            arrays["fps"] = np.asarray([fps], dtype=np.float32)
        arrays["duration_s"] = np.asarray([(end - start) / max(fps, 1.0e-6)], dtype=np.float32)
        arrays["crop_source_motion"] = np.asarray([str(pair.motion_file)])
        arrays["crop_source_terrain"] = np.asarray([str(pair.terrain_file)])
        arrays["crop_source_pair_name"] = np.asarray([pair.name])
        arrays["crop_start_frame"] = np.asarray([start], dtype=np.int32)
        arrays["crop_end_frame"] = np.asarray([end], dtype=np.int32)

        np.savez_compressed(motion_out, **arrays)

    shutil.copy2(pair.terrain_file, terrain_out)

    manifest_path = output_dir / manifest_name
    record = {
        "name": crop_name,
        "tracking_motion": str(motion_out.relative_to(output_dir)),
        "terrain_stl": str(terrain_out.relative_to(output_dir)),
        "source_name": pair.name,
        "source_motion": str(pair.motion_file),
        "source_terrain": str(pair.terrain_file),
        "crop_start_frame": start,
        "crop_end_frame": end,
        "frames": end - start,
        "duration_s": (end - start) / max(fps, 1.0e-6),
    }
    with manifest_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    return motion_out, terrain_out, manifest_path, crop_name


def _remove_loaded_scene(loaded: LoadedScene | None) -> None:
    if loaded is None:
        return
    handles = loaded.handles
    for handle in (
        handles.terrain,
        handles.terrain_wire,
        handles.robot_urdf,
        handles.robot_frame,
        handles.skeleton,
        handles.body_points,
        handles.height_scan_points,
        handles.root_frame,
        handles.label,
    ):
        if handle is None:
            continue
        try:
            handle.remove()
        except Exception:
            pass
    loaded.motion.npz.close()


def _create_loaded_scene(
    server: viser.ViserServer,
    pair: PairInfo,
    args: argparse.Namespace,
    urdf_path: Path,
    urdf_joint_names: list[str] | None,
    *,
    show_skeleton: bool,
    show_labels: bool,
) -> LoadedScene:
    motion = _load_motion(pair, None if args.skeleton else urdf_joint_names)
    mesh = _load_terrain_mesh(pair, args)
    mesh_bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center_xy = mesh_bounds.mean(axis=0)
    world_offset = np.zeros(3, dtype=np.float32)
    if not args.no_center:
        world_offset[:2] = -center_xy[:2].astype(np.float32)

    shifted_mesh = mesh.copy()
    shifted_mesh.apply_translation(world_offset.astype(np.float64))

    handles = SceneHandles()
    color = _mesh_color(pair.index)
    handles.terrain = server.scene.add_mesh_simple(
        "/pair/terrain",
        vertices=np.asarray(shifted_mesh.vertices, dtype=np.float32),
        faces=np.asarray(shifted_mesh.faces, dtype=np.uint32),
        color=color,
        opacity=float(args.terrain_opacity),
        flat_shading=False,
        side="double",
    )
    if args.terrain_wireframe:
        handles.terrain_wire = server.scene.add_mesh_simple(
            "/pair/terrain_wire",
            vertices=np.asarray(shifted_mesh.vertices, dtype=np.float32),
            faces=np.asarray(shifted_mesh.faces, dtype=np.uint32),
            color=(30, 35, 38),
            opacity=0.35,
            wireframe=True,
            flat_shading=True,
            side="double",
        )

    first_root = motion.body_pos[0, motion.root_body_index] + world_offset
    first_quat = motion.body_quat[0, motion.root_body_index]
    handles.robot_frame = server.scene.add_frame(
        "/pair/robot",
        position=tuple(first_root.tolist()),
        wxyz=tuple(first_quat.tolist()),
        show_axes=False,
    )
    if not args.skeleton:
        handles.robot_urdf = viser.extras.ViserUrdf(
            server,
            urdf_path,
            root_node_name="/pair/robot",
        )
        if motion.joint_to_urdf_perm is not None:
            handles.robot_urdf.update_cfg(motion.joint_pos[0, motion.joint_to_urdf_perm])

    edge_indices = _build_edge_indices(motion.body_names)
    body0 = motion.body_pos[0] + world_offset
    if edge_indices is not None:
        handles.skeleton = server.scene.add_line_segments(
            "/pair/skeleton",
            points=body0[edge_indices],
            colors=np.asarray([255, 210, 70], dtype=np.uint8),
            line_width=3.0,
            visible=bool(args.skeleton or show_skeleton),
        )
    handles.body_points = server.scene.add_point_cloud(
        "/pair/body_points",
        points=body0,
        colors=np.asarray([255, 235, 160], dtype=np.uint8),
        point_size=0.035,
        visible=bool(args.skeleton or show_skeleton),
    )
    height_points0 = _height_scan_points_for_frame(motion, 0, world_offset)
    if height_points0 is not None:
        handles.height_scan_points = server.scene.add_point_cloud(
            "/pair/height_scan",
            points=height_points0,
            colors=np.asarray([80, 190, 255], dtype=np.uint8),
            point_size=0.025,
            visible=bool(args.show_height_scan),
        )
    handles.root_frame = server.scene.add_frame(
        "/pair/root_frame",
        position=tuple(first_root.tolist()),
        wxyz=tuple(first_quat.tolist()),
        axes_length=0.35,
        axes_radius=0.012,
        show_axes=True,
    )

    label_pos = np.asarray([mesh_bounds[1, 0], mesh_bounds[1, 1], mesh_bounds[1, 2] + 0.45], dtype=np.float32)
    label_pos += world_offset
    handles.label = server.scene.add_label(
        "/pair/label",
        text=_terrain_status(pair, motion, mesh),
        position=tuple(label_pos.tolist()),
        font_size_mode="screen",
        font_screen_scale=0.85,
        depth_test=False,
        anchor="bottom-left",
        visible=show_labels,
    )

    print(_terrain_status(pair, motion, mesh), flush=True)
    return LoadedScene(
        pair=pair,
        motion=motion,
        mesh_bounds=mesh_bounds,
        world_offset=world_offset,
        handles=handles,
        edge_indices=edge_indices,
    )


def _summarize_pairs(pairs: list[PairInfo], limit: int | None = None) -> None:
    print(f"Resolved {len(pairs)} pair(s)")
    shown = pairs if limit is None else pairs[:limit]
    for pair in shown:
        with np.load(pair.motion_file, allow_pickle=True, mmap_mode="r") as raw:
            frames = int(np.asarray(raw["joint_pos"]).shape[0])
            fps = float(np.asarray(raw["fps"], dtype=np.float32).reshape(-1)[0])
        mesh = trimesh.load(pair.terrain_file, force="mesh", process=False)
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        print(
            f"  pair={pair.index:03d} name={pair.name} frames={frames} fps={fps:.1f} "
            f"terrain={pair.terrain_file.name} bounds_min={np.round(bounds[0], 3).tolist()} "
            f"bounds_max={np.round(bounds[1], 3).tolist()}"
        )


def main() -> None:  # noqa: C901
    args = parse_args()
    pairs_jsonl_path = resolve_repo_path(args.pairs_jsonl)
    crop_output_dir = (
        Path(args.crop_output_dir).expanduser().resolve()
        if args.crop_output_dir is not None
        else pairs_jsonl_path.parent / "cropped_pairs"
    )
    pairs = _collect_pairs(args.pairs_jsonl, args.pair_limit)
    if not 0 <= args.pair_index < len(pairs):
        raise IndexError(f"--pair_index {args.pair_index} is out of range for {len(pairs)} pairs")

    if args.check_only:
        _summarize_pairs(pairs)
        return

    urdf_path = Path(args.urdf_path).expanduser().resolve()
    urdf_joint_names: list[str] | None = None
    if not args.skeleton:
        if not urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        probe_urdf = yourdfpy.URDF.load(str(urdf_path), load_meshes=False, build_collision_scene_graph=False)
        urdf_joint_names = list(probe_urdf.actuated_joint_names)
        print(f"[INFO] URDF: {urdf_path}")
        print(f"[INFO] URDF actuated joints: {len(urdf_joint_names)}")

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.add_grid(
        "/ground_grid",
        width=10.0,
        height=10.0,
        cell_size=0.25,
        section_size=1.0,
        plane="xy",
        plane_opacity=0.0,
    )

    pair_options = _pair_options(pairs)
    state = {
        "pair_index": int(args.pair_index),
        "loaded": None,
        "frame_cursor": 0.0,
        "last_frame": -1,
        "suppress_dropdown": False,
    }

    with server.gui.add_folder("pair"):
        pair_dropdown = server.gui.add_dropdown(
            "pair",
            options=pair_options,
            initial_value=pair_options[args.pair_index],
        )
        prev_button = server.gui.add_button("prev")
        next_button = server.gui.add_button("next")
        reset_button = server.gui.add_button("reset frame")
        selected_text = server.gui.add_text("selected", initial_value="")
    with server.gui.add_folder("playback"):
        playing = server.gui.add_checkbox("playing", initial_value=not bool(args.start_paused))
        speed = server.gui.add_slider("speed", min=0.05, max=4.0, step=0.05, initial_value=float(args.playback_speed))
        show_skeleton = server.gui.add_checkbox("show skeleton", initial_value=bool(args.skeleton))
        show_height_scan = server.gui.add_checkbox("show height scan", initial_value=bool(args.show_height_scan))
        show_labels = server.gui.add_checkbox("show label", initial_value=True)
        frame_text = server.gui.add_text("frame", initial_value="0 / 0")
    with server.gui.add_folder("crop"):
        crop_start = server.gui.add_number("start frame", initial_value=0, min=0, step=1)
        crop_end = server.gui.add_number("end frame exclusive", initial_value=1, min=1, step=1)
        crop_suffix = server.gui.add_text("name suffix", initial_value=str(args.crop_name_suffix))
        set_start_button = server.gui.add_button("set start = current")
        set_end_button = server.gui.add_button("set end = current+1")
        save_crop_button = server.gui.add_button("save crop")
        crop_status = server.gui.add_text(
            "output",
            initial_value=f"manifest: {crop_output_dir / args.crop_manifest_name}",
        )

    def load_pair(pair_index: int) -> None:
        pair_index = int(pair_index) % len(pairs)
        if state["loaded"] is not None and pair_index == state["pair_index"]:
            return
        _remove_loaded_scene(state["loaded"])
        state["loaded"] = _create_loaded_scene(
            server,
            pairs[pair_index],
            args,
            urdf_path,
            urdf_joint_names,
            show_skeleton=bool(show_skeleton.value),
            show_labels=bool(show_labels.value),
        )
        state["pair_index"] = pair_index
        state["frame_cursor"] = 0.0
        state["last_frame"] = -1
        state["suppress_dropdown"] = True
        pair_dropdown.value = pair_options[pair_index]
        state["suppress_dropdown"] = False
        selected_text.value = (
            f"pair={pairs[pair_index].index:03d}  name={pairs[pair_index].name}\n"
            f"motion={pairs[pair_index].motion_file.name}\n"
            f"terrain={pairs[pair_index].terrain_file.name}"
        )
        if state["loaded"].motion.height_scan is not None:
            selected_text.value += f"\nheight_scan={state['loaded'].motion.height_scan.shape}"
        frame_count = int(state["loaded"].motion.joint_pos.shape[0])
        crop_start.value = 0
        crop_end.value = frame_count

    def set_pair_from_dropdown() -> None:
        value = str(pair_dropdown.value)
        try:
            pair_index = pair_options.index(value)
        except ValueError:
            return
        load_pair(pair_index)

    @pair_dropdown.on_update
    def _on_pair_dropdown(_event) -> None:
        if state["suppress_dropdown"]:
            return
        set_pair_from_dropdown()

    @prev_button.on_click
    def _on_prev(_event) -> None:
        load_pair(state["pair_index"] - 1)

    @next_button.on_click
    def _on_next(_event) -> None:
        load_pair(state["pair_index"] + 1)

    @reset_button.on_click
    def _on_reset(_event) -> None:
        state["frame_cursor"] = 0.0
        state["last_frame"] = -1

    @set_start_button.on_click
    def _on_set_crop_start(_event) -> None:
        loaded: LoadedScene | None = state["loaded"]
        if loaded is None:
            return
        frame_count = int(loaded.motion.joint_pos.shape[0])
        frame = int(state["frame_cursor"]) % max(frame_count, 1)
        crop_start.value = frame
        if int(crop_end.value) <= frame:
            crop_end.value = min(frame + 1, frame_count)

    @set_end_button.on_click
    def _on_set_crop_end(_event) -> None:
        loaded: LoadedScene | None = state["loaded"]
        if loaded is None:
            return
        frame_count = int(loaded.motion.joint_pos.shape[0])
        frame = int(state["frame_cursor"]) % max(frame_count, 1)
        crop_end.value = min(frame + 1, frame_count)
        if int(crop_start.value) >= int(crop_end.value):
            crop_start.value = max(0, int(crop_end.value) - 1)

    @save_crop_button.on_click
    def _on_save_crop(_event) -> None:
        loaded: LoadedScene | None = state["loaded"]
        if loaded is None:
            return
        try:
            motion_out, terrain_out, manifest_out, crop_name = _crop_pair(
                loaded.pair,
                start_frame=int(crop_start.value),
                end_frame=int(crop_end.value),
                output_dir=crop_output_dir,
                manifest_name=str(args.crop_manifest_name),
                name_suffix=str(crop_suffix.value),
            )
        except Exception as exc:
            crop_status.value = f"ERROR: {exc}"
            print(f"[CROP][ERROR] {exc}", flush=True)
            return

        crop_status.value = f"saved {crop_name}\nmotion={motion_out}\nterrain={terrain_out}\nmanifest={manifest_out}"
        print(
            f"[CROP] saved {crop_name}: motion={motion_out} terrain={terrain_out} manifest={manifest_out}",
            flush=True,
        )

    @show_skeleton.on_update
    def _on_show_skeleton(_event) -> None:
        loaded: LoadedScene | None = state["loaded"]
        if loaded is None or args.skeleton:
            return
        visible = bool(show_skeleton.value)
        if loaded.handles.skeleton is not None:
            loaded.handles.skeleton.visible = visible
        if loaded.handles.body_points is not None:
            loaded.handles.body_points.visible = visible

    @show_labels.on_update
    def _on_show_labels(_event) -> None:
        loaded: LoadedScene | None = state["loaded"]
        if loaded is not None and loaded.handles.label is not None:
            loaded.handles.label.visible = bool(show_labels.value)

    @show_height_scan.on_update
    def _on_show_height_scan(_event) -> None:
        loaded: LoadedScene | None = state["loaded"]
        if loaded is not None and loaded.handles.height_scan_points is not None:
            loaded.handles.height_scan_points.visible = bool(show_height_scan.value)

    load_pair(args.pair_index)
    print(f"[INFO] viser running at http://localhost:{args.port}")
    print("[INFO] Use the pair dropdown or prev/next buttons to inspect terrain-action pairs.")

    target_dt = 1.0 / max(float(args.target_fps), 1.0)
    t_prev = time.time()
    try:
        while True:
            loaded: LoadedScene | None = state["loaded"]
            if loaded is None:
                time.sleep(target_dt)
                continue

            now = time.time()
            dt = max(1.0e-4, now - t_prev)
            t_prev = now
            motion = loaded.motion
            frame_count = int(motion.joint_pos.shape[0])
            if frame_count <= 0:
                time.sleep(target_dt)
                continue

            if bool(playing.value):
                state["frame_cursor"] = (state["frame_cursor"] + dt * motion.fps * float(speed.value)) % frame_count
            frame = int(state["frame_cursor"]) % frame_count
            if frame != state["last_frame"]:
                state["last_frame"] = frame
                root_pos = motion.body_pos[frame, motion.root_body_index] + loaded.world_offset
                root_quat = motion.body_quat[frame, motion.root_body_index]
                body_pos = motion.body_pos[frame] + loaded.world_offset

                with server.atomic():
                    handles = loaded.handles
                    if handles.robot_frame is not None:
                        handles.robot_frame.position = tuple(root_pos.tolist())
                        handles.robot_frame.wxyz = tuple(root_quat.tolist())
                    if handles.robot_urdf is not None and motion.joint_to_urdf_perm is not None:
                        handles.robot_urdf.update_cfg(motion.joint_pos[frame, motion.joint_to_urdf_perm])
                    if handles.root_frame is not None:
                        handles.root_frame.position = tuple(root_pos.tolist())
                        handles.root_frame.wxyz = tuple(root_quat.tolist())
                    if handles.skeleton is not None and loaded.edge_indices is not None:
                        handles.skeleton.points = body_pos[loaded.edge_indices]
                    if handles.body_points is not None:
                        handles.body_points.points = body_pos
                    if handles.height_scan_points is not None:
                        height_points = _height_scan_points_for_frame(motion, frame, loaded.world_offset)
                        if height_points is not None:
                            handles.height_scan_points.points = height_points
                            handles.height_scan_points.visible = bool(show_height_scan.value)
                    frame_text.value = f"{frame} / {frame_count - 1}"

            time.sleep(target_dt)
    except KeyboardInterrupt:
        pass
    finally:
        _remove_loaded_scene(state["loaded"])
        server.stop()


if __name__ == "__main__":
    main()
