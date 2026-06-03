from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
import types
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def _bootstrap_motionlib_imports() -> None:
    motionlib_dir = Path(__file__).resolve().parents[2]
    if str(motionlib_dir) not in sys.path:
        sys.path.insert(0, str(motionlib_dir))

    smpl_motion_lib_dir = motionlib_dir / "smpl_motion_lib"
    if str(smpl_motion_lib_dir) not in sys.path:
        sys.path.insert(0, str(smpl_motion_lib_dir))

    smpl_math_utils_dir = motionlib_dir / "smpl_math_utils"
    if str(smpl_math_utils_dir) not in sys.path:
        sys.path.insert(0, str(smpl_math_utils_dir))

    import smpl_math_utils as smpl_math_utils_module

    if "whole_body_tracking" not in sys.modules:
        sys.modules["whole_body_tracking"] = types.ModuleType("whole_body_tracking")
    if "whole_body_tracking.utils" not in sys.modules:
        sys.modules["whole_body_tracking.utils"] = types.ModuleType("whole_body_tracking.utils")
    if "whole_body_tracking.utils.motionlib" not in sys.modules:
        sys.modules["whole_body_tracking.utils.motionlib"] = types.ModuleType("whole_body_tracking.utils.motionlib")

    whole_body_tracking_module = sys.modules["whole_body_tracking"]
    utils_module = sys.modules["whole_body_tracking.utils"]
    motionlib_module = sys.modules["whole_body_tracking.utils.motionlib"]

    whole_body_tracking_module.utils = utils_module
    utils_module.motionlib = motionlib_module

    smpl_math_pkg_name = "whole_body_tracking.utils.motionlib.smpl_math_utils"
    if smpl_math_pkg_name not in sys.modules:
        sys.modules[smpl_math_pkg_name] = types.ModuleType(smpl_math_pkg_name)
    smpl_math_pkg = sys.modules[smpl_math_pkg_name]
    smpl_math_pkg.smpl_math_utils = smpl_math_utils_module
    motionlib_module.smpl_math_utils = smpl_math_pkg
    sys.modules["whole_body_tracking.utils.motionlib.smpl_math_utils.smpl_math_utils"] = smpl_math_utils_module


_bootstrap_motionlib_imports()

import mujoco
import mujoco.viewer

from feat_motion_lib.facade.unified_motion_lib import UnifiedMotionLib
from feat_motion_lib.transform.pairing import pair_motion_paths


PACKAGE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_G1_MJCF = PACKAGE_ROOT / "assets" / "unitree_description" / "mjcf" / "g1.xml"
DEFAULT_ROBOT_MOTION_DIR = Path("/home/kiki/workspace/motion_data/mimic_data/g1/bones_seed")
DEFAULT_SMPL_DIR = Path("/home/kiki/workspace/motion_data/smpl_filtered")
FALLBACK_SMPL_DIR = Path("/home/kiki/workspace/motion_data/mimic_data/smpl_filtered")

G1_TRACKING_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

G1_ISAAC_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

SMPL_PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)

RAW_JOINT_RGBA = np.array([0.20, 0.85, 0.95, 0.85], dtype=np.float32)
RAW_BONE_RGBA = np.array([0.95, 0.70, 0.25, 0.85], dtype=np.float32)
GLOBAL_JOINT_RGBA = np.array([0.15, 1.00, 0.35, 0.90], dtype=np.float32)
GLOBAL_BONE_RGBA = np.array([0.10, 0.95, 0.10, 0.85], dtype=np.float32)
AXIS_RGBA = np.asarray(
    [
        [1.00, 0.20, 0.20, 0.95],
        [0.20, 1.00, 0.20, 0.95],
        [0.20, 0.45, 1.00, 0.95],
    ],
    dtype=np.float32,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Manual MuJoCo visualization for UnifiedMotionLib paired robot+SMPL clips. "
            "Loads 100 paired G1 motions by default, shows the robot, raw SMPL joints, "
            "SMPL global positions, and SMPL global rotations."
        )
    )
    parser.add_argument("--robot-dir", type=Path, default=DEFAULT_ROBOT_MOTION_DIR)
    parser.add_argument("--smpl-dir", type=Path, default=DEFAULT_SMPL_DIR)
    parser.add_argument("--robot-mjcf", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument(
        "--pair-list",
        type=Path,
        default=None,
        help="Optional text file listing desired stems or .npz file names, one per line.",
    )
    parser.add_argument("--target-fps", type=float, default=50.0)
    parser.add_argument("--max-frame-diff", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--smpl-local-offset",
        type=float,
        nargs=3,
        default=(0.0, -1.2, 0.0),
        metavar=("X", "Y", "Z"),
        help="Offset applied to raw SMPL joint visualization so it does not overlap the robot/global SMPL.",
    )
    parser.add_argument("--joint-radius", type=float, default=0.018)
    parser.add_argument("--bone-radius", type=float, default=0.006)
    parser.add_argument("--axis-radius", type=float, default=0.004)
    parser.add_argument("--axis-length", type=float, default=0.07)
    parser.add_argument(
        "--start-motion",
        type=int,
        default=-1,
        help="Start motion index. Use -1 to auto-pick the most dynamic clip among the loaded set.",
    )
    parser.add_argument("--loop", action="store_true", default=False)
    parser.add_argument("--sleep-scale", type=float, default=1.0)
    return parser.parse_args()


def _resolve_smpl_dir(path: Path) -> Path:
    if path.exists():
        return path
    if path == DEFAULT_SMPL_DIR and FALLBACK_SMPL_DIR.exists():
        print(f"[visualize] Default SMPL dir missing, fallback to {FALLBACK_SMPL_DIR}")
        return FALLBACK_SMPL_DIR
    raise FileNotFoundError(f"SMPL directory not found: {path}")


def _make_npz_to_mjcf_perm(mjcf_joint_names: list[str], isaac_joint_names: list[str]) -> np.ndarray:
    isaac_idx = {name: i for i, name in enumerate(isaac_joint_names)}
    missing = [name for name in mjcf_joint_names if name not in isaac_idx]
    if missing:
        raise ValueError(f"MJCF joints not found in Isaac joint order list: {missing}")
    return np.asarray([isaac_idx[name] for name in mjcf_joint_names], dtype=np.int64)


def _load_g1_model_metadata(mjcf_path: Path) -> tuple[list[str], list[str], list[int], np.ndarray]:
    root = ET.parse(mjcf_path).getroot()
    joint_names = [joint.attrib["name"] for joint in root.iter("joint") if "name" in joint.attrib]
    body_names = [body.attrib["name"] for body in root.iter("body") if "name" in body.attrib]
    body_indexes = [body_names.index(name) for name in G1_TRACKING_BODY_NAMES]
    npz_to_mjcf_perm = _make_npz_to_mjcf_perm(joint_names, G1_ISAAC_JOINT_NAMES)
    return joint_names, body_names, body_indexes, npz_to_mjcf_perm


def _read_requested_stems(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    stems = []
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry:
            continue
        stems.append(Path(entry).stem)
    return stems


def _select_pairs(robot_dir: Path, smpl_dir: Path, count: int, requested_stems: list[str] | None) -> list:
    robot_paths = sorted(str(path) for path in robot_dir.rglob("*.npz"))
    pairs = pair_motion_paths(robot_paths, smpl_dir)
    if requested_stems is not None:
        requested = set(requested_stems)
        pairs = [pair for pair in pairs if pair.stem in requested]
    if len(pairs) < count:
        raise ValueError(f"Requested {count} paired motions, but only found {len(pairs)} matching pairs.")
    return pairs[:count]


def _symlink_selected_npz(pairs: list, link_dir: Path) -> None:
    link_dir.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        dst = link_dir / pair.robot_path.name
        if dst.exists():
            continue
        try:
            os.symlink(pair.robot_path, dst)
        except OSError:
            shutil.copy2(pair.robot_path, dst)


def _build_cfg(motion_dir: Path, smpl_dir: Path, target_fps: float, max_frame_diff: int) -> SimpleNamespace:
    return SimpleNamespace(
        motion_file=str(motion_dir),
        max_motion_num=-1,
        dataset_txt=None,
        eval_mode=False,
        distributed=False,
        local_rank=0,
        total_rank=1,
        smpl_file_path=str(smpl_dir),
        target_fps=float(target_fps),
        up_axis="yup",
        max_frame_diff=int(max_frame_diff),
    )


def _load_unified_motion_lib(
    *,
    robot_mjcf: Path,
    motion_dir: Path,
    smpl_dir: Path,
    target_fps: float,
    max_frame_diff: int,
    device: str,
) -> UnifiedMotionLib:
    joint_names, all_body_names, body_indexes, _ = _load_g1_model_metadata(robot_mjcf)
    lib = UnifiedMotionLib(
        body_indexes=body_indexes,
        motion_anchor_body_index=0,
        joint_names=joint_names,
        motion_body_names=G1_TRACKING_BODY_NAMES,
        all_body_names=all_body_names,
        device=device,
    )
    report = lib.load_from_cfg(_build_cfg(motion_dir, smpl_dir, target_fps, max_frame_diff))
    if report.mode != "paired":
        raise RuntimeError(
            f"Expected paired robot+SMPL load, but UnifiedMotionLib reported mode={report.mode!r}. "
            f"fallback_used={report.fallback_used}"
        )
    return lib


def _get_smpl_global_rotations(lib: UnifiedMotionLib, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
    # UnifiedMotionLib currently forwards global positions but not global rotations,
    # so this manual visualization reaches the paired SMPL facade directly.
    smpl_lib = getattr(lib, "_smpl_lib", None)
    if smpl_lib is None:
        raise RuntimeError("UnifiedMotionLib has no paired SMPL data loaded.")
    return smpl_lib.get_smpl_global_rotations(motion_ids, motion_steps)


def _precompute_clips(lib: UnifiedMotionLib, local_offset: np.ndarray) -> list[dict[str, np.ndarray | str | float]]:
    clips: list[dict[str, np.ndarray | str | float]] = []
    if lib.file_names is None or lib.time_step_start_idx is None or lib.time_step_end_idx is None:
        raise RuntimeError("UnifiedMotionLib must be loaded before visualization.")

    for motion_id, motion_name in enumerate(lib.file_names):
        start = int(lib.time_step_start_idx[motion_id].item())
        end = int(lib.time_step_end_idx[motion_id].item())
        num_frames = end - start
        motion_ids = torch.full((num_frames,), motion_id, dtype=torch.long)
        motion_steps = torch.arange(num_frames, dtype=torch.long)

        raw_smpl = lib.get_smpl_joints(motion_ids, motion_steps).cpu().numpy()
        global_smpl = lib.get_smpl_global_position(motion_ids, motion_steps).cpu().numpy()
        global_axes = _get_smpl_global_rotations(lib, motion_ids, motion_steps).cpu().numpy()
        joint_pos = lib.joint_pos[start:end].cpu().numpy()
        anchor_pos = lib.anchor_pos_w[start:end].cpu().numpy()
        motion_score = float(np.linalg.norm(np.diff(joint_pos, axis=0), axis=1).mean())
        motion_score += float(np.linalg.norm(anchor_pos[-1] - anchor_pos[0]))

        clips.append(
            {
                "name": motion_name,
                "joint_pos": joint_pos,
                "anchor_pos": anchor_pos,
                "anchor_quat": lib.anchor_quat_w[start:end].cpu().numpy(),
                "raw_smpl": raw_smpl + local_offset[None, None, :],
                "global_smpl": global_smpl,
                "global_axes_origin": global_smpl,
                "global_axes_rot": global_axes,
                "fps": float(lib.fps or 50.0),
                "motion_score": motion_score,
            }
        )
    return clips


def _pick_start_motion(clips: list[dict[str, np.ndarray | str | float]], requested_index: int) -> int:
    if not clips:
        raise RuntimeError("No clips available for visualization.")
    if requested_index >= 0:
        if requested_index >= len(clips):
            raise IndexError(f"start_motion {requested_index} out of range [0, {len(clips)})")
        return requested_index
    return max(range(len(clips)), key=lambda idx: float(clips[idx]["motion_score"]))


def _set_robot_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    clip: dict[str, np.ndarray | str | float],
    frame_id: int,
    npz_to_mjcf_perm: np.ndarray,
) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    if model.nu > 0:
        data.ctrl[:] = 0.0

    joint_pos = clip["joint_pos"][frame_id]
    anchor_pos = clip["anchor_pos"][frame_id]
    anchor_quat = clip["anchor_quat"][frame_id]
    joint_pos_mjcf = joint_pos[npz_to_mjcf_perm]

    if model.nq < 7 + joint_pos_mjcf.shape[0]:
        raise RuntimeError(f"MuJoCo model nq={model.nq} is too small for G1 joint motion dim={joint_pos_mjcf.shape[0]}.")

    data.qpos[0:3] = anchor_pos.astype(np.float64)
    data.qpos[3:7] = anchor_quat.astype(np.float64)
    data.qpos[7 : 7 + joint_pos_mjcf.shape[0]] = joint_pos_mjcf.astype(np.float64)
    mujoco.mj_forward(model, data)


def _add_sphere(scene, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float64),
        pos.astype(np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba.astype(np.float32),
    )
    scene.ngeom += 1


def _add_capsule(scene, p0: np.ndarray, p1: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    geom = scene.geoms[scene.ngeom]
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba.astype(np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        float(radius),
        p0,
        p1,
    )
    scene.ngeom += 1


def _draw_skeleton(scene, joints: np.ndarray, joint_radius: float, bone_radius: float, joint_rgba: np.ndarray, bone_rgba: np.ndarray) -> None:
    for joint in joints:
        _add_sphere(scene, joint, joint_radius, joint_rgba)
    for child_idx, parent_idx in enumerate(SMPL_PARENTS):
        if parent_idx < 0:
            continue
        _add_capsule(scene, joints[parent_idx], joints[child_idx], bone_radius, bone_rgba)


def _draw_axes(scene, origins: np.ndarray, rotations: np.ndarray, axis_length: float, axis_radius: float) -> None:
    for joint_idx in range(origins.shape[0]):
        origin = origins[joint_idx]
        rotation = rotations[joint_idx]
        for axis_idx in range(3):
            direction = rotation[:, axis_idx] * axis_length
            _add_capsule(scene, origin, origin + direction, axis_radius, AXIS_RGBA[axis_idx])


def _update_overlay(
    scene,
    *,
    clip: dict[str, np.ndarray | str | float],
    frame_id: int,
    joint_radius: float,
    bone_radius: float,
    axis_radius: float,
    axis_length: float,
) -> None:
    scene.ngeom = 0
    raw_smpl = clip["raw_smpl"][frame_id]
    global_smpl = clip["global_smpl"][frame_id]
    global_axes_origin = clip["global_axes_origin"][frame_id]
    global_axes_rot = clip["global_axes_rot"][frame_id]

    _draw_skeleton(scene, raw_smpl, joint_radius, bone_radius, RAW_JOINT_RGBA, RAW_BONE_RGBA)
    _draw_skeleton(scene, global_smpl, joint_radius, bone_radius, GLOBAL_JOINT_RGBA, GLOBAL_BONE_RGBA)
    _draw_axes(scene, global_axes_origin, global_axes_rot, axis_length, axis_radius)


def _required_overlay_geoms() -> int:
    joints = 24
    bones = int((SMPL_PARENTS >= 0).sum())
    axes = joints * 3
    return (joints + bones) * 2 + axes


def main() -> None:
    args = _parse_args()
    robot_dir = args.robot_dir.expanduser().resolve()
    smpl_dir = _resolve_smpl_dir(args.smpl_dir.expanduser().resolve())
    robot_mjcf = args.robot_mjcf.expanduser().resolve()
    requested_stems = _read_requested_stems(args.pair_list.expanduser().resolve() if args.pair_list else None)
    selected_pairs = _select_pairs(robot_dir, smpl_dir, args.count, requested_stems)

    print(f"[visualize] Selected {len(selected_pairs)} paired motions.")
    print(f"[visualize] robot dir: {robot_dir}")
    print(f"[visualize] smpl dir : {smpl_dir}")
    print("[visualize] First 5 stems:")
    for pair in selected_pairs[:5]:
        print(f"  - {pair.stem}")
    print("[visualize] Colors: robot=MuJoCo mesh, raw SMPL=cyan/orange, global SMPL=green, global axes=RGB")

    with tempfile.TemporaryDirectory(prefix="unified_motion_mujoco_") as tmp_dir:
        link_dir = Path(tmp_dir) / "robot_subset"
        _symlink_selected_npz(selected_pairs, link_dir)

        lib = _load_unified_motion_lib(
            robot_mjcf=robot_mjcf,
            motion_dir=link_dir,
            smpl_dir=smpl_dir,
            target_fps=args.target_fps,
            max_frame_diff=args.max_frame_diff,
            device=args.device,
        )
        clips = _precompute_clips(lib, np.asarray(args.smpl_local_offset, dtype=np.float32))

        if not clips:
            raise RuntimeError("No paired clips were precomputed for visualization.")
        motion_idx = _pick_start_motion(clips, args.start_motion)
        top_motion_ids = sorted(range(len(clips)), key=lambda idx: float(clips[idx]["motion_score"]), reverse=True)[:5]
        print("[visualize] Top motion scores:")
        for idx in top_motion_ids:
            print(f"  - #{idx}: {clips[idx]['name']} score={float(clips[idx]['motion_score']):.6f}")
        print(f"[visualize] Starting from motion #{motion_idx}: {clips[motion_idx]['name']}")

        mjcf_joint_names, _, _, npz_to_mjcf_perm = _load_g1_model_metadata(robot_mjcf)
        print("[visualize] Using IsaacLab -> MuJoCo joint remap for G1.")
        print(f"[visualize] Example remap: MJCF[0]={mjcf_joint_names[0]} <- NPZ[{npz_to_mjcf_perm[0]}]={G1_ISAAC_JOINT_NAMES[npz_to_mjcf_perm[0]]}")

        model = mujoco.MjModel.from_xml_path(str(robot_mjcf))
        data = mujoco.MjData(model)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            if _required_overlay_geoms() > viewer.user_scn.maxgeom:
                raise RuntimeError(
                    f"Overlay needs {_required_overlay_geoms()} geoms but viewer.user_scn.maxgeom={viewer.user_scn.maxgeom}."
                )

            viewer.cam.distance = 2.8
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -18.0

            frame_idx = 0
            last_motion_idx = None

            while viewer.is_running():
                clip = clips[motion_idx]
                num_frames = clip["joint_pos"].shape[0]
                if last_motion_idx != motion_idx:
                    print(f"[visualize] Playing motion {motion_idx + 1}/{len(clips)}: {clip['name']}")
                    last_motion_idx = motion_idx

                _set_robot_pose(model, data, clip, frame_idx, npz_to_mjcf_perm)
                _update_overlay(
                    viewer.user_scn,
                    clip=clip,
                    frame_id=frame_idx,
                    joint_radius=args.joint_radius,
                    bone_radius=args.bone_radius,
                    axis_radius=args.axis_radius,
                    axis_length=args.axis_length,
                )
                viewer.sync()

                frame_idx += 1
                if frame_idx >= num_frames:
                    frame_idx = 0
                    motion_idx += 1
                    if motion_idx >= len(clips):
                        if args.loop:
                            motion_idx = 0
                        else:
                            break

                time.sleep((1.0 / max(float(clip["fps"]), 1e-6)) * args.sleep_scale)


if __name__ == "__main__":
    main()
