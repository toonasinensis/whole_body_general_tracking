"""Per-file loading and optional FPS resampling for SMPL motion pkl files."""

import numpy as np
import torch
from dataclasses import dataclass

import joblib
from smpl_math_utils import interpolate_linear, interpolate_pose


@dataclass
class MotionData:
    pose_aa: torch.Tensor  # (T, 72)   float32, axis-angle for 24 joints
    smpl_joints: torch.Tensor  # (T, 24, 3) float32, joint 3-D positions
    transl: torch.Tensor  # (T, 3)    float32, root translation
    fps: float  # effective fps after resampling
    source_fps: float  # original fps in the pkl file
    num_frames: int  # T
    duration: float  # (T-1) / fps  in seconds


def _to_tensor(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr, dtype=np.float32))


def load_pkl(path: str) -> dict:
    """Load a single SMPL pkl file and validate its contents."""
    data = joblib.load(path)

    if "pose_aa" not in data:
        raise ValueError(f"{path}: missing 'pose_aa' key")

    pose_aa = np.asarray(data["pose_aa"], dtype=np.float32)
    if pose_aa.ndim != 2 or pose_aa.shape[1] != 72:
        raise ValueError(f"{path}: pose_aa must be (T, 72), got {pose_aa.shape}")

    T = pose_aa.shape[0]

    smpl_joints = data.get("smpl_joints")
    if smpl_joints is not None:
        smpl_joints = np.asarray(smpl_joints, dtype=np.float32)
        if smpl_joints.shape != (T, 24, 3):
            smpl_joints = smpl_joints.reshape(T, 24, 3)
    else:
        smpl_joints = np.zeros((T, 24, 3), dtype=np.float32)

    transl = data.get("transl")
    if transl is not None:
        transl = np.asarray(transl, dtype=np.float32).reshape(T, 3)
    else:
        transl = np.zeros((T, 3), dtype=np.float32)

    fps = float(data.get("fps", 30))

    return {"pose_aa": pose_aa, "smpl_joints": smpl_joints, "transl": transl, "fps": fps}


def resample_motion(raw: dict, target_fps: float) -> MotionData:
    """Resample all motion arrays from raw['fps'] to target_fps."""
    src_fps = raw["fps"]

    pose_t = _to_tensor(raw["pose_aa"])  # (T, 72)
    joints_t = _to_tensor(raw["smpl_joints"])  # (T, 24, 3)
    transl_t = _to_tensor(raw["transl"])  # (T, 3)

    pose_r = interpolate_pose(pose_t, src_fps, target_fps, interpolation_type="slerp")
    joints_r = interpolate_linear(joints_t, src_fps, target_fps)
    transl_r = interpolate_linear(transl_t, src_fps, target_fps)

    T_new = pose_r.shape[0]
    return MotionData(
        pose_aa=pose_r,
        smpl_joints=joints_r,
        transl=transl_r,
        fps=target_fps,
        source_fps=src_fps,
        num_frames=T_new,
        duration=(T_new - 1) / target_fps,
    )


def load_motion_file(path: str, target_fps: float | None = None) -> MotionData:
    """Load an SMPL pkl file, resampling to target_fps if specified."""
    raw = load_pkl(path)
    src_fps = raw["fps"]

    if target_fps is not None and abs(target_fps - src_fps) > 1e-3:
        return resample_motion(raw, target_fps)

    pose_t = _to_tensor(raw["pose_aa"])
    joints_t = _to_tensor(raw["smpl_joints"])
    transl_t = _to_tensor(raw["transl"])
    T = pose_t.shape[0]
    return MotionData(
        pose_aa=pose_t,
        smpl_joints=joints_t,
        transl=transl_t,
        fps=src_fps,
        source_fps=src_fps,
        num_frames=T,
        duration=(T - 1) / src_fps,
    )
