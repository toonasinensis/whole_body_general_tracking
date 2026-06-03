from __future__ import annotations

import joblib
import numpy as np
import torch

from ..types import MotionData


def _to_tensor(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr, dtype=np.float32))


def load_pkl(path: str) -> dict:
    data = joblib.load(path)

    if "pose_aa" not in data:
        raise ValueError(f"{path}: missing 'pose_aa' key")

    pose_aa = np.asarray(data["pose_aa"], dtype=np.float32)
    if pose_aa.ndim != 2 or pose_aa.shape[1] != 72:
        raise ValueError(f"{path}: pose_aa must be (T, 72), got {pose_aa.shape}")

    num_frames = pose_aa.shape[0]

    smpl_joints = data.get("smpl_joints")
    if smpl_joints is not None:
        smpl_joints = np.asarray(smpl_joints, dtype=np.float32)
        if smpl_joints.shape != (num_frames, 24, 3):
            smpl_joints = smpl_joints.reshape(num_frames, 24, 3)
    else:
        smpl_joints = np.zeros((num_frames, 24, 3), dtype=np.float32)

    transl = data.get("transl")
    if transl is not None:
        transl = np.asarray(transl, dtype=np.float32).reshape(num_frames, 3)
    else:
        transl = np.zeros((num_frames, 3), dtype=np.float32)

    fps = float(data.get("fps", 30.0))

    return {
        "pose_aa": pose_aa,
        "smpl_joints": smpl_joints,
        "transl": transl,
        "fps": fps,
    }


def load_motion_file(path: str, target_fps: float | None = None) -> MotionData:
    from ..transform.resample import resample_motion

    raw = load_pkl(path)
    src_fps = float(raw["fps"])

    if target_fps is not None and abs(target_fps - src_fps) > 1e-3:
        return resample_motion(raw, target_fps)

    pose_t = _to_tensor(raw["pose_aa"])
    joints_t = _to_tensor(raw["smpl_joints"])
    transl_t = _to_tensor(raw["transl"])
    num_frames = pose_t.shape[0]
    return MotionData(
        pose_aa=pose_t,
        smpl_joints=joints_t,
        transl=transl_t,
        fps=src_fps,
        source_fps=src_fps,
        num_frames=num_frames,
        duration=(num_frames - 1) / src_fps if num_frames > 1 else 0.0,
    )
