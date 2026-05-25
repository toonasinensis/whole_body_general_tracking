from __future__ import annotations

import numpy as np
from pathlib import Path

from .math_utils import matrix_from_quat, quat_apply_inverse, quat_inv, quat_mul, resize_or_zero


class MotionData:
    def __init__(self, path: str):
        self.path = str(path)
        with np.load(path) as data:
            self.arrays = {name: self._load_value(data[name]) for name in data.files}
        self.files = list(self.arrays.keys())
        self.num_frames = int(self.arrays["joint_pos"].shape[0])
        self.total_bytes = sum(value.nbytes for value in self.arrays.values() if isinstance(value, np.ndarray))

    @staticmethod
    def _load_value(value):
        if isinstance(value, np.ndarray) and value.dtype.kind in ("S", "U", "O") and value.shape == ():
            return value.item()
        return np.asarray(value)

    def __getitem__(self, key: str):
        return self.arrays[key]

    def __contains__(self, key: str) -> bool:
        return key in self.arrays

    def print_config(self) -> None:
        print(
            f"[INFO] Preloaded motion arrays: frames={self.num_frames}, "
            f"fields={len(self.files)}, size={self.total_bytes / (1024 * 1024):.2f} MiB"
        )


def first_motion_file(motion_file: str, dataset_txt: str | None) -> str:
    root = Path(motion_file)
    if dataset_txt:
        for line in Path(dataset_txt).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            return str(p if p.is_absolute() else root / p)
        raise ValueError(f"No motion entries in dataset txt: {dataset_txt}")
    if root.is_file():
        return str(root)
    files = sorted(root.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under: {motion_file}")
    return str(files[0])


def future_indices(t: int, offsets: list[int], total: int) -> np.ndarray:
    return np.clip(t + np.asarray(offsets, dtype=np.int64), 0, total - 1)


def motion_body_index(meta: dict, body_name: str | None = None) -> tuple[str, int]:
    body_names = list(meta["motion_body_names"])
    name = body_name or meta.get("root_body_name") or body_names[0]
    if name not in body_names:
        raise ValueError(f"Motion body '{name}' is not in motion_body_names: {body_names}")
    return name, body_names.index(name)


def motion_frame_root_state(
    motion: MotionData,
    meta: dict,
    frame: int,
    root_body_name: str | None = None,
) -> tuple[str, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = int(np.clip(frame, 0, motion.num_frames - 1))
    name, body_idx = motion_body_index(meta, root_body_name)
    pos = np.asarray(motion["body_pos_w"][frame, body_idx], dtype=np.float64)
    quat = np.asarray(motion["body_quat_w"][frame, body_idx], dtype=np.float64)
    lin_vel = (
        np.asarray(motion["body_lin_vel_w"][frame, body_idx], dtype=np.float64)
        if "body_lin_vel_w" in motion
        else np.zeros(3, dtype=np.float64)
    )
    ang_vel = (
        np.asarray(motion["body_ang_vel_w"][frame, body_idx], dtype=np.float64)
        if "body_ang_vel_w" in motion
        else np.zeros(3, dtype=np.float64)
    )
    return name, body_idx, pos, quat, lin_vel, ang_vel


def motion_local_step_summary(
    motion: MotionData,
    meta: dict,
    root_body_name: str | None = None,
) -> dict[str, np.ndarray]:
    name, body_idx = motion_body_index(meta, root_body_name)
    pos = np.asarray(motion["body_pos_w"][:, body_idx], dtype=np.float64)
    quat = np.asarray(motion["body_quat_w"][:, body_idx], dtype=np.float64)
    if pos.shape[0] < 2:
        zeros = np.zeros(3, dtype=np.float64)
        return {"root_body": np.asarray(name), "mean": zeros, "min": zeros, "max": zeros, "total": zeros}
    local_step = quat_apply_inverse(quat[:-1], pos[1:] - pos[:-1])
    return {
        "root_body": np.asarray(name),
        "mean": local_step.mean(axis=0),
        "min": local_step.min(axis=0),
        "max": local_step.max(axis=0),
        "total": pos[-1] - pos[0],
    }


def motion_groups(
    motion: MotionData,
    t: int,
    meta: dict,
    robot_anchor_quat_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = meta["future_step_num"]
    future = future_indices(t, offsets, motion["joint_pos"].shape[0])
    target_quat = motion_anchor_ori_mf(motion, future, meta).reshape(1, len(offsets), 4)
    robot_anchor = np.repeat(robot_anchor_quat_w[:, None, :], len(offsets), axis=1)
    rel_quat = quat_mul(quat_inv(robot_anchor), target_quat)
    rel_6d = matrix_from_quat(rel_quat)[..., :2].reshape(1, -1).astype(np.float32)
    rbt_cmd_mf = np.concatenate(
        [
            motion["joint_pos"][future].reshape(1, -1),
            motion["joint_vel"][future].reshape(1, -1),
            rel_6d,
        ],
        axis=-1,
    ).astype(np.float32)
    rbt_dim = int(np.prod(meta["observation_shapes"].get("rbt_cmd_mf", rbt_cmd_mf.shape[1:])))
    if rbt_cmd_mf.shape[1] != rbt_dim:
        raise ValueError(
            f"Built rbt_cmd_mf with dim {rbt_cmd_mf.shape[1]}, but ONNX metadata expects {rbt_dim}. "
            "Check motion joint order/future_step_num and the exported task config."
        )
    smpl_dim = int(np.prod(meta["observation_shapes"].get("smpl_cmd_mf", [0])))
    smpl_cmd_mf = build_smpl_cmd_mf(motion, future, meta, robot_anchor_quat_w, smpl_dim)
    return rbt_cmd_mf, smpl_cmd_mf


def motion_anchor_ori_mf(motion: MotionData, future: np.ndarray, meta: dict) -> np.ndarray:
    body_names = list(meta["motion_body_names"])
    anchor_idx = body_names.index(meta["anchor_body_name"])
    return motion["body_quat_w"][future, anchor_idx, :]


def build_smpl_cmd_mf(
    motion: MotionData,
    future: np.ndarray,
    meta: dict,
    robot_anchor_quat_w: np.ndarray,
    smpl_dim: int,
) -> np.ndarray:
    if smpl_dim == 0:
        return np.zeros((1, 0), dtype=np.float32)
    if "smpl_joints" not in motion:
        return np.zeros((1, smpl_dim), dtype=np.float32)

    joints = np.asarray(motion["smpl_joints"][future], dtype=np.float32)
    joints = joints.reshape(1, joints.shape[0], 24, 3)
    root_quat_dim = 6 * len(future)
    if "smpl_poses" in motion.files:
        root_6d = np.zeros((1, root_quat_dim), dtype=np.float32)
    else:
        root_6d = np.zeros((1, root_quat_dim), dtype=np.float32)
    local_joints = joints.reshape(1, -1)
    return resize_or_zero(np.concatenate([local_joints, root_6d], axis=-1), smpl_dim)
