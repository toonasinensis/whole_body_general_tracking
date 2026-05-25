from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from .math_utils import quat_inv, quat_mul
from .motion import MotionData

METRIC_NAMES = (
    "error_anchor_pos",
    "error_anchor_rot",
    "error_anchor_lin_vel",
    "error_anchor_ang_vel",
    "error_body_pos",
    "error_body_rot",
    "error_body_lin_vel",
    "error_body_ang_vel",
    "error_joint_pos",
    "error_joint_vel",
)


def resolve_motion_body_indices(
    motion_body_count: int,
    selected_body_count: int,
    model_nbody: int,
    body_ids: np.ndarray,
) -> np.ndarray:
    """Map selected metadata body names to columns in a motion body array."""
    body_ids = np.asarray(body_ids, dtype=np.int64)
    if motion_body_count == selected_body_count:
        return np.arange(selected_body_count, dtype=np.int64)
    if motion_body_count == model_nbody - 1:
        return body_ids - 1
    if motion_body_count == model_nbody:
        return body_ids
    body_indices = body_ids - 1
    if body_indices.size > 0 and int(body_indices.max()) < motion_body_count:
        return body_indices
    raise ValueError(
        f"Cannot map motion body dim {motion_body_count} to selected body dim {selected_body_count} "
        f"with MuJoCo nbody={model_nbody} and body_ids={body_ids.tolist()}."
    )


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(norm, 1.0e-9, None)


def _quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    v = np.asarray(v, dtype=np.float64)
    xyz = q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + q[..., :1] * t + np.cross(xyz, t)


def _quat_error_magnitude(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = _quat_normalize(q1)
    q2 = _quat_normalize(q2)
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def _yaw_quat(q: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    zeros = np.zeros_like(half)
    return np.stack([np.cos(half), zeros, zeros, np.sin(half)], axis=-1)


def _motion_array_or_zeros(motion: MotionData, key: str, frame: int, shape: tuple[int, ...]) -> np.ndarray:
    if key not in motion:
        return np.zeros(shape, dtype=np.float64)
    return np.asarray(motion[key][frame], dtype=np.float64)


def _mujoco_body_velocities(model, data, body_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    lin_vel = np.zeros((len(body_ids), 3), dtype=np.float64)
    ang_vel = np.zeros((len(body_ids), 3), dtype=np.float64)
    spatial = np.zeros(6, dtype=np.float64)
    for i, body_id in enumerate(body_ids):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, int(body_id), spatial, 0)
        ang_vel[i] = spatial[:3]
        lin_vel[i] = spatial[3:]
    return lin_vel, ang_vel


def compute_tracking_metrics_from_state(
    *,
    ref_body_pos_w: np.ndarray,
    ref_body_quat_w: np.ndarray,
    ref_body_lin_vel_w: np.ndarray,
    ref_body_ang_vel_w: np.ndarray,
    ref_joint_pos: np.ndarray,
    ref_joint_vel: np.ndarray,
    robot_body_pos_w: np.ndarray,
    robot_body_quat_w: np.ndarray,
    robot_body_lin_vel_w: np.ndarray,
    robot_body_ang_vel_w: np.ndarray,
    robot_joint_pos: np.ndarray,
    robot_joint_vel: np.ndarray,
    anchor_index: int,
) -> dict[str, float]:
    ref_body_pos_w = np.asarray(ref_body_pos_w, dtype=np.float64)
    ref_body_quat_w = _quat_normalize(ref_body_quat_w)
    robot_body_pos_w = np.asarray(robot_body_pos_w, dtype=np.float64)
    robot_body_quat_w = _quat_normalize(robot_body_quat_w)
    anchor_index = int(anchor_index)

    ref_anchor_pos_w = ref_body_pos_w[anchor_index]
    ref_anchor_quat_w = ref_body_quat_w[anchor_index]
    robot_anchor_pos_w = robot_body_pos_w[anchor_index]
    robot_anchor_quat_w = robot_body_quat_w[anchor_index]

    delta_pos_w = np.repeat(robot_anchor_pos_w[None, :], ref_body_pos_w.shape[0], axis=0)
    delta_pos_w[:, 2] = ref_anchor_pos_w[2]
    delta_ori_w = _yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(ref_anchor_quat_w)))
    body_pos_relative_w = delta_pos_w + _quat_apply(delta_ori_w[None, :], ref_body_pos_w - ref_anchor_pos_w[None, :])
    body_quat_relative_w = quat_mul(delta_ori_w[None, :], ref_body_quat_w)

    ref_anchor_lin_vel_w = np.asarray(ref_body_lin_vel_w, dtype=np.float64)[anchor_index]
    ref_anchor_ang_vel_w = np.asarray(ref_body_ang_vel_w, dtype=np.float64)[anchor_index]
    robot_anchor_lin_vel_w = np.asarray(robot_body_lin_vel_w, dtype=np.float64)[anchor_index]
    robot_anchor_ang_vel_w = np.asarray(robot_body_ang_vel_w, dtype=np.float64)[anchor_index]

    return {
        "error_anchor_pos": float(np.linalg.norm(ref_anchor_pos_w - robot_anchor_pos_w)),
        "error_anchor_rot": float(_quat_error_magnitude(ref_anchor_quat_w, robot_anchor_quat_w)),
        "error_anchor_lin_vel": float(np.linalg.norm(ref_anchor_lin_vel_w - robot_anchor_lin_vel_w)),
        "error_anchor_ang_vel": float(np.linalg.norm(ref_anchor_ang_vel_w - robot_anchor_ang_vel_w)),
        "error_body_pos": float(np.linalg.norm(body_pos_relative_w - robot_body_pos_w, axis=-1).mean()),
        "error_body_rot": float(_quat_error_magnitude(body_quat_relative_w, robot_body_quat_w).mean()),
        "error_body_lin_vel": float(
            np.linalg.norm(np.asarray(ref_body_lin_vel_w) - np.asarray(robot_body_lin_vel_w), axis=-1).mean()
        ),
        "error_body_ang_vel": float(
            np.linalg.norm(np.asarray(ref_body_ang_vel_w) - np.asarray(robot_body_ang_vel_w), axis=-1).mean()
        ),
        "error_joint_pos": float(np.mean(np.abs(np.asarray(ref_joint_pos) - np.asarray(robot_joint_pos)))),
        "error_joint_vel": float(np.mean(np.abs(np.asarray(ref_joint_vel) - np.asarray(robot_joint_vel)))),
    }


def motion_tracking_metrics(
    model,
    data,
    motion: MotionData,
    frame: int,
    meta: dict,
    joint_qpos: np.ndarray,
    joint_qvel: np.ndarray,
    body_ids: np.ndarray,
) -> dict[str, float]:
    frame = int(np.clip(frame, 0, motion.num_frames - 1))
    body_names = list(meta["motion_body_names"])
    anchor_index = body_names.index(meta["anchor_body_name"])
    body_indices = np.asarray(meta.get("motion_body_indices", np.arange(len(body_names))), dtype=np.int64)
    # body_count = len(body_indices)
    robot_body_lin_vel_w, robot_body_ang_vel_w = _mujoco_body_velocities(model, data, body_ids)

    return compute_tracking_metrics_from_state(
        ref_body_pos_w=np.asarray(motion["body_pos_w"][frame, body_indices], dtype=np.float64),
        ref_body_quat_w=np.asarray(motion["body_quat_w"][frame, body_indices], dtype=np.float64),
        ref_body_lin_vel_w=_motion_array_or_zeros(motion, "body_lin_vel_w", frame, (motion["body_pos_w"].shape[1], 3))[
            body_indices
        ],
        ref_body_ang_vel_w=_motion_array_or_zeros(motion, "body_ang_vel_w", frame, (motion["body_pos_w"].shape[1], 3))[
            body_indices
        ],
        ref_joint_pos=np.asarray(motion["joint_pos"][frame], dtype=np.float64),
        ref_joint_vel=_motion_array_or_zeros(motion, "joint_vel", frame, np.asarray(motion["joint_pos"][frame]).shape),
        robot_body_pos_w=np.asarray(data.xpos[body_ids], dtype=np.float64),
        robot_body_quat_w=np.asarray(data.xquat[body_ids], dtype=np.float64),
        robot_body_lin_vel_w=robot_body_lin_vel_w,
        robot_body_ang_vel_w=robot_body_ang_vel_w,
        robot_joint_pos=np.asarray(data.qpos[joint_qpos], dtype=np.float64),
        robot_joint_vel=np.asarray(data.qvel[joint_qvel], dtype=np.float64),
        anchor_index=anchor_index,
    )


@dataclass
class MotionMetricAccumulator:
    motion_index: int
    motion_file: str
    num_frames: int
    sums: dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in METRIC_NAMES})
    samples: int = 0

    def update(self, metrics: dict[str, float]) -> None:
        for name in METRIC_NAMES:
            self.sums[name] += float(metrics[name])
        self.samples += 1

    def row(self) -> dict[str, float | int | str]:
        denom = max(self.samples, 1)
        row: dict[str, float | int | str] = {
            "motion_index": int(self.motion_index),
            "motion_file": self.motion_file,
            "num_frames": int(self.num_frames),
            "samples": int(self.samples),
        }
        row.update({name: self.sums[name] / denom for name in METRIC_NAMES})
        return row
