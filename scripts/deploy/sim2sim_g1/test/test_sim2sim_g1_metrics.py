from __future__ import annotations

import numpy as np

from sim2sim_g1.metrics import METRIC_NAMES, compute_tracking_metrics_from_state, resolve_motion_body_indices


def _identity_quats(count: int) -> np.ndarray:
    out = np.zeros((count, 4), dtype=np.float64)
    out[:, 0] = 1.0
    return out


def test_tracking_metrics_are_zero_for_matching_state() -> None:
    body_pos = np.array([[0.0, 0.0, 1.0], [0.2, 0.0, 0.5]], dtype=np.float64)
    body_quat = _identity_quats(2)
    body_vel = np.zeros((2, 3), dtype=np.float64)
    joint_pos = np.array([0.1, -0.2], dtype=np.float64)
    joint_vel = np.array([0.3, -0.4], dtype=np.float64)

    metrics = compute_tracking_metrics_from_state(
        ref_body_pos_w=body_pos,
        ref_body_quat_w=body_quat,
        ref_body_lin_vel_w=body_vel,
        ref_body_ang_vel_w=body_vel,
        ref_joint_pos=joint_pos,
        ref_joint_vel=joint_vel,
        robot_body_pos_w=body_pos,
        robot_body_quat_w=body_quat,
        robot_body_lin_vel_w=body_vel,
        robot_body_ang_vel_w=body_vel,
        robot_joint_pos=joint_pos,
        robot_joint_vel=joint_vel,
        anchor_index=0,
    )

    assert set(metrics) == set(METRIC_NAMES)
    assert all(value == 0.0 for value in metrics.values())


def test_tracking_metrics_keep_body_error_anchor_aligned() -> None:
    ref_body_pos = np.array([[0.0, 0.0, 1.0], [0.2, 0.0, 0.5]], dtype=np.float64)
    robot_body_pos = ref_body_pos + np.array([1.0, 2.0, 0.0])
    body_quat = _identity_quats(2)
    body_vel = np.zeros((2, 3), dtype=np.float64)
    joint = np.zeros(2, dtype=np.float64)

    metrics = compute_tracking_metrics_from_state(
        ref_body_pos_w=ref_body_pos,
        ref_body_quat_w=body_quat,
        ref_body_lin_vel_w=body_vel,
        ref_body_ang_vel_w=body_vel,
        ref_joint_pos=joint,
        ref_joint_vel=joint,
        robot_body_pos_w=robot_body_pos,
        robot_body_quat_w=body_quat,
        robot_body_lin_vel_w=body_vel,
        robot_body_ang_vel_w=body_vel,
        robot_joint_pos=joint,
        robot_joint_vel=joint,
        anchor_index=0,
    )

    assert np.isclose(metrics["error_anchor_pos"], np.sqrt(5.0))
    assert metrics["error_body_pos"] == 0.0


def test_resolve_motion_body_indices_maps_full_robot_motion_without_world_body() -> None:
    assert np.array_equal(
        resolve_motion_body_indices(
            motion_body_count=30,
            selected_body_count=3,
            model_nbody=31,
            body_ids=np.array([1, 8, 13], dtype=np.int64),
        ),
        np.array([0, 7, 12], dtype=np.int64),
    )
