from __future__ import annotations

import numpy as np
import torch

from feat_motion_lib.errors import MotionAlignmentError, MotionValidationError
from feat_motion_lib.io.robot_npz import load_robot_motion_file, load_robot_npz_raw, parse_robot_npz, read_npz_fps


def _write_robot_npz(path, *, body_count: int, joint_names=None, body_names=None) -> None:
    kwargs = {}
    if joint_names is not None:
        kwargs["joint_names"] = np.asarray(joint_names)
    if body_names is not None:
        kwargs["body_names"] = np.asarray(body_names)
    np.savez(
        path,
        fps=np.asarray([50.0], dtype=np.float32),
        joint_pos=np.asarray([[2.0, 1.0]], dtype=np.float32),
        joint_vel=np.asarray([[20.0, 10.0]], dtype=np.float32),
        body_pos_w=np.arange(body_count * 3, dtype=np.float32).reshape(1, body_count, 3),
        body_quat_w=np.zeros((1, body_count, 4), dtype=np.float32),
        body_lin_vel_w=np.zeros((1, body_count, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((1, body_count, 3), dtype=np.float32),
        **kwargs,
    )


def test_read_npz_fps_reads_scalar_value(tmp_path) -> None:
    path = tmp_path / "motion.npz"
    _write_robot_npz(path, body_count=1)
    raw = np.load(path, allow_pickle=True)
    assert read_npz_fps(raw, str(path)) == 50.0


def test_parse_robot_npz_aligns_joint_and_body_names(tmp_path) -> None:
    path = tmp_path / "motion.npz"
    _write_robot_npz(path, body_count=3, joint_names=["j2", "j1"], body_names=["b3", "b1", "b2"])
    raw, fps = load_robot_npz_raw(str(path))
    clip = parse_robot_npz(
        raw,
        str(path),
        fps,
        joint_names=["j1", "j2"],
        motion_body_names=["b1", "b2"],
        all_body_names=["b1", "b2", "b3"],
        body_indexes=[1, 2],
    )
    assert torch.allclose(clip.joint_pos, torch.tensor([[1.0, 2.0]]))
    assert torch.allclose(clip.joint_vel, torch.tensor([[10.0, 20.0]]))
    assert torch.allclose(clip.body_pos_w[0, :, 0], torch.tensor([3.0, 6.0]))
    assert clip.joint_names == ["j1", "j2"]
    assert clip.body_names == ["b1", "b2"]


def test_load_robot_motion_file_without_metadata_uses_all_body_names(tmp_path) -> None:
    path = tmp_path / "motion.npz"
    _write_robot_npz(path, body_count=3)
    clip = load_robot_motion_file(
        str(path),
        joint_names=["j1", "j2"],
        motion_body_names=["b1", "b2"],
        all_body_names=["root", "b1", "b2"],
        body_indexes=[1, 2],
    )
    assert torch.allclose(clip.body_pos_w[0, :, 0], torch.tensor([3.0, 6.0]))
    assert clip.num_frames == 1
    assert clip.duration == 0.0


def test_load_robot_motion_file_raises_when_joint_dim_mismatch_without_names(tmp_path) -> None:
    path = tmp_path / "motion.npz"
    _write_robot_npz(path, body_count=1)
    try:
        load_robot_motion_file(str(path), joint_names=["j1", "j2", "j3"])
    except MotionAlignmentError as exc:
        assert "joint_pos dim" in str(exc)
    else:
        raise AssertionError("Expected MotionAlignmentError")


def test_parse_robot_npz_raises_on_missing_robot_tensor(tmp_path) -> None:
    path = tmp_path / "broken.npz"
    np.savez(
        path,
        fps=np.asarray([50.0], dtype=np.float32),
        joint_pos=np.asarray([[2.0, 1.0]], dtype=np.float32),
        body_pos_w=np.zeros((1, 1, 3), dtype=np.float32),
        body_quat_w=np.zeros((1, 1, 4), dtype=np.float32),
        body_lin_vel_w=np.zeros((1, 1, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((1, 1, 3), dtype=np.float32),
    )
    raw, fps = load_robot_npz_raw(str(path))
    try:
        parse_robot_npz(raw, str(path), fps)
    except MotionValidationError as exc:
        assert "missing required robot tensor key" in str(exc)
    else:
        raise AssertionError("Expected MotionValidationError")
