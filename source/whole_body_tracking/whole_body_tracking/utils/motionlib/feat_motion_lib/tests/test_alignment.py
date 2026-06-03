from __future__ import annotations

import numpy as np
import torch

from feat_motion_lib.errors import MotionAlignmentError
from feat_motion_lib.transform.alignment import align_body_tensors, align_joint_tensors, read_name_list


def _build_raw_npz(tmp_path, *, joint_names=None, body_names=None):
    path = tmp_path / "motion.npz"
    kwargs = {
        "fps": np.asarray([50.0], dtype=np.float32),
        "joint_pos": np.asarray([[2.0, 1.0]], dtype=np.float32),
        "joint_vel": np.asarray([[20.0, 10.0]], dtype=np.float32),
        "body_pos_w": np.arange(9, dtype=np.float32).reshape(1, 3, 3),
        "body_quat_w": np.zeros((1, 3, 4), dtype=np.float32),
        "body_lin_vel_w": np.zeros((1, 3, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((1, 3, 3), dtype=np.float32),
    }
    if joint_names is not None:
        kwargs["joint_names"] = np.asarray(joint_names)
    if body_names is not None:
        kwargs["body_names"] = np.asarray(body_names)
    np.savez(path, **kwargs)
    return np.load(path, allow_pickle=True)


def test_read_name_list_decodes_scalar_bytes(tmp_path) -> None:
    path = tmp_path / "scalar_names.npz"
    np.savez(
        path,
        fps=np.asarray([50.0], dtype=np.float32),
        joint_pos=np.zeros((1, 1), dtype=np.float32),
        joint_vel=np.zeros((1, 1), dtype=np.float32),
        body_pos_w=np.zeros((1, 1, 3), dtype=np.float32),
        body_quat_w=np.zeros((1, 1, 4), dtype=np.float32),
        body_lin_vel_w=np.zeros((1, 1, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((1, 1, 3), dtype=np.float32),
        joint_names=np.asarray(b"hip"),
    )
    raw = np.load(path, allow_pickle=True)
    assert read_name_list(raw, ("joint_names",)) == ["hip"]


def test_align_joint_tensors_reorders_by_joint_names(tmp_path) -> None:
    raw = _build_raw_npz(tmp_path, joint_names=["j2", "j1"])
    tensors = {
        "joint_pos": torch.tensor([[2.0, 1.0]]),
        "joint_vel": torch.tensor([[20.0, 10.0]]),
    }
    tensors, names = align_joint_tensors(tensors, raw, str(tmp_path / "motion.npz"), ["j1", "j2"])
    assert names == ["j1", "j2"]
    assert torch.allclose(tensors["joint_pos"], torch.tensor([[1.0, 2.0]]))
    assert torch.allclose(tensors["joint_vel"], torch.tensor([[10.0, 20.0]]))


def test_align_body_tensors_uses_body_names(tmp_path) -> None:
    raw = _build_raw_npz(tmp_path, body_names=["b3", "b1", "b2"])
    tensors = {
        "body_pos_w": torch.arange(9, dtype=torch.float32).reshape(1, 3, 3),
        "body_quat_w": torch.zeros((1, 3, 4), dtype=torch.float32),
        "body_lin_vel_w": torch.zeros((1, 3, 3), dtype=torch.float32),
        "body_ang_vel_w": torch.zeros((1, 3, 3), dtype=torch.float32),
    }
    tensors, names = align_body_tensors(
        tensors,
        raw,
        str(tmp_path / "motion.npz"),
        motion_body_names=["b1", "b2"],
        all_body_names=["b1", "b2", "b3"],
        body_indexes=[1, 2],
    )
    assert names == ["b1", "b2"]
    assert torch.allclose(tensors["body_pos_w"][0, :, 0], torch.tensor([3.0, 6.0]))


def test_align_joint_tensors_raises_on_missing_name(tmp_path) -> None:
    raw = _build_raw_npz(tmp_path, joint_names=["j1", "j2"])
    tensors = {
        "joint_pos": torch.tensor([[2.0, 1.0]]),
        "joint_vel": torch.tensor([[20.0, 10.0]]),
    }
    try:
        align_joint_tensors(tensors, raw, str(tmp_path / "motion.npz"), ["j1", "missing"])
    except MotionAlignmentError as exc:
        assert "missing required names" in str(exc)
    else:
        raise AssertionError("Expected MotionAlignmentError")
