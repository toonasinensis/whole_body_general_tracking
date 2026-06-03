from __future__ import annotations

import numpy as np
import torch
from types import SimpleNamespace

from smpl_motion_lib import UnifiedMotionLib


def _cfg(motion_file: str) -> SimpleNamespace:
    return SimpleNamespace(
        motion_file=motion_file,
        max_motion_num=-1,
        dataset_txt=None,
        eval_mode=False,
        distributed=False,
        local_rank=0,
        total_rank=1,
        smpl_file_path=None,
    )


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
        body_pos_w=np.arange(body_count, dtype=np.float32).reshape(1, body_count, 1).repeat(3, axis=2),
        body_quat_w=np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (1, body_count, 1)),
        body_lin_vel_w=np.zeros((1, body_count, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((1, body_count, 3), dtype=np.float32),
        **kwargs,
    )


def test_unified_motion_lib_uses_npz_joint_and_body_names(tmp_path) -> None:
    path = tmp_path / "motion.npz"
    _write_robot_npz(
        path,
        body_count=3,
        joint_names=["j2", "j1"],
        body_names=["b3", "b1", "b2"],
    )

    lib = UnifiedMotionLib(
        body_indexes=[1, 2],
        motion_anchor_body_index=0,
        joint_names=["j1", "j2"],
        motion_body_names=["b1", "b2"],
        all_body_names=["b1", "b2", "b3"],
        device="cpu",
    )
    lib.load_from_cfg(_cfg(str(tmp_path)))

    assert torch.allclose(lib.joint_pos, torch.tensor([[1.0, 2.0]]))
    assert torch.allclose(lib.joint_vel, torch.tensor([[10.0, 20.0]]))
    assert torch.allclose(lib.body_pos_w[0, :, 0], torch.tensor([1.0, 2.0]))


def test_unified_motion_lib_legacy_selected_body_order(tmp_path) -> None:
    path = tmp_path / "motion.npz"
    _write_robot_npz(path, body_count=2)

    lib = UnifiedMotionLib(
        body_indexes=[1, 2],
        motion_anchor_body_index=0,
        joint_names=["j1", "j2"],
        motion_body_names=["b1", "b2"],
        all_body_names=["root", "b1", "b2"],
        device="cpu",
    )
    lib.load_from_cfg(_cfg(str(tmp_path)))

    assert torch.allclose(lib.body_pos_w[0, :, 0], torch.tensor([0.0, 1.0]))


def test_unified_motion_lib_legacy_full_body_order(tmp_path) -> None:
    path = tmp_path / "motion.npz"
    _write_robot_npz(path, body_count=3)

    lib = UnifiedMotionLib(
        body_indexes=[1, 2],
        motion_anchor_body_index=0,
        joint_names=["j1", "j2"],
        motion_body_names=["b1", "b2"],
        all_body_names=["root", "b1", "b2"],
        device="cpu",
    )
    lib.load_from_cfg(_cfg(str(tmp_path)))

    assert torch.allclose(lib.body_pos_w[0, :, 0], torch.tensor([1.0, 2.0]))
