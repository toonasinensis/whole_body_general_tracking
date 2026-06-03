from __future__ import annotations

import torch

from feat_motion_lib.facade.unified_motion_lib import UnifiedMotionLib as NewUnifiedMotionLib
from smpl_motion_lib import UnifiedMotionLib as OldUnifiedMotionLib

from ._compat_helpers import build_cfg, write_robot_npz, write_smpl_pkl


def _build_libs():
    new_lib = NewUnifiedMotionLib(
        body_indexes=[1, 2],
        motion_anchor_body_index=0,
        joint_names=["j1", "j2"],
        motion_body_names=["anchor", "b1"],
        all_body_names=["root", "anchor", "b1"],
        device="cpu",
    )
    old_lib = OldUnifiedMotionLib(
        body_indexes=[1, 2],
        motion_anchor_body_index=0,
        joint_names=["j1", "j2"],
        motion_body_names=["anchor", "b1"],
        all_body_names=["root", "anchor", "b1"],
        device="cpu",
    )
    return new_lib, old_lib


def _assert_core_outputs_match(new_lib: NewUnifiedMotionLib, old_lib: OldUnifiedMotionLib) -> None:
    assert new_lib.motion_num == old_lib.motion_num
    assert int(new_lib.time_step_total) == int(old_lib.time_step_total)
    assert new_lib.file_names == old_lib.file_names
    torch.testing.assert_close(new_lib.frame_list, old_lib.frame_list)
    torch.testing.assert_close(new_lib.time_step_start_idx, old_lib.time_step_start_idx)
    torch.testing.assert_close(new_lib.time_step_end_idx, old_lib.time_step_end_idx)
    torch.testing.assert_close(new_lib.joint_pos, old_lib.joint_pos)
    torch.testing.assert_close(new_lib.joint_vel, old_lib.joint_vel)
    torch.testing.assert_close(new_lib.body_pos_w, old_lib.body_pos_w)
    torch.testing.assert_close(new_lib.body_quat_w, old_lib.body_quat_w)
    torch.testing.assert_close(new_lib.body_lin_vel_w, old_lib.body_lin_vel_w)
    torch.testing.assert_close(new_lib.body_ang_vel_w, old_lib.body_ang_vel_w)
    torch.testing.assert_close(new_lib.anchor_pos_w, old_lib.anchor_pos_w)
    torch.testing.assert_close(new_lib.anchor_quat_w, old_lib.anchor_quat_w)
    torch.testing.assert_close(new_lib.anchor_lin_vel_w, old_lib.anchor_lin_vel_w)
    torch.testing.assert_close(new_lib.anchor_ang_vel_w, old_lib.anchor_ang_vel_w)
    torch.testing.assert_close(new_lib.anchor_pos_z, old_lib.anchor_pos_z)

    timestamps = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    torch.testing.assert_close(
        new_lib.motion_ids_from_timestamps(timestamps),
        old_lib.motion_ids_from_timestamps(timestamps),
    )


def test_unified_motion_lib_robot_outputs_match_legacy(tmp_path) -> None:
    motion_dir = tmp_path / "robot"
    motion_dir.mkdir()
    write_robot_npz(
        motion_dir / "clip_a.npz",
        num_frames=3,
        joint_dim=2,
        body_count=3,
        frame_offset=0,
        joint_names=["j2", "j1"],
        body_names=["b2", "anchor", "b1"],
    )
    write_robot_npz(
        motion_dir / "clip_b.npz",
        num_frames=2,
        joint_dim=2,
        body_count=3,
        frame_offset=3,
        joint_names=["j2", "j1"],
        body_names=["b2", "anchor", "b1"],
    )
    cfg = build_cfg(str(motion_dir))

    new_lib, old_lib = _build_libs()
    new_lib.load_from_cfg(cfg)
    old_lib.load_from_cfg(cfg)

    _assert_core_outputs_match(new_lib, old_lib)


def test_unified_motion_lib_paired_outputs_match_legacy(tmp_path) -> None:
    motion_dir = tmp_path / "robot"
    smpl_dir = tmp_path / "smpl"
    motion_dir.mkdir()
    smpl_dir.mkdir()
    write_robot_npz(
        motion_dir / "clip_a.npz",
        num_frames=3,
        joint_dim=2,
        body_count=3,
        frame_offset=0,
        joint_names=["j2", "j1"],
        body_names=["b2", "anchor", "b1"],
    )
    write_smpl_pkl(smpl_dir / "clip_a.pkl", num_frames=3, frame_offset=0)
    cfg = build_cfg(str(motion_dir), smpl_file_path=str(smpl_dir))

    new_lib, old_lib = _build_libs()
    new_lib.load_from_cfg(cfg)
    old_lib.load_from_cfg(cfg)

    _assert_core_outputs_match(new_lib, old_lib)
    assert new_lib.smpl_joints is not None
    assert new_lib.smpl_transl is not None
    assert new_lib.smpl_poses is not None
    assert old_lib.smpl_joints is not None
    assert old_lib.smpl_transl is not None
    assert old_lib.smpl_poses is not None

    torch.testing.assert_close(new_lib.smpl_joints, old_lib.smpl_joints)
    torch.testing.assert_close(new_lib.smpl_transl, old_lib.smpl_transl, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(new_lib.smpl_poses, old_lib.smpl_poses, atol=1e-5, rtol=1e-5)

    motion_ids = torch.tensor([0, 0, 0], dtype=torch.long)
    local_steps = torch.tensor([0, 1, 2], dtype=torch.long)
    torch.testing.assert_close(new_lib.get_smpl_joints(motion_ids, local_steps), old_lib.get_smpl_joints(motion_ids, local_steps))
    torch.testing.assert_close(
        new_lib.get_smpl_transl(motion_ids, local_steps),
        old_lib.get_smpl_transl(motion_ids, local_steps),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        new_lib.get_smpl_pose(motion_ids, local_steps),
        old_lib.get_smpl_pose(motion_ids, local_steps),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        new_lib.get_smpl_global_position(motion_ids, local_steps),
        old_lib.get_smpl_global_position(motion_ids, local_steps),
        atol=1e-5,
        rtol=1e-5,
    )
