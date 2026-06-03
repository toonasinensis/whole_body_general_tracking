from __future__ import annotations

import torch

from feat_motion_lib.facade.unified_motion_lib import UnifiedMotionLib

from ._compat_helpers import build_cfg, write_robot_npz, write_smpl_pkl


def _global_time_steps(lib: UnifiedMotionLib, motion_ids: torch.Tensor, local_steps: torch.Tensor) -> torch.Tensor:
    return lib.time_step_start_idx[motion_ids] + local_steps


def test_commands_view_robot_access_patterns_are_supported(tmp_path) -> None:
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

    lib = UnifiedMotionLib(
        body_indexes=[1, 2],
        motion_anchor_body_index=0,
        joint_names=["j1", "j2"],
        motion_body_names=["anchor", "b1"],
        all_body_names=["root", "anchor", "b1"],
        device="cpu",
    )
    report = lib.load_from_cfg(build_cfg(str(motion_dir)))

    assert report.mode == "robot_only"
    assert lib.motion_num == 2
    assert int(lib.time_step_total) == 5
    assert lib.frame_list.tolist() == [3, 2]
    assert lib.time_step_start_idx.tolist() == [0, 3]
    assert lib.time_step_end_idx.tolist() == [3, 5]

    motion_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    local_steps = torch.tensor([0, 2, 0, 1], dtype=torch.long)
    global_steps = _global_time_steps(lib, motion_ids, local_steps)

    torch.testing.assert_close(lib.joint_pos[global_steps], torch.tensor([[1.0, 0.0], [5.0, 4.0], [4.0, 3.0], [6.0, 5.0]]))
    torch.testing.assert_close(
        lib.joint_vel[global_steps],
        torch.tensor([[1001.0, 1000.0], [1005.0, 1004.0], [1004.0, 1003.0], [1006.0, 1005.0]]),
    )

    expected_anchor_pos = lib.body_pos_w[global_steps, 0]
    torch.testing.assert_close(lib.anchor_pos_w[global_steps], expected_anchor_pos)
    torch.testing.assert_close(lib.anchor_quat_w[global_steps], lib.body_quat_w[global_steps, 0])
    torch.testing.assert_close(lib.anchor_lin_vel_w[global_steps], lib.body_lin_vel_w[global_steps, 0])
    torch.testing.assert_close(lib.anchor_ang_vel_w[global_steps], lib.body_ang_vel_w[global_steps, 0])
    torch.testing.assert_close(lib.anchor_pos_z[global_steps], expected_anchor_pos[:, 2:3])

    timestamps = torch.tensor([0, 2, 3, 4], dtype=torch.long)
    torch.testing.assert_close(lib.motion_ids_from_timestamps(timestamps), torch.tensor([0, 0, 1, 1], dtype=torch.long))


def test_commands_view_smpl_access_patterns_are_supported(tmp_path) -> None:
    motion_dir = tmp_path / "robot"
    smpl_dir = tmp_path / "smpl"
    motion_dir.mkdir()
    smpl_dir.mkdir()
    write_robot_npz(
        motion_dir / "clip_a.npz",
        num_frames=3,
        joint_dim=2,
        body_count=2,
        frame_offset=0,
        body_names=["anchor", "b1"],
    )
    write_smpl_pkl(smpl_dir / "clip_a.pkl", num_frames=3, frame_offset=0)

    lib = UnifiedMotionLib(
        body_indexes=[0, 1],
        motion_anchor_body_index=0,
        joint_names=None,
        motion_body_names=["anchor", "b1"],
        all_body_names=["anchor", "b1"],
        device="cpu",
    )
    report = lib.load_from_cfg(build_cfg(str(motion_dir), smpl_file_path=str(smpl_dir)))

    assert report.mode == "paired"
    assert lib.smpl_joints is not None
    assert lib.smpl_transl is not None
    assert lib.smpl_poses is not None

    motion_ids = torch.tensor([0, 0, 0], dtype=torch.long)
    local_steps = torch.tensor([0, 1, 2], dtype=torch.long)
    global_steps = _global_time_steps(lib, motion_ids, local_steps)

    torch.testing.assert_close(lib.smpl_joints[global_steps], lib.get_smpl_joints(motion_ids, local_steps))
    torch.testing.assert_close(lib.smpl_transl[global_steps], lib.get_smpl_transl(motion_ids, local_steps))
    torch.testing.assert_close(lib.smpl_poses[global_steps], lib.get_smpl_pose(motion_ids, local_steps))
    torch.testing.assert_close(
        lib.get_smpl_global_position(motion_ids, local_steps),
        lib.get_smpl_joints(motion_ids, local_steps) + lib.get_smpl_transl(motion_ids, local_steps).unsqueeze(1),
        atol=1e-5,
        rtol=1e-5,
    )
