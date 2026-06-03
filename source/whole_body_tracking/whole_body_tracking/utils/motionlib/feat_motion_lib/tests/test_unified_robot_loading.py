from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from feat_motion_lib.config import DiscoveryOptions, LoadOptions, UnifiedLoadConfig
from feat_motion_lib.facade.unified_motion_lib import UnifiedMotionLib


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
        target_fps=50.0,
    )


def _write_robot_npz(path, *, body_count: int, num_frames: int = 3) -> None:
    np.savez(
        path,
        fps=np.asarray([50.0], dtype=np.float32),
        joint_pos=np.zeros((num_frames, 2), dtype=np.float32),
        joint_vel=np.zeros((num_frames, 2), dtype=np.float32),
        body_pos_w=np.zeros((num_frames, body_count, 3), dtype=np.float32),
        body_quat_w=np.zeros((num_frames, body_count, 4), dtype=np.float32),
        body_lin_vel_w=np.zeros((num_frames, body_count, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((num_frames, body_count, 3), dtype=np.float32),
    )


def test_unified_motion_lib_robot_only_report_and_state(tmp_path) -> None:
    motion_dir = tmp_path / "robot"
    motion_dir.mkdir()
    _write_robot_npz(motion_dir / "sample_a.npz", body_count=3, num_frames=3)
    _write_robot_npz(motion_dir / "sample_b.npz", body_count=3, num_frames=2)

    lib = UnifiedMotionLib(
        body_indexes=[0, 1],
        motion_anchor_body_index=0,
        device="cpu",
    )
    report = lib.load_from_cfg(_cfg(str(motion_dir)))

    assert report.mode == "robot_only"
    assert report.fallback_used is False
    assert lib.motion_num == 2
    assert int(lib.time_step_total) == 5
    assert lib.file_names == ["sample_a.npz", "sample_b.npz"]
    assert lib.frame_list.tolist() == [3, 2]
    assert lib.time_step_start_idx.tolist() == [0, 3]
    assert lib.time_step_end_idx.tolist() == [3, 5]


def test_unified_motion_lib_load_accepts_unified_load_config(tmp_path) -> None:
    motion_dir = tmp_path / "robot"
    motion_dir.mkdir()
    _write_robot_npz(motion_dir / "sample.npz", body_count=3, num_frames=3)

    lib = UnifiedMotionLib(body_indexes=[0, 1], motion_anchor_body_index=0, device="cpu")
    report = lib.load(
        UnifiedLoadConfig(
            discovery=DiscoveryOptions(motion_dir=motion_dir),
            load=LoadOptions(device="cpu"),
        )
    )

    assert report.mode == "robot_only"
    assert lib.motion_num == 1
    assert lib.file_names == ["sample.npz"]
