from __future__ import annotations

import numpy as np

from sim2sim_g1.motion import MotionData, first_motion_file, future_indices, motion_files, motion_groups


def test_motion_data_preloads_npz(tmp_path) -> None:
    path = tmp_path / "motion.npz"
    np.savez(
        path,
        joint_pos=np.ones((3, 2), dtype=np.float32),
        joint_vel=np.zeros((3, 2), dtype=np.float32),
        body_quat_w=np.zeros((3, 1, 4), dtype=np.float32),
    )

    motion = MotionData(str(path))

    assert motion.num_frames == 3
    assert "joint_pos" in motion
    assert motion["joint_pos"].shape == (3, 2)


def test_first_motion_file_uses_dataset_txt(tmp_path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    dataset = tmp_path / "dataset.txt"
    dataset.write_text("a/b/demo.npz\n")

    assert first_motion_file(str(root), str(dataset)) == str(root / "a/b/demo.npz")


def test_motion_files_expands_dataset_txt_and_ignores_comments(tmp_path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    dataset = tmp_path / "dataset.txt"
    absolute = tmp_path / "absolute.npz"
    dataset.write_text(f"\n# skipped\na/b/demo.npz\n{absolute}  # inline comment\n")

    assert motion_files(str(root), str(dataset)) == [str(root / "a/b/demo.npz"), str(absolute)]


def test_future_indices_clips_to_motion_length() -> None:
    assert np.array_equal(future_indices(3, [0, 2, 100], total=6), np.array([3, 5, 5]))


def test_motion_groups_shape_and_robot_relative_orientation() -> None:
    class Motion:
        files = ["joint_pos", "joint_vel", "body_quat_w"]
        num_frames = 4

        def __init__(self):
            self.values = {
                "joint_pos": np.ones((4, 2), dtype=np.float32),
                "joint_vel": np.zeros((4, 2), dtype=np.float32),
                "body_quat_w": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (4, 1, 1)),
            }

        def __getitem__(self, key):
            return self.values[key]

        def __contains__(self, key):
            return key in self.values

    meta = {
        "future_step_num": [0, 1],
        "motion_body_names": ["pelvis"],
        "anchor_body_name": "pelvis",
        "observation_shapes": {"rbt_cmd_mf": [20], "smpl_cmd_mf": [0]},
    }

    rbt_cmd_mf, smpl_cmd_mf = motion_groups(Motion(), 0, meta, np.array([[1.0, 0.0, 0.0, 0.0]]))

    assert rbt_cmd_mf.shape == (1, 20)
    assert smpl_cmd_mf.shape == (1, 0)
