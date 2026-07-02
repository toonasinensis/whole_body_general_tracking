from __future__ import annotations

import numpy as np

from sim2sim_g1.observations import ImuReader, TermMajorHistory, build_obs, prop_terms_from_metadata


def test_imu_reader_uses_named_mujoco_xml_sensors() -> None:
    class SensorView:
        def __init__(self, data):
            self.data = data

    class Data:
        def sensor(self, name):
            values = {
                "base_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                "base_gyro": np.array([0.2, -0.3, 0.4], dtype=np.float64),
            }
            return SensorView(values[name])

    reader = ImuReader.__new__(ImuReader)
    reader.quat_sensor_name = "base_quat"
    reader.gyro_sensor_name = "base_gyro"

    quat, gyro, gravity = reader.read(Data())

    assert np.array_equal(quat, np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64))
    assert np.array_equal(gyro, np.array([[0.2, -0.3, 0.4]], dtype=np.float64))
    assert np.allclose(gravity, np.array([[0.0, 0.0, -1.0]], dtype=np.float64))


def test_term_major_history_matches_isaaclab_layout() -> None:
    history = TermMajorHistory([("a", 2, 3), ("b", 1, 3)])

    first = history.update({"a": np.array([[1.0, 2.0]]), "b": np.array([[10.0]])})
    assert np.array_equal(first, np.array([[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 10.0, 10.0, 10.0]], dtype=np.float32))

    history.update({"a": np.array([[3.0, 4.0]]), "b": np.array([[20.0]])})
    third = history.update({"a": np.array([[5.0, 6.0]]), "b": np.array([[30.0]])})

    expected_term_major = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0]], dtype=np.float32)
    frame_major = np.array([[1.0, 2.0, 10.0, 3.0, 4.0, 20.0, 5.0, 6.0, 30.0]], dtype=np.float32)
    assert np.array_equal(third, expected_term_major)
    assert not np.array_equal(third, frame_major)


def test_prop_terms_from_new_metadata() -> None:
    meta = {
        "observation_shapes": {"prop": [48]},
        "observation_terms": {
            "prop": {
                "terms": [
                    {
                        "name": "projected_gravity",
                        "shape": [6],
                        "base_shape": [3],
                        "history_length": 2,
                        "flatten_history_dim": True,
                    },
                    {
                        "name": "joint_pos",
                        "shape": [42],
                        "base_shape": [21],
                        "history_length": 2,
                        "flatten_history_dim": True,
                    },
                ]
            }
        },
    }

    assert prop_terms_from_metadata(meta, num_joints=21) == [("projected_gravity", 3, 2), ("joint_pos", 21, 2)]


def test_prop_terms_fallback_for_old_metadata() -> None:
    meta = {"observation_shapes": {"prop": [840]}}

    assert prop_terms_from_metadata(meta, num_joints=26) == [
        ("projected_gravity", 3, 10),
        ("base_ang_vel", 3, 10),
        ("joint_pos", 26, 10),
        ("joint_vel", 26, 10),
        ("actions", 26, 10),
    ]


def test_build_obs_keeps_robot_future_command_available_for_dynamic_policy(monkeypatch) -> None:
    class SensorView:
        def __init__(self, data):
            self.data = data

    class Data:
        qpos = np.zeros(2, dtype=np.float64)
        qvel = np.zeros(2, dtype=np.float64)

        def sensor(self, name):
            values = {
                "base_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                "base_gyro": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            }
            return SensorView(values[name])

    def fake_motion_groups(motion, t, meta, root_quat):
        return np.ones((1, 4), dtype=np.float32), np.ones((1, 2), dtype=np.float32)

    import sim2sim_g1.observations as observations

    monkeypatch.setattr(observations, "motion_groups", fake_motion_groups)
    meta = {"default_joint_pos": [0.0], "default_joint_vel": [0.0]}
    history = TermMajorHistory(
        [
            ("projected_gravity", 3, 1),
            ("base_ang_vel", 3, 1),
            ("joint_pos", 1, 1),
            ("joint_vel", 1, 1),
            ("actions", 1, 1),
        ]
    )
    reader = ImuReader.__new__(ImuReader)
    reader.quat_sensor_name = "base_quat"
    reader.gyro_sensor_name = "base_gyro"

    obs = build_obs(
        Data(),
        motion=None,
        t=0,
        meta=meta,
        imu_reader=reader,
        joint_qpos=np.array([0], dtype=np.int32),
        joint_qvel=np.array([0], dtype=np.int32),
        last_action=np.zeros((1, 1), dtype=np.float32),
        prop_history=history,
    )

    assert np.array_equal(obs["rbt_cmd_mf"], np.ones((1, 4), dtype=np.float32))
    assert np.array_equal(obs["smpl_cmd_mf"], np.ones((1, 2), dtype=np.float32))


def test_build_obs_uses_terrain_scanner_when_metadata_has_terrain(monkeypatch) -> None:
    class SensorView:
        def __init__(self, data):
            self.data = data

    class Data:
        qpos = np.zeros(1, dtype=np.float64)
        qvel = np.zeros(1, dtype=np.float64)

        def sensor(self, name):
            values = {
                "base_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                "base_gyro": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            }
            return SensorView(values[name])

    class TerrainScanner:
        def __init__(self, value):
            self.value = value

        def scan(self, data):
            return self.value

    def fake_motion_groups(motion, t, meta, root_quat):
        return np.zeros((1, 0), dtype=np.float32), np.zeros((1, 0), dtype=np.float32)

    import sim2sim_g1.observations as observations

    monkeypatch.setattr(observations, "motion_groups", fake_motion_groups)
    reader = ImuReader.__new__(ImuReader)
    reader.quat_sensor_name = "base_quat"
    reader.gyro_sensor_name = "base_gyro"
    history = TermMajorHistory(
        [
            ("projected_gravity", 3, 1),
            ("base_ang_vel", 3, 1),
            ("joint_pos", 1, 1),
            ("joint_vel", 1, 1),
            ("actions", 1, 1),
        ]
    )
    terrain = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32)

    obs = build_obs(
        Data(),
        motion=None,
        t=0,
        meta={"observation_shapes": {"terrain": [3]}, "default_joint_pos": [0.0], "default_joint_vel": [0.0]},
        imu_reader=reader,
        joint_qpos=np.array([0], dtype=np.int32),
        joint_qvel=np.array([0], dtype=np.int32),
        last_action=np.zeros((1, 1), dtype=np.float32),
        prop_history=history,
        terrain_scanner=TerrainScanner(terrain),
    )

    assert np.array_equal(obs["terrain"], terrain)


def test_build_obs_rejects_terrain_scanner_dim_mismatch(monkeypatch) -> None:
    class SensorView:
        def __init__(self, data):
            self.data = data

    class Data:
        qpos = np.zeros(1, dtype=np.float64)
        qvel = np.zeros(1, dtype=np.float64)

        def sensor(self, name):
            values = {
                "base_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                "base_gyro": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            }
            return SensorView(values[name])

    class BadTerrainScanner:
        def scan(self, data):
            return np.zeros((1, 2), dtype=np.float32)

    def fake_motion_groups(motion, t, meta, root_quat):
        return np.zeros((1, 0), dtype=np.float32), np.zeros((1, 0), dtype=np.float32)

    import pytest
    import sim2sim_g1.observations as observations

    monkeypatch.setattr(observations, "motion_groups", fake_motion_groups)
    reader = ImuReader.__new__(ImuReader)
    reader.quat_sensor_name = "base_quat"
    reader.gyro_sensor_name = "base_gyro"
    history = TermMajorHistory(
        [
            ("projected_gravity", 3, 1),
            ("base_ang_vel", 3, 1),
            ("joint_pos", 1, 1),
            ("joint_vel", 1, 1),
            ("actions", 1, 1),
        ]
    )

    with pytest.raises(ValueError, match="Height scan dim 2 does not match ONNX terrain dim 3"):
        build_obs(
            Data(),
            motion=None,
            t=0,
            meta={"observation_shapes": {"terrain": [3]}, "default_joint_pos": [0.0], "default_joint_vel": [0.0]},
            imu_reader=reader,
            joint_qpos=np.array([0], dtype=np.int32),
            joint_qvel=np.array([0], dtype=np.int32),
            last_action=np.zeros((1, 1), dtype=np.float32),
            prop_history=history,
            terrain_scanner=BadTerrainScanner(),
        )


def test_build_obs_uses_velcommand_override(monkeypatch) -> None:
    class SensorView:
        def __init__(self, data):
            self.data = data

    class Data:
        qpos = np.zeros(1, dtype=np.float64)
        qvel = np.zeros(1, dtype=np.float64)
        xpos = np.asarray([[0.5, 1.0, 0.3]], dtype=np.float64)
        xquat = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)

        def sensor(self, name):
            values = {
                "base_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                "base_gyro": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            }
            return SensorView(values[name])

    class Motion:
        num_frames = 1

        arrays = {
            "joint_pos": np.asarray([[0.2]], dtype=np.float64),
            "joint_vel": np.asarray([[-0.1]], dtype=np.float64),
            "body_pos_w": np.asarray([[[1.0, 2.0, 0.8]]], dtype=np.float64),
            "body_quat_w": np.asarray([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float64),
            "body_lin_vel_w": np.asarray([[[0.4, 0.0, 0.0]]], dtype=np.float64),
            "body_ang_vel_w": np.asarray([[[0.0, 0.0, 0.7]]], dtype=np.float64),
        }

        def __getitem__(self, key):
            return self.arrays[key]

        def __contains__(self, key):
            return key in self.arrays

    def fake_motion_groups(motion, t, meta, root_quat):
        return np.zeros((1, 0), dtype=np.float32), np.zeros((1, 0), dtype=np.float32)

    import sim2sim_g1.observations as observations

    monkeypatch.setattr(observations, "motion_groups", fake_motion_groups)
    reader = ImuReader.__new__(ImuReader)
    reader.quat_sensor_name = "base_quat"
    reader.gyro_sensor_name = "base_gyro"
    history = TermMajorHistory(
        [
            ("projected_gravity", 3, 1),
            ("base_ang_vel", 3, 1),
            ("joint_pos", 1, 1),
            ("joint_vel", 1, 1),
            ("actions", 1, 1),
        ]
    )
    velcommand = np.asarray([[0.4, -0.2, 0.7]], dtype=np.float32)

    obs = build_obs(
        Data(),
        motion=Motion(),
        t=0,
        meta={
            "observation_shapes": {"velcommand": [3], "vel_task_mask": [1], "wbc_cmd": [21]},
            "observation_terms": {
                "wbc_cmd": {
                    "terms": [
                        {"name": "command"},
                        {"name": "motion_anchor_pos_b"},
                        {"name": "motion_anchor_ori_b"},
                    ]
                }
            },
            "motion_body_names": ["torso_link"],
            "anchor_body_name": "torso_link",
            "default_joint_pos": [0.0],
            "default_joint_vel": [0.0],
        },
        imu_reader=reader,
        joint_qpos=np.array([0], dtype=np.int32),
        joint_qvel=np.array([0], dtype=np.int32),
        last_action=np.zeros((1, 1), dtype=np.float32),
        prop_history=history,
        body_ids=np.array([0], dtype=np.int32),
        velcommand_override=velcommand,
    )

    assert np.array_equal(obs["velcommand"], velcommand)
    assert np.array_equal(obs["vel_task_mask"], np.ones((1, 1), dtype=np.float32))
    expected_wbc_cmd = np.asarray(
        [[0.2, -0.1, 0.4, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, -1.0, 0.8, 0.5, 1.0, 0.5, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    assert np.allclose(obs["wbc_cmd"], expected_wbc_cmd)


def test_build_obs_uses_zero_velocity_task_mask_for_motion_command(monkeypatch) -> None:
    class SensorView:
        def __init__(self, data):
            self.data = data

    class Data:
        qpos = np.zeros(1, dtype=np.float64)
        qvel = np.zeros(1, dtype=np.float64)

        def sensor(self, name):
            values = {
                "base_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                "base_gyro": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            }
            return SensorView(values[name])

    import sim2sim_g1.observations as observations

    monkeypatch.setattr(
        observations,
        "build_velcommand_obs",
        lambda motion, t, meta: np.zeros((1, 3), dtype=np.float32),
    )
    monkeypatch.setattr(
        observations,
        "motion_groups",
        lambda motion, t, meta, root_quat: (
            np.zeros((1, 0), dtype=np.float32),
            np.zeros((1, 0), dtype=np.float32),
        ),
    )
    reader = ImuReader.__new__(ImuReader)
    reader.quat_sensor_name = "base_quat"
    reader.gyro_sensor_name = "base_gyro"
    history = TermMajorHistory(
        [
            ("projected_gravity", 3, 1),
            ("base_ang_vel", 3, 1),
            ("joint_pos", 1, 1),
            ("joint_vel", 1, 1),
            ("actions", 1, 1),
        ]
    )

    obs = build_obs(
        Data(),
        motion=None,
        t=0,
        meta={
            "observation_shapes": {"velcommand": [3], "vel_task_mask": [1]},
            "default_joint_pos": [0.0],
            "default_joint_vel": [0.0],
        },
        imu_reader=reader,
        joint_qpos=np.array([0], dtype=np.int32),
        joint_qvel=np.array([0], dtype=np.int32),
        last_action=np.zeros((1, 1), dtype=np.float32),
        prop_history=history,
    )

    assert np.array_equal(obs["vel_task_mask"], np.zeros((1, 1), dtype=np.float32))
