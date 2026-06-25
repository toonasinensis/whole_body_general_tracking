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


def test_build_obs_keeps_robot_future_command_zeroed_like_training(monkeypatch) -> None:
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

    assert np.array_equal(obs["rbt_cmd_mf"], np.zeros((1, 4), dtype=np.float32))
    assert np.array_equal(obs["smpl_cmd_mf"], np.ones((1, 2), dtype=np.float32))
