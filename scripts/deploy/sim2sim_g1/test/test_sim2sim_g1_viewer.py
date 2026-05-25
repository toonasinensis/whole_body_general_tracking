from __future__ import annotations

import numpy as np
import sys

from sim2sim_g1.viewer import ReferenceMotionPlayer


def test_reference_motion_player_writes_motion_frame_to_qpos() -> None:
    class Model:
        qpos0 = np.zeros(10, dtype=np.float64)

    class Motion:
        files = ["joint_pos", "body_pos_w", "body_quat_w"]

        def __init__(self):
            self.values = {
                "joint_pos": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
                "body_pos_w": np.array([[[0.4, 0.5, 0.6], [1.4, 1.5, 1.6]]], dtype=np.float32),
                "body_quat_w": np.array([[[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]]], dtype=np.float32),
            }

        def __getitem__(self, key):
            return self.values[key]

    player = ReferenceMotionPlayer.__new__(ReferenceMotionPlayer)
    player.model = Model()
    player.motion = Motion()
    player.data = type("Data", (), {})()
    player.data.qpos = np.zeros(10, dtype=np.float64)
    player.data.qvel = np.ones(9, dtype=np.float64)
    player.joint_qpos = np.array([7, 8, 9], dtype=np.int32)
    player.frame_count = 1
    player.root_body_index = 1
    player._last_frame = None

    original_mujoco = sys.modules.get("mujoco")
    fake_mujoco = type("FakeMujoco", (), {"mj_forward": staticmethod(lambda model, data: None)})
    sys.modules["mujoco"] = fake_mujoco
    try:
        player.update_data(0)
    finally:
        if original_mujoco is None:
            del sys.modules["mujoco"]
        else:
            sys.modules["mujoco"] = original_mujoco

    assert np.allclose(player.data.qpos[:3], np.array([1.4, 1.5, 1.6]))
    assert np.allclose(player.data.qpos[3:7], np.array([0.5, 0.5, 0.5, 0.5]))
    assert np.allclose(player.data.qpos[7:10], np.array([1.0, 2.0, 3.0]))
    assert np.allclose(player.data.qvel, 0.0)
