from __future__ import annotations

import numpy as np

from sim2sim_g1.mujoco_robot import action_to_target


def test_action_to_target_uses_policy_output() -> None:
    raw_action = np.array([[0.5, -2.0]], dtype=np.float32)
    action_scale = np.array([2.0, 3.0], dtype=np.float64)
    action_offset = np.array([1.0, -1.0], dtype=np.float64)

    target = action_to_target(raw_action, action_scale, action_offset)

    assert np.allclose(target, np.array([2.0, -7.0]))
