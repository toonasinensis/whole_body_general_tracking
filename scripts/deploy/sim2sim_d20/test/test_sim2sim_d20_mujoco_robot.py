from __future__ import annotations

import numpy as np

from sim2sim_d20.mujoco_robot import (
    D20_MJCF,
    action_to_target,
    actuator_name_for_joint,
    name_to_actuator_ids,
    name_to_joint_qvel_addrs,
)


def test_actuator_name_for_joint() -> None:
    assert actuator_name_for_joint("waist_yaw_joint") == "waist_yaw_motor"
    assert actuator_name_for_joint("left_hip_pitch_joint") == "left_hip_pitch_motor"


def test_action_to_target_uses_policy_output() -> None:
    raw_action = np.array([[0.5, -2.0]], dtype=np.float32)
    action_scale = np.array([2.0, 3.0], dtype=np.float64)
    action_offset = np.array([1.0, -1.0], dtype=np.float64)

    target = action_to_target(raw_action, action_scale, action_offset)

    assert np.allclose(target, np.array([2.0, -7.0]))


def test_d20_mjcf_actuator_mapping() -> None:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(D20_MJCF))
    joint_names = [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "right_ankle_roll_joint",
    ]
    actuator_ids = name_to_actuator_ids(model, joint_names)
    joint_qpos, joint_qvel = name_to_joint_qvel_addrs(model, joint_names)

    assert actuator_ids.shape == (len(joint_names),)
    assert joint_qpos.shape == (len(joint_names),)
    assert joint_qvel.shape == (len(joint_names),)
    assert int(model.nu) == 25

    expected_actuators = [actuator_name_for_joint(name) for name in joint_names]
    for aid, expected_name in zip(actuator_ids, expected_actuators):
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(aid)) == expected_name
