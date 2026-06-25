from __future__ import annotations

import numpy as np
from pathlib import Path

from sim2sim_g1.math_utils import as_vector
from sim2sim_g1.mujoco_robot import (
    action_to_target,
    apply_pd_control,
    gains_from_metadata,
    initialize_from_motion,
    name_to_body_ids,
    name_to_joint_ids,
    name_to_joint_qvel_addrs,
    print_joint_map,
    validate_sensor,
)

D20_MJCF = (
    Path(__file__).resolve().parents[3]
    / "source/whole_body_tracking/whole_body_tracking/assets/d20_v2/mjcf/D20.xml"
)

D20_DEFAULT_ROOT_Z = 1.05


def actuator_name_for_joint(joint_name: str) -> str:
    if joint_name.endswith("_joint"):
        return f"{joint_name[:-6]}_motor"
    return f"{joint_name}_motor"


def name_to_actuator_ids(model, joint_names: list[str]) -> np.ndarray:
    import mujoco

    ids = []
    for name in joint_names:
        candidates = (name, actuator_name_for_joint(name))
        aid = -1
        for candidate in candidates:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, candidate)
            if aid >= 0:
                break
        if aid < 0:
            for actuator_id in range(model.nu):
                joint_id = model.actuator_trnid[actuator_id, 0]
                if joint_id < 0:
                    continue
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                if joint_name == name:
                    aid = actuator_id
                    break
        if aid < 0:
            raise ValueError(
                f"Actuator for joint '{name}' not found in MuJoCo model. "
                f"Tried actuator names {list(candidates)} and joint-attached actuators."
            )
        ids.append(aid)
    return np.asarray(ids, dtype=np.int32)


def initialize_default_pose(data, meta: dict, joint_names: list[str], joint_qpos: np.ndarray) -> None:
    default_joint_pos = as_vector(meta, "default_joint_pos", len(joint_names), 0.0)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[2] = D20_DEFAULT_ROOT_Z
    data.qpos[3] = 1.0
    data.qpos[joint_qpos] = default_joint_pos


__all__ = [
    "D20_DEFAULT_ROOT_Z",
    "D20_MJCF",
    "action_to_target",
    "actuator_name_for_joint",
    "apply_pd_control",
    "gains_from_metadata",
    "initialize_default_pose",
    "initialize_from_motion",
    "name_to_actuator_ids",
    "name_to_body_ids",
    "name_to_joint_ids",
    "name_to_joint_qvel_addrs",
    "print_joint_map",
    "validate_sensor",
]
