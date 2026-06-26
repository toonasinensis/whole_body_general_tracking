from __future__ import annotations

import numpy as np

from .math_utils import as_vector, matrix_from_quat, quat_apply_inverse, quat_inv, quat_mul, shape_dim
from .motion import MotionData, build_rbt_cmd_mf, build_smpl_cmd_mf, future_indices, motion_body_index, motion_groups
from .mujoco_robot import validate_sensor

G1_PROP_TERM_ORDER = ("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions")


class ImuReader:
    def __init__(
        self,
        model,
        quat_sensor_name: str = "base_quat",
        gyro_sensor_name: str = "base_gyro",
    ):
        self.quat_sensor_name = quat_sensor_name
        self.gyro_sensor_name = gyro_sensor_name
        validate_sensor(model, quat_sensor_name, expected_dim=4)
        validate_sensor(model, gyro_sensor_name, expected_dim=3)

    def read(self, data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        root_quat = np.asarray(data.sensor(self.quat_sensor_name).data, dtype=np.float64)[None, :]
        root_ang_vel_b = np.asarray(data.sensor(self.gyro_sensor_name).data, dtype=np.float64)[None, :]
        gravity_b = quat_apply_inverse(root_quat, np.asarray([[0.0, 0.0, -1.0]], dtype=np.float64))
        return root_quat, root_ang_vel_b, gravity_b

    def print_config(self) -> None:
        print(f"[INFO] IMU source: MuJoCo XML sensors quat={self.quat_sensor_name} gyro={self.gyro_sensor_name}")


class HistoryBuffer:
    def __init__(self, dim: int, length: int):
        self.dim = int(dim)
        self.length = max(1, int(length))
        self._values = np.zeros((self.length, self.dim), dtype=np.float32)
        self._filled = False

    def update(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if value.shape[0] != self.dim:
            raise ValueError(f"History value dim {value.shape[0]} does not match expected dim {self.dim}.")
        if not self._filled:
            self._values[:] = value
            self._filled = True
        else:
            self._values[:-1] = self._values[1:]
            self._values[-1] = value
        return self.value()

    def value(self) -> np.ndarray:
        return self._values.reshape(1, -1).astype(np.float32)


class TermMajorHistory:
    def __init__(self, term_dims: list[tuple[str, int, int]]):
        self.term_dims = term_dims
        self.buffers = {name: HistoryBuffer(dim, history_len) for name, dim, history_len in term_dims}

    @property
    def dim(self) -> int:
        return sum(dim * history_len for _, dim, history_len in self.term_dims)

    @property
    def max_history_length(self) -> int:
        return max((history_len for _, _, history_len in self.term_dims), default=1)

    def update(self, values: dict[str, np.ndarray]) -> np.ndarray:
        pieces = []
        for name, _, _ in self.term_dims:
            if name not in values:
                raise KeyError(f"Missing observation term '{name}'.")
            pieces.append(self.buffers[name].update(values[name]))
        return np.concatenate(pieces, axis=-1).astype(np.float32)


def fallback_prop_terms(prop_dim: int, num_joints: int) -> list[tuple[str, int, int]]:
    base_dims = {
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "joint_pos": num_joints,
        "joint_vel": num_joints,
        "actions": num_joints,
    }
    single_dim = sum(base_dims.values())
    if prop_dim % single_dim != 0:
        raise ValueError(f"prop dim {prop_dim} is not divisible by fallback single-frame dim {single_dim}.")
    history_len = prop_dim // single_dim
    return [(name, base_dims[name], history_len) for name in G1_PROP_TERM_ORDER]


def prop_terms_from_metadata(meta: dict, num_joints: int) -> list[tuple[str, int, int]]:
    return observation_terms_from_metadata(meta, "prop", num_joints)


def observation_terms_from_metadata(meta: dict, group_name: str, num_joints: int) -> list[tuple[str, int, int]]:
    group_dim = int(np.prod(meta["observation_shapes"].get(group_name, [0])))
    observation_terms = meta.get("observation_terms", {})
    group_terms = observation_terms.get(group_name, {}).get("terms", [])
    if not group_terms:
        if group_name == "prop":
            return fallback_prop_terms(group_dim, num_joints)
        raise ValueError(
            f"ONNX metadata has no observation_terms for group '{group_name}'. "
            "Re-export with the updated grouped play.py exporter."
        )

    out = []
    for term in group_terms:
        name = term["name"]
        shape = term.get("shape", [])
        base_shape = term.get("base_shape", shape)
        history_len = max(1, int(term.get("history_length", 0)))
        term_dim = shape_dim(base_shape)
        if term_dim <= 0:
            term_dim = shape_dim(shape)
            if history_len > 1 and term_dim % history_len == 0:
                term_dim //= history_len
        out.append((name, term_dim, history_len))

    total = sum(dim * history_len for _, dim, history_len in out)
    if total != group_dim:
        raise ValueError(f"{group_name} metadata describes dim {total}, but ONNX metadata expects {group_dim}: {out}")
    return out


def histories_from_metadata(meta: dict, input_names: list[str], num_joints: int) -> dict[str, TermMajorHistory]:
    histories = {}
    for name in input_names:
        if name in ("policy", "prop"):
            histories[name] = TermMajorHistory(observation_terms_from_metadata(meta, name, num_joints))
    return histories


def print_obs_layout(meta: dict, group_histories: dict[str, TermMajorHistory], input_names: list[str]) -> None:
    shapes = meta.get("observation_shapes", {})
    print("[INFO] ONNX inputs:", input_names)
    for group_name, history in group_histories.items():
        print(f"[INFO] {group_name} dim: {history.dim}, history max length: {history.max_history_length}")
        for name, dim, history_len in history.term_dims:
            print(f"[INFO]   {group_name}/{name}: base_dim={dim}, history={history_len}, flat_dim={dim * history_len}")
    for name in input_names:
        print(f"[INFO]   input/{name}: expected_shape={tuple(shapes.get(name, []))}")


def print_imu_debug(data, imu_reader: ImuReader) -> None:
    sensor_quat = np.asarray(data.sensor(imu_reader.quat_sensor_name).data, dtype=np.float64)
    sensor_gyro = np.asarray(data.sensor(imu_reader.gyro_sensor_name).data, dtype=np.float64)
    sensor_gravity_b = quat_apply_inverse(sensor_quat[None, :], np.asarray([[0.0, 0.0, -1.0]], dtype=np.float64))[0]
    print("========== MUJOCO IMU DEBUG ==========")
    print("[IMUDBG] active source: MuJoCo XML sensors")
    print(f"[IMUDBG] sensor {imu_reader.quat_sensor_name} quat wxyz: {sensor_quat.tolist()}")
    print(f"[IMUDBG] sensor {imu_reader.gyro_sensor_name} gyro: {sensor_gyro.tolist()}")
    print(f"[IMUDBG] sensor projected gravity: {sensor_gravity_b.tolist()}")
    print("======================================")


def _robot_terms(
    data,
    meta: dict,
    joint_qpos: np.ndarray,
    joint_qvel: np.ndarray,
    last_action: np.ndarray,
    imu_reader: ImuReader,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    root_quat, root_ang_vel_b, gravity_b = imu_reader.read(data)
    default_joint_pos = as_vector(meta, "default_joint_pos", len(joint_qpos), 0.0)
    default_joint_vel = as_vector(meta, "default_joint_vel", len(joint_qvel), 0.0)
    qpos = data.qpos[joint_qpos][None, :]
    qvel = data.qvel[joint_qvel][None, :]
    return root_quat, {
        "projected_gravity": gravity_b,
        "base_ang_vel": root_ang_vel_b,
        "joint_pos": qpos - default_joint_pos[None, :],
        "joint_vel": qvel - default_joint_vel[None, :],
        "actions": last_action,
    }


def _motion_current_terms(motion: MotionData, t: int, meta: dict, robot_anchor_quat_w: np.ndarray) -> dict[str, np.ndarray]:
    frame = int(np.clip(t, 0, motion.num_frames - 1))
    _, anchor_idx = motion_body_index(meta, meta.get("anchor_body_name"))
    anchor_quat_w = np.asarray(motion["body_quat_w"][frame, anchor_idx], dtype=np.float64).reshape(1, 4)
    anchor_pos_w = np.asarray(motion["body_pos_w"][frame, anchor_idx], dtype=np.float64).reshape(1, 3)
    anchor_lin_vel_w = (
        np.asarray(motion["body_lin_vel_w"][frame, anchor_idx], dtype=np.float64).reshape(1, 3)
        if "body_lin_vel_w" in motion
        else np.zeros((1, 3), dtype=np.float64)
    )
    anchor_ang_vel_w = (
        np.asarray(motion["body_ang_vel_w"][frame, anchor_idx], dtype=np.float64).reshape(1, 3)
        if "body_ang_vel_w" in motion
        else np.zeros((1, 3), dtype=np.float64)
    )
    rel_quat = quat_mul(quat_inv(robot_anchor_quat_w), anchor_quat_w)
    return {
        "motion_joint_pos": np.asarray(motion["joint_pos"][frame], dtype=np.float32).reshape(1, -1),
        "motion_joint_vel": np.asarray(motion["joint_vel"][frame], dtype=np.float32).reshape(1, -1),
        "motion_anchor_lin_vel_b": quat_apply_inverse(anchor_quat_w, anchor_lin_vel_w).astype(np.float32),
        "motion_anchor_ang_vel_b": quat_apply_inverse(anchor_quat_w, anchor_ang_vel_w).astype(np.float32),
        "motion_anchor_project_gravity": quat_apply_inverse(
            anchor_quat_w, np.asarray([[0.0, 0.0, -1.0]], dtype=np.float64)
        ).astype(np.float32),
        "motion_anchor_pos_z": anchor_pos_w[:, 2:3].astype(np.float32),
        "motion_anchor_ori_b": matrix_from_quat(rel_quat)[..., :2].reshape(1, -1).astype(np.float32),
    }


def _build_history_group(
    group_name: str,
    values: dict[str, np.ndarray],
    group_histories: dict[str, TermMajorHistory],
) -> np.ndarray:
    if group_name not in group_histories:
        raise KeyError(f"Observation group '{group_name}' needs history metadata but no history buffer was created.")
    return group_histories[group_name].update(values)


def _build_motion_group(motion: MotionData, t: int, meta: dict, root_quat: np.ndarray, group_name: str) -> np.ndarray:
    offsets = meta["future_step_num"]
    future = future_indices(t, offsets, motion["joint_pos"].shape[0])
    shapes = meta.get("observation_shapes", {})
    if group_name in ("rbt_cmd_mf", "zrbt_cmd_mf"):
        group_dim = int(np.prod(shapes.get(group_name, [0])))
        return build_rbt_cmd_mf(motion, future, meta, root_quat, group_dim, group_name=group_name)
    if group_name == "smpl_cmd_mf":
        group_dim = int(np.prod(shapes.get(group_name, [0])))
        return build_smpl_cmd_mf(motion, future, meta, root_quat, group_dim)
    raise KeyError(f"Unsupported motion observation group '{group_name}'.")


def build_obs(
    data,
    motion: MotionData,
    t: int,
    meta: dict,
    imu_reader: ImuReader,
    joint_qpos: np.ndarray,
    joint_qvel: np.ndarray,
    last_action: np.ndarray,
    prop_history: TermMajorHistory | None = None,
    input_names: list[str] | None = None,
    group_histories: dict[str, TermMajorHistory] | None = None,
) -> dict[str, np.ndarray]:
    if input_names is None:
        input_names = ["prop", "rbt_cmd_mf", "smpl_cmd_mf"]
    if group_histories is None:
        group_histories = {}
    if prop_history is not None and "prop" not in group_histories:
        group_histories = {**group_histories, "prop": prop_history}

    root_quat, robot_terms = _robot_terms(data, meta, joint_qpos, joint_qvel, last_action, imu_reader)
    obs: dict[str, np.ndarray] = {}
    for group_name in input_names:
        if group_name == "prop":
            obs[group_name] = _build_history_group(group_name, robot_terms, group_histories)
        elif group_name == "policy":
            values = {**_motion_current_terms(motion, t, meta, root_quat), **robot_terms}
            obs[group_name] = _build_history_group(group_name, values, group_histories)
        elif group_name in ("rbt_cmd_mf", "zrbt_cmd_mf", "smpl_cmd_mf"):
            if input_names == ["prop", "rbt_cmd_mf", "smpl_cmd_mf"] and group_name in ("rbt_cmd_mf", "smpl_cmd_mf"):
                rbt_cmd_mf, smpl_cmd_mf = motion_groups(motion, t, meta, root_quat)
                if "rbt_cmd_mf" not in meta.get("observation_shapes", {}):
                    rbt_cmd_mf = np.zeros_like(rbt_cmd_mf)
                obs["rbt_cmd_mf"] = rbt_cmd_mf
                obs["smpl_cmd_mf"] = smpl_cmd_mf
            elif group_name not in obs:
                obs[group_name] = _build_motion_group(motion, t, meta, root_quat, group_name)
        else:
            raise KeyError(f"Unsupported ONNX observation group '{group_name}'.")
    return {name: obs[name].astype(np.float32) for name in input_names}


def validate_inputs(obs: dict[str, np.ndarray], input_names: list[str], meta: dict) -> None:
    shapes = meta.get("observation_shapes", {})
    missing = [name for name in input_names if name not in obs]
    if missing:
        raise KeyError(f"ONNX expects unsupported input groups: {missing}")
    for name in input_names:
        expected = tuple(shapes.get(name, obs[name].shape[1:]))
        got = tuple(obs[name].shape[1:])
        if got != expected:
            raise ValueError(f"Observation group '{name}' has shape {got}, expected {expected}.")
