from __future__ import annotations

import numpy as np

from .math_utils import as_vector, quat_apply_inverse, shape_dim
from .motion import MotionData, motion_groups
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
                raise KeyError(f"Missing prop observation term '{name}'.")
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
    prop_dim = int(np.prod(meta["observation_shapes"].get("prop", [0])))
    observation_terms = meta.get("observation_terms", {})
    prop_terms = observation_terms.get("prop", {}).get("terms", [])
    if not prop_terms:
        return fallback_prop_terms(prop_dim, num_joints)

    out = []
    for term in prop_terms:
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
    if total != prop_dim:
        raise ValueError(f"prop metadata describes dim {total}, but ONNX metadata expects {prop_dim}: {out}")
    return out


def print_obs_layout(meta: dict, prop_history: TermMajorHistory, input_names: list[str]) -> None:
    shapes = meta.get("observation_shapes", {})
    print("[INFO] ONNX inputs:", input_names)
    print(f"[INFO] prop dim: {prop_history.dim}, history max length: {prop_history.max_history_length}")
    for name, dim, history_len in prop_history.term_dims:
        print(f"[INFO]   prop/{name}: base_dim={dim}, history={history_len}, flat_dim={dim * history_len}")
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


def build_obs(
    data,
    motion: MotionData,
    t: int,
    meta: dict,
    imu_reader: ImuReader,
    joint_qpos: np.ndarray,
    joint_qvel: np.ndarray,
    last_action: np.ndarray,
    prop_history: TermMajorHistory,
    robot_anchor_quat_w: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    root_quat, root_ang_vel_b, gravity_b = imu_reader.read(data)
    default_joint_pos = as_vector(meta, "default_joint_pos", len(joint_qpos), 0.0)
    default_joint_vel = as_vector(meta, "default_joint_vel", len(joint_qvel), 0.0)
    qpos = data.qpos[joint_qpos][None, :]
    qvel = data.qvel[joint_qvel][None, :]
    prop_terms = {
        "projected_gravity": gravity_b,
        "base_ang_vel": root_ang_vel_b,
        "joint_pos": qpos - default_joint_pos[None, :],
        "joint_vel": qvel - default_joint_vel[None, :],
        "actions": last_action,
    }

    prop = prop_history.update(prop_terms)
    if robot_anchor_quat_w is None:
        robot_anchor_quat_w = root_quat
    rbt_cmd_mf, smpl_cmd_mf = motion_groups(motion, t, meta, robot_anchor_quat_w)
    return {
        "prop": prop.astype(np.float32),
        "rbt_cmd_mf": rbt_cmd_mf,
        "smpl_cmd_mf": smpl_cmd_mf,
    }


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
