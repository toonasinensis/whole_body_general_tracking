from __future__ import annotations

import argparse
import json
import numpy as np
import time
from pathlib import Path

import onnx

G1_PROP_TERM_ORDER = ("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions")

G1_MJCF = (
    Path(__file__).resolve().parents[2]
    / "source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml"
)


def _metadata(path: str) -> dict:
    model = onnx.load(path)
    out = {}
    for entry in model.metadata_props:
        try:
            out[entry.key] = json.loads(entry.value)
        except json.JSONDecodeError:
            out[entry.key] = entry.value
    return out


def _onnx_input_names(path: str) -> list[str]:
    model = onnx.load(path)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    return [inp.name for inp in model.graph.input if inp.name not in initializer_names]


def _first_motion_file(motion_file: str, dataset_txt: str | None) -> str:
    root = Path(motion_file)
    if dataset_txt:
        for line in Path(dataset_txt).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            return str(p if p.is_absolute() else root / p)
        raise ValueError(f"No motion entries in dataset txt: {dataset_txt}")
    if root.is_file():
        return str(root)
    files = sorted(root.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under: {motion_file}")
    return str(files[0])


def _name_to_joint_qvel_addrs(model, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    qpos_addrs = []
    qvel_addrs = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint '{name}' from ONNX metadata not found in MuJoCo model.")
        qpos_addrs.append(model.jnt_qposadr[jid])
        qvel_addrs.append(model.jnt_dofadr[jid])
    return np.asarray(qpos_addrs, dtype=np.int32), np.asarray(qvel_addrs, dtype=np.int32)


def _name_to_joint_ids(model, joint_names: list[str]) -> np.ndarray:
    import mujoco

    ids = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint '{name}' from ONNX metadata not found in MuJoCo model.")
        ids.append(jid)
    return np.asarray(ids, dtype=np.int32)


def _name_to_body_ids(model, body_names: list[str]) -> np.ndarray:
    import mujoco

    ids = []
    for name in body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body '{name}' from ONNX metadata not found in MuJoCo model.")
        ids.append(bid)
    return np.asarray(ids, dtype=np.int32)


def _as_vector(meta: dict, key: str, size: int, default: float = 0.0) -> np.ndarray:
    value = meta.get(key, default)
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(size, float(arr), dtype=np.float64)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if arr.size != size:
        raise ValueError(f"Metadata '{key}' has {arr.size} values, expected {size}.")
    return arr


def _resize_or_zero(values: np.ndarray, dim: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(1, -1)
    if values.shape[1] == dim:
        return values
    out = np.zeros((1, dim), dtype=np.float32)
    width = min(dim, values.shape[1])
    out[:, :width] = values[:, :width]
    return out


def _quat_inv(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    norm_sq = np.sum(q * q, axis=-1, keepdims=True)
    return out / np.clip(norm_sq, 1.0e-9, None)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def _quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    xyz = q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v - q[..., :1] * t + np.cross(xyz, t)


def _matrix_from_quat(q: np.ndarray) -> np.ndarray:
    r, i, j, k = np.moveaxis(q, -1, 0)
    two_s = 2.0 / np.clip(np.sum(q * q, axis=-1), 1.0e-9, None)
    mat = np.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        axis=-1,
    )
    return mat.reshape(q.shape[:-1] + (3, 3))


def _future_indices(t: int, offsets: list[int], total: int) -> np.ndarray:
    return np.clip(t + np.asarray(offsets, dtype=np.int64), 0, total - 1)


def _motion_groups(
    motion: np.lib.npyio.NpzFile,
    t: int,
    meta: dict,
    robot_anchor_quat_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = meta["future_step_num"]
    future = _future_indices(t, offsets, motion["joint_pos"].shape[0])
    target_quat = _motion_anchor_ori_mf(motion, future, meta).reshape(1, len(offsets), 4)
    robot_anchor = np.repeat(robot_anchor_quat_w[:, None, :], len(offsets), axis=1)
    rel_quat = _quat_mul(_quat_inv(robot_anchor), target_quat)
    rel_6d = _matrix_from_quat(rel_quat)[..., :2].reshape(1, -1).astype(np.float32)
    rbt_cmd_mf = np.concatenate(
        [
            motion["joint_pos"][future].reshape(1, -1),
            motion["joint_vel"][future].reshape(1, -1),
            rel_6d,
        ],
        axis=-1,
    ).astype(np.float32)
    rbt_dim = int(np.prod(meta["observation_shapes"].get("rbt_cmd_mf", rbt_cmd_mf.shape[1:])))
    if rbt_cmd_mf.shape[1] != rbt_dim:
        raise ValueError(
            f"Built rbt_cmd_mf with dim {rbt_cmd_mf.shape[1]}, but ONNX metadata expects {rbt_dim}. "
            "Check motion joint order/future_step_num and the exported task config."
        )
    smpl_dim = int(np.prod(meta["observation_shapes"].get("smpl_cmd_mf", [0])))
    smpl_cmd_mf = _build_smpl_cmd_mf(motion, future, meta, robot_anchor_quat_w, smpl_dim)
    return rbt_cmd_mf, smpl_cmd_mf


def _motion_anchor_ori_mf(motion: np.lib.npyio.NpzFile, future: np.ndarray, meta: dict) -> np.ndarray:
    body_names = list(meta["motion_body_names"])
    anchor_idx = body_names.index(meta["anchor_body_name"])
    # MuJoCo root orientation is used as robot anchor orientation in the caller.  Here we encode target anchor
    # orientation in world frame; robot-relative correction is applied in build_obs().
    return motion["body_quat_w"][future, anchor_idx, :]


def _build_smpl_cmd_mf(
    motion: np.lib.npyio.NpzFile,
    future: np.ndarray,
    meta: dict,
    robot_anchor_quat_w: np.ndarray,
    smpl_dim: int,
) -> np.ndarray:
    if smpl_dim == 0:
        return np.zeros((1, 0), dtype=np.float32)
    if "smpl_joints" not in motion.files:
        return np.zeros((1, smpl_dim), dtype=np.float32)

    joints = np.asarray(motion["smpl_joints"][future], dtype=np.float32)
    joints = joints.reshape(1, joints.shape[0], 24, 3)
    root_quat_dim = 6 * len(future)
    if "smpl_poses" in motion.files:
        # Full SMPL root orientation parity with IsaacLab needs its axis-angle conversion path.
        # Keep the dimensional contract explicit until that deployment dependency is wired in.
        root_6d = np.zeros((1, root_quat_dim), dtype=np.float32)
    else:
        root_6d = np.zeros((1, root_quat_dim), dtype=np.float32)
    local_joints = joints.reshape(1, -1)
    return _resize_or_zero(np.concatenate([local_joints, root_6d], axis=-1), smpl_dim)


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


def _shape_dim(shape: list[int] | tuple[int, ...]) -> int:
    return int(np.prod(shape)) if shape else 0


def _fallback_prop_terms(prop_dim: int, num_joints: int) -> list[tuple[str, int, int]]:
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


def _prop_terms_from_metadata(meta: dict, num_joints: int) -> list[tuple[str, int, int]]:
    prop_dim = int(np.prod(meta["observation_shapes"].get("prop", [0])))
    observation_terms = meta.get("observation_terms", {})
    prop_terms = observation_terms.get("prop", {}).get("terms", [])
    if not prop_terms:
        return _fallback_prop_terms(prop_dim, num_joints)

    out = []
    for term in prop_terms:
        name = term["name"]
        shape = term.get("shape", [])
        base_shape = term.get("base_shape", shape)
        history_len = max(1, int(term.get("history_length", 0)))
        term_dim = _shape_dim(base_shape)
        if term_dim <= 0:
            term_dim = _shape_dim(shape)
            if history_len > 1 and term_dim % history_len == 0:
                term_dim //= history_len
        out.append((name, term_dim, history_len))

    total = sum(dim * history_len for _, dim, history_len in out)
    if total != prop_dim:
        raise ValueError(f"prop metadata describes dim {total}, but ONNX metadata expects {prop_dim}: {out}")
    return out


def _print_obs_layout(meta: dict, prop_history: TermMajorHistory, input_names: list[str]) -> None:
    shapes = meta.get("observation_shapes", {})
    print("[INFO] ONNX inputs:", input_names)
    print(f"[INFO] prop dim: {prop_history.dim}, history max length: {prop_history.max_history_length}")
    for name, dim, history_len in prop_history.term_dims:
        print(f"[INFO]   prop/{name}: base_dim={dim}, history={history_len}, flat_dim={dim * history_len}")
    for name in input_names:
        print(f"[INFO]   input/{name}: expected_shape={tuple(shapes.get(name, []))}")


def _action_to_target(raw_action: np.ndarray, action_scale: np.ndarray, action_offset: np.ndarray) -> np.ndarray:
    return raw_action[0].astype(np.float64) * action_scale + action_offset


def _build_obs(
    data,
    motion,
    t: int,
    meta: dict,
    body_ids: np.ndarray,
    joint_qpos: np.ndarray,
    joint_qvel: np.ndarray,
    last_action: np.ndarray,
    prop_history: TermMajorHistory,
):
    root_quat = data.xquat[body_ids[0]][None, :]
    root_ang_vel_b = _quat_apply_inverse(root_quat, data.qvel[3:6][None, :])
    gravity_b = _quat_apply_inverse(root_quat, np.asarray([[0.0, 0.0, -1.0]], dtype=np.float64))
    default_joint_pos = _as_vector(meta, "default_joint_pos", len(joint_qpos), 0.0)
    default_joint_vel = _as_vector(meta, "default_joint_vel", len(joint_qvel), 0.0)
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
    rbt_cmd_mf, smpl_cmd_mf = _motion_groups(motion, t, meta, root_quat)
    return {
        "prop": prop.astype(np.float32),
        "rbt_cmd_mf": rbt_cmd_mf,
        "smpl_cmd_mf": smpl_cmd_mf,
    }


def _validate_inputs(obs: dict[str, np.ndarray], input_names: list[str], meta: dict) -> None:
    shapes = meta.get("observation_shapes", {})
    missing = [name for name in input_names if name not in obs]
    if missing:
        raise KeyError(f"ONNX expects unsupported input groups: {missing}")
    for name in input_names:
        expected = tuple(shapes.get(name, obs[name].shape[1:]))
        got = tuple(obs[name].shape[1:])
        if got != expected:
            raise ValueError(f"Observation group '{name}' has shape {got}, expected {expected}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G1 ONNX policy in MuJoCo.")
    parser.add_argument("--onnx_path", required=True)
    parser.add_argument("--motion_file", required=True)
    parser.add_argument("--dataset_txt", default=None)
    parser.add_argument("--xml_path", default=str(G1_MJCF))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--decimation", type=int, default=None)
    parser.add_argument("--kp", type=float, default=60.0)
    parser.add_argument("--kd", type=float, default=2.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Build and validate one observation, then exit.")
    args = parser.parse_args()

    import mujoco

    meta = _metadata(args.onnx_path)
    if meta.get("encoder_mode") not in (None, "robot", "encoder_g1", "g1"):
        print(
            f"[WARN] ONNX encoder_mode={meta.get('encoder_mode')} needs non-zero smpl_cmd_mf. "
            "This script currently feeds zero SMPL observations."
        )

    motion_path = _first_motion_file(args.motion_file, args.dataset_txt)
    motion = np.load(motion_path)
    print(f"[INFO] Motion: {motion_path}")

    model = mujoco.MjModel.from_xml_path(args.xml_path)
    model.opt.timestep = float(meta.get("sim_dt", model.opt.timestep))
    data = mujoco.MjData(model)
    joint_names = list(meta["action_joint_names"])
    joint_ids = _name_to_joint_ids(model, joint_names)
    joint_qpos, joint_qvel = _name_to_joint_qvel_addrs(model, joint_names)
    body_ids = _name_to_body_ids(model, list(meta["motion_body_names"]))
    torque_limits = np.asarray(model.jnt_actfrcrange[joint_ids], dtype=np.float64)

    default_joint_pos = _as_vector(meta, "default_joint_pos", len(joint_names), 0.0)
    data.qpos[2] = 0.793
    data.qpos[joint_qpos] = default_joint_pos
    mujoco.mj_forward(model, data)

    input_names = _onnx_input_names(args.onnx_path)
    action_scale = _as_vector(meta, "action_scale", len(joint_names), 1.0)
    action_offset = _as_vector(meta, "action_offset", len(joint_names), 0.0)
    decimation = args.decimation or int(meta.get("decimation", 1))
    last_action = np.zeros((1, len(joint_names)), dtype=np.float32)
    prop_history = TermMajorHistory(_prop_terms_from_metadata(meta, len(joint_names)))
    _print_obs_layout(meta, prop_history, input_names)
    if args.dry_run:
        obs = _build_obs(data, motion, 0, meta, body_ids, joint_qpos, joint_qvel, last_action, prop_history)
        _validate_inputs(obs, input_names, meta)
        print("[INFO] dry_run observation validation passed.")
        return

    import onnxruntime as ort

    sess = ort.InferenceSession(args.onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_names = [inp.name for inp in sess.get_inputs()]
    output_name = sess.get_outputs()[0].name

    viewer_cm = None
    viewer = None
    if args.render:
        import mujoco.viewer

        viewer_cm = mujoco.viewer.launch_passive(model, data)
        viewer = viewer_cm.__enter__()

    try:
        for step in range(args.steps):
            t = min(step, motion["joint_pos"].shape[0] - 1)
            obs = _build_obs(data, motion, t, meta, body_ids, joint_qpos, joint_qvel, last_action, prop_history)
            _validate_inputs(obs, input_names, meta)
            ort_inputs = {name: obs[name] for name in input_names}
            raw_action = sess.run([output_name], ort_inputs)[0].astype(np.float32)
            target = _action_to_target(raw_action, action_scale, action_offset)
            last_action = raw_action
            for _ in range(decimation):
                q = data.qpos[joint_qpos]
                qd = data.qvel[joint_qvel]
                tau = args.kp * (target - q) - args.kd * qd
                tau = np.clip(tau, torque_limits[:, 0], torque_limits[:, 1])
                data.qfrc_applied[joint_qvel] = tau
                mujoco.mj_step(model, data)
                time.sleep(model.opt.timestep)
                if viewer is not None:
                    viewer.sync()
    finally:
        if viewer_cm is not None:
            viewer_cm.__exit__(None, None, None)


if __name__ == "__main__":
    main()
