#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path

WHOLE_BODY_TRACKING_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOTION_PATH = WHOLE_BODY_TRACKING_ROOT / "data/g1_amp/WalkandRun"
DEFAULT_XML_PATH = (
    WHOLE_BODY_TRACKING_ROOT / "source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml"
)


def _decode_names(raw: np.ndarray) -> list[str]:
    names = []
    for value in raw.reshape(-1).tolist():
        if isinstance(value, bytes):
            names.append(value.decode("utf-8"))
        else:
            names.append(str(value))
    return names


def _collect_npz_files(path: Path, max_files: int | None) -> list[Path]:
    if path.is_dir():
        files = sorted(path.rglob("*.npz"))
    elif path.is_file() and path.suffix.lower() == ".npz":
        files = [path]
    else:
        raise FileNotFoundError(f"No npz file or directory found at {path}")
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No npz files found under {path}")
    return files


def _mujoco_names(model, obj_type, count: int) -> list[str]:
    import mujoco

    return [mujoco.mj_id2name(model, obj_type, index) for index in range(count)]


def _mujoco_hinge_joint_names(model) -> list[str]:
    import mujoco

    pairs = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        pairs.append((int(model.jnt_qposadr[joint_id]), name))
    return [name for _, name in sorted(pairs)]


def _joint_qpos_addresses(model, joint_names: list[str]) -> np.ndarray:
    import mujoco

    addresses = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint '{name}' from npz metadata is not in the MuJoCo model.")
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Joint '{name}' is a free joint; npz joint_pos should contain hinge joints only.")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(addresses, dtype=np.int32)


def _body_ids(model, body_names: list[str]) -> np.ndarray:
    import mujoco

    ids = []
    for name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Body '{name}' from npz metadata is not in the MuJoCo model.")
        ids.append(body_id)
    return np.asarray(ids, dtype=np.int32)


def _frame_indices(num_frames: int, frames_per_file: int) -> np.ndarray:
    if num_frames <= frames_per_file:
        return np.arange(num_frames, dtype=np.int64)
    return np.unique(np.linspace(0, num_frames - 1, frames_per_file, dtype=np.int64))


def _quat_abs_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    same = np.max(np.abs(a - b), axis=-1)
    flipped = np.max(np.abs(a + b), axis=-1)
    return np.minimum(same, flipped)


def _validate_file(path: Path, model, data, args: argparse.Namespace) -> tuple[float, float, int]:
    import mujoco

    with np.load(path, allow_pickle=True) as raw:
        required = ("joint_pos", "body_pos_w", "body_quat_w", "joint_names", "body_names")
        missing = [name for name in required if name not in raw.files]
        if missing:
            raise KeyError(f"{path}: missing keys {missing}; run scripts/attach_npz_names.py first.")
        joint_pos = np.asarray(raw["joint_pos"])
        body_pos_w = np.asarray(raw["body_pos_w"])
        body_quat_w = np.asarray(raw["body_quat_w"])
        joint_names = _decode_names(np.asarray(raw["joint_names"]))
        body_names = _decode_names(np.asarray(raw["body_names"]))

    if joint_pos.shape[1] != len(joint_names):
        raise ValueError(f"{path}: joint_names length does not match joint_pos dim.")
    if body_pos_w.shape[1] != len(body_names):
        raise ValueError(f"{path}: body_names length does not match body_pos_w dim.")
    if body_quat_w.shape[:2] != body_pos_w.shape[:2]:
        raise ValueError(f"{path}: body_quat_w shape does not match body_pos_w shape.")
    if args.root_body_name not in body_names:
        raise ValueError(f"{path}: root body '{args.root_body_name}' is not in body_names.")

    joint_qpos = _joint_qpos_addresses(model, joint_names)
    body_id_array = _body_ids(model, body_names)
    root_index = body_names.index(args.root_body_name)

    max_pos_err = 0.0
    max_quat_err = 0.0
    frames = _frame_indices(joint_pos.shape[0], args.frames_per_file)
    for frame in frames:
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[:3] = body_pos_w[frame, root_index]
        data.qpos[3:7] = body_quat_w[frame, root_index]
        data.qpos[joint_qpos] = joint_pos[frame]
        mujoco.mj_forward(model, data)

        pos_err = np.max(np.abs(data.xpos[body_id_array] - body_pos_w[frame]))
        quat_err = np.max(_quat_abs_error(data.xquat[body_id_array], body_quat_w[frame]))
        max_pos_err = max(max_pos_err, float(pos_err))
        max_quat_err = max(max_quat_err, float(quat_err))

    return max_pos_err, max_quat_err, len(frames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that G1 AMP npz joint/body arrays are named in MuJoCo order by "
            "replaying joint_pos through the MuJoCo model and comparing body_pos_w/body_quat_w."
        )
    )
    parser.add_argument("--motion_path", default=str(DEFAULT_MOTION_PATH), help="NPZ file or directory to validate.")
    parser.add_argument("--xml_path", default=str(DEFAULT_XML_PATH), help="G1 MuJoCo XML path.")
    parser.add_argument("--root_body_name", default="pelvis", help="Body used to fill the free-joint root pose.")
    parser.add_argument("--frames_per_file", type=int, default=16, help="Evenly sampled frames per npz.")
    parser.add_argument("--max_files", type=int, default=None, help="Limit number of npz files checked.")
    parser.add_argument("--pos_tol", type=float, default=1.0e-5)
    parser.add_argument("--quat_tol", type=float, default=1.0e-5)
    return parser.parse_args()


def main() -> None:
    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit("This script requires the 'mujoco' Python package.") from exc

    args = parse_args()
    motion_path = Path(args.motion_path).expanduser()
    xml_path = Path(args.xml_path).expanduser()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    files = _collect_npz_files(motion_path, args.max_files)
    model_body_order = _mujoco_names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)[1:]
    model_joint_order = _mujoco_hinge_joint_names(model)

    global_pos_err = 0.0
    global_quat_err = 0.0
    total_frames = 0
    for path in files:
        with np.load(path, allow_pickle=True) as raw:
            body_names = _decode_names(np.asarray(raw["body_names"]))
            joint_names = _decode_names(np.asarray(raw["joint_names"]))
        if body_names != model_body_order:
            raise ValueError(f"{path}: body_names metadata is not MuJoCo body[1:] order.")
        if joint_names != model_joint_order:
            raise ValueError(f"{path}: joint_names metadata is not MuJoCo hinge qpos order.")

        pos_err, quat_err, num_frames = _validate_file(path, model, data, args)
        total_frames += num_frames
        global_pos_err = max(global_pos_err, pos_err)
        global_quat_err = max(global_quat_err, quat_err)
        print(f"[OK] {path.name}: frames={num_frames} max_pos_err={pos_err:.3e} max_quat_err={quat_err:.3e}")

    print(
        f"[SUMMARY] files={len(files)} frames={total_frames} "
        f"max_pos_err={global_pos_err:.3e} max_quat_err={global_quat_err:.3e}"
    )
    if global_pos_err > args.pos_tol or global_quat_err > args.quat_tol:
        raise SystemExit(
            f"Validation failed: max_pos_err={global_pos_err:.3e} max_quat_err={global_quat_err:.3e} "
            f"tolerances=({args.pos_tol:.3e}, {args.quat_tol:.3e})"
        )


if __name__ == "__main__":
    main()
