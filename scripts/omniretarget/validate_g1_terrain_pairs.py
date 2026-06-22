#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import numpy as np
import trimesh
from pathlib import Path

WBT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = WBT_ROOT / "data" / "omniretarget" / "g1_terrain"
REQUIRED_TRACKING_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "joint_names",
    "body_names",
)


def _read_pairs(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_pair_path(pairs_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = pairs_path.parent / path
    return path.resolve()


def _validate_raw(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as raw:
        if "qpos" not in raw.files or "fps" not in raw.files:
            raise ValueError(f"{path}: expected qpos and fps keys")
        qpos = np.asarray(raw["qpos"])
        if qpos.ndim != 2 or qpos.shape[1] != 36:
            raise ValueError(f"{path}: expected qpos shape [T, 36], got {qpos.shape}")
        return int(qpos.shape[0]), int(qpos.shape[1])


def _validate_tracking(path: Path) -> tuple[int, int, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as raw:
        missing = [key for key in REQUIRED_TRACKING_KEYS if key not in raw.files]
        if missing:
            raise ValueError(f"{path}: missing tracking keys {missing}")
        joint_pos = np.asarray(raw["joint_pos"])
        body_pos_w = np.asarray(raw["body_pos_w"])
        if joint_pos.ndim != 2:
            raise ValueError(f"{path}: joint_pos must be [T, J], got {joint_pos.shape}")
        if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
            raise ValueError(f"{path}: body_pos_w must be [T, B, 3], got {body_pos_w.shape}")
        if body_pos_w.shape[0] != joint_pos.shape[0]:
            raise ValueError(f"{path}: joint/body frame mismatch {joint_pos.shape[0]} vs {body_pos_w.shape[0]}")
        return int(joint_pos.shape[0]), int(joint_pos.shape[1]), int(body_pos_w.shape[1])


def _validate_stl(path: Path) -> tuple[int, int, list[list[float]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    mesh = trimesh.load(path, force="mesh", process=False)
    if mesh.is_empty or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"{path}: empty STL mesh")
    return int(len(mesh.vertices)), int(len(mesh.faces)), np.asarray(mesh.bounds, dtype=float).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate prepared/converted OmniRetarget G1 terrain pairs.")
    parser.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--pairs_jsonl", default=None, help="Defaults to data_root/pairs.jsonl.")
    parser.add_argument("--require_tracking", action="store_true", help="Require converted tracking npz files.")
    parser.add_argument("--max_pairs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    pairs_path = Path(args.pairs_jsonl).expanduser().resolve() if args.pairs_jsonl else data_root / "pairs.jsonl"
    pairs = _read_pairs(pairs_path)
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        raise RuntimeError(f"No pairs in {pairs_path}")

    total_frames = 0
    for index, pair in enumerate(pairs):
        raw_frames, raw_dim = _validate_raw(_resolve_pair_path(pairs_path, pair["raw_motion"]))
        vertices, faces, bounds = _validate_stl(_resolve_pair_path(pairs_path, pair["terrain_stl"]))
        total_frames += raw_frames
        message = (
            f"[{index:04d}] {pair['name']}: raw={raw_frames}x{raw_dim}, "
            f"stl_vertices={vertices}, stl_faces={faces}, bounds={bounds}"
        )
        tracking_path = _resolve_pair_path(pairs_path, pair["tracking_motion"])
        if tracking_path.is_file():
            frames, joints, bodies = _validate_tracking(tracking_path)
            message += f", tracking={frames}x{joints}, bodies={bodies}"
        elif args.require_tracking:
            raise FileNotFoundError(tracking_path)
        print(message)

    print(f"Validated {len(pairs)} pairs, raw frames total={total_frames}")


if __name__ == "__main__":
    main()
