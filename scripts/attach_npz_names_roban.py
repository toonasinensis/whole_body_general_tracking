#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROBAN_S22_JOINT_NAMES = [
    "waist_yaw_joint",
    "leg_l1_joint",
    "leg_l2_joint",
    "leg_l3_joint",
    "leg_l4_joint",
    "leg_l5_joint",
    "leg_l6_joint",
    "leg_r1_joint",
    "leg_r2_joint",
    "leg_r3_joint",
    "leg_r4_joint",
    "leg_r5_joint",
    "leg_r6_joint",
    "zarm_l1_joint",
    "zarm_l2_joint",
    "zarm_l3_joint",
    "zarm_l4_joint",
    "zarm_r1_joint",
    "zarm_r2_joint",
    "zarm_r3_joint",
    "zarm_r4_joint",
]

ROBAN_S22_FULL_BODY_NAMES = [
    "base_link",
    "waist_yaw_link",
    "zarm_l1_link",
    "zarm_r1_link",
    "zhead_1_link",
    "leg_l1_link",
    "leg_r1_link",
    "zarm_l2_link",
    "zarm_r2_link",
    "head_radar",
    "zhead_2_link",
    "leg_l2_link",
    "leg_r2_link",
    "zarm_l3_link",
    "zarm_r3_link",
    "camera_base",
    "leg_l3_link",
    "leg_r3_link",
    "zarm_l4_link",
    "zarm_r4_link",
    "leg_l4_link",
    "leg_r4_link",
    "zarm_l5_link",
    "zarm_r5_link",
    "leg_l5_link",
    "leg_r5_link",
    "leg_l6_link",
    "leg_r6_link",
]


def _read_names_file(path: str | None) -> list[str] | None:
    if path is None:
        return None
    source = Path(path).expanduser()
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("names", "joint_names", "body_names"):
                if key in data:
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError(f"{source}: expected a JSON list or object containing a names list.")
        return [str(value) for value in data]
    names = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(line)
    return names


def _collect_npz_files(paths: list[str], motion_root: str | None = None) -> list[Path]:
    result: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            result.extend(sorted(path.rglob("*.npz")))
        elif path.is_file():
            if path.suffix.lower() == ".txt":
                root = Path(motion_root).expanduser() if motion_root is not None else path.parent
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.split("#", 1)[0].strip()
                    if not line:
                        continue
                    item = Path(line).expanduser()
                    result.append(item if item.is_absolute() else root / item)
            elif path.suffix.lower() == ".npz":
                result.append(path)
            else:
                raise ValueError(f"Unsupported input file: {path}")
        else:
            raise FileNotFoundError(path)
    return result


def _default_output_path(path: Path, suffix: str) -> Path:
    return path.with_name(path.stem + suffix + path.suffix)


def _output_path(path: Path, args: argparse.Namespace) -> Path:
    if args.output_dir is None:
        return _default_output_path(path, args.suffix)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / (path.stem + args.suffix + path.suffix)


def _names_for_args(args: argparse.Namespace, joint_dim: int, body_dim: int) -> tuple[list[str], list[str]]:
    joint_names = _read_names_file(args.joint_names_file) or list(ROBAN_S22_JOINT_NAMES)
    body_names = _read_names_file(args.body_names_file) or list(ROBAN_S22_FULL_BODY_NAMES)

    if len(joint_names) != joint_dim:
        raise ValueError(f"joint_names length {len(joint_names)} does not match joint_pos dim {joint_dim}.")
    if len(body_names) != body_dim:
        raise ValueError(f"body_names length {len(body_names)} does not match body_pos_w dim {body_dim}.")
    return joint_names, body_names


def _attach_one(path: Path, args: argparse.Namespace) -> Path:
    with np.load(path, allow_pickle=True) as raw:
        arrays = {name: raw[name] for name in raw.files}

    if "joint_pos" not in arrays or "body_pos_w" not in arrays:
        raise ValueError(f"{path}: expected at least joint_pos and body_pos_w arrays.")

    joint_dim = int(np.asarray(arrays["joint_pos"]).shape[1])
    body_dim = int(np.asarray(arrays["body_pos_w"]).shape[1])
    joint_names, body_names = _names_for_args(args, joint_dim, body_dim)

    arrays["joint_names"] = np.asarray(joint_names)
    arrays["body_names"] = np.asarray(body_names)

    output_path = path if args.in_place else _output_path(path, args)
    if output_path.exists() and not args.in_place and not args.force:
        raise FileExistsError(f"{output_path} already exists. Use --force to replace it.")
    np.savez_compressed(output_path, **arrays)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach Roban joint_names/body_names metadata to motion npz files.")
    parser.add_argument("paths", nargs="+", help="NPZ files, directories, or dataset txt files.")
    parser.add_argument(
        "--motion_root",
        default=None,
        help="Root used to resolve relative entries when an input path is a dataset txt.",
    )
    parser.add_argument("--suffix", default=".named", help="Sidecar suffix before .npz. Default: .named")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional directory for generated sidecars. Defaults to each input file's directory.",
    )
    parser.add_argument(
        "--in_place",
        action="store_true",
        help="Overwrite input npz files instead of writing sidecars.",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing sidecar outputs.")
    parser.add_argument("--joint_names_file", default=None, help="TXT/JSON file containing joint names in npz order.")
    parser.add_argument("--body_names_file", default=None, help="TXT/JSON file containing body names in npz order.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.in_place and args.output_dir is not None:
        raise ValueError("--output_dir cannot be used together with --in_place.")
    paths = _collect_npz_files(args.paths, args.motion_root)
    if not paths:
        raise ValueError("No npz files found.")

    for index, path in enumerate(paths, start=1):
        output_path = _attach_one(path, args)
        print(f"[{index}/{len(paths)}] {path} -> {output_path}")


if __name__ == "__main__":
    main()
