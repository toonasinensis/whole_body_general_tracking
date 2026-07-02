#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path

from attach_npz_names import G1_FULL_ISAAC_BODY_NAMES, G1_ISAAC_JOINT_NAMES

WHOLE_BODY_TRACKING_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = WHOLE_BODY_TRACKING_ROOT / "data/g1_amp/WalkandRun"
DEFAULT_OUTPUT_DIR = WHOLE_BODY_TRACKING_ROOT / "data/g1_amp/WalkandRun_isaaclab"

JOINT_ARRAY_KEYS = ("joint_pos", "joint_vel")
BODY_ARRAY_KEYS = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def _decode_names(raw: np.ndarray) -> list[str]:
    names: list[str] = []
    for value in raw.reshape(-1).tolist():
        if isinstance(value, bytes):
            names.append(value.decode("utf-8"))
        else:
            names.append(str(value))
    return names


def _collect_npz_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")
    return files


def _permutation(source_names: list[str], target_names: list[str], path: Path, label: str) -> np.ndarray:
    source_index = {name: i for i, name in enumerate(source_names)}
    missing = [name for name in target_names if name not in source_index]
    extra = [name for name in source_names if name not in set(target_names)]
    if missing:
        raise ValueError(f"{path}: missing {label} names required by IsaacLab order: {missing}")
    if extra:
        raise ValueError(f"{path}: source has unexpected {label} names not in IsaacLab order: {extra}")
    return np.asarray([source_index[name] for name in target_names], dtype=np.int64)


def _reorder_file(path: Path, output_dir: Path) -> Path:
    with np.load(path, allow_pickle=True) as raw:
        arrays = {key: raw[key] for key in raw.files}

    for key in ("joint_names", "body_names"):
        if key not in arrays:
            raise KeyError(f"{path}: missing required key {key!r}")

    source_joint_names = _decode_names(np.asarray(arrays["joint_names"]))
    source_body_names = _decode_names(np.asarray(arrays["body_names"]))
    joint_perm = _permutation(source_joint_names, G1_ISAAC_JOINT_NAMES, path, "joint")
    body_perm = _permutation(source_body_names, G1_FULL_ISAAC_BODY_NAMES, path, "body")

    converted: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        value = np.asarray(value)
        if key in JOINT_ARRAY_KEYS:
            if value.ndim < 2 or value.shape[1] != len(source_joint_names):
                raise ValueError(f"{path}: {key} shape {value.shape} does not match joint_names.")
            converted[key] = value[:, joint_perm, ...]
        elif key in BODY_ARRAY_KEYS:
            if value.ndim < 2 or value.shape[1] != len(source_body_names):
                raise ValueError(f"{path}: {key} shape {value.shape} does not match body_names.")
            converted[key] = value[:, body_perm, ...]
        elif key == "joint_names":
            converted[key] = np.asarray(G1_ISAAC_JOINT_NAMES)
        elif key == "body_names":
            converted[key] = np.asarray(G1_FULL_ISAAC_BODY_NAMES)
        else:
            converted[key] = value

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / path.name
    np.savez(output_path, **converted)
    return output_path


def _write_dataset_txt(output_dir: Path, files: list[Path]) -> Path:
    dataset_path = output_dir / "dataset.txt"
    dataset_path.write_text("".join(f"{path.name}\n" for path in files), encoding="utf-8")
    return dataset_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reorder G1 AMP npz joint/body arrays from metadata order to IsaacLab joint/body order."
    )
    parser.add_argument(
        "--input_dir", default=str(DEFAULT_INPUT_DIR), help="Input directory with MuJoCo-order npz files."
    )
    parser.add_argument(
        "--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for IsaacLab-order npz files."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    input_files = _collect_npz_files(input_dir)
    output_files = []
    for path in input_files:
        output_path = _reorder_file(path, output_dir)
        output_files.append(output_path)
        print(f"[OK] {path.name} -> {output_path}")

    dataset_path = _write_dataset_txt(output_dir, output_files)
    print(f"[SUMMARY] files={len(output_files)} output_dir={output_dir}")
    print(f"[SUMMARY] dataset_txt={dataset_path}")


if __name__ == "__main__":
    main()
