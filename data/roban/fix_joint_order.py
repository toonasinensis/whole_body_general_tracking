import argparse
import numpy as np
from collections.abc import Sequence
from pathlib import Path


def parse_joint_order(order: str | Sequence[str]) -> list[str]:
    """Parse joint order from either string or sequence of names."""
    if isinstance(order, str):
        normalized = order.replace("\n", " ").replace("\t", " ").replace(",", " ")
        joints = [name.strip() for name in normalized.split(" ") if name.strip()]
    else:
        joints = [str(name).strip() for name in order if str(name).strip()]
    if not joints:
        raise ValueError("Joint order string is empty after parsing.")
    if len(set(joints)) != len(joints):
        raise ValueError("Joint order contains duplicated joint names.")
    return joints


def build_reorder_index(src_order: list[str], dst_order: list[str]) -> np.ndarray:
    """Build index so data[:, index] maps src order to dst order."""
    missing = [j for j in dst_order if j not in src_order]
    extra = [j for j in src_order if j not in dst_order]
    if missing:
        raise ValueError(f"Target joints missing in source order: {missing}")
    if extra:
        raise ValueError(f"Source has joints not in target order: {extra}")

    src_to_idx = {name: i for i, name in enumerate(src_order)}
    return np.asarray([src_to_idx[name] for name in dst_order], dtype=np.int64)


def reorder_npz(
    npz_path: Path, out_path: Path, src_order_in: str | Sequence[str], dst_order_in: str | Sequence[str]
) -> None:
    src_order = parse_joint_order(src_order_in)
    dst_order = parse_joint_order(dst_order_in)

    reorder_index = build_reorder_index(src_order, dst_order)

    data = np.load(npz_path, allow_pickle=True)
    data_dict = {k: data[k] for k in data.files}
    # import ipdb; ipdb.set_trace()

    for key in ("joint_pos", "joint_vel"):
        if key not in data_dict:
            raise KeyError(f"{key} not found in {npz_path}")
        if data_dict[key].ndim != 2:
            raise ValueError(f"{key} must be 2D [T, J], got shape {data_dict[key].shape}")
        if data_dict[key].shape[1] != len(src_order):
            raise ValueError(
                f"{key} joint dim mismatch: data has {data_dict[key].shape[1]}, src_order has {len(src_order)}"
            )
        data_dict[key] = data_dict[key][:, reorder_index]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **data_dict)


def reorder_npz_in_folder(
    input_dir: Path,
    output_dir: Path,
    src_order_in: str | Sequence[str],
    dst_order_in: str | Sequence[str],
    recursive: bool = False,
) -> tuple[int, int]:
    """Reorder all NPZ files in a folder and save to output dir.

    Returns:
            (success_count, fail_count)
    """
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")

    pattern = "**/*.npz" if recursive else "*.npz"
    npz_files = sorted(input_dir.glob(pattern))
    if not npz_files:
        print(f"No npz files found in: {input_dir}")
        return 0, 0

    success_count = 0
    fail_count = 0
    for npz_path in npz_files:
        rel_path = npz_path.relative_to(input_dir)
        out_path = output_dir / rel_path
        try:
            reorder_npz(npz_path, out_path, src_order_in, dst_order_in)
            success_count += 1
        except Exception as exc:
            fail_count += 1
            print(f"[FAILED] {npz_path}: {exc}")

    return success_count, fail_count


src_joint_order = [
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

dst_joint_order = [
    "waist_yaw_joint",
    "zarm_l1_joint",
    "zarm_r1_joint",
    "leg_l1_joint",
    "leg_r1_joint",
    "zarm_l2_joint",
    "zarm_r2_joint",
    "leg_l2_joint",
    "leg_r2_joint",
    "zarm_l3_joint",
    "zarm_r3_joint",
    "leg_l3_joint",
    "leg_r3_joint",
    "zarm_l4_joint",
    "zarm_r4_joint",
    "leg_l4_joint",
    "leg_r4_joint",
    "leg_l5_joint",
    "leg_r5_joint",
    "leg_l6_joint",
    "leg_r6_joint",
]

input_dir = "/home/thl/Documents/210531"
output_dir = "/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/data/roban/roban2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Reorder joint_pos/joint_vel in NPZ by joint name strings")
    parser.add_argument(
        "--input-dir",
        default=input_dir,
        help="Input directory containing npz files (default: script directory)",
    )
    parser.add_argument(
        "--output-dir", required=False, default=output_dir, help="Output directory for reordered npz files"
    )
    parser.add_argument("--src-order", required=False, default=src_joint_order, help="Source joint order string")
    parser.add_argument("--dst-order", required=False, default=dst_joint_order, help="Target joint order string")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan subfolders for npz files")

    args = parser.parse_args()
    success_count, fail_count = reorder_npz_in_folder(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        src_order_in=args.src_order,
        dst_order_in=args.dst_order,
        recursive=args.recursive,
    )
    print(f"Done. success={success_count}, failed={fail_count}, output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
