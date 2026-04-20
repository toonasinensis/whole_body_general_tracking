"""Report adaptive bins stats from one JSON file or a folder of JSON files.

This script reads adaptive bins JSON exported during training and prints:
- the maximum bin_failed_count
- all bins that match that maximum
- top-K bins by bin_failed_count

It can also scan a directory, process all JSON files in it, and write one merged
top-files txt.

Examples:
python scripts/eval/stat_adaptive_bins.py \
    --bins_json train_logs/adaptive_bins_step_000024000.json

python scripts/eval/stat_adaptive_bins.py \
    --bins_dir logs/rsl_rl/g1_flat --top_k 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _format_segment(segment: dict) -> str:
    return (
        f"motion_id={segment['motion_id']} "
        f"file={segment['motion_file']} "
        f"global=[{segment['global_frame_start']},{segment['global_frame_end_exclusive']}) "
        f"local=[{segment['motion_local_frame_start']},{segment['motion_local_frame_end_exclusive']})"
    )


def _collect_top_files(top_bins: list[dict]) -> list[str]:
    top_files: list[str] = []
    for bin_item in top_bins:
        motion_file = str(bin_item.get("motion_file", "")).strip()
        if motion_file:
            top_files.append(motion_file)
    return top_files


def _dedup_keep_order(paths: list[str]) -> list[str]:
    unique_paths: list[str] = []
    for path_str in paths:
        if path_str not in unique_paths:
            unique_paths.append(path_str)
    return unique_paths


def _process_one_json(bins_path: Path, top_k: int) -> tuple[list[str], list[dict]]:
    payload = json.loads(bins_path.read_text(encoding="utf-8"))
    bins = payload.get("bins", [])
    if not bins:
        raise ValueError(f"No bins found in {bins_path}")

    max_fail_count = max(float(bin_item.get("bin_failed_count", 0.0)) for bin_item in bins)
    max_bins = [bin_item for bin_item in bins if float(bin_item.get("bin_failed_count", 0.0)) == max_fail_count]
    top_bins = sorted(bins, key=lambda item: float(item.get("bin_failed_count", 0.0)), reverse=True)[:top_k]

    print(f"[INFO] file={bins_path}")
    print(f"[INFO] total_bins={len(bins)}")
    print(f"[INFO] max_bin_failed_count={max_fail_count:.12g}")
    print(f"[INFO] bins_with_max={len(max_bins)}")
    print()

    print("[MAX BINS]")
    for bin_item in max_bins:
        print(
            f"bin_index={bin_item['bin_index']} "
            f"sampling_probability={bin_item.get('sampling_probability', 0.0):.12g} "
            f"global=[{bin_item.get('global_frame_start', -1)},{bin_item.get('global_frame_end_exclusive', -1)})"
        )
        segments = bin_item.get("motion_segments", [])
        if segments:
            for segment in segments:
                print(f"  - {_format_segment(segment)}")
        else:
            print(
                "  - "
                f"motion_id={bin_item.get('motion_id_at_bin_start', -1)} "
                f"file={bin_item.get('motion_file')} "
                f"local=[{bin_item.get('motion_local_frame_start', -1)},"
                f"{bin_item.get('motion_local_frame_end_exclusive', -1)})"
            )
        print()

    print(f"[TOP {len(top_bins)} BY bin_failed_count]")
    for rank, bin_item in enumerate(top_bins, start=1):
        print(
            f"#{rank} bin_index={bin_item['bin_index']} "
            f"bin_failed_count={bin_item.get('bin_failed_count', 0.0):.12g} "
            f"sampling_probability={bin_item.get('sampling_probability', 0.0):.12g} "
            f"file={bin_item.get('motion_file')}"
        )

    return _collect_top_files(top_bins), bins


def main() -> None:
    parser = argparse.ArgumentParser(description="Report max bin_failed_count from adaptive bins JSON.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--bins_json", type=str, help="Path to one adaptive bins JSON file.")
    source_group.add_argument("--bins_dir", type=str, help="Directory that contains adaptive bins JSON files.")
    parser.add_argument("--top_k", type=int, default=10, help="Show top-K bins by bin_failed_count (default: 10).")
    parser.add_argument(
        "--out_txt",
        type=str,
        default=None,
        help=(
            "Optional txt output path for top motion files. "
            "Single-file mode default: <bins_json_stem>_top_files.txt. "
            "Directory mode default: <bins_dir>/all_top_files.txt."
        ),
    )
    parser.add_argument(
        "--unique_only",
        action="store_true",
        help="If set, remove duplicate motion files before writing txt.",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top_k must be > 0")

    all_top_files: list[str] = []

    if args.bins_json:
        bins_path = Path(args.bins_json).expanduser().resolve()
        top_files, _ = _process_one_json(bins_path, args.top_k)
        all_top_files.extend(top_files)

        out_txt_path = (
            Path(args.out_txt).expanduser().resolve()
            if args.out_txt
            else bins_path.with_name(f"{bins_path.stem}_top_files.txt")
        )
    else:
        bins_dir = Path(args.bins_dir).expanduser().resolve()
        if not bins_dir.is_dir():
            raise ValueError(f"--bins_dir is not a directory: {bins_dir}")

        json_files = sorted(bins_dir.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in {bins_dir}")

        for idx, json_file in enumerate(json_files, start=1):
            print(f"\n{'=' * 80}")
            print(f"[FILE {idx}/{len(json_files)}] {json_file.name}")
            print(f"{'=' * 80}")
            top_files, _ = _process_one_json(json_file, args.top_k)
            all_top_files.extend(top_files)

        out_txt_path = Path(args.out_txt).expanduser().resolve() if args.out_txt else bins_dir / "all_top_files.txt"

    if args.unique_only:
        all_top_files = _dedup_keep_order(all_top_files)

    out_txt_path.write_text("\n".join(all_top_files) + "\n", encoding="utf-8")
    print(f"\n[INFO] merged top files saved to {out_txt_path}")
    print(f"[INFO] total written lines={len(all_top_files)}")
    if not args.unique_only:
        print(f"[INFO] total unique motion files={len(_dedup_keep_order(all_top_files))}")


if __name__ == "__main__":
    main()
