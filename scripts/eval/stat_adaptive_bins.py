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
from typing import Iterable


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


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _iter_motion_segments(bin_item: dict) -> Iterable[tuple[int, int]]:
    """Yield (motion_id, segment_len_frames) pairs for a bin.

    Supports both styles:
    - multi-motion bins with `motion_segments`
    - single-motion bins with `motion_id_at_bin_start` and local frame range
    """

    segments = bin_item.get("motion_segments", [])
    if segments:
        for seg in segments:
            motion_id = _safe_int(seg.get("motion_id", -1), default=-1)
            seg_len = _safe_int(
                _safe_int(seg.get("global_frame_end_exclusive", 0)) - _safe_int(seg.get("global_frame_start", 0))
            )
            if seg_len <= 0:
                seg_len = _safe_int(
                    _safe_int(seg.get("motion_local_frame_end_exclusive", 0))
                    - _safe_int(seg.get("motion_local_frame_start", 0))
                )
            if motion_id >= 0 and seg_len > 0:
                yield motion_id, seg_len
        return

    motion_id = _safe_int(bin_item.get("motion_id_at_bin_start", -1), default=-1)
    seg_len = _safe_int(
        _safe_int(bin_item.get("motion_local_frame_end_exclusive", 0))
        - _safe_int(bin_item.get("motion_local_frame_start", 0))
    )
    if seg_len <= 0:
        seg_len = _safe_int(
            _safe_int(bin_item.get("global_frame_end_exclusive", 0)) - _safe_int(bin_item.get("global_frame_start", 0))
        )
    if motion_id >= 0 and seg_len > 0:
        yield motion_id, seg_len


def _plot_motion_distribution(bins: list[dict], out_png: Path, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("matplotlib is required for --plot. Install it via `pip install matplotlib`.") from exc

    # Plot per-bin (not per-motion aggregated): this disambiguates repeated motions.
    # X label uses `motion_name__bin_index`.
    items: list[tuple[int, float, float, str]] = []
    for bin_item in bins:
        bin_index = _safe_int(bin_item.get("bin_index", -1), default=-1)
        fail_count = _safe_float(bin_item.get("bin_failed_count", 0.0), default=0.0)
        prob = _safe_float(bin_item.get("sampling_probability", 0.0), default=0.0)

        motion_file = str(bin_item.get("motion_file", "")).strip()
        if not motion_file:
            # best-effort fallback: take first segment's motion_file
            segments = bin_item.get("motion_segments", [])
            if segments:
                motion_file = str(segments[0].get("motion_file", "")).strip()
        motion_name = Path(motion_file).stem if motion_file else "unknown_motion"
        label = f"{motion_name}__{bin_index}" if bin_index >= 0 else motion_name
        items.append((bin_index, float(fail_count), float(prob), label))

    if not items:
        raise ValueError("No bins found; cannot plot.")

    # Sort high -> low by fail_count (then prob) for easier inspection.
    items.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)

    fail_vals = [x[1] for x in items]
    prob_vals = [x[2] for x in items]
    labels = [x[3] for x in items]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, len(items) * 0.16), 8), sharex=True)
    fig.suptitle(title)

    x = list(range(len(items)))
    ax1.bar(x, fail_vals, width=0.9)
    ax1.set_ylabel("fail_bin_count")
    ax1.grid(True, alpha=0.3)

    ax2.bar(x, prob_vals, width=0.9)
    ax2.set_xlabel("motion__bin_index")
    ax2.set_ylabel("sampling_prob")
    ax2.grid(True, alpha=0.3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=90, fontsize=6)

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report max bin_failed_count from adaptive bins JSON.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--bins_json", type=str, help="Path to one adaptive bins JSON file.")
    source_group.add_argument("--bins_dir", type=str, help="Directory that contains adaptive bins JSON files.")
    parser.add_argument("--top_k", type=int, default=10, help="Show top-K bins by bin_failed_count (default: 10).")
    parser.add_argument(
        "--plot",
        action="store_true",
        help=(
            "If set, plot per-bin distributions for fail_bin_count and sampling_probability "
            "(x label: motion_name__bin_index, sorted by fail_bin_count) "
            "and save a PNG next to the JSON (requires matplotlib)."
        ),
    )
    parser.add_argument(
        "--plot_out",
        type=str,
        default=None,
        help=(
            "Optional PNG output path for --plot. Default: <bins_json_stem>_motion_dist.png (single JSON) or"
            " <bins_dir>/motion_dist_all_jsons.png (dir mode uses all JSON files)."
        ),
    )
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
    plot_bins: list[dict] | None = None
    plot_title: str | None = None
    plot_default_out: Path | None = None
    plot_source_json_count: int = 0

    if args.bins_json:
        bins_path = Path(args.bins_json).expanduser().resolve()
        top_files, bins = _process_one_json(bins_path, args.top_k)
        all_top_files.extend(top_files)

        if args.plot:
            plot_bins = bins
            plot_title = f"adaptive_bins: {bins_path.name}"
            plot_default_out = bins_path.with_name(f"{bins_path.stem}_motion_dist.png")
            plot_source_json_count = 1

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
            top_files, bins = _process_one_json(json_file, args.top_k)
            all_top_files.extend(top_files)

            # In directory mode, plot all JSON files merged together.
            if args.plot:
                if plot_bins is None:
                    plot_bins = []
                plot_bins.extend(bins)
                plot_source_json_count += 1

        if args.plot:
            plot_title = f"adaptive_bins (all {len(json_files)} JSON files)"
            plot_default_out = bins_dir / "motion_dist_all_jsons.png"

        out_txt_path = Path(args.out_txt).expanduser().resolve() if args.out_txt else bins_dir / "all_top_files.txt"

    if args.unique_only:
        all_top_files = _dedup_keep_order(all_top_files)

    out_txt_path.write_text("\n".join(all_top_files) + "\n", encoding="utf-8")
    print(f"\n[INFO] merged top files saved to {out_txt_path}")
    print(f"[INFO] total written lines={len(all_top_files)}")
    if not args.unique_only:
        print(f"[INFO] total unique motion files={len(_dedup_keep_order(all_top_files))}")

    if args.plot:
        if plot_bins is None or plot_title is None or plot_default_out is None:
            raise RuntimeError("--plot was set but no bins were selected for plotting.")
        plot_out = Path(args.plot_out).expanduser().resolve() if args.plot_out else plot_default_out
        print(f"[INFO] plotting bins from {plot_source_json_count} JSON file(s), total bins={len(plot_bins)}")
        _plot_motion_distribution(plot_bins, plot_out, title=plot_title)
        print(f"[INFO] motion distribution plot saved to {plot_out}")


if __name__ == "__main__":
    main()
