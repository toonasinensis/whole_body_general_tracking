from __future__ import annotations

import argparse
import csv
import html
import numpy as np
import os
from pathlib import Path

from sim2sim_g1.metrics import METRIC_NAMES

DEFAULT_DATASET_TXT = "/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/g1_flat/all_top_files.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize per-motion sim2sim tracking metrics.")
    parser.add_argument(
        "--csv",
        default=None,
        help=(
            "Metrics CSV from sim2sim_g1_mujoco.py. Defaults to METRICS_CSV, "
            "or the newest DATASET_TXT + '.sim2sim_metrics*.csv' file."
        ),
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory. Defaults to a '<csv_stem>_viz' directory next to the CSV.",
    )
    parser.add_argument("--top_k", type=int, default=30, help="Number of worst motions to show in detail.")
    parser.add_argument(
        "--score_metrics",
        default="",
        help="Comma-separated metric names used for the aggregate score. Defaults to all error_* metrics in the CSV.",
    )
    parser.add_argument("--title", default=None, help="Optional title prefix for figures.")
    return parser.parse_args()


def resolve_csv_path(args: argparse.Namespace) -> Path:
    if args.csv:
        return Path(args.csv).expanduser().resolve()
    if os.environ.get("METRICS_CSV"):
        return Path(os.environ["METRICS_CSV"]).expanduser().resolve()
    dataset_txt = Path(os.environ.get("DATASET_TXT", DEFAULT_DATASET_TXT)).expanduser()
    pattern = f"{dataset_txt.name}.sim2sim_metrics*.csv"
    candidates = sorted(dataset_txt.parent.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0].resolve()
    return dataset_txt.with_suffix(dataset_txt.suffix + ".sim2sim_metrics.csv").resolve()


def resolve_out_dir(csv_path: Path, args: argparse.Namespace) -> Path:
    if args.out_dir:
        return Path(args.out_dir).expanduser().resolve()
    return csv_path.with_name(f"{csv_path.stem}_viz").resolve()


def read_metrics_csv(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"No rows found in metrics CSV: {csv_path}")
    return rows, fieldnames


def metric_names_from_fields(fieldnames: list[str]) -> list[str]:
    return [name for name in METRIC_NAMES if name in fieldnames]


def parse_metric_selection(selection: str, available: list[str]) -> list[str]:
    if not selection.strip():
        return list(available)
    wanted = [part.strip() for part in selection.split(",") if part.strip()]
    missing = [name for name in wanted if name not in available]
    if missing:
        raise ValueError(f"Selected score metrics not found in CSV: {missing}. Available: {available}")
    return wanted


def metric_matrix(rows: list[dict[str, str]], metric_names: list[str]) -> np.ndarray:
    values = np.zeros((len(rows), len(metric_names)), dtype=np.float64)
    for row_i, row in enumerate(rows):
        for col_i, name in enumerate(metric_names):
            values[row_i, col_i] = float(row.get(name, "nan"))
    return values


def normalized_score(values: np.ndarray, metric_names: list[str], score_metric_names: list[str]) -> np.ndarray:
    score_indices = [metric_names.index(name) for name in score_metric_names]
    selected = values[:, score_indices]
    mins = np.nanmin(selected, axis=0)
    maxs = np.nanmax(selected, axis=0)
    denom = np.maximum(maxs - mins, 1.0e-12)
    normalized = (selected - mins) / denom
    return np.nanmean(normalized, axis=1)


def motion_labels(rows: list[dict[str, str]]) -> list[str]:
    raw_paths = [row.get("motion_file", "") for row in rows]
    names = [Path(path).name if path else f"motion_{i:05d}" for i, path in enumerate(raw_paths)]
    duplicated = {name for name in names if names.count(name) > 1}
    labels = []
    for path, name in zip(raw_paths, names):
        if name in duplicated and path:
            p = Path(path)
            labels.append(str(Path(p.parent.name) / p.name))
        else:
            labels.append(name)
    return labels


def import_pyplot():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required. Install it or run inside my_env.") from exc
    return plt


def save_sorted_csv(
    out_path: Path,
    rows: list[dict[str, str]],
    scores: np.ndarray,
    metric_names: list[str],
    order: np.ndarray,
) -> None:
    fieldnames = ["rank", "score", "motion_index", "motion_file", "num_frames", "samples", *metric_names]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row_i in enumerate(order, start=1):
            row = rows[int(row_i)]
            writer.writerow(
                {
                    "rank": rank,
                    "score": f"{float(scores[row_i]):.9g}",
                    "motion_index": row.get("motion_index", ""),
                    "motion_file": row.get("motion_file", ""),
                    "num_frames": row.get("num_frames", ""),
                    "samples": row.get("samples", ""),
                    **{name: row.get(name, "") for name in metric_names},
                }
            )


def plot_score_bars(
    out_path: Path,
    labels: list[str],
    scores: np.ndarray,
    order: np.ndarray,
    top_k: int,
    title: str,
) -> None:
    plt = import_pyplot()
    selected = order[:top_k][::-1]
    fig_h = max(6.0, 0.34 * len(selected) + 1.4)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.barh(np.arange(len(selected)), scores[selected], color="#3a7ca5")
    ax.set_yticks(np.arange(len(selected)))
    ax.set_yticklabels([labels[i] for i in selected], fontsize=8)
    ax.set_xlabel("normalized aggregate error")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_metric_means(out_path: Path, metric_names: list[str], values: np.ndarray, title: str) -> None:
    plt = import_pyplot()
    means = np.nanmean(values, axis=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(metric_names)), means, color="#5b8c5a")
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=35, ha="right")
    ax.set_ylabel("mean error")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_heatmap(
    out_path: Path,
    labels: list[str],
    metric_names: list[str],
    values: np.ndarray,
    order: np.ndarray,
    top_k: int,
    title: str,
) -> None:
    plt = import_pyplot()
    selected = order[:top_k]
    chosen = values[selected]
    mins = np.nanmin(values, axis=0)
    maxs = np.nanmax(values, axis=0)
    normalized = (chosen - mins) / np.maximum(maxs - mins, 1.0e-12)

    fig_h = max(6.0, 0.34 * len(selected) + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    image = ax.imshow(normalized, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(selected)))
    ax.set_yticklabels([labels[i] for i in selected], fontsize=8)
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("normalized error")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_metric_grid(
    out_path: Path,
    labels: list[str],
    metric_names: list[str],
    values: np.ndarray,
    top_k: int,
    title: str,
) -> None:
    plt = import_pyplot()
    cols = 2
    rows_n = int(np.ceil(len(metric_names) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(15, max(4.0, rows_n * 3.3)))
    axes = np.asarray(axes).reshape(-1)
    for ax, name, col in zip(axes, metric_names, range(len(metric_names))):
        order = np.argsort(values[:, col])[::-1][:top_k][::-1]
        ax.barh(np.arange(len(order)), values[order, col], color="#b45f4d")
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels([labels[i] for i in order], fontsize=6)
        ax.set_title(name)
        ax.grid(axis="x", alpha=0.25)
    for ax in axes[len(metric_names) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_index_html(
    out_path: Path,
    csv_path: Path,
    sorted_csv: Path,
    images: list[Path],
    rows: list[dict[str, str]],
    scores: np.ndarray,
    order: np.ndarray,
) -> None:
    top_rows = []
    for rank, row_i in enumerate(order[:20], start=1):
        row = rows[int(row_i)]
        top_rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td>{float(scores[row_i]):.6f}</td>"
            f"<td>{html.escape(Path(row.get('motion_file', '')).name)}</td>"
            f"<td>{html.escape(row.get('samples', ''))}</td>"
            "</tr>"
        )
    image_tags = "\n".join(
        f'<section><h2>{html.escape(image.stem)}</h2><img src="{html.escape(image.name)}" /></section>'
        for image in images
    )
    out_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset='utf-8' />",
                "<title>sim2sim metrics</title>",
                "<style>",
                "body{font-family:Arial,sans-serif;margin:24px;line-height:1.4;color:#222}",
                "img{max-width:100%;height:auto;border:1px solid #ddd}",
                "table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:4px 8px}",
                "code{background:#f3f3f3;padding:2px 4px}",
                "</style></head><body>",
                "<h1>sim2sim metrics</h1>",
                f"<p>Source CSV: <code>{html.escape(str(csv_path))}</code></p>",
                f"<p>Sorted CSV: <a href='{html.escape(sorted_csv.name)}'>{html.escape(sorted_csv.name)}</a></p>",
                "<h2>Top Aggregate Errors</h2>",
                "<table><tr><th>rank</th><th>score</th><th>motion</th><th>samples</th></tr>",
                *top_rows,
                "</table>",
                image_tags,
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    csv_path = resolve_csv_path(args)
    out_dir = resolve_out_dir(csv_path, args)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, fieldnames = read_metrics_csv(csv_path)
    metric_names = metric_names_from_fields(fieldnames)
    if not metric_names:
        raise ValueError(f"No known sim2sim metric columns found in {csv_path}")
    score_metric_names = parse_metric_selection(args.score_metrics, metric_names)
    values = metric_matrix(rows, metric_names)
    scores = normalized_score(values, metric_names, score_metric_names)
    order = np.argsort(scores)[::-1]
    labels = motion_labels(rows)
    top_k = max(1, min(int(args.top_k), len(rows)))
    title_prefix = args.title or csv_path.stem

    sorted_csv = out_dir / "metrics_sorted_by_score.csv"
    score_png = out_dir / "top_aggregate_error.png"
    means_png = out_dir / "metric_means.png"
    heatmap_png = out_dir / "top_metrics_heatmap.png"
    grid_png = out_dir / "per_metric_top_errors.png"
    index_html = out_dir / "index.html"

    save_sorted_csv(sorted_csv, rows, scores, metric_names, order)
    plot_score_bars(score_png, labels, scores, order, top_k, f"{title_prefix}: worst motions")
    plot_metric_means(means_png, metric_names, values, f"{title_prefix}: metric means")
    plot_heatmap(heatmap_png, labels, metric_names, values, order, top_k, f"{title_prefix}: top {top_k} heatmap")
    plot_metric_grid(grid_png, labels, metric_names, values, top_k=min(top_k, 15), title=f"{title_prefix}: per metric")
    write_index_html(
        index_html, csv_path, sorted_csv, [score_png, means_png, heatmap_png, grid_png], rows, scores, order
    )

    print(f"[INFO] source csv: {csv_path}")
    print(f"[INFO] output dir: {out_dir}")
    print(f"[INFO] wrote: {sorted_csv}")
    print(f"[INFO] wrote: {index_html}")
    for image in (score_png, means_png, heatmap_png, grid_png):
        print(f"[INFO] wrote: {image}")


if __name__ == "__main__":
    main()
