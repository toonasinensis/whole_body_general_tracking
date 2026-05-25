from __future__ import annotations

import argparse
import csv
import html
import numpy as np
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sim2sim_g1.metrics import METRIC_NAMES

DEFAULT_DATASET_TXT = "/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/g1_flat/all_top_files.txt"
TIMESTAMP_RE = re.compile(r"(?:^|_)(\d{8}_\d{6})$")


@dataclass
class PolicyMetrics:
    label: str
    csv_path: Path
    rows: list[dict[str, str]]
    metric_names: list[str]
    values: np.ndarray
    scores: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare sim2sim metrics CSVs from different policies.")
    parser.add_argument("csvs", nargs="*", help="Metrics CSV paths. Shell globs are OK.")
    parser.add_argument("--csv", action="append", default=[], help="Metrics CSV path. Can be repeated.")
    parser.add_argument(
        "--metrics_dir",
        default=None,
        help="Directory used when no CSV paths are provided. Defaults to DATASET_TXT's directory.",
    )
    parser.add_argument("--latest", type=int, default=5, help="Compare latest N metrics CSVs when no paths are given.")
    parser.add_argument(
        "--labels",
        default="",
        help="Comma-separated policy labels. Defaults to labels inferred from CSV names.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory. Defaults to sim2sim_metrics_compare_<timestamp> under the first CSV directory.",
    )
    parser.add_argument(
        "--score_metrics",
        default="",
        help="Comma-separated metric names used for aggregate score. Defaults to all common error metrics.",
    )
    parser.add_argument("--top_k", type=int, default=40, help="Number of motions shown in detailed plots.")
    parser.add_argument("--title", default="sim2sim policy comparison", help="Title used in figures and HTML.")
    return parser.parse_args()


def metrics_dir_from_env() -> Path:
    if os.environ.get("METRICS_DIR"):
        return Path(os.environ["METRICS_DIR"]).expanduser().resolve()
    dataset_txt = Path(os.environ.get("DATASET_TXT", DEFAULT_DATASET_TXT)).expanduser()
    return dataset_txt.parent.resolve()


def resolve_csv_paths(args: argparse.Namespace) -> list[Path]:
    raw_paths = [*args.csvs, *args.csv]
    if raw_paths:
        paths = [Path(path).expanduser().resolve() for path in raw_paths]
    else:
        metrics_dir = Path(args.metrics_dir).expanduser().resolve() if args.metrics_dir else metrics_dir_from_env()
        candidates = sorted(
            metrics_dir.glob("*.sim2sim_metrics*.csv"), key=lambda path: path.stat().st_mtime, reverse=True
        )
        paths = candidates[: max(1, int(args.latest))]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Metrics CSV not found: {missing}")
    if len(paths) < 2:
        raise ValueError("Need at least two metrics CSVs to compare.")
    return paths


def infer_label(csv_path: Path) -> str:
    stem = csv_path.stem
    marker = ".sim2sim_metrics"
    if marker in stem:
        suffix = stem.split(marker, 1)[1].strip("_")
        if suffix:
            timestamp_match = TIMESTAMP_RE.search(suffix)
            if timestamp_match:
                without_timestamp = suffix[: timestamp_match.start()].strip("_")
                return without_timestamp or timestamp_match.group(1)
            return suffix
    return stem


def unique_labels(labels: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    result = []
    for label in labels:
        base = label or "policy"
        counts[base] = counts.get(base, 0) + 1
        result.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return result


def resolve_labels(args: argparse.Namespace, csv_paths: list[Path]) -> list[str]:
    if args.labels.strip():
        labels = [part.strip() for part in args.labels.split(",") if part.strip()]
        if len(labels) != len(csv_paths):
            raise ValueError(f"--labels count {len(labels)} does not match CSV count {len(csv_paths)}.")
        return unique_labels(labels)
    return unique_labels([infer_label(path) for path in csv_paths])


def read_metrics_csv(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"No rows found in metrics CSV: {csv_path}")
    return rows, fieldnames


def metric_names_from_fields(fieldnames: list[str]) -> list[str]:
    return [name for name in METRIC_NAMES if name in fieldnames]


def metric_matrix(rows: list[dict[str, str]], metric_names: list[str]) -> np.ndarray:
    values = np.full((len(rows), len(metric_names)), np.nan, dtype=np.float64)
    for row_i, row in enumerate(rows):
        for col_i, name in enumerate(metric_names):
            try:
                values[row_i, col_i] = float(row.get(name, "nan"))
            except ValueError:
                values[row_i, col_i] = np.nan
    return values


def load_policy_metrics(csv_paths: list[Path], labels: list[str]) -> list[PolicyMetrics]:
    policies = []
    for label, csv_path in zip(labels, csv_paths):
        rows, fieldnames = read_metrics_csv(csv_path)
        metric_names = metric_names_from_fields(fieldnames)
        if not metric_names:
            raise ValueError(f"No known sim2sim metric columns found in {csv_path}")
        policies.append(
            PolicyMetrics(
                label=label,
                csv_path=csv_path,
                rows=rows,
                metric_names=metric_names,
                values=metric_matrix(rows, metric_names),
            )
        )
    return policies


def common_metric_names(policies: list[PolicyMetrics]) -> list[str]:
    available = [set(policy.metric_names) for policy in policies]
    return [name for name in METRIC_NAMES if all(name in names for names in available)]


def parse_metric_selection(selection: str, available: list[str]) -> list[str]:
    if not selection.strip():
        return list(available)
    wanted = [part.strip() for part in selection.split(",") if part.strip()]
    missing = [name for name in wanted if name not in available]
    if missing:
        raise ValueError(f"Selected score metrics not found in every CSV: {missing}. Common: {available}")
    return wanted


def restrict_values(policy: PolicyMetrics, metric_names: list[str]) -> np.ndarray:
    indices = [policy.metric_names.index(name) for name in metric_names]
    return policy.values[:, indices]


def compute_scores(policies: list[PolicyMetrics], metric_names: list[str], score_metric_names: list[str]) -> None:
    values_by_policy = [restrict_values(policy, metric_names) for policy in policies]
    stacked = np.vstack(values_by_policy)
    score_indices = [metric_names.index(name) for name in score_metric_names]
    selected = stacked[:, score_indices]
    mins = np.nanmin(selected, axis=0)
    maxs = np.nanmax(selected, axis=0)
    denom = np.maximum(maxs - mins, 1.0e-12)

    offset = 0
    for policy, values in zip(policies, values_by_policy):
        selected_values = values[:, score_indices]
        normalized = (selected_values - mins) / denom
        policy.scores = np.nanmean(normalized, axis=1)
        offset += len(policy.rows)


def motion_key(row: dict[str, str], row_index: int) -> str:
    return row.get("motion_file") or row.get("motion_index") or f"row_{row_index:05d}"


def motion_label(key: str) -> str:
    return Path(key).name if key else "motion"


def safe_column(label: str) -> str:
    chars = [char if char.isalnum() or char in ("_", "-") else "_" for char in label.strip()]
    return "".join(chars).strip("_") or "policy"


def write_policy_summary(out_path: Path, policies: list[PolicyMetrics], metric_names: list[str]) -> None:
    fieldnames = ["label", "csv_path", "num_motions", "total_samples", "aggregate_score", *metric_names]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for policy in policies:
            values = restrict_values(policy, metric_names)
            samples = [int(float(row.get("samples", "0") or 0)) for row in policy.rows]
            row = {
                "label": policy.label,
                "csv_path": str(policy.csv_path),
                "num_motions": len(policy.rows),
                "total_samples": sum(samples),
                "aggregate_score": f"{float(np.nanmean(policy.scores)):.9g}",
            }
            row.update({name: f"{float(np.nanmean(values[:, i])):.9g}" for i, name in enumerate(metric_names)})
            writer.writerow(row)


def per_motion_records(
    policies: list[PolicyMetrics],
    metric_names: list[str],
) -> list[dict[str, str | float]]:
    by_key: dict[str, dict[str, dict[str, str | float]]] = {}
    for policy in policies:
        values = restrict_values(policy, metric_names)
        assert policy.scores is not None
        for row_i, row in enumerate(policy.rows):
            key = motion_key(row, row_i)
            entry: dict[str, str | float] = {
                "motion_file": row.get("motion_file", key),
                "score": float(policy.scores[row_i]),
            }
            entry.update({name: float(values[row_i, col_i]) for col_i, name in enumerate(metric_names)})
            by_key.setdefault(key, {})[policy.label] = entry

    records = []
    for key in sorted(by_key):
        policy_entries = by_key[key]
        score_items = [
            (label, float(entry["score"]))
            for label, entry in policy_entries.items()
            if np.isfinite(float(entry["score"]))
        ]
        best_label = min(score_items, key=lambda item: item[1])[0] if score_items else ""
        record: dict[str, str | float] = {
            "motion_key": key,
            "motion": motion_label(key),
            "best_policy": best_label,
        }
        for policy in policies:
            prefix = safe_column(policy.label)
            entry = policy_entries.get(policy.label)
            if entry is None:
                record[f"{prefix}__score"] = ""
                for name in metric_names:
                    record[f"{prefix}__{name}"] = ""
                continue
            record[f"{prefix}__score"] = f"{float(entry['score']):.9g}"
            for name in metric_names:
                record[f"{prefix}__{name}"] = f"{float(entry[name]):.9g}"
        records.append(record)
    return records


def write_per_motion_comparison(
    out_path: Path, records: list[dict[str, str | float]], policies: list[PolicyMetrics], metric_names: list[str]
) -> None:
    fieldnames = ["motion_key", "motion", "best_policy"]
    for policy in policies:
        prefix = safe_column(policy.label)
        fieldnames.append(f"{prefix}__score")
        fieldnames.extend(f"{prefix}__{name}" for name in metric_names)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def import_pyplot():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required. Install it or run inside my_env.") from exc
    return plt


def plot_aggregate_scores(out_path: Path, policies: list[PolicyMetrics], title: str) -> None:
    plt = import_pyplot()
    means = np.asarray([float(np.nanmean(policy.scores)) for policy in policies], dtype=np.float64)
    order = np.argsort(means)[::-1]
    labels = [policies[i].label for i in order]
    fig_h = max(4.0, 0.45 * len(policies) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(np.arange(len(order)), means[order], color="#3a7ca5")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("mean normalized aggregate error (lower is better)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_metric_means_raw(out_path: Path, policies: list[PolicyMetrics], metric_names: list[str], title: str) -> None:
    plt = import_pyplot()
    means = np.vstack([np.nanmean(restrict_values(policy, metric_names), axis=0) for policy in policies])
    x = np.arange(len(metric_names))
    width = min(0.8 / len(policies), 0.24)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for policy_i, policy in enumerate(policies):
        offset = (policy_i - (len(policies) - 1) / 2.0) * width
        ax.bar(x + offset, means[policy_i], width=width, label=policy.label)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=35, ha="right")
    ax.set_ylabel("mean error")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_metric_means_heatmap(
    out_path: Path, policies: list[PolicyMetrics], metric_names: list[str], title: str
) -> None:
    plt = import_pyplot()
    means = np.vstack([np.nanmean(restrict_values(policy, metric_names), axis=0) for policy in policies])
    mins = np.nanmin(means, axis=0)
    maxs = np.nanmax(means, axis=0)
    normalized = (means - mins) / np.maximum(maxs - mins, 1.0e-12)

    fig_h = max(4.0, 0.45 * len(policies) + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    image = ax.imshow(normalized, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(policies)))
    ax.set_yticklabels([policy.label for policy in policies])
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("normalized mean error (lower is better)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_best_policy_counts(
    out_path: Path, records: list[dict[str, str | float]], policies: list[PolicyMetrics], title: str
) -> None:
    plt = import_pyplot()
    counts = {policy.label: 0 for policy in policies}
    for record in records:
        best = str(record.get("best_policy", ""))
        if best in counts:
            counts[best] += 1
    labels = list(counts)
    values = np.asarray([counts[label] for label in labels], dtype=np.float64)
    order = np.argsort(values)[::-1]
    fig_h = max(4.0, 0.45 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.barh(np.arange(len(labels)), values[order], color="#5b8c5a")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels([labels[i] for i in order])
    ax.set_xlabel("motions with lowest aggregate score")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_motion_score_heatmap(
    out_path: Path,
    records: list[dict[str, str | float]],
    policies: list[PolicyMetrics],
    top_k: int,
    title: str,
) -> None:
    plt = import_pyplot()
    labels = [safe_column(policy.label) for policy in policies]
    rows = []
    names = []
    for record in records:
        scores = []
        for label in labels:
            value = record.get(f"{label}__score", "")
            scores.append(float(value) if value != "" else np.nan)
        finite = np.asarray(scores, dtype=np.float64)
        if np.isfinite(finite).sum() >= 2:
            rows.append(finite)
            names.append(str(record.get("motion", "")))
    if not rows:
        return
    matrix = np.vstack(rows)
    spread = np.nanmax(matrix, axis=1) - np.nanmin(matrix, axis=1)
    selected = np.argsort(spread)[::-1][: max(1, min(top_k, len(spread)))]
    chosen = matrix[selected]

    fig_h = max(5.0, 0.34 * len(selected) + 1.8)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    image = ax.imshow(chosen, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(policies)))
    ax.set_xticklabels([policy.label for policy in policies], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(selected)))
    ax.set_yticklabels([names[i] for i in selected], fontsize=7)
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("normalized aggregate error")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_index_html(
    out_path: Path,
    title: str,
    policies: list[PolicyMetrics],
    summary_csv: Path,
    per_motion_csv: Path,
    images: list[Path],
) -> None:
    rows = []
    for policy in policies:
        rows.append(
            "<tr>"
            f"<td>{html.escape(policy.label)}</td>"
            f"<td>{len(policy.rows)}</td>"
            f"<td>{float(np.nanmean(policy.scores)):.6f}</td>"
            f"<td><code>{html.escape(str(policy.csv_path))}</code></td>"
            "</tr>"
        )
    image_tags = "\n".join(
        f'<section><h2>{html.escape(image.stem)}</h2><img src="{html.escape(image.name)}" /></section>'
        for image in images
        if image.exists()
    )
    out_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset='utf-8' />",
                f"<title>{html.escape(title)}</title>",
                "<style>",
                "body{font-family:Arial,sans-serif;margin:24px;line-height:1.4;color:#222}",
                "img{max-width:100%;height:auto;border:1px solid #ddd}",
                "table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #ddd;padding:4px 8px}",
                "code{background:#f3f3f3;padding:2px 4px}",
                ".wide{overflow-x:auto}",
                "</style></head><body>",
                f"<h1>{html.escape(title)}</h1>",
                f"<p>Summary CSV: <a href='{html.escape(summary_csv.name)}'>{html.escape(summary_csv.name)}</a></p>",
                (
                    "<p>Per-motion CSV: <a"
                    f" href='{html.escape(per_motion_csv.name)}'>{html.escape(per_motion_csv.name)}</a></p>"
                ),
                "<div class='wide'><table>",
                "<tr><th>policy</th><th>motions</th><th>aggregate score</th><th>source</th></tr>",
                *rows,
                "</table></div>",
                image_tags,
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )


def resolve_out_dir(args: argparse.Namespace, csv_paths: list[Path]) -> Path:
    if args.out_dir:
        return Path(args.out_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (csv_paths[0].parent / f"sim2sim_metrics_compare_{timestamp}").resolve()


def main() -> None:
    args = parse_args()
    csv_paths = resolve_csv_paths(args)
    labels = resolve_labels(args, csv_paths)
    policies = load_policy_metrics(csv_paths, labels)
    metric_names = common_metric_names(policies)
    if not metric_names:
        raise ValueError("No common sim2sim metric columns found across CSVs.")
    score_metric_names = parse_metric_selection(args.score_metrics, metric_names)
    compute_scores(policies, metric_names, score_metric_names)

    out_dir = resolve_out_dir(args, csv_paths)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "policy_summary.csv"
    per_motion_csv = out_dir / "per_motion_comparison.csv"
    aggregate_png = out_dir / "aggregate_score_by_policy.png"
    means_raw_png = out_dir / "metric_means_by_policy.png"
    means_heatmap_png = out_dir / "metric_means_normalized_heatmap.png"
    best_counts_png = out_dir / "best_policy_counts.png"
    score_heatmap_png = out_dir / "per_motion_score_heatmap.png"
    index_html = out_dir / "index.html"

    records = per_motion_records(policies, metric_names)
    write_policy_summary(summary_csv, policies, metric_names)
    write_per_motion_comparison(per_motion_csv, records, policies, metric_names)
    plot_aggregate_scores(aggregate_png, policies, f"{args.title}: aggregate score")
    plot_metric_means_raw(means_raw_png, policies, metric_names, f"{args.title}: raw metric means")
    plot_metric_means_heatmap(means_heatmap_png, policies, metric_names, f"{args.title}: normalized metric means")
    plot_best_policy_counts(best_counts_png, records, policies, f"{args.title}: best policy per motion")
    plot_motion_score_heatmap(
        score_heatmap_png,
        records,
        policies,
        max(1, int(args.top_k)),
        f"{args.title}: motions with largest policy spread",
    )
    images = [aggregate_png, means_raw_png, means_heatmap_png, best_counts_png, score_heatmap_png]
    write_index_html(index_html, args.title, policies, summary_csv, per_motion_csv, images)

    print(f"[INFO] compared CSVs: {len(csv_paths)}")
    for policy in policies:
        print(f"[INFO] {policy.label}: {policy.csv_path}")
    print(f"[INFO] output dir: {out_dir}")
    print(f"[INFO] wrote: {summary_csv}")
    print(f"[INFO] wrote: {per_motion_csv}")
    print(f"[INFO] wrote: {index_html}")
    for image in images:
        if image.exists():
            print(f"[INFO] wrote: {image}")


if __name__ == "__main__":
    main()
