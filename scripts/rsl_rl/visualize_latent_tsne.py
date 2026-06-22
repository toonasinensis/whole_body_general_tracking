"""Visualize a distilled student's latent space with t-SNE."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Collect latent codes from a distillation student and plot t-SNE.")
parser.add_argument("--task", type=str, default="LatentDistill-Flat-G1-v0", help="Isaac Lab task name.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel envs used for collection.")
parser.add_argument("--resume_path", type=str, required=True, help="Path to the distilled student checkpoint.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion root or npz file.")
parser.add_argument("--dataset_txt", type=str, default=None, help="Optional motion list used to filter motion_file.")
parser.add_argument(
    "--smpl_file_path", type=str, default=None, help="Optional SMPL directory for paired motion loading."
)
parser.add_argument("--out_dir", type=str, default="logs/rsl_rl/latent_tsne", help="Output directory.")
parser.add_argument("--output_prefix", type=str, default="lafan_latent_tsne", help="Output filename prefix.")
parser.add_argument("--num_samples", type=int, default=4096, help="Stop collection after this many samples.")
parser.add_argument(
    "--num_steps", type=int, default=None, help="Maximum env steps to collect. Defaults from samples/envs."
)
parser.add_argument("--warmup_steps", type=int, default=5, help="Policy/env steps before recording latents.")
parser.add_argument("--latent_key", type=str, default="posterior_mu", help="Student output key to visualize.")
parser.add_argument(
    "--label_mode",
    type=str,
    default="lafan_action",
    choices=["lafan_action", "motion_name", "motion_id"],
    help="How to derive point labels for color/metrics.",
)
parser.add_argument(
    "--model_train_mode",
    action="store_true",
    default=False,
    help="Call student with train_mode=True. Useful when visualizing sampled latent with sample_latent_train.",
)
parser.add_argument(
    "--motion_eval_mode",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Set env_cfg.commands.motion.eval_mode before loading motions.",
)
parser.add_argument("--episode_length_s", type=float, default=9999.0, help="Long episode length for collection.")
parser.add_argument("--max_tsne_points", type=int, default=4096, help="Subsample collected points before t-SNE.")
parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity; clipped to sample count.")
parser.add_argument("--tsne_iter", type=int, default=1000, help="t-SNE optimization iterations.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling and t-SNE.")
parser.add_argument("--point_size", type=float, default=8.0, help="Scatter point size.")
parser.add_argument("--alpha", type=float, default=0.75, help="Scatter alpha.")
parser.add_argument("--max_legend_items", type=int, default=20, help="Use legend when motion count is at most this.")
parser.add_argument("--knn_neighbors", type=int, default=5, help="kNN neighbors for label separability metric.")
parser.add_argument("--verbose_tsne", type=int, default=1, help="sklearn TSNE verbose level.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import whole_body_tracking.tasks  # noqa: E402,F401


def _as_actions(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, dict):
        if "actions" not in output:
            raise KeyError(f"Student output is missing 'actions'. Available keys: {list(output.keys())}")
        return output["actions"]
    return output


def _output_tensor(output: torch.Tensor | dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if not isinstance(output, dict):
        if key == "actions":
            return output
        raise TypeError(f"Student returned a tensor; only key='actions' is available, requested {key!r}.")
    if key not in output:
        raise KeyError(f"Student output is missing {key!r}. Available keys: {list(output.keys())}")
    value = output[key]
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2:
        raise ValueError(f"Expected {key!r} to be 2D after flattening batch, got shape {tuple(value.shape)}")
    return value


def _motion_files_for_ids(motion_cmd, motion_ids: torch.Tensor) -> list[str]:
    file_names = list(getattr(motion_cmd.motion, "file_names", []) or [])
    if not file_names:
        file_names = [f"motion_{i:05d}.npz" for i in range(int(motion_cmd.motion.motion_num))]
    ids = motion_ids.detach().cpu().numpy().astype(int).tolist()
    return [file_names[i] if 0 <= i < len(file_names) else f"motion_{i:05d}.npz" for i in ids]


def _infer_label(motion_file: str, motion_id: int, mode: str) -> str:
    if mode == "motion_id":
        return f"motion_{motion_id:03d}"
    stem = Path(motion_file).stem
    if mode == "motion_name":
        return stem

    # LAFAN filenames usually look like "dance2_subject1.npz".
    action = stem.split("_subject", 1)[0]
    action = re.sub(r"\d+$", "", action)
    return action or stem


def _attach_labels(meta: dict[str, np.ndarray], mode: str) -> None:
    labels = [
        _infer_label(str(motion_file), int(motion_id), mode)
        for motion_file, motion_id in zip(meta["motion_file"], meta["motion_id"], strict=False)
    ]
    meta["label"] = np.asarray(labels, dtype=str)


def _collect_latents(env, policy, motion_cmd, args) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    device = env.unwrapped.device
    obs = env.get_observations().to(device)

    for _ in range(max(0, int(args.warmup_steps))):
        with torch.inference_mode():
            output = policy(obs, train_mode=args.model_train_mode)
            obs, _, _, _ = env.step(_as_actions(output))

    target_steps = args.num_steps
    if target_steps is None:
        target_steps = int(math.ceil(max(args.num_samples, 1) / max(env.num_envs, 1)))
    target_steps = max(1, int(target_steps))

    latent_chunks: list[np.ndarray] = []
    kl_chunks: list[np.ndarray] = []
    env_chunks: list[np.ndarray] = []
    step_chunks: list[np.ndarray] = []
    motion_id_chunks: list[np.ndarray] = []
    local_frame_chunks: list[np.ndarray] = []
    global_frame_chunks: list[np.ndarray] = []
    motion_file_chunks: list[np.ndarray] = []

    total = 0
    for step in range(target_steps):
        with torch.inference_mode():
            output = policy(obs, train_mode=args.model_train_mode)
            latent = _output_tensor(output, args.latent_key)
            actions = _as_actions(output)

            motion_ids = motion_cmd.motion_ids.detach().clone()
            local_frames = motion_cmd.local_time_steps.detach().clone()
            global_frames = motion_cmd.global_time_steps.detach().clone()
            kl = output.get("kl") if isinstance(output, dict) else None
            if kl is None:
                kl = torch.full((env.num_envs,), float("nan"), device=device)

            latent_chunks.append(latent.detach().cpu().numpy().astype(np.float32, copy=False))
            kl_chunks.append(kl.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False))
            env_chunks.append(np.arange(env.num_envs, dtype=np.int64))
            step_chunks.append(np.full(env.num_envs, step, dtype=np.int64))
            motion_id_chunks.append(motion_ids.cpu().numpy().astype(np.int64, copy=False))
            local_frame_chunks.append(local_frames.cpu().numpy().astype(np.int64, copy=False))
            global_frame_chunks.append(global_frames.cpu().numpy().astype(np.int64, copy=False))
            motion_file_chunks.append(np.asarray(_motion_files_for_ids(motion_cmd, motion_ids), dtype=str))

            total += env.num_envs
            obs, _, _, _ = env.step(actions)

        if total >= args.num_samples:
            break

    latents = np.concatenate(latent_chunks, axis=0)[: args.num_samples]
    meta = {
        "kl": np.concatenate(kl_chunks, axis=0)[: args.num_samples],
        "env_id": np.concatenate(env_chunks, axis=0)[: args.num_samples],
        "step": np.concatenate(step_chunks, axis=0)[: args.num_samples],
        "motion_id": np.concatenate(motion_id_chunks, axis=0)[: args.num_samples],
        "local_frame": np.concatenate(local_frame_chunks, axis=0)[: args.num_samples],
        "global_frame": np.concatenate(global_frame_chunks, axis=0)[: args.num_samples],
        "motion_file": np.concatenate(motion_file_chunks, axis=0)[: args.num_samples],
    }
    return latents, meta


def _select_points(latents: np.ndarray, meta: dict[str, np.ndarray], max_points: int, seed: int):
    count = latents.shape[0]
    if count <= max_points:
        idx = np.arange(count)
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(count, size=max_points, replace=False))
    return latents[idx], {key: value[idx] for key, value in meta.items()}, idx


def _resolve_perplexity(requested: float, count: int) -> float:
    if count < 4:
        raise ValueError(f"Need at least 4 points for t-SNE, got {count}. Increase --num_samples.")
    return float(min(requested, max(2.0, (count - 1) / 3.0), count - 1e-3))


def _run_tsne(latents: np.ndarray, args) -> np.ndarray:
    if latents.ndim != 2 or latents.shape[1] < 2:
        raise ValueError(f"t-SNE needs vector latents with dim >= 2, got shape {latents.shape}")
    perplexity = _resolve_perplexity(args.perplexity, latents.shape[0])
    print(
        f"[INFO] Running t-SNE: points={latents.shape[0]}, dim={latents.shape[1]}, "
        f"perplexity={perplexity:.3f}, max_iter={args.tsne_iter}"
    )
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=int(args.tsne_iter),
        random_state=int(args.seed),
        verbose=int(args.verbose_tsne),
    )
    return tsne.fit_transform(latents).astype(np.float32, copy=False)


def _write_csv(path: Path, points: np.ndarray, meta: dict[str, np.ndarray], fps: float) -> None:
    fieldnames = [
        "tsne_x",
        "tsne_y",
        "label",
        "motion_id",
        "motion_file",
        "motion_name",
        "local_frame",
        "global_frame",
        "time_s",
        "env_id",
        "step",
        "kl",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(points.shape[0]):
            motion_file = str(meta["motion_file"][i])
            writer.writerow(
                {
                    "tsne_x": float(points[i, 0]),
                    "tsne_y": float(points[i, 1]),
                    "label": str(meta["label"][i]),
                    "motion_id": int(meta["motion_id"][i]),
                    "motion_file": motion_file,
                    "motion_name": Path(motion_file).stem,
                    "local_frame": int(meta["local_frame"][i]),
                    "global_frame": int(meta["global_frame"][i]),
                    "time_s": float(meta["local_frame"][i] / fps) if fps > 0 else float("nan"),
                    "env_id": int(meta["env_id"][i]),
                    "step": int(meta["step"][i]),
                    "kl": float(meta["kl"][i]),
                }
            )


def _plot_by_motion(path: Path, points: np.ndarray, meta: dict[str, np.ndarray], args) -> None:
    motion_ids = meta["motion_id"].astype(int)
    motion_files = meta["motion_file"].astype(str)
    unique_ids = np.unique(motion_ids)
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    if len(unique_ids) <= args.max_legend_items:
        cmap = plt.get_cmap("tab20", max(len(unique_ids), 1))
        for color_idx, motion_id in enumerate(unique_ids):
            mask = motion_ids == motion_id
            first_file = motion_files[np.where(mask)[0][0]]
            label = f"{motion_id}: {Path(first_file).stem}"
            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                s=args.point_size,
                alpha=args.alpha,
                color=cmap(color_idx),
                label=label,
                linewidths=0,
            )
        ax.legend(loc="best", fontsize=7, markerscale=1.5, frameon=False)
    else:
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            c=motion_ids,
            cmap="turbo",
            s=args.point_size,
            alpha=args.alpha,
            linewidths=0,
        )
        fig.colorbar(scatter, ax=ax, label="motion_id")
    ax.set_title(f"Latent t-SNE by motion ({args.latent_key})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_by_label(path: Path, points: np.ndarray, meta: dict[str, np.ndarray], args) -> None:
    labels = meta["label"].astype(str)
    counts = Counter(labels.tolist())
    unique_labels = [label for label, _ in counts.most_common()]
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))
    for color_idx, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=args.point_size,
            alpha=args.alpha,
            color=cmap(color_idx % cmap.N),
            label=f"{label} ({int(mask.sum())})",
            linewidths=0,
        )
    if len(unique_labels) <= args.max_legend_items:
        ax.legend(loc="best", fontsize=8, markerscale=1.5, frameon=False)
    ax.set_title(f"Latent t-SNE by label ({args.latent_key}, {args.label_mode})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_by_time(path: Path, points: np.ndarray, meta: dict[str, np.ndarray], fps: float, args) -> None:
    time_s = meta["local_frame"].astype(np.float32) / float(fps) if fps > 0 else meta["local_frame"].astype(np.float32)
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=time_s,
        cmap="viridis",
        s=args.point_size,
        alpha=args.alpha,
        linewidths=0,
    )
    fig.colorbar(scatter, ax=ax, label="motion local time (s)" if fps > 0 else "motion local frame")
    ax.set_title(f"Latent t-SNE by local time ({args.latent_key})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _encode_labels(labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    unique = sorted(set(labels.astype(str).tolist()))
    index = {label: idx for idx, label in enumerate(unique)}
    encoded = np.asarray([index[label] for label in labels.astype(str)], dtype=np.int64)
    return encoded, unique


def _knn_cv_accuracy(
    features: np.ndarray, labels: np.ndarray, neighbors: int, seed: int
) -> dict[str, float | int | None]:
    encoded, unique = _encode_labels(labels)
    counts = np.bincount(encoded)
    min_count = int(counts.min()) if counts.size else 0
    if len(unique) < 2 or min_count < 2:
        return {"mean": None, "std": None, "folds": 0, "neighbors": 0}
    folds = min(5, min_count)
    n_neighbors = min(max(1, int(neighbors)), max(1, min_count - 1))
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
    scores = cross_val_score(classifier, features, encoded, cv=cv)
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "folds": int(folds),
        "neighbors": int(n_neighbors),
    }


def _silhouette(features: np.ndarray, labels: np.ndarray) -> float | None:
    encoded, unique = _encode_labels(labels)
    if len(unique) < 2 or len(unique) >= features.shape[0]:
        return None
    return float(silhouette_score(features, encoded))


def _kmeans_alignment(features: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, float | int | None]:
    encoded, unique = _encode_labels(labels)
    if len(unique) < 2 or features.shape[0] < len(unique):
        return {"n_clusters": len(unique), "nmi": None, "ari": None}
    kmeans = KMeans(n_clusters=len(unique), n_init=10, random_state=int(seed))
    clusters = kmeans.fit_predict(features)
    return {
        "n_clusters": int(len(unique)),
        "nmi": float(normalized_mutual_info_score(encoded, clusters)),
        "ari": float(adjusted_rand_score(encoded, clusters)),
    }


def _write_metrics(path: Path, latents: np.ndarray, points: np.ndarray, meta: dict[str, np.ndarray], args) -> None:
    labels = meta["label"].astype(str)
    counts = Counter(labels.tolist())
    metrics = {
        "label_mode": args.label_mode,
        "latent_key": args.latent_key,
        "num_points": int(points.shape[0]),
        "latent_dim": int(latents.shape[1]),
        "label_counts": dict(sorted(counts.items())),
        "silhouette": {
            "latent": _silhouette(latents, labels),
            "tsne": _silhouette(points, labels),
        },
        "knn_cv_accuracy": {
            "latent": _knn_cv_accuracy(latents, labels, args.knn_neighbors, args.seed),
            "tsne": _knn_cv_accuracy(points, labels, args.knn_neighbors, args.seed),
        },
        "kmeans_alignment": {
            "latent": _kmeans_alignment(latents, labels, args.seed),
            "tsne": _kmeans_alignment(points, labels, args.seed),
        },
    }
    with path.open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.episode_length_s = float(args_cli.episode_length_s)

    if args_cli.motion_file is not None:
        env_cfg.commands.motion.motion_file = args_cli.motion_file
    if args_cli.dataset_txt is not None:
        env_cfg.commands.motion.dataset_txt = args_cli.dataset_txt
    if args_cli.smpl_file_path is not None:
        env_cfg.commands.motion.smpl_file_path = args_cli.smpl_file_path
    env_cfg.commands.motion.eval_mode = bool(args_cli.motion_eval_mode)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    runner_device = getattr(agent_cfg, "device", None) or getattr(args_cli, "device", None) or env.unwrapped.device
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=runner_device)
    print(f"[INFO] Loading student checkpoint: {args_cli.resume_path}")
    runner.load(args_cli.resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        motion_cmd = env.unwrapped.command_manager.get_term("motion")
    except Exception as exc:
        env.close()
        raise RuntimeError("This script requires a command term named 'motion'.") from exc

    latents, meta = _collect_latents(env, policy, motion_cmd, args_cli)
    fps = float(getattr(motion_cmd.motion, "fps", 0.0) or 0.0)
    env.close()

    latents, meta, selected_idx = _select_points(latents, meta, int(args_cli.max_tsne_points), int(args_cli.seed))
    _attach_labels(meta, args_cli.label_mode)
    points = _run_tsne(latents, args_cli)

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args_cli.output_prefix
    csv_path = out_dir / f"{prefix}_points.csv"
    npz_path = out_dir / f"{prefix}_latents.npz"
    motion_png = out_dir / f"{prefix}_motion.png"
    label_png = out_dir / f"{prefix}_label.png"
    time_png = out_dir / f"{prefix}_time.png"
    metrics_path = out_dir / f"{prefix}_metrics.json"

    _write_csv(csv_path, points, meta, fps)
    np.savez_compressed(
        npz_path,
        latent=latents,
        tsne=points,
        selected_index=selected_idx,
        fps=np.asarray([fps], dtype=np.float32),
        **meta,
    )
    _plot_by_motion(motion_png, points, meta, args_cli)
    _plot_by_label(label_png, points, meta, args_cli)
    _plot_by_time(time_png, points, meta, fps, args_cli)
    _write_metrics(metrics_path, latents, points, meta, args_cli)

    print(f"[INFO] Wrote CSV: {csv_path}")
    print(f"[INFO] Wrote latent archive: {npz_path}")
    print(f"[INFO] Wrote motion plot: {motion_png}")
    print(f"[INFO] Wrote label plot: {label_png}")
    print(f"[INFO] Wrote time plot: {time_png}")
    print(f"[INFO] Wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
