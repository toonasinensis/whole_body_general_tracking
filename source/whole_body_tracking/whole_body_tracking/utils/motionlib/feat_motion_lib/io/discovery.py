from __future__ import annotations

import random
import warnings
from pathlib import Path

from ..config import DiscoveryOptions


def find_npz_files(
    dir_path: str,
    motion_num: int,
    dataset_txt: str | None,
    eval_mode: bool,
    distributed: bool,
    local_rank: int,
    total_rank: int,
    sample_counter: int,
) -> tuple[list[str], int]:
    """Return selected npz files plus the next eval sample counter."""
    if dataset_txt is not None:
        with open(dataset_txt) as f:
            relative_paths = [line.strip() for line in f if line.strip()]
        npz_files: list[str] = []
        skipped = 0
        for rel_path in relative_paths:
            candidate = Path(rel_path)
            if not candidate.is_absolute():
                candidate = Path(dir_path) / rel_path
            if candidate.suffix.lower() != ".npz":
                warnings.warn(f"Skip non-npz entry in dataset_txt: {rel_path}")
                skipped += 1
                continue
            if not candidate.is_file():
                warnings.warn(f"Skip missing npz file in dataset_txt: {candidate}")
                skipped += 1
                continue
            npz_files.append(str(candidate))
        if not npz_files:
            raise FileNotFoundError(
                f"No valid .npz files found from dataset_txt={dataset_txt}. Skipped entries: {skipped}"
            )
        random.seed(42)
        random.shuffle(npz_files)
    else:
        npz_files = sorted(str(p) for p in Path(dir_path).rglob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {dir_path}")

    next_counter = sample_counter
    if len(npz_files) > motion_num != -1:
        if eval_mode:
            start = min(sample_counter * motion_num, len(npz_files) - 1)
            end = min(start + motion_num, len(npz_files) - 1)
            if start >= len(npz_files) - 1:
                raise ValueError("eval_mode sampling exhausted all data.")
            sampled = npz_files[start:end]
            print("eval_mode process: ", float(start) / len(npz_files))
            next_counter += 1
        else:
            sampled = sorted(random.sample(npz_files, motion_num))
    elif distributed:
        per_rank = len(npz_files) // total_rank
        start = per_rank * local_rank
        end = per_rank * (local_rank + 1)
        sampled = npz_files[start:end]
        print(f"[GPU {local_rank}/{total_rank}]; files range: {start}→{end}; total number: {per_rank}")
    else:
        sampled = npz_files
        print(f"A total of {len(npz_files)} NPZ motion files are loaded")

    return sampled, next_counter


def discover_npz_files(options: DiscoveryOptions) -> tuple[list[str], int]:
    return find_npz_files(
        dir_path=str(options.motion_dir),
        motion_num=options.max_motion_num,
        dataset_txt=str(options.dataset_txt) if options.dataset_txt is not None else None,
        eval_mode=options.eval_mode,
        distributed=options.distributed,
        local_rank=options.local_rank,
        total_rank=options.total_rank,
        sample_counter=options.sample_counter,
    )


def discover_pkl_files(input_path: str | Path) -> list[str]:
    input_path = Path(input_path)
    if input_path.is_dir():
        files = sorted(str(path) for path in input_path.glob("*.pkl"))
        if not files:
            raise FileNotFoundError(f"No .pkl files found in directory: {input_path}")
        return files
    if input_path.is_file():
        if input_path.suffix.lower() != ".pkl":
            raise ValueError(f"Expected a .pkl file, got: {input_path}")
        return [str(input_path)]
    raise FileNotFoundError(f"Path not found: {input_path}")
