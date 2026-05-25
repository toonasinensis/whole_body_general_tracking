"""Randomly sample lines from a hard dataset txt into a smaller sample txt.

Example:
    python whole_body_tracking/scripts/sample_from_hard.py --num 100
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset_txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomly sample motion entries from hard.txt.")
    parser.add_argument(
        "--input_txt",
        type=Path,
        default=DEFAULT_DATASET_DIR / "hard.txt",
        help="Source txt file, one motion path per line.",
    )
    parser.add_argument(
        "--output_txt",
        type=Path,
        default=DEFAULT_DATASET_DIR / "sample.txt",
        help="Output txt file for sampled motion paths.",
    )
    parser.add_argument("--num", type=int, default=100, help="Number of lines to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--sort", action="store_true", help="Sort sampled lines before writing.")
    return parser.parse_args()


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    input_txt = args.input_txt.expanduser().resolve()
    output_txt = args.output_txt.expanduser().resolve()

    if not input_txt.is_file():
        raise FileNotFoundError(f"input_txt does not exist: {input_txt}")
    if args.num <= 0:
        raise ValueError("--num must be greater than 0")

    lines = read_lines(input_txt)
    if not lines:
        raise ValueError(f"input_txt has no non-empty lines: {input_txt}")

    rng = random.Random(args.seed)
    sampled = rng.sample(lines, min(args.num, len(lines)))
    if args.sort:
        sampled = sorted(sampled)

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n".join(sampled) + "\n")

    print(f"[INFO] input_txt: {input_txt}")
    print(f"[INFO] output_txt: {output_txt}")
    print(f"[INFO] total lines: {len(lines)}, sampled: {len(sampled)}, seed: {args.seed}")


if __name__ == "__main__":
    main()
