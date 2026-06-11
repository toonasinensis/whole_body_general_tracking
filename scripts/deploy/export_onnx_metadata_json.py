#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def load_onnx_metadata(onnx_path: Path) -> dict:
    try:
        import onnx
    except ImportError as exc:
        raise ImportError("The 'onnx' package is required to read ONNX metadata.") from exc

    model = onnx.load(str(onnx_path))
    metadata = {}

    for entry in model.metadata_props:
        try:
            metadata[entry.key] = json.loads(entry.value)
        except json.JSONDecodeError:
            metadata[entry.key] = entry.value

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ONNX metadata_props as a JSON file.")
    parser.add_argument("onnx_path", type=Path, help="Path to the input ONNX model.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to the ONNX path with a .json suffix.",
    )
    args = parser.parse_args()

    onnx_path = args.onnx_path.expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    output_path = args.output.expanduser().resolve() if args.output else onnx_path.with_suffix(".json")
    metadata = load_onnx_metadata(onnx_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Exported {len(metadata)} metadata fields to: {output_path}")


if __name__ == "__main__":
    main()
