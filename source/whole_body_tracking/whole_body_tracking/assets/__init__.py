"""Package containing paths to local asset files."""

from __future__ import annotations

from pathlib import Path

# Absolute path to this package's directory.
ASSET_DIR: Path = Path(__file__).resolve().parent

__all__ = ["ASSET_DIR"]
