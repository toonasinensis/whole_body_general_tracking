from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DiscoveryOptions:
    motion_dir: Path
    max_motion_num: int = -1
    dataset_txt: Path | None = None
    eval_mode: bool = False
    distributed: bool = False
    local_rank: int = 0
    total_rank: int = 1
    sample_counter: int = 0


@dataclass
class LoadOptions:
    target_fps: float = 50.0
    device: str = "cpu"
    up_axis: str = "yup"


@dataclass
class SmplLoadConfig:
    motion_files: list[Path]
    load: LoadOptions


@dataclass
class UnifiedLoadConfig:
    discovery: DiscoveryOptions
    load: LoadOptions
    smpl_dir: Path | None = None
    max_frame_diff: int = 2
