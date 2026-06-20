from __future__ import annotations

from pathlib import Path

from ..config import DiscoveryOptions, LoadOptions, SmplLoadConfig, UnifiedLoadConfig
from ..io import discover_pkl_files


"""
将传递的 cfg 转化为 config.py 中定义的标准 cfg
"""

def cfg_to_unified_load_config(
    cfg,
    *,
    sample_counter: int,
    device: str,
    default_target_fps: float = 50.0,
    default_up_axis: str = "yup",
    default_max_frame_diff: int = 2,
) -> UnifiedLoadConfig:
    dataset_txt = getattr(cfg, "dataset_txt", None)
    smpl_dir = getattr(cfg, "smpl_file_path", None)
    max_frame_diff = int(getattr(cfg, "max_frame_diff", default_max_frame_diff))

    discovery = DiscoveryOptions(
        motion_dir=Path(cfg.motion_file),
        max_motion_num=int(getattr(cfg, "max_motion_num", -1)),
        dataset_txt=Path(dataset_txt) if dataset_txt else None,
        eval_mode=bool(getattr(cfg, "eval_mode", False)),
        distributed=bool(getattr(cfg, "distributed", False)),
        local_rank=int(getattr(cfg, "local_rank", 0)),
        total_rank=int(getattr(cfg, "total_rank", 1)),
        sample_counter=sample_counter,
    )
    load = LoadOptions(
        target_fps=float(getattr(cfg, "target_fps", default_target_fps)),
        device=device,
        up_axis=str(getattr(cfg, "up_axis", default_up_axis)),
    )
    return UnifiedLoadConfig(
        discovery=discovery,
        load=load,
        smpl_dir=Path(smpl_dir) if smpl_dir else None,
        max_frame_diff=max_frame_diff,
    )


def cfg_to_smpl_load_config(
    cfg,
    *,
    device: str,
    default_target_fps: float = 50.0,
    default_up_axis: str = "yup",
) -> SmplLoadConfig:
    motion_files = getattr(cfg, "motion_files", None)
    if motion_files is not None:
        files = [Path(path) for path in motion_files]
    else:
        input_path = getattr(cfg, "motion_file", None)
        if input_path is None:
            raise ValueError("cfg must provide 'motion_files' or 'motion_file' for SmplMotionLib.")
        files = [Path(path) for path in discover_pkl_files(input_path)]

    load = LoadOptions(
        target_fps=float(getattr(cfg, "target_fps", default_target_fps)),
        device=device,
        up_axis=str(getattr(cfg, "up_axis", default_up_axis)),
    )
    return SmplLoadConfig(motion_files=files, load=load)
