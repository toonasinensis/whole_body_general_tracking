from __future__ import annotations

from types import SimpleNamespace

from feat_motion_lib.facade.config_adapter import cfg_to_unified_load_config


def test_cfg_to_unified_load_config_maps_fields() -> None:
    cfg = SimpleNamespace(
        motion_file="/tmp/motions",
        max_motion_num=8,
        dataset_txt="/tmp/dataset.txt",
        eval_mode=True,
        distributed=True,
        local_rank=1,
        total_rank=4,
        smpl_file_path="/tmp/smpl",
        target_fps=60.0,
        up_axis="zup",
        max_frame_diff=5,
    )

    load_config = cfg_to_unified_load_config(cfg, sample_counter=3, device="cuda:0")

    assert str(load_config.discovery.motion_dir) == "/tmp/motions"
    assert str(load_config.discovery.dataset_txt) == "/tmp/dataset.txt"
    assert load_config.discovery.max_motion_num == 8
    assert load_config.discovery.eval_mode is True
    assert load_config.discovery.distributed is True
    assert load_config.discovery.local_rank == 1
    assert load_config.discovery.total_rank == 4
    assert load_config.discovery.sample_counter == 3
    assert load_config.load.target_fps == 60.0
    assert load_config.load.device == "cuda:0"
    assert load_config.load.up_axis == "zup"
    assert str(load_config.smpl_dir) == "/tmp/smpl"
    assert load_config.max_frame_diff == 5


def test_cfg_to_unified_load_config_uses_defaults() -> None:
    cfg = SimpleNamespace(motion_file="/tmp/motions")

    load_config = cfg_to_unified_load_config(cfg, sample_counter=0, device="cpu")

    assert load_config.discovery.max_motion_num == -1
    assert load_config.discovery.dataset_txt is None
    assert load_config.load.target_fps == 50.0
    assert load_config.load.up_axis == "yup"
    assert load_config.smpl_dir is None
    assert load_config.max_frame_diff == 2
