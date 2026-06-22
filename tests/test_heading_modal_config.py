from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
G1_CONFIG_PATH = ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_heading_walk_amp_modal_task_is_registered() -> None:
    source = _read(G1_CONFIG_PATH / "__init__.py")

    assert 'id="HeadingWalkAMP-Modal-G1"' in source
    assert "velocity_amp_env_cfg.G1VelocityFlatAMPModalEnvCfg" in source
    assert "G1VelocityFlatAMPModalRunnerCfg" in source


def test_heading_walk_amp_modal_env_uses_local_flat_velocity_mdp_plus_amp() -> None:
    source = _read(G1_CONFIG_PATH / "velocity_amp_env_cfg.py")
    tree = ast.parse(source)
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "G1VelocityFlatAMPModalEnvCfg"
    )

    assert isinstance(class_node.bases[0], ast.Name)
    assert class_node.bases[0].id == "ManagerBasedRLEnvCfg"
    assert "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg" not in source
    assert "from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG" in source
    assert "self.scene.robot = G1_CYLINDER_CFG.replace" in source
    assert "self.actions.joint_pos.scale = G1_ACTION_SCALE" in source
    assert 'self.scene.terrain.terrain_type = "plane"' in source
    assert "self.scene.height_scanner = None" in source
    assert "track_lin_vel_xy_yaw_frame_exp" in source
    assert "track_ang_vel_z_world_exp" in source
    assert "tracking_mdp.root_height_recovery_exp" in source
    assert '"active_only_on_delayed_termination": False' in source
    assert "tracking_mdp.body_orientation_l2" in source
    assert 'body_names=["torso_link"]' in source
    assert "tracking_mdp.variable_posture" in source
    assert '"std_standing": {".*": 0.05}' in source
    assert '"walking_threshold": 0.1' in source
    assert '"running_threshold": 1.5' in source
    assert "prop: PropCfg = PropCfg()" in source
    assert "base_velocity_cmd: BaseVelocityCommandCfg = BaseVelocityCommandCfg()" in source
    assert "self.observations.amp = TrackingObservationsCfg.AmpCfg()" in source


def test_heading_walk_amp_modal_runner_uses_lafan_walk_amp_data() -> None:
    source = _read(G1_CONFIG_PATH / "agents/rsl_rl_velocity_amp_cfg.py")

    assert '"init_std": 1.0' in source
    assert "hidden_dims: list = [1024, 512]" in source
    assert (
        'G1_LAFAN_WALK_AMP_MOTION_DIR = "/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/data/g1_amp/LAFAN_WALK"'
        in source
    )
    assert '"amp_motion_files": G1_LAFAN_WALK_AMP_MOTION_DIR' in source
    assert '"amp_discr_hidden_dims": [1024, 512]' in source
    assert '"amp_body_names": G1_AMP_BODY_NAMES' in source


def test_heading_walk_amp_modal_env_cfg_instantiates_local_project_g1_velocity_amp() -> None:
    import sys

    pytest.importorskip("toml")

    source_root = str(ROOT / "source/whole_body_tracking")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

    from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
    from whole_body_tracking.tasks.tracking.config.g1.velocity_amp_env_cfg import (
        G1_AMP_BODY_NAMES,
        G1VelocityFlatAMPModalEnvCfg,
    )

    cfg = G1VelocityFlatAMPModalEnvCfg()

    assert cfg.scene.robot.spawn.asset_path == G1_CYLINDER_CFG.spawn.asset_path
    assert cfg.actions.joint_pos.scale == G1_ACTION_SCALE
    assert cfg.scene.terrain.terrain_type == "plane"
    assert cfg.scene.terrain.terrain_generator is None
    assert cfg.scene.height_scanner is None
    assert cfg.curriculum.terrain_levels is None
    assert cfg.commands.base_velocity.ranges.lin_vel_x == (0.0, 1.0)
    assert cfg.commands.base_velocity.ranges.lin_vel_y == (-0.5, 0.5)
    assert cfg.rewards.track_ang_vel_z_exp.weight == 1.0
    assert cfg.rewards.track_root_height.weight == 1.0
    assert cfg.rewards.track_root_height.params["std"] == 0.3
    assert cfg.rewards.track_root_height.params["active_only_on_delayed_termination"] is False
    assert cfg.rewards.body_orientation_l2.weight == -1.0
    assert cfg.rewards.body_orientation_l2.params["asset_cfg"].body_names == ["torso_link"]
    assert cfg.rewards.pose.weight == 1.0
    assert cfg.rewards.pose.params["command_name"] == "base_velocity"
    assert cfg.rewards.pose.params["std_standing"] == {".*": 0.05}
    assert cfg.rewards.pose.params["std_walking"][r".*knee.*"] == 0.5
    assert cfg.rewards.pose.params["std_running"][r".*hip_roll.*"] == 0.25
    assert cfg.rewards.pose.params["walking_threshold"] == 0.1
    assert cfg.rewards.pose.params["running_threshold"] == 1.5
    assert cfg.rewards.feet_air_time.weight == 0.75
    assert cfg.observations.prop.velocity_commands is None
    assert cfg.observations.prop.height_scan is None
    assert cfg.observations.amp.body_pos_b.params["anchor_body_name"] == "torso_link"
    assert cfg.observations.amp.body_pos_b.params["body_names"] == tuple(G1_AMP_BODY_NAMES)
