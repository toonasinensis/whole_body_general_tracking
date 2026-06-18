from pathlib import Path

from sim2sim_g1.robot_config import available_robot_configs, load_robot_config


def test_builtin_robot_configs_resolve_existing_mjcf() -> None:
    assert {"g1", "roban"}.issubset(available_robot_configs())
    for name in ("g1", "roban"):
        config = load_robot_config(name)
        assert config.name == name
        assert config.xml_path.is_file()
    assert len(load_robot_config("roban").legacy_motion_body_names) == 28


def test_custom_robot_config_supports_actuator_pattern(tmp_path: Path) -> None:
    config_path = tmp_path / "robot.json"
    config_path.write_text(
        """
{
  "name": "custom",
  "xml_path": "source/whole_body_tracking/whole_body_tracking/assets/roban_s22/xml/scene.xml",
  "imu_quat_sensor": "q",
  "imu_gyro_sensor": "g",
  "actuator_name_pattern": "motor_{joint_name}",
  "default_root_height": 1.0
}
""".strip()
    )
    config = load_robot_config(str(config_path))
    assert config.actuator_name("hip") == "motor_hip"
    assert config.default_root_height == 1.0
