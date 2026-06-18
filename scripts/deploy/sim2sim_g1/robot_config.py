from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"


@dataclass(frozen=True)
class RobotConfig:
    name: str
    xml_path: Path
    imu_quat_sensor: str
    imu_gyro_sensor: str
    actuator_name_pattern: str
    default_root_height: float
    legacy_motion_body_names: tuple[str, ...] = ()
    reference_geom_groups: tuple[int, ...] = (2,)
    joint_armature: dict[str, float] | None = None
    joint_velocity_limit: dict[str, float] | None = None
    joint_static_friction: dict[str, float] | None = None
    passive_joint_damping: float | None = None

    def actuator_name(self, joint_name: str) -> str:
        return self.actuator_name_pattern.format(joint_name=joint_name)


def available_robot_configs() -> list[str]:
    return sorted(path.stem for path in CONFIG_DIR.glob("*.json"))


def load_robot_config(name_or_path: str) -> RobotConfig:
    path = Path(name_or_path).expanduser()
    if not path.is_file():
        path = CONFIG_DIR / f"{name_or_path}.json"
    if not path.is_file():
        available = ", ".join(available_robot_configs())
        raise FileNotFoundError(f"Robot config not found: {name_or_path!r}. Available presets: {available}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    xml_path = Path(raw["xml_path"]).expanduser()
    if not xml_path.is_absolute():
        xml_path = REPO_ROOT / xml_path
    return RobotConfig(
        name=str(raw["name"]),
        xml_path=xml_path.resolve(),
        imu_quat_sensor=str(raw["imu_quat_sensor"]),
        imu_gyro_sensor=str(raw["imu_gyro_sensor"]),
        actuator_name_pattern=str(raw.get("actuator_name_pattern", "{joint_name}")),
        default_root_height=float(raw["default_root_height"]),
        legacy_motion_body_names=tuple(str(name) for name in raw.get("legacy_motion_body_names", [])),
        reference_geom_groups=tuple(int(group) for group in raw.get("reference_geom_groups", [2])),
        joint_armature={str(name): float(value) for name, value in raw.get("joint_armature", {}).items()},
        joint_velocity_limit={
            str(name): float(value) for name, value in raw.get("joint_velocity_limit", {}).items()
        },
        joint_static_friction={
            str(name): float(value) for name, value in raw.get("joint_static_friction", {}).items()
        },
        passive_joint_damping=(
            float(raw["passive_joint_damping"]) if "passive_joint_damping" in raw else None
        ),
    )
