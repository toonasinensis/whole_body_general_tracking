from __future__ import annotations

import importlib.util
import json
import sys
import torch
import types
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TERRAINS_ROOT = ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "terrains"

sys.modules.setdefault("whole_body_tracking", types.ModuleType("whole_body_tracking"))
sys.modules.setdefault("whole_body_tracking.terrains", types.ModuleType("whole_body_tracking.terrains"))

PM_SPEC = importlib.util.spec_from_file_location(
    "whole_body_tracking.terrains.paired_manifest", TERRAINS_ROOT / "paired_manifest.py"
)
paired_manifest = importlib.util.module_from_spec(PM_SPEC)
assert PM_SPEC is not None and PM_SPEC.loader is not None
sys.modules["whole_body_tracking.terrains.paired_manifest"] = paired_manifest
PM_SPEC.loader.exec_module(paired_manifest)

SPEC = importlib.util.spec_from_file_location(
    "whole_body_tracking.terrains.mixed_domains", TERRAINS_ROOT / "mixed_domains.py"
)
mixed_domains = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules["whole_body_tracking.terrains.mixed_domains"] = mixed_domains
SPEC.loader.exec_module(mixed_domains)

allocate_domain_env_counts = mixed_domains.allocate_domain_env_counts
domain_env_starts = mixed_domains.domain_env_starts
build_mixed_domain_layout = mixed_domains.build_mixed_domain_layout
build_mixed_terrain_cfg = mixed_domains.build_mixed_terrain_cfg


@dataclass
class FlatCfg:
    name: str = "flat_lafan"
    env_ratio: float = 0.25
    pairing: str = "unpaired"
    terrain_kind: str = "flat"
    motion_file: str = ""
    dataset_txt: str = ""
    max_motion_num: int = -1
    terrain_cell_count: int = 2


@dataclass
class MeshCfg:
    name: str = "mesh_pairs"
    env_ratio: float = 0.75
    pairing: str = "strict_pair"
    terrain_kind: str = "stl_grid"
    pairs_jsonl: str = ""
    pair_limit: int | None = None


@dataclass
class HfCfg:
    name: str = "velocity_terrain"
    env_ratio: float = 0.20
    pairing: str = "unpaired"
    terrain_kind: str = "hf_grid"
    motion_file: str = ""
    dataset_txt: str = ""
    max_motion_num: int = -1
    terrain_cell_count: int = 4
    terrain_profile: str = "velocity_runway_steps"
    task_profile: str = "velocity_terrain"


def _write_flat_dataset(tmp_path: Path, names: list[str]) -> tuple[Path, Path]:
    data = tmp_path / "data"
    motion_dir = data / "flat"
    motion_dir.mkdir(parents=True)
    for name in names:
        (motion_dir / f"{name}.npz").write_bytes(b"npz")
    dataset = data / "flat.txt"
    dataset.write_text("\n".join(f"flat/{name}.npz" for name in names) + "\n", encoding="utf-8")
    return data, dataset


def _write_pairs(tmp_path: Path, names: list[str]) -> Path:
    root = tmp_path / "pairs"
    motion_dir = root / "motions"
    terrain_dir = root / "terrains"
    motion_dir.mkdir(parents=True)
    terrain_dir.mkdir(parents=True)
    lines = []
    for name in names:
        (motion_dir / f"{name}.npz").write_bytes(b"npz")
        (terrain_dir / f"{name}.stl").write_bytes(b"solid terrain\nendsolid terrain\n")
        lines.append(
            json.dumps(
                {
                    "name": name,
                    "tracking_motion": f"motions/{name}.npz",
                    "terrain_stl": f"terrains/{name}.stl",
                }
            )
        )
    manifest = root / "pairs.jsonl"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def test_allocate_domain_env_counts_uses_ratios() -> None:
    assert allocate_domain_env_counts(32, (0.25, 0.75)) == (8, 24)
    assert allocate_domain_env_counts(10, (1.0, 1.0, 2.0)) == (3, 2, 5)


def test_mesh_domain_local_indices_start_after_flat_envs() -> None:
    counts = allocate_domain_env_counts(16, (0.25, 0.75))
    starts = domain_env_starts(counts)

    assert counts == (4, 12)
    assert starts == (0, 4)
    mesh_pair_count = 8
    mesh_env_ids = list(range(starts[1], starts[1] + mesh_pair_count))
    mesh_local_pair_ids = [(env_id - starts[1]) % mesh_pair_count for env_id in mesh_env_ids]

    assert mesh_local_pair_ids == list(range(mesh_pair_count))


def test_mixed_layout_ranges_do_not_overlap(tmp_path: Path) -> None:
    flat_root, flat_dataset = _write_flat_dataset(tmp_path, ["walk", "run"])
    pairs = _write_pairs(tmp_path, ["climb_a", "climb_b", "climb_c"])

    flat_cfg = FlatCfg(motion_file=str(flat_root), dataset_txt=str(flat_dataset), terrain_cell_count=2)
    mesh_cfg = MeshCfg(pairs_jsonl=str(pairs))
    layout = build_mixed_domain_layout([flat_cfg, mesh_cfg])

    flat, mesh = layout.domains
    assert flat.motion_start == 0
    assert flat.motion_count == 2
    assert mesh.motion_start == 2
    assert mesh.motion_count == 3
    assert flat.terrain_start == 0
    assert flat.terrain_count == 2
    assert mesh.terrain_start == 2
    assert mesh.terrain_count == 3
    assert set(range(flat.motion_start, flat.motion_start + flat.motion_count)).isdisjoint(
        range(mesh.motion_start, mesh.motion_start + mesh.motion_count)
    )
    assert set(range(flat.terrain_start, flat.terrain_start + flat.terrain_count)).isdisjoint(
        range(mesh.terrain_start, mesh.terrain_start + mesh.terrain_count)
    )


def test_mixed_layout_supports_flat_wbc_flat_velocity_and_mesh(tmp_path: Path) -> None:
    flat_root, flat_dataset = _write_flat_dataset(tmp_path, ["walk", "run"])
    pairs = _write_pairs(tmp_path, ["climb_a", "climb_b", "climb_c"])

    flat_wbc = FlatCfg(name="flat_wbc", env_ratio=0.25, motion_file=str(flat_root), dataset_txt=str(flat_dataset))
    flat_velocity = FlatCfg(
        name="flat_velocity",
        env_ratio=0.25,
        motion_file=str(flat_root),
        dataset_txt=str(flat_dataset),
    )
    mesh = MeshCfg(name="omniretarget_g1_terrain", env_ratio=0.50, pairs_jsonl=str(pairs))

    layout = build_mixed_domain_layout([flat_wbc, flat_velocity, mesh])

    assert tuple(domain.name for domain in layout.domains) == (
        "flat_wbc",
        "flat_velocity",
        "omniretarget_g1_terrain",
    )
    assert layout.domains[0].terrain_start == 0
    assert layout.domains[1].terrain_start == layout.domains[0].terrain_start + layout.domains[0].terrain_count
    assert layout.domains[2].terrain_start == layout.domains[1].terrain_start + layout.domains[1].terrain_count
    assert layout.domains[0].motion_start == 0
    assert layout.domains[1].motion_start == layout.domains[0].motion_start + layout.domains[0].motion_count
    assert layout.domains[2].motion_start == layout.domains[1].motion_start + layout.domains[1].motion_count


def test_mixed_layout_supports_velocity_hf_grid_domain(tmp_path: Path) -> None:
    flat_root, flat_dataset = _write_flat_dataset(tmp_path, ["walk", "run"])

    hf_cfg = HfCfg(
        motion_file=str(flat_root),
        dataset_txt=str(flat_dataset),
        terrain_cell_count=3,
        terrain_profile="large_step_platform",
    )
    layout = build_mixed_domain_layout([hf_cfg])

    domain = layout.domains[0]
    assert domain.name == "velocity_terrain"
    assert domain.terrain_kind == "hf_grid"
    assert domain.task_profile == "velocity_terrain"
    assert domain.motion_count == 2
    assert domain.terrain_count == 3
    assert layout.terrain_names == (
        "velocity_terrain_large_step_platform_000",
        "velocity_terrain_large_step_platform_001",
        "velocity_terrain_large_step_platform_002",
    )


def test_mixed_layout_can_insert_ground_separator_cells(tmp_path: Path) -> None:
    flat_root, flat_dataset = _write_flat_dataset(tmp_path, ["walk", "run"])
    pairs = _write_pairs(tmp_path, ["climb_a", "climb_b", "climb_c"])

    flat_cfg = FlatCfg(motion_file=str(flat_root), dataset_txt=str(flat_dataset), terrain_cell_count=2)
    mesh_cfg = MeshCfg(pairs_jsonl=str(pairs))
    layout = build_mixed_domain_layout(
        [flat_cfg, mesh_cfg],
        domain_separator_cell_count=4,
        pad_to_grid=True,
    )

    flat, mesh = layout.domains
    assert flat.terrain_start == 0
    assert flat.terrain_count == 2
    assert layout.terrain_names[2:6] == tuple(f"flat_lafan_separator_ground_{index:03d}" for index in range(4))
    assert mesh.terrain_start == 6
    assert mesh.terrain_count == 3
    assert layout.terrain_names[mesh.terrain_start : mesh.terrain_start + mesh.terrain_count] == (
        "climb_a",
        "climb_b",
        "climb_c",
    )
    assert layout.terrain_count == layout.num_rows * layout.num_cols


def test_strict_pair_local_terrain_index_matches_motion_index(tmp_path: Path) -> None:
    flat_root, flat_dataset = _write_flat_dataset(tmp_path, ["walk"])
    pairs = _write_pairs(tmp_path, ["climb_a", "climb_b"])

    layout = build_mixed_domain_layout(
        [
            FlatCfg(motion_file=str(flat_root), dataset_txt=str(flat_dataset), terrain_cell_count=1),
            MeshCfg(pairs_jsonl=str(pairs)),
        ]
    )

    mesh = layout.domains[1]
    for local_index, name in enumerate(("climb_a", "climb_b")):
        motion_id = mesh.motion_start + local_index
        terrain_id = mesh.terrain_start + local_index
        assert Path(layout.motion_paths[motion_id]).stem == name
        assert layout.terrain_names[terrain_id] == name


def test_domain_motion_command_splits_velocity_metrics_by_domain() -> None:
    domain_spec = importlib.util.spec_from_file_location(
        "whole_body_tracking.tasks.tracking.mdp.domain_commands",
        ROOT
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "mdp"
        / "domain_commands.py",
    )
    domain_module = importlib.util.module_from_spec(domain_spec)
    assert domain_spec is not None and domain_spec.loader is not None

    managers = types.ModuleType("isaaclab.managers")
    managers.EventTermCfg = type("EventTermCfg", (), {})
    managers.SceneEntityCfg = type("SceneEntityCfg", (), {})
    utils = types.ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    commands_module = types.ModuleType("whole_body_tracking.tasks.tracking.mdp.commands")
    commands_module.MotionCommand = object

    @dataclass
    class MotionCommandCfg:
        pass

    commands_module.MotionCommandCfg = MotionCommandCfg
    sys.modules["isaaclab.managers"] = managers
    sys.modules["isaaclab.utils"] = utils
    sys.modules["whole_body_tracking.tasks.tracking.mdp.commands"] = commands_module
    sys.modules["whole_body_tracking.tasks.tracking.mdp.domain_commands"] = domain_module
    domain_spec.loader.exec_module(domain_module)

    command = domain_module.DomainMotionCommand.__new__(domain_module.DomainMotionCommand)
    command.num_envs = 5
    command.device = torch.device("cpu")
    command.metrics = {}
    command.env_domain_ids = torch.tensor([0, 0, 1, 1, 2])
    command.domain_layout = types.SimpleNamespace(
        domains=(
            types.SimpleNamespace(name="flat_velocity", task_profile="velocity_flat"),
            types.SimpleNamespace(name="velocity_terrain", task_profile="velocity_terrain"),
            types.SimpleNamespace(name="mesh", task_profile="wbc_tracking"),
        )
    )
    command._env = types.SimpleNamespace(
        command_manager=types.SimpleNamespace(
            get_term=lambda name: types.SimpleNamespace(
                metrics={
                    "error_vel_xy": torch.tensor([1.0, 3.0, 10.0, 14.0, 100.0]),
                    "error_vel_yaw": torch.tensor([2.0, 4.0, 20.0, 24.0, 200.0]),
                }
            )
        )
    )

    command._update_velocity_domain_metrics()

    assert torch.allclose(command.metrics["error_vel_xy_flat_velocity"], torch.full((5,), 2.0))
    assert torch.allclose(command.metrics["error_vel_xy_velocity_terrain"], torch.full((5,), 12.0))
    assert "error_vel_xy_mesh" not in command.metrics


def test_velocity_hf_terrain_functions_scale_with_difficulty() -> None:
    hf_spec = importlib.util.spec_from_file_location(
        "whole_body_tracking.terrains.height_field.hf_terrains",
        TERRAINS_ROOT / "height_field" / "hf_terrains.py",
    )
    hf_module = importlib.util.module_from_spec(hf_spec)
    assert hf_spec is not None and hf_spec.loader is not None
    package = types.ModuleType("whole_body_tracking.terrains.height_field")
    package.__path__ = [str(TERRAINS_ROOT / "height_field")]
    sys.modules["whole_body_tracking.terrains.height_field"] = package
    sys.modules["whole_body_tracking.terrains.height_field.hf_terrains"] = hf_module
    hf_spec.loader.exec_module(hf_module)

    cfg = types.SimpleNamespace(
        size=(4.0, 2.0),
        horizontal_scale=0.5,
        vertical_scale=0.05,
        step_height_range=(0.05, 0.25),
        step_depth=1.0,
        start_platform_length=1.0,
        max_steps=2,
    )
    low = hf_module.large_step_platform_terrain.__wrapped__(0.0, cfg)
    high = hf_module.large_step_platform_terrain.__wrapped__(1.0, cfg)

    assert low.shape == (8, 4)
    assert high.shape == (8, 4)
    assert low.max() == 2
    assert high.max() == 10
    assert high.max() > low.max()

    runway_cfg = types.SimpleNamespace(
        size=(4.0, 2.0),
        horizontal_scale=0.5,
        vertical_scale=0.05,
        shoulder_width=0.5,
        shoulder_height_range=(0.0, 0.10),
    )
    runway = hf_module.long_runway_terrain.__wrapped__(1.0, runway_cfg)
    assert runway.shape == (8, 4)
    assert runway[:, 0].max() == 2
    assert runway[:, 1:3].max() == 0


def test_build_mixed_terrain_cfg_can_mix_flat_hf_and_stl(tmp_path: Path, monkeypatch) -> None:
    flat_root, flat_dataset = _write_flat_dataset(tmp_path, ["walk"])
    pairs = _write_pairs(tmp_path, ["climb_a"])

    @dataclass
    class DummyTrimeshPlatformCfg:
        proportion: float
        stl_file_path: str | None = None
        add_ground_plane: bool = False
        ground_size: tuple[float, float] | None = None
        ground_thickness: float = 0.01
        ground_z: float = 0.0

    @dataclass
    class DummyHfLongRunwayTerrainCfg:
        proportion: float
        shoulder_width: float = 0.0
        shoulder_height_range: tuple[float, float] = (0.0, 0.0)

    @dataclass
    class DummyHfLargeStepPlatformTerrainCfg:
        proportion: float
        step_height_range: tuple[float, float]
        step_depth: float
        start_platform_length: float
        max_steps: int

    @dataclass
    class DummySTLTerrainGeneratorCfg:
        size: tuple[float, float]
        border_width: float
        num_rows: int
        num_cols: int
        horizontal_scale: float
        vertical_scale: float
        slope_threshold: float
        use_cache: bool
        sub_terrains: dict

    stl_cfg_module = types.ModuleType("whole_body_tracking.terrains.stl_trimesh_cfg")
    stl_cfg_module.TrimeshPlatformCfg = DummyTrimeshPlatformCfg
    generator_cfg_module = types.ModuleType("whole_body_tracking.terrains.stl_terrain_generator_cfg")
    generator_cfg_module.STLTerrainGeneratorCfg = DummySTLTerrainGeneratorCfg
    height_field_module = types.ModuleType("whole_body_tracking.terrains.height_field")
    height_field_module.HfLongRunwayTerrainCfg = DummyHfLongRunwayTerrainCfg
    height_field_module.HfLargeStepPlatformTerrainCfg = DummyHfLargeStepPlatformTerrainCfg
    monkeypatch.setitem(sys.modules, "whole_body_tracking.terrains.stl_trimesh_cfg", stl_cfg_module)
    monkeypatch.setitem(sys.modules, "whole_body_tracking.terrains.stl_terrain_generator_cfg", generator_cfg_module)
    monkeypatch.setitem(sys.modules, "whole_body_tracking.terrains.height_field", height_field_module)

    terrain_cfg, layout = build_mixed_terrain_cfg(
        [
            FlatCfg(
                name="flat_velocity", motion_file=str(flat_root), dataset_txt=str(flat_dataset), terrain_cell_count=1
            ),
            HfCfg(motion_file=str(flat_root), dataset_txt=str(flat_dataset), terrain_cell_count=1),
            MeshCfg(pairs_jsonl=str(pairs)),
        ],
        terrain_size=(24.0, 8.0),
    )

    assert layout.terrain_count == 4
    assert terrain_cfg.size == (24.0, 8.0)
    assert any(isinstance(cfg, DummyTrimeshPlatformCfg) for cfg in terrain_cfg.sub_terrains.values())
    assert any(isinstance(cfg, DummyHfLargeStepPlatformTerrainCfg) for cfg in terrain_cfg.sub_terrains.values())


def _load_mixed_domain_profiles(monkeypatch):
    from dataclasses import dataclass, field

    @dataclass
    class SceneEntityCfg:
        name: str
        joint_names: object = None
        body_names: object = None
        preserve_order: bool = False

    @dataclass
    class CommandTermCfg:
        params: dict = field(default_factory=dict)

    @dataclass
    class EventTermCfg:
        func: object
        mode: str = "reset"
        interval_range_s: tuple[float, float] | None = None
        params: dict = field(default_factory=dict)

    @dataclass
    class RewardTermCfg:
        func: object
        weight: float = 0.0
        params: dict = field(default_factory=dict)

    @dataclass
    class TerminationTermCfg:
        func: object
        params: dict = field(default_factory=dict)
        time_out: bool = False

    @dataclass
    class UniformVelocityCommandCfg(CommandTermCfg):
        @dataclass
        class Ranges:
            lin_vel_x: tuple[float, float] = (0.0, 0.0)
            lin_vel_y: tuple[float, float] = (0.0, 0.0)
            ang_vel_z: tuple[float, float] = (0.0, 0.0)
            heading: tuple[float, float] = (0.0, 0.0)

        asset_name: str = "robot"
        resampling_time_range: tuple[float, float] = (0.0, 0.0)
        rel_standing_envs: float = 0.0
        rel_heading_envs: float = 0.0
        heading_command: bool = False
        heading_control_stiffness: float = 0.0
        debug_vis: bool = False
        ranges: Ranges = field(default_factory=Ranges)

    def fake_func(*args, **kwargs):
        return None

    class variable_posture:
        pass

    managers = types.ModuleType("isaaclab.managers")
    managers.CommandTermCfg = CommandTermCfg
    managers.EventTermCfg = EventTermCfg
    managers.RewardTermCfg = RewardTermCfg
    managers.SceneEntityCfg = SceneEntityCfg
    managers.TerminationTermCfg = TerminationTermCfg

    utils = types.ModuleType("isaaclab.utils")
    utils.configclass = dataclass

    velocity_mdp = types.ModuleType("isaaclab_tasks.manager_based.locomotion.velocity.mdp")
    velocity_mdp.UniformVelocityCommandCfg = UniformVelocityCommandCfg
    for name in (
        "time_out",
        "illegal_contact",
        "is_terminated",
        "reset_root_state_uniform",
        "reset_joints_by_scale",
        "push_by_setting_velocity",
        "track_lin_vel_xy_yaw_frame_exp",
        "track_ang_vel_z_world_exp",
        "lin_vel_z_l2",
        "ang_vel_xy_l2",
        "joint_torques_l2",
        "joint_acc_l2",
        "action_rate_l2",
        "feet_air_time_positive_biped",
        "feet_slide",
        "joint_pos_limits",
        "joint_deviation_l1",
    ):
        setattr(velocity_mdp, name, fake_func)

    mdp = types.ModuleType("whole_body_tracking.tasks.tracking.mdp")
    for name in (
        "motion_global_anchor_position_error_exp",
        "motion_global_anchor_position_z_error_exp",
        "motion_global_anchor_orientation_error_exp",
        "motion_relative_body_position_error_exp",
        "motion_relative_body_orientation_error_exp",
        "motion_global_body_linear_velocity_error_exp",
        "motion_global_body_angular_velocity_error_exp",
        "action_rate_l2",
        "joint_pos_limits",
        "undesired_contacts",
        "feet_stumble",
        "feet_flat",
        "time_out",
        "bad_anchor_pos_z_only",
        "bad_anchor_pos_xyz",
        "bad_anchor_ori",
        "bad_motion_body_pos_z_only",
        "parkour_reach_timeout",
        "root_height_recovery_exp",
        "root_height_below_desired",
        "body_orientation_l2",
        "domain_masked_reward",
        "domain_masked_termination",
        "domain_masked_event",
    ):
        setattr(mdp, name, fake_func)
    mdp.variable_posture = variable_posture

    tracking = types.ModuleType("whole_body_tracking.tasks.tracking")
    tracking.__path__ = []
    tracking.mdp = mdp

    for module_name in (
        "isaaclab",
        "isaaclab_tasks",
        "isaaclab_tasks.manager_based",
        "isaaclab_tasks.manager_based.locomotion",
        "isaaclab_tasks.manager_based.locomotion.velocity",
        "whole_body_tracking.tasks",
        "whole_body_tracking.tasks.tracking.config",
    ):
        package = types.ModuleType(module_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, module_name, package)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", utils)
    monkeypatch.setitem(sys.modules, "isaaclab_tasks.manager_based.locomotion.velocity.mdp", velocity_mdp)
    monkeypatch.setitem(sys.modules, "whole_body_tracking.tasks.tracking", tracking)
    monkeypatch.setitem(sys.modules, "whole_body_tracking.tasks.tracking.mdp", mdp)

    applier_name = "whole_body_tracking.tasks.tracking.config.domain_profile_applier"
    monkeypatch.delitem(sys.modules, applier_name, raising=False)
    applier_spec = importlib.util.spec_from_file_location(
        applier_name,
        ROOT
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "config"
        / "domain_profile_applier.py",
    )
    applier = importlib.util.module_from_spec(applier_spec)
    assert applier_spec is not None and applier_spec.loader is not None
    sys.modules[applier_name] = applier
    applier_spec.loader.exec_module(applier)

    module_name = "_mixed_domain_profiles_under_test"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "config"
        / "g1"
        / "mixed_domain_profiles.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    profile_api = types.SimpleNamespace(
        build_default_mixed_profiles=module.build_default_mixed_profiles,
        apply_domain_profiles=applier.apply_domain_profiles,
        collect_domain_profile_routes=applier.collect_domain_profile_routes,
    )
    return profile_api, managers


def test_profile_applier_registers_prefixed_masked_terms(monkeypatch) -> None:
    profiles, managers = _load_mixed_domain_profiles(monkeypatch)
    domains = [
        types.SimpleNamespace(name="flat_wbc", task_profile="wbc_tracking"),
        types.SimpleNamespace(name="flat_velocity", task_profile="velocity_flat"),
        types.SimpleNamespace(name="omniretarget_g1_terrain", task_profile="wbc_tracking"),
    ]
    env_cfg = types.SimpleNamespace(
        commands=types.SimpleNamespace(),
        rewards=types.SimpleNamespace(old=managers.RewardTermCfg(func=lambda env: None)),
        terminations=types.SimpleNamespace(old=managers.TerminationTermCfg(func=lambda env: None)),
    )

    registry = profiles.build_default_mixed_profiles()
    profiles.apply_domain_profiles(env_cfg, domains, registry)

    assert not hasattr(registry["wbc_tracking"], "reset_cfg")
    assert not hasattr(registry["velocity_flat"], "reset_cfg")
    assert env_cfg.rewards.old is None
    assert env_cfg.terminations.old is None
    assert env_cfg.commands.base_velocity is not registry["velocity_flat"].commands_cfg.base_velocity
    assert env_cfg.terminations.velocity_flat_time_out.params["enabled_domain_names"] == ("flat_velocity",)
    assert env_cfg.terminations.velocity_flat_time_out.time_out is True
    assert env_cfg.terminations.wbc_tracking_anchor_pos.params["enabled_domain_names"] == (
        "flat_wbc",
        "omniretarget_g1_terrain",
    )
    assert env_cfg.rewards.velocity_flat_track_lin_vel_xy_exp.params["enabled_domain_names"] == ("flat_velocity",)
    assert env_cfg.rewards.velocity_flat_pose.params["enabled_domain_names"] == ("flat_velocity",)
    assert env_cfg.rewards.velocity_flat_pose.func is registry["velocity_flat"].rewards_cfg.pose.func
    assert not hasattr(env_cfg.events, "velocity_flat_reset_base")
    assert env_cfg.events.velocity_flat_push_robot.params["enabled_domain_names"] == ("flat_velocity",)
    assert (
        env_cfg.events.velocity_flat_push_robot.params["source_func"]
        is registry["velocity_flat"].events_cfg.push_robot.func
    )
    assert env_cfg.events.velocity_flat_push_robot.mode == "interval"

    routes = profiles.collect_domain_profile_routes(domains, registry)
    assert tuple(routes.profile_reset_cfgs) == ("velocity_flat",)
    assert hasattr(routes.profile_reset_cfgs["velocity_flat"], "reset_base")
    assert routes.profile_reset_cfgs["velocity_flat"].reset_base is not registry["velocity_flat"].events_cfg.reset_base
    assert routes.profile_reset_cfgs["velocity_flat"].reset_base.params["pose_range"]["z"] == (0.05, 0.1)


def test_profile_applier_deep_copies_terms_and_is_idempotent(monkeypatch) -> None:
    profiles, _ = _load_mixed_domain_profiles(monkeypatch)
    domains = [types.SimpleNamespace(name="flat_velocity", task_profile="velocity_flat")]
    env_cfg = types.SimpleNamespace(
        commands=types.SimpleNamespace(),
        rewards=types.SimpleNamespace(),
        terminations=types.SimpleNamespace(),
    )

    registry = profiles.build_default_mixed_profiles()
    profiles.apply_domain_profiles(env_cfg, domains, registry)
    env_cfg.rewards.velocity_flat_track_lin_vel_xy_exp.params["source_params"]["std"] = 99.0
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (9.0, 9.0)

    assert registry["velocity_flat"].rewards_cfg.track_lin_vel_xy_exp.params["std"] == 0.5
    assert registry["velocity_flat"].commands_cfg.base_velocity.ranges.lin_vel_x == (-0.5, 1.5)

    profiles.apply_domain_profiles(env_cfg, domains, registry)
    assert env_cfg.rewards.velocity_flat_track_lin_vel_xy_exp.params["source_params"]["std"] == 0.5
    assert env_cfg.commands.base_velocity.ranges.lin_vel_x == (-0.5, 1.5)

    profiles.apply_domain_profiles(
        env_cfg,
        [types.SimpleNamespace(name="flat_wbc", task_profile="wbc_tracking")],
        registry,
    )
    assert env_cfg.commands.base_velocity is None


def test_profile_route_collection_includes_velocity_terrain(monkeypatch) -> None:
    profiles, _ = _load_mixed_domain_profiles(monkeypatch)
    domains = [
        types.SimpleNamespace(name="flat_wbc", task_profile="wbc_tracking"),
        types.SimpleNamespace(name="flat_velocity", task_profile="velocity_flat"),
        types.SimpleNamespace(name="velocity_terrain", task_profile="velocity_terrain"),
        types.SimpleNamespace(name="omniretarget_g1_terrain", task_profile="wbc_tracking"),
    ]

    routes = profiles.collect_domain_profile_routes(domains, profiles.build_default_mixed_profiles())

    assert routes.velocity_domain_names == ("flat_velocity", "velocity_terrain")
    assert routes.aux_mask_domain_names == ("flat_wbc", "omniretarget_g1_terrain")
    assert routes.amp_mask_domain_names == (
        "flat_wbc",
        "flat_velocity",
        "velocity_terrain",
        "omniretarget_g1_terrain",
    )


def test_mixed_task_registers_amp_runner_without_polluting_paired_runner() -> None:
    init_source = (
        ROOT
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "config"
        / "g1"
        / "__init__.py"
    ).read_text(encoding="utf-8")
    runner_source = (
        ROOT
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "config"
        / "g1"
        / "agents"
        / "rsl_rl_paired_terrain_cfg.py"
    ).read_text(encoding="utf-8")
    env_source = (
        ROOT
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "config"
        / "g1"
        / "mixed_terrain_env_cfg.py"
    ).read_text(encoding="utf-8")

    assert (
        '"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_paired_terrain_cfg:G1PairedTerrainHeightScanRunnerCfg"'
        in init_source
    )
    assert (
        '"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_paired_terrain_cfg:G1MixedTerrainHeightScanAMPRunnerCfg"'
        in init_source
    )
    assert (
        '"rsl_rl_cfg_entry_point":'
        ' f"{agents.__name__}.rsl_rl_paired_terrain_cfg:G1MixedTerrainHeightScanAMPModalityFusionRunnerCfg"'
        in init_source
    )
    paired_runner_block = runner_source.split("class G1MixedTerrainHeightScanAMPRunnerCfg", maxsplit=1)[0]
    assert '"class_name": "AMPPlugin"' not in paired_runner_block
    assert '"class_name": "AMPPlugin"' in runner_source
    assert '"amp_mask_group": "amp_mask"' in runner_source
    assert '"vel_task_mask"' in runner_source
    assert "self.observations.vel_task_mask = VelTaskMaskCfg()" in env_source
    assert (
        'self.observations.vel_task_mask.vel_task_mask.params["aux_domain_names"] = routes.velocity_domain_names'
        in env_source
    )
    assert "self.observations.amp_mask = AmpMaskCfg()" in env_source
    assert "self._configure_amp_observations()" in env_source
    assert "profile_reset_cfgs=routes.profile_reset_cfgs" in env_source
