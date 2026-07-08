from __future__ import annotations

import inspect
import os
import torch
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from whole_body_tracking.terrains.mixed_domains import (
    MixedDomainLayout,
    allocate_domain_env_counts,
    build_mixed_domain_layout,
    domain_env_starts,
)

from .commands import MotionCommand, MotionCommandCfg


@dataclass
class FlatMotionDomainCfg:
    name: str = "flat_lafan"
    env_ratio: float = 0.25
    pairing: str = "unpaired"
    terrain_kind: str = "flat"
    motion_file: str = "data"
    dataset_txt: str = "data/tracking_npz_data/lafan_named.txt"
    max_motion_num: int = -1
    terrain_cell_count: int = 1
    task_profile: str = "wbc_tracking"


@dataclass
class PairedMeshMotionDomainCfg:
    name: str = "omniretarget_g1_terrain"
    env_ratio: float = 0.75
    pairing: str = "strict_pair"
    terrain_kind: str = "stl_grid"
    pairs_jsonl: str = "data/omniretarget/g1_terrain/pairs.jsonl"
    pair_limit: int | None = None
    task_profile: str = "wbc_tracking"


@dataclass
class ProceduralVelocityTerrainDomainCfg:
    name: str = "velocity_terrain"
    env_ratio: float = 0.20
    pairing: str = "unpaired"
    terrain_kind: str = "hf_grid"
    motion_file: str = "data"
    dataset_txt: str = "data/tracking_npz_data/lafan_named.txt"
    max_motion_num: int = -1
    terrain_cell_count: int = 8
    terrain_profile: str = "velocity_runway_steps"
    curriculum: bool = True
    task_profile: str = "velocity_terrain"


class DomainMotionCommand(MotionCommand):
    cfg: DomainMotionCommandCfg

    def __init__(self, cfg: DomainMotionCommandCfg, env):
        self.domain_layout = self._build_domain_layout(cfg)
        self._domain_mapping_logged = False
        super().__init__(cfg, env)
        self._init_domain_tensors()
        self._sync_env_origins_to_domains(torch.arange(self.num_envs, dtype=torch.long, device=self.device))

    def _build_domain_layout(self, cfg: DomainMotionCommandCfg) -> MixedDomainLayout:
        return build_mixed_domain_layout(
            cfg.domains,
            domain_separator_cell_count=int(cfg.domain_separator_cell_count),
            pad_to_grid=True,
        )

    def resample_motion_files(self, env, motion_cfg=None):
        if motion_cfg is None:
            motion_cfg = self.cfg
        self.domain_layout = self._build_domain_layout(motion_cfg)
        load_cfg = SimpleNamespace(**vars(motion_cfg))
        load_cfg.motion_files = list(self.domain_layout.motion_paths)
        load_cfg.motion_file = str(Path(self.domain_layout.motion_paths[0]).parent)
        load_cfg.dataset_txt = None
        load_cfg.max_motion_num = -1
        load_cfg.distributed = False
        load_cfg.local_rank = -1
        load_cfg.total_rank = -1
        load_cfg.preserve_dataset_order = True
        super().resample_motion_files(env, load_cfg)

        loaded_stems = [Path(file_name).stem for file_name in getattr(self.motion, "file_names", [])]
        expected_stems = [Path(path).stem for path in self.domain_layout.motion_paths]
        if loaded_stems != expected_stems:
            raise RuntimeError(
                "Domain motion loader order mismatch. "
                f"Expected first={expected_stems[:8]}, loaded first={loaded_stems[:8]}"
            )

    def _init_domain_tensors(self) -> None:
        counts = allocate_domain_env_counts(
            self.num_envs,
            tuple(domain.env_ratio for domain in self.domain_layout.domains),
        )
        env_domain_ids = torch.empty(self.num_envs, dtype=torch.long, device=self.device)
        cursor = 0
        for domain_id, count in enumerate(counts):
            if count <= 0:
                continue
            env_domain_ids[cursor : cursor + count] = domain_id
            cursor += count
        if cursor != self.num_envs:
            raise RuntimeError(f"Domain env allocation filled {cursor}/{self.num_envs} envs")
        self.env_domain_ids = env_domain_ids
        self.domain_env_start = torch.tensor(domain_env_starts(counts), dtype=torch.long, device=self.device)
        self.domain_env_count = torch.tensor(counts, dtype=torch.long, device=self.device)

        self.domain_motion_start = torch.tensor(
            [domain.motion_start for domain in self.domain_layout.domains], dtype=torch.long, device=self.device
        )
        self.domain_motion_count = torch.tensor(
            [domain.motion_count for domain in self.domain_layout.domains], dtype=torch.long, device=self.device
        )
        self.domain_terrain_start = torch.tensor(
            [domain.terrain_start for domain in self.domain_layout.domains], dtype=torch.long, device=self.device
        )
        self.domain_terrain_count = torch.tensor(
            [domain.terrain_count for domain in self.domain_layout.domains], dtype=torch.long, device=self.device
        )
        self.domain_pairing = tuple(domain.pairing for domain in self.domain_layout.domains)

    @property
    def _terrain_num_cols(self) -> int:
        terrain = getattr(self._env.scene, "terrain", None)
        generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
        if generator is not None:
            return int(generator.num_cols)
        return int(self.domain_layout.num_cols)

    def _global_terrain_ids_for_envs(self, env_ids: torch.Tensor) -> torch.Tensor:
        domain_ids = self.env_domain_ids[env_ids]
        terrain_ids = torch.empty_like(domain_ids)
        for domain_id, domain in enumerate(self.domain_layout.domains):
            mask = domain_ids == domain_id
            if not torch.any(mask):
                continue
            local_env_ids = env_ids[mask]
            local_env_ids = local_env_ids - self.domain_env_start[domain_id]
            terrain_count = int(domain.terrain_count)
            local_terrain_ids = torch.remainder(local_env_ids, terrain_count)
            terrain_ids[mask] = int(domain.terrain_start) + local_terrain_ids
        return terrain_ids

    def _sync_env_origins_to_domains(self, env_ids: torch.Tensor) -> None:
        terrain = getattr(self._env.scene, "terrain", None)
        if terrain is None or getattr(terrain, "terrain_origins", None) is None:
            return
        terrain_ids = self._global_terrain_ids_for_envs(env_ids)
        rows = torch.div(terrain_ids, int(self._terrain_num_cols), rounding_mode="floor")
        cols = torch.remainder(terrain_ids, int(self._terrain_num_cols))
        terrain.terrain_levels[env_ids] = rows.to(terrain.terrain_levels.dtype)
        terrain.terrain_types[env_ids] = cols.to(terrain.terrain_types.dtype)
        terrain.env_origins[env_ids] = terrain.terrain_origins[rows, cols]

    def _motion_ids_for_envs(self, env_ids: torch.Tensor) -> torch.Tensor:
        domain_ids = self.env_domain_ids[env_ids]
        terrain_ids = self._global_terrain_ids_for_envs(env_ids)
        motion_ids = torch.empty_like(domain_ids)
        for domain_id, domain in enumerate(self.domain_layout.domains):
            mask = domain_ids == domain_id
            if not torch.any(mask):
                continue
            count = int(domain.motion_count)
            start = int(domain.motion_start)
            if domain.pairing == "strict_pair":
                local_terrain_ids = terrain_ids[mask] - int(domain.terrain_start)
                if count != int(domain.terrain_count):
                    raise RuntimeError(f"{domain.name}: strict_pair requires motion_count == terrain_count")
                motion_ids[mask] = start + torch.remainder(local_terrain_ids, count)
            else:
                motion_ids[mask] = start + torch.randint(count, (int(mask.sum().item()),), device=self.device)
        return motion_ids

    def _env_ids_for_task_profile(self, env_ids: torch.Tensor, task_profile: str) -> torch.Tensor:
        if not task_profile:
            return env_ids[:0]
        profile_domain_ids = [
            domain_id
            for domain_id, domain in enumerate(self.domain_layout.domains)
            if getattr(domain, "task_profile", "wbc_tracking") == task_profile
        ]
        if not profile_domain_ids:
            return env_ids[:0]
        mask = torch.zeros(env_ids.shape, dtype=torch.bool, device=self.device)
        env_domain_ids = self.env_domain_ids[env_ids]
        for domain_id in profile_domain_ids:
            mask |= env_domain_ids == domain_id
        return env_ids[mask]

    @staticmethod
    def _iter_reset_terms(reset_cfg) -> list[EventTermCfg]:
        if reset_cfg is None:
            return []
        terms = []
        seen = set()
        for source in (vars(reset_cfg), vars(type(reset_cfg))):
            for name, value in source.items():
                if name.startswith("_") or name in seen:
                    continue
                seen.add(name)
                if isinstance(value, EventTermCfg) and value.mode == "reset":
                    terms.append(value)
        return terms

    def _apply_profile_reset_cfg(self, reset_cfg, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        for term in self._iter_reset_terms(reset_cfg):
            term.func(self._env, env_ids, **term.params)

    def _resample_command(self, env_ids: Sequence[int], record_failures: bool = True):
        super()._resample_command(env_ids, record_failures=record_failures)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return
        for task_profile, reset_cfg in self.cfg.profile_reset_cfgs.items():
            profile_env_ids = self._env_ids_for_task_profile(env_ids, task_profile)
            self._apply_profile_reset_cfg(reset_cfg, profile_env_ids)

    def _uniform_local_times_for_motions(self, motion_ids: torch.Tensor) -> torch.Tensor:
        starts = self.motion.time_step_start_idx[motion_ids]
        ends = self.motion.time_step_end_idx[motion_ids]
        local_lengths = (ends - starts).clamp(min=1)
        min_frame = int(self.cfg.motion_sampling_start_frame)
        max_future = int(self.cfg.max_future_step)
        valid_max = (local_lengths - max_future - 1).clamp(min=min_frame)
        span = (valid_max - min_frame + 1).clamp(min=1)
        return min_frame + torch.floor(torch.rand_like(span, dtype=torch.float32) * span.float()).to(torch.long)

    def _prepare_adaptive_sampling(self, env_ids: torch.Tensor) -> None:
        self._sync_env_origins_to_domains(env_ids)

    def _accumulate_motion_bin_failures(self, fail_motion_ids: torch.Tensor, fail_local_frames: torch.Tensor) -> None:
        del fail_motion_ids, fail_local_frames

    def _sample_adaptive_motion_times(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        self._compute_sampling_probabilities()
        sampled_motion_ids = self._motion_ids_for_envs(env_ids)
        local_t = self._uniform_local_times_for_motions(sampled_motion_ids)

        self.motion_ids[env_ids] = sampled_motion_ids
        self.local_time_steps[env_ids] = local_t
        self.frame_end_per_env[env_ids] = (
            self.motion.time_step_end_idx[sampled_motion_ids] - self.motion.time_step_start_idx[sampled_motion_ids]
        )
        if self.cfg.eval_mode:
            self.local_time_steps[env_ids] = int(max(0, self.cfg.motion_sampling_start_frame))

        self._log_domain_mapping_once(env_ids)
        sampling_probabilities, valid_motion_bin_mask = self._compute_motion_bin_sampling_probabilities()
        return sampling_probabilities, valid_motion_bin_mask, sampled_motion_ids

    def _update_adaptive_sampling_metrics(
        self,
        sampling_probabilities: torch.Tensor,
        valid_motion_bin_mask: torch.Tensor,
        sampled_motion_ids: torch.Tensor | None,
    ) -> None:
        del sampling_probabilities, valid_motion_bin_mask
        if sampled_motion_ids is None:
            return
        counts = torch.bincount(sampled_motion_ids, minlength=int(self.motion.motion_num)).float()
        active = counts > 0
        if torch.any(active):
            probs = counts[active] / counts[active].sum().clamp_min(1.0)
            entropy = -(probs * (probs + 1e-12).log()).sum() / torch.log(
                torch.tensor(float(max(int(active.sum().item()), 2)), device=self.device)
            )
            self.metrics["sampling_entropy"][:] = entropy
            self.metrics["sampling_top1_prob_max"][:] = probs.max()
            self.metrics["sampling_top1_prob_mean"][:] = probs.mean()
            self.metrics["sampling_top1_prob_min"][:] = probs.min()
            self.metrics["prob_uniform"][:] = 1.0 / float(max(int(active.sum().item()), 1))
            self.metrics["prob_max_over_uniform"][:] = probs.max() / self.metrics["prob_uniform"][0].clamp_min(1e-12)
            self.metrics["num_concentrate_bins"][:] = active.sum()

    def _update_metrics(self):
        super()._update_metrics()
        self._update_velocity_domain_metrics()

    def _update_velocity_domain_metrics(self) -> None:
        try:
            base_velocity = self._env.command_manager.get_term("base_velocity")
        except Exception:
            return
        base_metrics = getattr(base_velocity, "metrics", None)
        if not base_metrics:
            return
        for metric_name in ("error_vel_xy", "error_vel_yaw"):
            value = base_metrics.get(metric_name)
            if value is None or not torch.is_tensor(value):
                continue
            value = value.to(device=self.device)
            if value.ndim == 0:
                continue
            for domain_id, domain in enumerate(self.domain_layout.domains):
                if getattr(domain, "task_profile", "wbc_tracking") not in ("velocity_flat", "velocity_terrain"):
                    continue
                mask = self.env_domain_ids == domain_id
                if torch.any(mask):
                    mean_value = value[mask].mean()
                else:
                    mean_value = torch.zeros((), device=self.device, dtype=value.dtype)
                self.metrics[f"{metric_name}_{domain.name}"] = mean_value.repeat(self.num_envs)

    def _mapping_rows(self, env_ids: torch.Tensor) -> list[dict[str, object]]:
        terrain_ids = self._global_terrain_ids_for_envs(env_ids).detach().cpu().tolist()
        motion_ids = self.motion_ids[env_ids].detach().cpu().tolist()
        domain_ids = self.env_domain_ids[env_ids].detach().cpu().tolist()
        rows = []
        file_names = list(getattr(self.motion, "file_names", []) or [])
        for env_id, domain_id, terrain_id, motion_id in zip(
            env_ids.detach().cpu().tolist(), domain_ids, terrain_ids, motion_ids, strict=False
        ):
            domain = self.domain_layout.domains[domain_id]
            terrain_name = self.domain_layout.terrain_names[terrain_id]
            motion_name = Path(file_names[motion_id]).stem if file_names else f"motion_{motion_id:05d}"
            pair_id = terrain_id - domain.terrain_start if domain.pairing == "strict_pair" else None
            rows.append(
                {
                    "env": int(env_id),
                    "domain": domain.name,
                    "terrain": terrain_name,
                    "motion": motion_name,
                    "pair_id": pair_id,
                }
            )
        return rows

    def get_debug_mapping_rows(self, count: int | None = None) -> list[dict[str, object]]:
        sample_count = self.num_envs if count is None else min(int(count), self.num_envs)
        env_ids = torch.arange(sample_count, dtype=torch.long, device=self.device)
        return self._mapping_rows(env_ids)

    def _log_domain_mapping_once(self, env_ids: torch.Tensor) -> None:
        if self._domain_mapping_logged:
            return
        sample_count = min(int(self.cfg.debug_domain_log_count), self.num_envs)
        if sample_count <= 0:
            return
        sample_env_ids = torch.arange(sample_count, dtype=torch.long, device=self.device)
        sampled = (sample_env_ids[:, None] == env_ids[None, :]).any(dim=1)
        if not bool(torch.all(sampled)):
            return
        self._domain_mapping_logged = True
        print("[DomainMotionCommand] env -> domain -> terrain -> motion preview:")
        for row in self._mapping_rows(sample_env_ids):
            print(
                f"  env={row['env']:03d} domain={row['domain']} pair={row['pair_id']} "
                f"terrain={row['terrain']} motion={row['motion']}"
            )


@configclass
class DomainMotionCommandCfg(MotionCommandCfg):
    class_type: type = DomainMotionCommand

    domains: list = [
        FlatMotionDomainCfg(),
        PairedMeshMotionDomainCfg(),
    ]
    domain_separator_cell_count: int = 0
    debug_domain_log_count: int = 16
    profile_reset_cfgs: dict[str, object] = {}


# 根据指定的 domain name，返回哪些 env 属于这些 domain 的 mask


def domain_name_mask(
    env,
    command_name: str = "motion",
    enabled_domain_names: Sequence[str] = (),
) -> torch.Tensor:
    """Return a boolean env mask for selected DomainMotionCommand domain names."""
    command = env.command_manager.get_term(command_name)
    if not hasattr(command, "env_domain_ids") or not enabled_domain_names:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    layout = getattr(command, "domain_layout", None)
    domains = tuple(getattr(layout, "domains", ()))
    name_to_id = {domain.name: domain_id for domain_id, domain in enumerate(domains)}
    missing = [name for name in enabled_domain_names if name not in name_to_id]
    if missing:
        raise ValueError(f"Unknown domain names {missing}. Available domains: {list(name_to_id)}")

    enabled = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in enabled_domain_names:
        enabled |= command.env_domain_ids == int(name_to_id[name])
    return enabled


def domain_name_float_mask(
    env,
    command_name: str = "motion",
    enabled_domain_names: Sequence[str] = (),
) -> torch.Tensor:
    return domain_name_mask(env, command_name, enabled_domain_names).to(dtype=torch.float32).unsqueeze(-1)


def _resolve_masked_source_params(env, source_params: dict | None) -> dict:
    if not source_params:
        return {}
    scene_id = id(env.scene)
    for value in source_params.values():
        if not isinstance(value, SceneEntityCfg):
            continue
        if getattr(value, "_domain_mask_resolved_scene_id", None) == scene_id:
            continue
        value.resolve(env.scene)
        value._domain_mask_resolved_scene_id = scene_id
    return source_params


def domain_masked_reward(
    env,
    source_func,
    source_params: dict | None = None,
    domain_command_name: str = "motion",
    enabled_domain_names: Sequence[str] = (),
) -> torch.Tensor:
    source_params = _resolve_masked_source_params(env, source_params)
    reward = source_func(env, **source_params)
    if not enabled_domain_names:
        return reward
    mask = domain_name_mask(env, domain_command_name, enabled_domain_names)
    _debug_domain_masked_reward_once(env, source_func, enabled_domain_names, mask, reward)
    return torch.where(mask, reward, torch.zeros_like(reward))


def _debug_domain_masked_reward_once(
    env,
    source_func,
    enabled_domain_names: Sequence[str],
    mask: torch.Tensor,
    reward: torch.Tensor,
) -> None:
    if os.getenv("WBT_DEBUG_VELCOMMAND", "0") in ("", "0", "false", "False"):
        return
    func_name = getattr(source_func, "__name__", source_func.__class__.__name__)
    debug_key = (func_name, tuple(enabled_domain_names))
    printed = getattr(env, "_wbt_debug_masked_rewards_printed", set())
    if debug_key in printed:
        return
    printed = set(printed)
    printed.add(debug_key)
    setattr(env, "_wbt_debug_masked_rewards_printed", printed)
    mask = mask.to(dtype=torch.bool)
    active_count = int(mask.sum().item())
    inactive_count = int((~mask).sum().item())
    active_mean = float(reward[mask].mean().item()) if active_count > 0 else 0.0
    inactive_mean = float(reward[~mask].mean().item()) if inactive_count > 0 else 0.0
    print(
        "[WBT_DEBUG_REWARD_MASK] "
        f"func={func_name} enabled_domains={tuple(enabled_domain_names)} "
        f"active_envs={active_count} inactive_envs={inactive_count} "
        f"reward_mean_active={active_mean:.6f} reward_mean_inactive_before_zero={inactive_mean:.6f}"
    )


def domain_masked_termination(
    env,
    source_func,
    source_params: dict | None = None,
    domain_command_name: str = "motion",
    enabled_domain_names: Sequence[str] = (),
) -> torch.Tensor:
    source_params = _resolve_masked_source_params(env, source_params)
    terminated = source_func(env, **source_params)
    if not enabled_domain_names:
        return terminated
    return terminated & domain_name_mask(env, domain_command_name, enabled_domain_names)


def domain_masked_event(
    env,
    env_ids: torch.Tensor | None,
    source_func,
    source_params: dict | None = None,
    domain_command_name: str = "motion",
    enabled_domain_names: Sequence[str] = (),
) -> None:
    """Run an event only for envs that belong to selected mixed-domain names."""
    source_params = _resolve_masked_source_params(env, source_params)
    if not enabled_domain_names:
        _call_domain_masked_event_source(env, env_ids, source_func, source_params)
        return

    filtered_env_ids = _filtered_domain_env_ids(env, env_ids, domain_command_name, enabled_domain_names)
    if filtered_env_ids.numel() == 0:
        return
    _call_domain_masked_event_source(env, filtered_env_ids, source_func, source_params)


def domain_masked_curriculum(
    env,
    env_ids: torch.Tensor | None,
    source_func,
    source_params: dict | None = None,
    domain_command_name: str = "motion",
    enabled_domain_names: Sequence[str] = (),
):
    """Run a curriculum term only on reset envs that belong to selected mixed-domain names."""
    source_params = _resolve_masked_source_params(env, source_params)
    if not enabled_domain_names:
        return source_func(env, env_ids, **source_params)
    filtered_env_ids = _filtered_domain_env_ids(env, env_ids, domain_command_name, enabled_domain_names)
    if filtered_env_ids.numel() == 0:
        return {}
    return source_func(env, filtered_env_ids, **source_params)


def _filtered_domain_env_ids(
    env,
    env_ids: torch.Tensor | None,
    domain_command_name: str,
    enabled_domain_names: Sequence[str],
) -> torch.Tensor:
    mask = domain_name_mask(env, domain_command_name, enabled_domain_names)
    if env_ids is None:
        return torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
    return env_ids[mask[env_ids]]


def _call_domain_masked_event_source(env, env_ids: torch.Tensor | None, source_func, source_params: dict) -> None:
    call_params = {key: value for key, value in source_params.items() if not str(key).startswith("_domain_masked_")}
    if inspect.isclass(source_func):
        cache_key = "_domain_masked_event_instance"
        cache = source_params.setdefault(cache_key, {})
        scene_id = id(env.scene)
        instance = cache.get(scene_id)
        if instance is None:
            term_cfg = EventTermCfg(func=source_func, params=call_params, mode="startup")
            instance = source_func(cfg=term_cfg, env=env)
            cache[scene_id] = instance
        source_func = instance
    source_func(env, env_ids, **call_params)
