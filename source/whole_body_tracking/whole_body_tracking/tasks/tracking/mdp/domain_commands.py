from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

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


@dataclass
class PairedMeshMotionDomainCfg:
    name: str = "omniretarget_g1_terrain"
    env_ratio: float = 0.75
    pairing: str = "strict_pair"
    terrain_kind: str = "stl_grid"
    pairs_jsonl: str = "data/omniretarget/g1_terrain/pairs.jsonl"
    pair_limit: int | None = None


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
