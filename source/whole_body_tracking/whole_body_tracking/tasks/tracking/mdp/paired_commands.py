from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from pathlib import Path

from isaaclab.utils import configclass

from whole_body_tracking.terrains.paired_manifest import DEFAULT_OMNIRETARGET_G1_PAIRS_JSONL, TerrainMotionPairManifest

from .commands import MotionCommand, MotionCommandCfg
from .motion_sampling import accumulate_rewinded_motion_sample_bin_counts, sample_motion_local_times_from_bins


class PairedTerrainMotionCommand(MotionCommand):
    cfg: PairedTerrainMotionCommandCfg

    def __init__(self, cfg: PairedTerrainMotionCommandCfg, env):
        self.pair_manifest = TerrainMotionPairManifest.from_jsonl(cfg.pairs_jsonl, limit=cfg.pair_limit)
        super().__init__(cfg, env)
        self._sync_env_origins_to_pairs(torch.arange(self.num_envs, dtype=torch.long, device=self.device))
        self._log_pair_mapping_once()

    def resample_motion_files(self, env, motion_cfg=None):
        if motion_cfg is None:
            motion_cfg = self.cfg
        super().resample_motion_files(env, motion_cfg)
        self._build_pair_motion_mapping()

    def _build_pair_motion_mapping(self) -> None:
        file_names = list(getattr(self.motion, "file_names", []) or [])
        if not file_names:
            raise RuntimeError("PairedTerrainMotionCommand requires motion.file_names from the motion loader.")

        loaded_names = [Path(file_name).stem for file_name in file_names]
        expected_names = list(self.pair_manifest.names)
        if len(loaded_names) != len(expected_names):
            raise ValueError(
                "Paired terrain motion order requires the motion loader to contain exactly the manifest motions: "
                f"expected={len(expected_names)}, loaded={len(loaded_names)}. "
                "Do not override --motion_file/--dataset_txt unless it preserves pairs.jsonl order."
            )
        if loaded_names != expected_names:
            mismatches = [
                (index, expected, loaded)
                for index, (expected, loaded) in enumerate(zip(expected_names, loaded_names, strict=True))
                if expected != loaded
            ]
            raise ValueError(
                "Paired terrain motion order mismatch. "
                "Do not override --motion_file/--dataset_txt unless it preserves pairs.jsonl order. "
                f"First mismatches: {mismatches[:8]}"
            )
        self.pair_motion_ids = torch.arange(len(expected_names), dtype=torch.long, device=self.device)

    @property
    def _terrain_num_cols(self) -> int:
        terrain = getattr(self._env.scene, "terrain", None)
        generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
        if generator is not None:
            return int(generator.num_cols)
        return int(self.cfg.terrain_num_cols)

    @property
    def _terrain_num_rows(self) -> int:
        terrain = getattr(self._env.scene, "terrain", None)
        generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
        if generator is not None:
            return int(generator.num_rows)
        return int(self.cfg.terrain_num_rows)

    def _pair_ids_for_envs(self, env_ids: torch.Tensor) -> torch.Tensor:
        terrain = getattr(self._env.scene, "terrain", None)
        if terrain is None or getattr(terrain, "terrain_origins", None) is None:
            raise RuntimeError("PairedTerrainMotionCommand requires generator terrain origins.")

        if self.cfg.env_id_pairing:
            return torch.remainder(env_ids, len(self.pair_manifest.pairs)).to(torch.long)

        terrain_levels = terrain.terrain_levels[env_ids].to(torch.long)
        terrain_types = terrain.terrain_types[env_ids].to(torch.long)
        pair_ids = terrain_levels * int(self._terrain_num_cols) + terrain_types
        return torch.remainder(pair_ids, len(self.pair_manifest.pairs)).to(torch.long)

    def _sync_env_origins_to_pairs(self, env_ids: torch.Tensor) -> None:
        terrain = getattr(self._env.scene, "terrain", None)
        if terrain is None or getattr(terrain, "terrain_origins", None) is None:
            return
        pair_ids = self._pair_ids_for_envs(env_ids)
        rows = torch.div(pair_ids, int(self._terrain_num_cols), rounding_mode="floor")
        cols = torch.remainder(pair_ids, int(self._terrain_num_cols))
        terrain.terrain_levels[env_ids] = rows.to(terrain.terrain_levels.dtype)
        terrain.terrain_types[env_ids] = cols.to(terrain.terrain_types.dtype)
        terrain.env_origins[env_ids] = terrain.terrain_origins[rows, cols]

    def _motion_ids_for_envs(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self.pair_motion_ids[self._pair_ids_for_envs(env_ids)]

    def _prepare_adaptive_sampling(self, env_ids: torch.Tensor) -> None:
        self._sync_env_origins_to_pairs(env_ids)

    def _accumulate_motion_bin_failures(self, fail_motion_ids: torch.Tensor, fail_local_frames: torch.Tensor) -> None:
        self._accumulate_paired_motion_failures(fail_motion_ids, fail_local_frames)

    def _sample_adaptive_motion_times(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        self._compute_sampling_probabilities()
        sampled_motion_ids = self._motion_ids_for_envs(env_ids)
        sampled_local_bins = self._sample_local_bins_for_paired_motions(sampled_motion_ids)
        sampled_motion_ids, local_t = sample_motion_local_times_from_bins(
            motion_ids=sampled_motion_ids,
            local_bin_ids=sampled_local_bins,
            time_step_start_idx=self.motion.time_step_start_idx,
            time_step_end_idx=self.motion.time_step_end_idx,
            min_local_frame=int(self.cfg.motion_sampling_start_frame),
            max_future_step=int(self.cfg.max_future_step),
            bin_frame_width=int(self.adaptive_bin_frame_width),
        )

        self.motion_ids[env_ids] = sampled_motion_ids
        self.local_time_steps[env_ids] = local_t
        self.frame_end_per_env[env_ids] = (
            self.motion.time_step_end_idx[sampled_motion_ids] - self.motion.time_step_start_idx[sampled_motion_ids]
        )
        if self.cfg.eval_mode:
            self.local_time_steps[env_ids] = int(max(0, self.cfg.motion_sampling_start_frame))

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
        self._update_paired_sampling_metrics(sampled_motion_ids)

    def _sample_local_bins_for_paired_motions(self, motion_ids: torch.Tensor) -> torch.Tensor:
        local_bins = torch.zeros_like(motion_ids)
        arange_bins = torch.arange(self.max_motion_sample_bin_count, device=self.device)
        for motion_id in torch.unique(motion_ids):
            env_mask = motion_ids == motion_id
            valid_count = int(self.motion_sample_bin_counts[motion_id].item())
            if valid_count <= 0:
                raise RuntimeError(f"No valid local sampling bins for paired motion_id={int(motion_id.item())}")

            counts = self.motion_bin_failed_count[motion_id, :valid_count]
            valid_mean = counts.mean()
            if torch.isfinite(valid_mean) and valid_mean > 1e-12 and counts.sum() > 1e-12:
                clipped = torch.clamp(counts, max=self.cfg.failure_most_hard_cap_beta * valid_mean)
                padded = torch.nn.functional.pad(
                    clipped.view(1, 1, -1),
                    (0, self.cfg.adaptive_kernel_size - 1),
                    mode="constant",
                    value=0.0,
                )
                smoothed = torch.nn.functional.conv1d(padded, self.kernel.view(1, 1, -1)).view(-1)[:valid_count]
                probs = smoothed / (smoothed.sum() + 1e-12)
                uniform = torch.full_like(probs, 1.0 / float(valid_count))
                probs = self.cfg.motion_ratio[0] * uniform + (1.0 - self.cfg.motion_ratio[0]) * probs
                probs = probs / (probs.sum() + 1e-12)
            else:
                probs = torch.full((valid_count,), 1.0 / float(valid_count), device=self.device)

            sampled = torch.multinomial(probs, int(env_mask.sum().item()), replacement=True)
            local_bins[env_mask] = arange_bins[:valid_count][sampled]
        return local_bins

    def _accumulate_paired_motion_failures(
        self, fail_motion_ids: torch.Tensor, fail_local_frames: torch.Tensor
    ) -> None:
        self._current_motion_bin_failed.zero_()
        if fail_motion_ids.numel() == 0:
            return
        self._current_motion_bin_failed[:] = accumulate_rewinded_motion_sample_bin_counts(
            fail_motion_ids=fail_motion_ids,
            fail_local_frames=fail_local_frames,
            time_step_start_idx=self.motion.time_step_start_idx,
            time_step_end_idx=self.motion.time_step_end_idx,
            min_local_frame=int(self.cfg.motion_sampling_start_frame),
            max_future_step=int(self.cfg.max_future_step),
            bin_frame_width=int(self.adaptive_bin_frame_width),
            rewind_bins=int(self.cfg.adaptive_sample_rewind_bins),
            min_rewind_bins=int(self.cfg.adaptive_sample_rewind_min_bins),
            max_sample_bin_count=int(self.max_motion_sample_bin_count),
        )

    def _update_paired_sampling_metrics(self, sampled_motion_ids: torch.Tensor) -> None:
        counts = torch.bincount(sampled_motion_ids, minlength=int(self.motion.motion_num)).float()
        active = counts > 0
        if torch.any(active):
            probs = counts[active] / counts[active].sum().clamp_min(1.0)
            entropy = -(probs * (probs + 1e-12).log()).sum()
            entropy = entropy / math.log(max(int(active.sum().item()), 2))
            self.metrics["sampling_entropy"][:] = entropy
            self.metrics["sampling_top1_prob_max"][:] = probs.max()
            self.metrics["sampling_top1_prob_mean"][:] = probs.mean()
            self.metrics["sampling_top1_prob_min"][:] = probs.min()
            self.metrics["prob_uniform"][:] = 1.0 / float(max(int(active.sum().item()), 1))
            self.metrics["prob_max_over_uniform"][:] = probs.max() / self.metrics["prob_uniform"][0].clamp_min(1e-12)
            self.metrics["num_concentrate_bins"][:] = active.sum()

    def _log_pair_mapping_once(self) -> None:
        sample_count = min(int(self.cfg.debug_pair_log_count), self.num_envs)
        if sample_count <= 0:
            return
        env_ids = torch.arange(sample_count, dtype=torch.long, device=self.device)
        pair_ids = self._pair_ids_for_envs(env_ids).detach().cpu().tolist()
        motion_ids = self._motion_ids_for_envs(env_ids).detach().cpu().tolist()
        print("[PairedTerrainMotionCommand] env -> pair -> motion mapping preview:")
        for env_id, pair_id, motion_id in zip(env_ids.detach().cpu().tolist(), pair_ids, motion_ids, strict=False):
            pair = self.pair_manifest.pairs[pair_id]
            terrain_name = Path(pair.terrain_path).stem
            print(
                f"  env={env_id:03d} pair={pair_id:03d} motion={motion_id:03d} "
                f"motion_name={pair.name} terrain_name={terrain_name}"
            )


@configclass
class PairedTerrainMotionCommandCfg(MotionCommandCfg):
    class_type: type = PairedTerrainMotionCommand

    pairs_jsonl: str = DEFAULT_OMNIRETARGET_G1_PAIRS_JSONL
    terrain_num_rows: int = 1
    terrain_num_cols: int = 1
    strict_pairing: bool = True
    pair_limit: int | None = None
    debug_pair_log_count: int = 8
    env_id_pairing: bool = True
    preserve_dataset_order: bool = True
