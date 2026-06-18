from __future__ import annotations
import torch
from typing import Sequence, Protocol

from .interface import MotionDataSource, MotionSelection
from .motion_cache import MotionReferenceCache
from .motion_timeline import MotionCommandTimeline
from .motion_sampling import AdaptiveMotionSampler


#region motion sampling policy interface
class MotionSelectionPolicy(Protocol):
    fixed_eval_motion_ids: torch.Tensor | None

    def bind_motion_source(
        self,
        motion_source: MotionDataSource,
        *,
        timeline: MotionCommandTimeline,
        decimation: int,
        sim_dt: float,
    ) -> None:
        """Reset policy state after motion-source reload."""

    def select(
        self,
        env_ids: Sequence[int],
        *,
        motion_source: MotionDataSource,
        timeline: MotionCommandTimeline,
        terminated: torch.Tensor,
        metrics: dict[str, torch.Tensor],
        allow_failure_accounting: bool = True,
    ): #-> MotionSelection:
        """Produce a selection for the requested environments."""

    def step_post_update(
        self,
        *,
        motion_source: MotionDataSource,
        command_step_count: int,
    ) -> None:
        """Update policy state after each command step."""

    @property
    def counts_eval_cycles(self): #-> bool:
        """Whether the policy represents deterministic per-clip eval restarts."""


class FixedEvalMotionSelectionPolicy(MotionSelectionPolicy):
    """Deterministically bind each env to one motion id during evaluation."""

    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device
        self.fixed_eval_motion_ids: torch.Tensor | None = None

    @property
    def counts_eval_cycles(self) -> bool:
        return True

    def bind_motion_source(
        self,
        motion_source: MotionDataSource,
        *,
        timeline: MotionCommandTimeline,
        decimation: int,
        sim_dt: float,
    ) -> None:
        motion_num = int(getattr(motion_source, "motion_num", 0) or 0)
        if motion_num <= 0:
            raise RuntimeError("Motion source has no motions loaded; cannot set up fixed eval motion mapping.")
        if int(self.num_envs) != motion_num:
            raise ValueError(
                f"fixed_eval_motion_ids requires num_envs == motion_num, got num_envs={int(self.num_envs)} motion_num={motion_num}."
            )

        self.fixed_eval_motion_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        selection = timeline.selection_from_motion_ids(motion_source, self.fixed_eval_motion_ids)
        timeline.apply_selection(env_ids, selection)
        timeline.eval_cycle_count.zero_()

    def select(
        self,
        env_ids: Sequence[int],
        *,
        motion_source: MotionDataSource,
        timeline: MotionCommandTimeline,
        terminated: torch.Tensor,
        metrics: dict[str, torch.Tensor],
        allow_failure_accounting: bool = True,
    ) -> MotionSelection:
        if self.fixed_eval_motion_ids is None:
            raise RuntimeError("Fixed eval policy is not bound. Call bind_motion_source() first.")
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        motion_ids = self.fixed_eval_motion_ids[env_ids_t]
        return timeline.selection_from_motion_ids(motion_source, motion_ids)

    def step_post_update(
        self,
        *,
        motion_source: MotionDataSource,
        command_step_count: int,
    ) -> None:
        return


class AdaptiveMotionSelectionPolicy(MotionSelectionPolicy):
    """Policy adapter that exposes the adaptive sampler through a common selection interface."""

    def __init__(self, sampler: AdaptiveMotionSampler):
        self.sampler = sampler
        self.fixed_eval_motion_ids: torch.Tensor | None = None

    @property
    def counts_eval_cycles(self) -> bool:
        return False

    def bind_motion_source(
        self,
        motion_source: MotionDataSource,
        *,
        timeline: MotionCommandTimeline,
        decimation: int,
        sim_dt: float,
    ) -> None:
        self.fixed_eval_motion_ids = None
        self.sampler.reset_for_motion_source(motion_source, decimation=decimation, sim_dt=sim_dt)

    def select(
        self,
        env_ids: Sequence[int],
        *,
        motion_source: MotionDataSource,
        timeline: MotionCommandTimeline,
        terminated: torch.Tensor,
        metrics: dict[str, torch.Tensor],
        allow_failure_accounting: bool = True,
    ) -> MotionSelection:
        return self.sampler.sample_selection(
            env_ids,
            motion_source=motion_source,
            timeline=timeline,
            terminated=terminated,
            metrics=metrics,
            allow_failure_accounting=allow_failure_accounting,
        )

    def step_post_update(
        self,
        *,
        motion_source: MotionDataSource,
        command_step_count: int,
    ) -> None:
        self.sampler.step_post_update(
            motion_source=motion_source,
            command_step_count=command_step_count,
        )


class UnsupportedMotionSelectionPolicy(MotionSelectionPolicy):
    """Preserve the previous not-implemented behavior for unsupported configs."""

    def __init__(self):
        self.fixed_eval_motion_ids: torch.Tensor | None = None

    @property
    def counts_eval_cycles(self) -> bool:
        return False

    def bind_motion_source(
        self,
        motion_source: MotionDataSource,
        *,
        timeline: MotionCommandTimeline,
        decimation: int,
        sim_dt: float,
    ) -> None:
        return

    def select(
        self,
        env_ids: Sequence[int],
        *,
        motion_source: MotionDataSource,
        timeline: MotionCommandTimeline,
        terminated: torch.Tensor,
        metrics: dict[str, torch.Tensor],
        allow_failure_accounting: bool = True,
    ) -> MotionSelection:
        raise NotImplementedError("Only fixed-eval and adaptive motion selection policies are implemented.")

    def step_post_update(
        self,
        *,
        motion_source: MotionDataSource,
        command_step_count: int,
    ) -> None:
        return


def create_motion_selection_policy(cfg, *, num_envs: int, device: str) -> MotionSelectionPolicy:
    if bool(getattr(cfg, "fixed_eval_motion_ids", False)) and bool(getattr(cfg, "eval_mode", False)):
        return FixedEvalMotionSelectionPolicy(num_envs=num_envs, device=device)
    if bool(getattr(cfg, "adaptive_sample", False)):
        return AdaptiveMotionSelectionPolicy(AdaptiveMotionSampler(cfg, device))
    return UnsupportedMotionSelectionPolicy()
#endregion motion sampling policy interface
