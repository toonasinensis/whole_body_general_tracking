from __future__ import annotations

import torch

from isaaclab.managers import TerminationManager


###
# Termination wrapper
###

class DelayedTerminationManager(TerminationManager):
    """Wrap ``TerminationManager`` and delay early terminations for a subset of envs."""

    def __init__(
        self,
        base: TerminationManager,
        delay_env_mask: torch.Tensor,
        max_delay_steps: int,
    ) -> None:
        self.__dict__.update(base.__dict__)
        self._max_delay_steps = int(max_delay_steps)

        self._delay_env_mask = delay_env_mask
        self.delayed_termination_env_mask = delay_env_mask

        self._delay_counters = torch.zeros_like(delay_env_mask, dtype=torch.long)
        self.delayed_termination_active_mask = torch.zeros_like(delay_env_mask, dtype=torch.bool)

    def reset(self, env_ids=None) -> dict[str, torch.Tensor]:
        extras = super().reset(env_ids=env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._delay_counters[env_ids] = 0
        self.delayed_termination_active_mask[env_ids] = False
        return extras 

    def compute(self) -> torch.Tensor:
        """
            将由于 bad state 导致 termiation 的环境，
            依据其特权情况修改为 dones = False 
        """
        dones = super().compute()
        self.delayed_termination_active_mask[:] = False
        if self._max_delay_steps <= 0:
            return dones
        
        # 有 delay 权限，同时又处于 bad termination 状态的环境 ID
        delay_and_terminated = self._delay_env_mask & self._terminated_buf
        self._delay_counters[delay_and_terminated] += 1

        # 有 delay 权限，同时又处于 bad termination 状态, 而且在 delay 周期内 的环境 ID
        not_ready = delay_and_terminated & (self._delay_counters < self._max_delay_steps)
        # Expose only the currently suppressed termination window. Rewards use this
        # to switch from full tracking to recovery tracking on the same step.
        self.delayed_termination_active_mask[not_ready] = True
        # 将 not_ready 的环境 terminate mark 设置为 False
        self._terminated_buf[not_ready] = False

        ready = delay_and_terminated & (self._delay_counters >= self._max_delay_steps)
        self._delay_counters[ready] = 0
        self._delay_counters[self._delay_env_mask & ~self._terminated_buf & ~not_ready] = 0
        
        return self._truncated_buf | self._terminated_buf
