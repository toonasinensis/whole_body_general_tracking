"""Isaac Lab adapter for project-local ``rsl_rl`` (TensorDict ``VecEnv`` API).

Isaac Lab's ``isaaclab_rl.rsl_rl.RslRlVecEnvWrapper`` targets older rsl-rl releases that
expect a flat ``policy`` tensor. The fork in ``rsl_rl/`` expects ``TensorDict`` observations
with one key per observation group (``prop``, ``critic``, ``amp``, ...).
"""

from __future__ import annotations

import gymnasium as gym
import torch
from tensordict import TensorDict

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from rsl_rl.env import VecEnv


def _obs_group_to_tensor(value: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        return torch.cat([value[key] for key in value], dim=-1)
    raise TypeError(f"Unsupported observation group type: {type(value)!r}")


def obs_dict_to_tensordict(obs_dict: dict, num_envs: int, device: torch.device | str) -> TensorDict:
    """Convert Isaac Lab observation groups to an ``rsl_rl``-compatible ``TensorDict``."""
    data: dict[str, torch.Tensor] = {}
    for group_name, group_obs in obs_dict.items():
        if group_obs is None:
            continue
        data[group_name] = _obs_group_to_tensor(group_obs)
    return TensorDict(data, batch_size=[num_envs], device=device)


class IsaacLabVecEnvWrapper(VecEnv):
    """Wrap Isaac Lab envs for ``OnPolicyRunner`` in the local ``rsl_rl`` package."""

    def __init__(self, env: gym.Env, clip_actions: float | None = None):
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must inherit from ManagerBasedRLEnv or DirectRLEnv. "
                f"Got: {type(env.unwrapped)!r}"
            )

        self.env = env
        self.clip_actions = clip_actions

        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        self._modify_action_space()

        # RSL-RL runner does not call reset before the first rollout.
        self.env.reset()

    @property
    def cfg(self) -> object:
        return self.unwrapped.cfg

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.unwrapped.episode_length_buf = value

    def _compute_obs_dict(self) -> dict:
        if hasattr(self.unwrapped, "observation_manager"):
            return self.unwrapped.observation_manager.compute()
        return self.unwrapped._get_observations()

    def get_observations(self) -> TensorDict:
        return obs_dict_to_tensordict(self._compute_obs_dict(), self.num_envs, self.device)

    def reset(self) -> TensorDict:
        obs_dict, _ = self.env.reset()
        return obs_dict_to_tensordict(obs_dict, self.num_envs, self.device)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs_dict, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        obs = obs_dict_to_tensordict(obs_dict, self.num_envs, self.device)

        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        return obs, rewards, dones, extras

    def close(self):
        return self.env.close()

    def _modify_action_space(self) -> None:
        if self.clip_actions is None:
            return

        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions,
            high=self.clip_actions,
            shape=(self.num_actions,),
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space,
            self.num_envs,
        )
