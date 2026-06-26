# Termination Modules Agent Contract

`tmt_modules` owns termination term functions and delayed termination behavior.
Public API is the `__all__` list in `tmt_modules/__init__.py`.

## Public Exports

Delayed termination:

- `DelayedTerminationManager`
- `install_delayed_termination`
- `tmt_cnd_disable_wrapper`

Wrapped termination terms:

- `bad_anchor_pos`
- `bad_anchor_pos_z_only`
- `bad_anchor_ori`
- `bad_motion_body_pos`
- `bad_motion_body_pos_z_only`

Configs should use wrapped exports from `tmt_modules` or facade `mdp`, not raw
functions from `tmt_functions.py`.

## `DelayedTerminationManager`

`DelayedTerminationManager` wraps an IsaacLab `TerminationManager` and delays bad
termination for a selected env subset.

Public attributes intentionally read by other modules:

- `delayed_termination_env_mask`: bool mask for envs eligible for delay.
- `delayed_termination_active_mask`: bool mask for envs whose termination is currently
  suppressed during the delay window.

Internal state:

- `_delay_env_mask`
- `_delay_counters`
- `_max_delay_steps`

Public methods:

- `reset(env_ids=None)`
- `compute()`

`compute()` calls the base termination computation, suppresses `_terminated_buf` for
eligible envs until `_max_delay_steps`, and returns truncated-or-terminated dones.

## Installation Contract

`install_delayed_termination(env, env_ids, delay_reset_env_ratio, max_delay_steps,
use_motion_pose_range_mask=True)` is an event function intended for startup mode.

When `use_motion_pose_range_mask=True`, it derives delayed envs from
`env.envs_classes_mask["lying"]`. If that runtime mask is not available yet, it may
fall back to `env.cfg.commands.motion.envs_classes_ratio["lying"]` and build the
same contiguous prefix mask that `MotionCommandResetter` would build. This must stay
aligned with `MotionCommandResetter` and `cfg.envs_classes_ratio`.

When `use_motion_pose_range_mask=False`, it selects the first
`int(env.num_envs * delay_reset_env_ratio)` envs.

The function replaces `env.termination_manager` with `DelayedTerminationManager` if
enabled and not already installed.

## Wrapper Contract

`tmt_cnd_disable_wrapper()` wraps raw termination functions and adds:

```python
disable_on_delayed_termination_envs: bool = False
```

When enabled, the wrapped termination result is masked out for envs where
`env.termination_manager.delayed_termination_env_mask` is true.

This is a config-facing behavior. Do not bypass wrappers in environment configs.

## Raw Function Ownership

`tmt_functions.py` contains pure termination predicates. They may read:

- `env.command_manager.get_term(command_name)` returning `MotionCommand`.
- `env.scene[asset_cfg.name]` for gravity/orientation checks.

They must return boolean tensors with shape `(env.num_envs,)`.

Forbidden side effects:

- Raw termination functions must not mutate env, command state, simulator state,
  metrics, rewards, or event state.
- Only `install_delayed_termination()` may replace `env.termination_manager`.

## Dependencies on `MotionCommand`

Termination terms depend on:

- `anchor_pos_w`, `anchor_quat_w`
- `robot_anchor_pos_w`, `robot_anchor_quat_w`
- `body_pos_relative_w`
- `robot_body_pos_w`
- `cfg.body_names`

Update this file when changing those command property shapes or semantics.

## Follow-Ups

- If env class ownership moves away from `MotionCommandResetter`, update
  `install_delayed_termination()` and this contract together.
- The fallback path duplicates contiguous-mask construction logic; prefer a shared
  provider if more modules need env class masks.
