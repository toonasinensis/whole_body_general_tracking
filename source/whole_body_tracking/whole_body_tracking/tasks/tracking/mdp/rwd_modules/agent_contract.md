# Reward Modules Agent Contract

`rwd_modules` owns reward term functions and delayed-termination reward masking.
Public API is the `__all__` list in `rwd_modules/__init__.py`.

## Public Exports

Wrapped reward terms:

- `motion_global_anchor_position_error_exp`
- `motion_global_anchor_position_z_error_exp`
- `motion_global_anchor_orientation_error_exp`
- `motion_relative_body_position_error_exp`
- `motion_relative_body_orientation_error_exp`
- `motion_global_body_linear_velocity_error_exp`
- `motion_global_body_angular_velocity_error_exp`
- `feet_contact_time`

Factory/helper:

- `rwd_zero_delayed_wrapper`

Configs should import the wrapped exports from `rwd_modules` or the package facade
`mdp`, not raw functions from `rwd_functions.py`.

## Wrapper Contract

`rwd_factory.rwd_zero_delayed_wrapper()` wraps raw reward functions and adds the
keyword-only parameter:

```python
disable_on_delayed_termination: bool = False
```

When enabled, rewards are zeroed for envs where
`env.termination_manager.delayed_termination_active_mask` is true.

This wrapper preserves the wrapped function name, qualname, docstring, and extends
the visible signature for IsaacLab config validation.

## Raw Function Ownership

`rwd_functions.py` contains pure reward computations. They may read:

- `env.command_manager.get_term(command_name)` returning `MotionCommand`.
- `env.scene.sensors[sensor_cfg.name]` for contact rewards.

They must return a tensor with shape `(env.num_envs,)`.

Forbidden side effects:

- Do not mutate env, command state, timeline, resetter, sampler, metrics, or extras.
- Do not write simulator state.
- Do not install delayed termination. That belongs to `tmt_modules`.

## Dependencies on `MotionCommand`

Reward terms depend on:

- `anchor_pos_w`, `anchor_quat_w`
- `robot_anchor_pos_w`, `robot_anchor_quat_w`
- `body_pos_relative_w`, `body_quat_relative_w`
- `body_lin_vel_w`, `body_ang_vel_w`
- `robot_body_pos_w`, `robot_body_quat_w`
- `robot_body_lin_vel_w`, `robot_body_ang_vel_w`
- `cfg.body_names`

`body_pos_relative_w` and `body_quat_relative_w` are aligned world-frame reference
cache outputs, despite the historical `relative` name.

## Extension Rules

- Add new reward math in `rwd_functions.py`.
- Export config-facing reward terms through `rwd_factory.py` with
  `rwd_zero_delayed_wrapper()` unless there is a documented reason not to support
  delayed-termination masking.
- Update `rwd_modules/__init__.py` and this file when adding public terms.
