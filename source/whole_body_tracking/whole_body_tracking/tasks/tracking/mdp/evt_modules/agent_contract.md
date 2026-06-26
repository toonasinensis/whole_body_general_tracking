# Event Modules Agent Contract

`evt_modules` owns IsaacLab event functions for startup and interval effects. Public
API is the `__all__` list in `evt_modules/__init__.py`.

## Public Exports

- `randomize_joint_default_pos`
- `randomize_rigid_body_com`
- `assist_fallen_robots_with_upward_force`

## Ownership

Event functions are the only MDP terms in this package that intentionally mutate env,
asset defaults, PhysX properties, or external force buffers outside command reset.

Allowed side effects:

- `randomize_joint_default_pos()` may mutate `asset.data.default_joint_pos`,
  `asset.data.default_joint_pos_nominal`, and the joint position action offset.
- `randomize_rigid_body_com()` may mutate PhysX COMs through
  `asset.root_physx_view.set_coms()`.
- `assist_fallen_robots_with_upward_force()` may write external forces/torques through
  wrench composers or `asset.set_external_force_and_torque()`.
- `assist_fallen_robots_with_upward_force()` may write log metrics into
  `env.extras["log"]` and may own `env._fallen_upward_assist_timeout_stats`.

Forbidden side effects:

- Do not mutate `MotionCommand` timeline, selection policy, sampler, or reference
  cache from event functions.
- Do not perform robot reset here; command-driven reset belongs to
  `MotionCommandResetter`.
- Do not replace `env.termination_manager`; delayed termination installation belongs
  to `tmt_modules`.

## `assist_fallen_robots_with_upward_force`

This interval event reads `MotionCommand` through
`env.command_manager.get_term(command_name)`.

Command dependencies:

- `cfg.anchor_body_name`
- `anchor_pos_w`

Asset dependencies:

- `asset.body_names`
- `asset.data.body_pos_w`
- `asset.data.body_quat_w`
- `asset.data.body_lin_vel_w`
- `asset.data.GRAVITY_VEC_W`
- Optional `asset.instantaneous_wrench_composer`
- Optional `asset.permanent_wrench_composer`
- Fallback `asset.set_external_force_and_torque()`

Shared env attributes:

- Reads `env.envs_classes_mask["lying"]` when `use_motion_pose_range_mask=True`.
  This mask is owned by `MotionCommand`/`MotionCommandResetter` and should select the
  same recovery env class as delayed termination.
- Owns `env._fallen_upward_assist_timeout_stats`.

Behavior expectations:

- The assist force is only considered for envs selected by the active mask.
- The function computes fallen state from command reference height/orientation,
  current robot anchor pose/velocity, and projected gravity.
- Timeout statistics are used to decay the assist force scale across assisted resets.
- The function should remain a recovery aid; it must not change motion selection,
  reset policy, reference cache, or termination masks.

Performance contract:

- The function may run every control step. Keep tensor allocations and external-force
  writes scoped to the minimum required env subset.
- Debug printing must default off through `debug_steps=0` for production training.

## Extension Rules

- Add new event functions in `evt_functions.py`.
- Export public event functions through `evt_modules/__init__.py`.
- Document any new env attribute, simulator write, or cross-module dependency here
  and in `mdp/agent_contract.md`.
- If env class masks move to a different owner, update this contract and
  `tmt_modules/agent_contract.md` together.
