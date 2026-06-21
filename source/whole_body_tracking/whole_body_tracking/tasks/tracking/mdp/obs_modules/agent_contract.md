# Observation Modules Agent Contract

`obs_modules` owns IsaacLab observation term functions for the tracking MDP. Public
API is the `__all__` list in `obs_modules/__init__.py`.

## Public Exports

Robot/reference state observations:

- `robot_anchor_ori_w`
- `robot_anchor_lin_vel_w`
- `robot_anchor_ang_vel_w`
- `robot_body_pos_b`
- `robot_body_ori_b`

Motion command observations:

- `motion_command`
- `motion_joint_pos`
- `motion_joint_vel`
- `motion_anchor_pos_b`
- `motion_anchor_ori_b`
- `motion_anchor_lin_vel_b`
- `motion_anchor_ang_vel_b`
- `motion_anchor_project_gravity`
- `motion_anchor_pos_z`

Multi-future observations:

- `motion_joint_pos_mf`
- `motion_joint_vel_mf`
- `motion_anchor_ori_b_mf`
- `motion_anchor_z_mf`
- `smpl_joints_local_multi_future`
- `smpl_root_quat_w_dif_l_multi_future`

AMP observations:

- `amp_robot_body_pos_b`
- `amp_robot_body_ori_b`
- `amp_robot_body_lin_vel_b`
- `amp_robot_body_ang_vel_b`

## Ownership

Observation functions read env state and return tensors. They do not own persistent
training state except the private `_AMP_BODY_ID_CACHE` used to cache asset body index
resolution.

Allowed reads:

- `env.command_manager.get_term(command_name)` returning `MotionCommand`.
- `env.scene[asset_name]` and `asset.data`.
- IsaacLab math helpers.

Forbidden side effects:

- Do not mutate `MotionCommand`, timeline, resetter, sampler, metrics, env extras, or
  simulator state.
- Do not write external forces or randomize asset properties.
- Do not install or modify termination managers.

## Tensor Contract

- Return tensors are env-major: first dimension is `env.num_envs`.
- Observation functions must keep tensors on the command/asset simulation device.
- Flattened multi-body or multi-future observations must preserve the layout expected
  by `cfg_observations.py` and downstream policies.
- Quaternion-derived orientation observations use 6D rotation layout from the first
  two matrix columns unless a function name explicitly says otherwise.

## Dependencies on `MotionCommand`

The observation functions depend on the following command properties:

- Current reference: `joint_pos`, `joint_vel`, `anchor_pos_w`, `anchor_quat_w`,
  `anchor_lin_vel_b`, `anchor_ang_vel_b`, `anchor_project_gravity`, `anchor_pos_z`.
- Future reference: `joint_pos_future`, `joint_vel_future`, `anchor_quat_w_future`,
  `anchor_pos_w_future`, `num_future_frames`.
- Robot state views: `robot_anchor_pos_w`, `robot_anchor_quat_w`,
  `robot_anchor_lin_vel_w`, `robot_anchor_ang_vel_w`, `robot_body_pos_w`,
  `robot_body_quat_w`.
- SMPL state: `has_smpl_data`, `smpl_joints_local_multi_future`,
  `smpl_root_quat_w_dif_l_multi_future`.

If these command properties change shape or semantics, update this contract and
`cfg_observations.py` together.

## Performance Boundary

Observation groups are computed by IsaacLab when active in env config, even if a
runner later ignores some TensorDict keys. Avoid keeping expensive groups such as
SMPL or multi-future observations active unless the actor, critic, plugin, or teacher
actually consumes them.
