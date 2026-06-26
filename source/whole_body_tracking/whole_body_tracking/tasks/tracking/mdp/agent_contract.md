# MDP Package Agent Contract

This package is the public MDP facade for the tracking task. Task configs should import
runtime functions and config-facing classes through `whole_body_tracking.tasks.tracking.mdp`
unless they intentionally need a submodule-internal type.

## Public Facade

`mdp/__init__.py` re-exports:

- IsaacLab built-in MDP functions from `isaaclab.envs.mdp`.
- Motion command APIs from `cmd_modules`.
- Event APIs from `evt_modules`.
- Observation APIs from `obs_modules`.
- Reward APIs from `rwd_modules`.
- Termination APIs from `tmt_modules`.

The facade is intentionally broad for IsaacLab config ergonomics. The real ownership
boundaries are defined by each `*_modules/agent_contract.md` file.

## Module Ownership

- `cmd_modules`: owns reference motion loading, timeline state, motion selection,
  robot reset from reference poses, aligned reference cache, motion command features,
  and debug visualization.
- `obs_modules`: owns observation term functions. It reads env, assets, and command
  state, and returns tensors. It does not mutate simulator or command state.
- `rwd_modules`: owns reward term functions and delayed-termination reward masking.
  Configs should use the wrapped exports from `rwd_modules`, not raw functions.
- `tmt_modules`: owns termination term functions, delayed termination installation,
  and `DelayedTerminationManager`.
- `evt_modules`: owns startup/interval event functions that intentionally mutate env,
  robot defaults, PhysX properties, or external force buffers.

## Cross-Module Rules

- Config files may import `mdp` facade symbols.
- Submodules should prefer explicit relative imports from their owning package.
- Observation, reward, termination, and event functions may read `MotionCommand`
  through `env.command_manager.get_term(command_name)`.
- Only `cmd_modules.motion_reset` should write robot root/joint state during command
  resets.
- Only event functions should perform event-manager side effects such as randomization
  or assist force application.
- Reward and termination configs should import from `rwd_modules` and `tmt_modules`
  wrappers so delayed-termination flags remain available.

## Shared Runtime State

The following env attributes are intentionally shared across modules:

- `env.envs_classes_mask`: written by `MotionCommand`, read by delayed termination
  and fallen upward assist. `envs_classes_mask["lying"]` selects recovery/lying envs.
  The mask is derived from `MotionCommandResetter.envs_classes_mask`, which is built
  from `cfg.commands.motion.envs_classes_ratio`.
- `env.termination_manager.delayed_termination_env_mask`: written by
  `DelayedTerminationManager`, read by termination wrappers.
- `env.termination_manager.delayed_termination_active_mask`: written by
  `DelayedTerminationManager.compute()`, read by reward wrappers.
- `env._fallen_upward_assist_timeout_stats`: owned by the fallen upward assist event.

Any new shared env attribute must be documented here and in the owning module
contract.

## Command Runtime Expectations

The command runtime is split into smaller modules:

- `MotionCommand` orchestrates loading, selection, reset, cache refresh, metrics, and
  visualization.
- `MotionCommandTimeline` owns only per-env motion cursor state.
- `MotionCommandResetter` is the only command module that writes robot root/joint
  state to simulation.
- `MotionReferenceCache` owns aligned reference body pose cache.
- `MotionSelectionPolicy` chooses motion frames and returns `MotionSelection`.
- `AdaptiveMotionSampler` owns adaptive failed-bin statistics and sampling metrics.

Observation, reward, termination, and event modules may read `MotionCommand`
properties, but they must not mutate timeline, sampler, resetter, motion source, or
reference cache state.

## Compatibility Notes

- Quaternion tensors use IsaacLab `wxyz` order.
- Env-indexed tensors should live on the simulation device unless an Isaac/PhysX API
  specifically requires CPU tensors.
- Per-env outputs from observation, reward, and termination terms must have first
  dimension `env.num_envs`.
- Public functions exported through `__all__` are config-facing API. Renaming or
  changing signatures can break Hydra/IsaacLab configs and checkpoints.
- `cfg.commands.motion.adaptive_uniform_ratio` still exists for compatibility, but
  current adaptive sampling uses `motion_ratio[0]` as its uniform-probability weight.
