from __future__ import annotations

import copy
import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace

from isaaclab.managers import CommandTermCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from whole_body_tracking.tasks.tracking import mdp


@configclass
class ProfileObsRouteCfg:
    # Selects which command source feeds the shared "velcommand" observation group:
    # - "motion": use the motion-tracking command, i.e. anchor velocity in yaw frame.
    # - "base_velocity": use the generated velocity command from the velocity profile.
    velcommand_source: str = "motion"
    # Enables the scalar "aux_mask" observation for this profile.
    #
    # Mechanism:
    # 1. aux_mask_domain_names() collects domain names whose profile sets this to True.
    # 2. mixed_terrain_env_cfg.py passes those names to mdp.domain_aux_mask().
    # 3. At runtime, mdp.domain_aux_mask() outputs a float tensor with shape [num_envs, 1]:
    #    1.0 for envs in enabled domains, 0.0 for the rest.
    # 4. The policy config uses that observation group as encoder/aux-loss mask.
    #
    # Keep this as bool in config. The float conversion belongs to the observation
    # function because only the policy input tensor needs numeric 0/1 values.
    aux_mask_enabled: bool = True
    # Enables the scalar "amp_mask" observation for this profile.
    #
    # Mechanism:
    # 1. collect_domain_profile_routes() converts profile-level booleans to concrete domain names.
    # 2. mixed_terrain_env_cfg.py passes those names to a runtime mask observation.
    # 3. At runtime, the observation outputs a float tensor with shape [num_envs, 1]:
    #    1.0 for envs in enabled domains, 0.0 for the rest.
    # 4. AMPPlugin reads that tensor, turns it into a bool mask, and applies AMP only to masked envs.
    #
    # Algorithm details such as expert data, reward coefficients, and replay buffers stay in runner/plugin config.
    amp_mask_enabled: bool = False


@dataclass(frozen=True)
class DomainProfileRoutes:
    velocity_domain_names: tuple[str, ...]
    aux_mask_domain_names: tuple[str, ...]
    amp_mask_domain_names: tuple[str, ...]
    profile_reset_cfgs: dict[str, object]


def domains_for_profile(domains: Sequence[object], task_profile: str) -> tuple[str, ...]:
    return tuple(domain.name for domain in domains if getattr(domain, "task_profile", "wbc_tracking") == task_profile)


def domain_names_by_obs_source(domains: Sequence[object], profiles: dict[str, object], source: str) -> tuple[str, ...]:
    names = []
    for domain in domains:
        profile = profiles[getattr(domain, "task_profile", "wbc_tracking")]
        route = getattr(profile, "obs_route_cfg", ProfileObsRouteCfg())
        if route.velcommand_source == source:
            names.append(domain.name)
    return tuple(names)


def aux_mask_domain_names(domains: Sequence[object], profiles: dict[str, object]) -> tuple[str, ...]:
    # Convert profile-level aux_mask_enabled flags into concrete domain names.
    # The runtime observation later turns this domain-name list into a per-env 0/1 mask.
    names = []
    for domain in domains:
        profile = profiles[getattr(domain, "task_profile", "wbc_tracking")]
        route = getattr(profile, "obs_route_cfg", ProfileObsRouteCfg())
        if bool(route.aux_mask_enabled):
            names.append(domain.name)
    return tuple(names)


def amp_mask_domain_names(domains: Sequence[object], profiles: dict[str, object]) -> tuple[str, ...]:
    # Convert profile-level amp_mask_enabled flags into concrete domain names.
    # The runtime observation later turns this domain-name list into a per-env 0/1 mask.
    names = []
    for domain in domains:
        profile = profiles[getattr(domain, "task_profile", "wbc_tracking")]
        route = getattr(profile, "obs_route_cfg", ProfileObsRouteCfg())
        if bool(route.amp_mask_enabled):
            names.append(domain.name)
    return tuple(names)


def profile_reset_cfgs_for_domains(domains: Sequence[object], profiles: dict[str, object]) -> dict[str, object]:
    out = {}
    for domain in domains:
        task_profile = getattr(domain, "task_profile", "wbc_tracking")
        if task_profile in out:
            continue
        profile = profiles[task_profile]
        reset_cfg = _reset_events_cfg(getattr(profile, "events_cfg", None))
        if reset_cfg is not None:
            out[task_profile] = reset_cfg
    return out


def collect_domain_profile_routes(domains: Sequence[object], profiles: dict[str, object]) -> DomainProfileRoutes:
    _validate_profiles(domains, profiles)
    return DomainProfileRoutes(
        velocity_domain_names=domain_names_by_obs_source(domains, profiles, "base_velocity"),
        aux_mask_domain_names=aux_mask_domain_names(domains, profiles),
        amp_mask_domain_names=amp_mask_domain_names(domains, profiles),
        profile_reset_cfgs=profile_reset_cfgs_for_domains(domains, profiles),
    )


def apply_domain_profiles(env_cfg, domains: Sequence[object], profiles: dict[str, object]) -> None:
    _validate_profiles(domains, profiles)
    _clear_profile_commands(env_cfg.commands, profiles)
    if not hasattr(env_cfg, "events"):
        env_cfg.events = SimpleNamespace()
    _clear_profile_events(env_cfg.events, profiles)
    if not hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum = SimpleNamespace()
    _clear_profile_curriculums(env_cfg.curriculum, profiles)
    _clear_terms(env_cfg.rewards, RewTerm)
    _clear_terms(env_cfg.terminations, DoneTerm)
    for task_profile, profile in profiles.items():
        domain_names = domains_for_profile(domains, task_profile)
        if not domain_names:
            continue
        _apply_command_cfg(env_cfg.commands, getattr(profile, "commands_cfg", None))
        _apply_event_cfg(env_cfg.events, task_profile, getattr(profile, "events_cfg", None), domain_names)
        _apply_curriculum_cfg(
            env_cfg.curriculum,
            task_profile,
            getattr(profile, "curriculum_cfg", None),
            domain_names,
        )
        _apply_reward_cfg(env_cfg.rewards, task_profile, getattr(profile, "rewards_cfg", None), domain_names)
        _apply_termination_cfg(
            env_cfg.terminations,
            task_profile,
            getattr(profile, "terminations_cfg", None),
            domain_names,
        )


def _validate_profiles(domains: Sequence[object], profiles: dict[str, object]) -> None:
    missing = sorted(
        {
            getattr(domain, "task_profile", "wbc_tracking")
            for domain in domains
            if getattr(domain, "task_profile", "wbc_tracking") not in profiles
        }
    )
    if missing:
        raise ValueError(f"Missing mixed-domain task profiles {missing}. Available profiles: {sorted(profiles)}")


def _iter_cfg_items(cfg, term_type: type | None = None):
    if cfg is None:
        return
    seen = set()
    for source in (vars(cfg), vars(type(cfg))):
        for name, value in source.items():
            if name in seen:
                continue
            seen.add(name)
            if name.startswith("_") or value is None:
                continue
            if term_type is not None and not isinstance(value, term_type):
                continue
            yield name, value


def _iter_cfg_term_names(cfg, term_type: type) -> tuple[str, ...]:
    if cfg is None:
        return ()
    names = []
    for source in (vars(cfg), vars(type(cfg))):
        for name, value in source.items():
            if name.startswith("_") or name in names:
                continue
            if isinstance(value, term_type):
                names.append(name)
    return tuple(names)


def _clear_terms(cfg, term_type: type) -> None:
    for name in _iter_cfg_term_names(cfg, term_type):
        setattr(cfg, name, None)


def _clear_profile_commands(target_cfg, profiles: dict[str, object]) -> None:
    command_names = set()
    for profile in profiles.values():
        for name, _ in _iter_cfg_items(getattr(profile, "commands_cfg", None), CommandTermCfg):
            command_names.add(name)
    for name in command_names:
        setattr(target_cfg, name, None)


def _clear_profile_events(target_cfg, profiles: dict[str, object]) -> None:
    event_names = set()
    for profile_name, profile in profiles.items():
        for name, term in _iter_cfg_items(getattr(profile, "events_cfg", None), EventTerm):
            if _event_is_registered_to_manager(term):
                event_names.add(f"{profile_name}_{name}")
    for name in event_names:
        setattr(target_cfg, name, None)


def _clear_profile_curriculums(target_cfg, profiles: dict[str, object]) -> None:
    curriculum_names = set()
    for profile_name, profile in profiles.items():
        for name, _ in _iter_cfg_items(getattr(profile, "curriculum_cfg", None), CurrTerm):
            curriculum_names.add(f"{profile_name}_{name}")
    for name in curriculum_names:
        setattr(target_cfg, name, None)


def _apply_command_cfg(target_cfg, commands_cfg) -> None:
    for name, value in _iter_cfg_items(commands_cfg, CommandTermCfg):
        setattr(target_cfg, name, copy.deepcopy(value))


def _apply_event_cfg(target_cfg, profile_name: str, events_cfg, domain_names: Sequence[str]) -> None:
    for name, term in _iter_cfg_items(events_cfg, EventTerm):
        if not _event_is_registered_to_manager(term):
            continue
        setattr(target_cfg, f"{profile_name}_{name}", _domain_masked_event_term(term, domain_names))


def _apply_curriculum_cfg(target_cfg, profile_name: str, curriculum_cfg, domain_names: Sequence[str]) -> None:
    for name, term in _iter_cfg_items(curriculum_cfg, CurrTerm):
        setattr(target_cfg, f"{profile_name}_{name}", _domain_masked_curriculum_term(term, domain_names))


def _apply_reward_cfg(target_cfg, profile_name: str, rewards_cfg, domain_names: Sequence[str]) -> None:
    for name, term in _iter_cfg_items(rewards_cfg, RewTerm):
        setattr(target_cfg, f"{profile_name}_{name}", _domain_masked_reward_term(term, domain_names))


def _apply_termination_cfg(target_cfg, profile_name: str, terminations_cfg, domain_names: Sequence[str]) -> None:
    for name, term in _iter_cfg_items(terminations_cfg, DoneTerm):
        setattr(target_cfg, f"{profile_name}_{name}", _domain_masked_termination_term(term, domain_names))


def _term_func_is_class(func) -> bool:
    return inspect.isclass(func)


def _event_is_registered_to_manager(term: EventTerm) -> bool:
    return getattr(term, "mode", None) != "reset"


def _reset_events_cfg(events_cfg):
    reset_terms = {
        name: copy.deepcopy(term)
        for name, term in _iter_cfg_items(events_cfg, EventTerm)
        if getattr(term, "mode", None) == "reset"
    }
    if not reset_terms:
        return None
    return SimpleNamespace(**reset_terms)


def _domain_masked_reward_term(term: RewTerm, domain_names: Sequence[str]) -> RewTerm:
    out = copy.deepcopy(term)
    if _term_func_is_class(out.func):
        out.params["domain_command_name"] = "motion"
        out.params["enabled_domain_names"] = tuple(domain_names)
        return out
    source_func = out.func
    source_params = out.params
    out.func = mdp.domain_masked_reward
    out.params = {
        "source_func": source_func,
        "source_params": source_params,
        "domain_command_name": "motion",
        "enabled_domain_names": tuple(domain_names),
    }
    return out


def _domain_masked_event_term(term: EventTerm, domain_names: Sequence[str]) -> EventTerm:
    out = copy.deepcopy(term)
    source_func = out.func
    source_params = out.params
    out.func = mdp.domain_masked_event
    out.params = {
        "source_func": source_func,
        "source_params": source_params,
        "domain_command_name": "motion",
        "enabled_domain_names": tuple(domain_names),
    }
    return out


def _domain_masked_curriculum_term(term: CurrTerm, domain_names: Sequence[str]) -> CurrTerm:
    out = copy.deepcopy(term)
    source_func = out.func
    source_params = out.params
    out.func = mdp.domain_masked_curriculum
    out.params = {
        "source_func": source_func,
        "source_params": source_params,
        "domain_command_name": "motion",
        "enabled_domain_names": tuple(domain_names),
    }
    return out


def _domain_masked_termination_term(term: DoneTerm, domain_names: Sequence[str]) -> DoneTerm:
    out = copy.deepcopy(term)
    source_func = out.func
    source_params = out.params
    out.func = mdp.domain_masked_termination
    out.params = {
        "source_func": source_func,
        "source_params": source_params,
        "domain_command_name": "motion",
        "enabled_domain_names": tuple(domain_names),
    }
    return out
