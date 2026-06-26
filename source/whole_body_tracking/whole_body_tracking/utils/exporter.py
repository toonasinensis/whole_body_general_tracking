# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import json
import numbers
import os
import torch

import onnx

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter

from whole_body_tracking.tasks.tracking.mdp import MotionCommand


def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


def export_grouped_motion_policy_as_onnx(
    env,
    actor: torch.nn.Module,
    path: str,
    filename="policy.onnx",
    verbose=False,
    metadata: dict | None = None,
) -> str:
    """Export a policy that consumes named observation groups."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _GroupedOnnxMotionPolicyExporter(env, actor, verbose)
    onnx_path = policy_exporter.export(path, filename)
    if metadata is not None:
        export_metadata = {
            "input_names": policy_exporter.input_names,
            "input_groups": policy_exporter.input_names,
            "input_shapes": {name: list(shape) for name, shape in policy_exporter.input_shapes.items()},
        }
        export_metadata.update(metadata)
        append_onnx_metadata(onnx_path, export_metadata)
    return onnx_path


def resolve_policy_observation_groups(actor: torch.nn.Module, obs) -> list[str]:
    """Return observation groups consumed by the policy actor in export order."""
    module = getattr(actor, "backbone", actor)
    groups = getattr(module, "obs_groups", None)
    if groups is None:
        groups = getattr(actor, "obs_groups", None)
    if groups is None:
        return list(obs.keys())

    groups = list(groups)
    obs_keys = set(obs.keys())
    missing = [name for name in groups if name not in obs_keys]
    if missing:
        raise KeyError(
            f"Policy expects observation groups {missing}, but env observations only have {list(obs.keys())}."
        )
    return groups


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)
        cmd: MotionCommand = env.command_manager.get_term("motion")
        
        # 只需要time_step_total用于归一化
        self.time_step_total = cmd.motion.time_step_total

    def forward(self, x):
        # 只返回actions输出
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.actor[0].in_features)
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )


class _GroupedOnnxMotionPolicyExporter(torch.nn.Module):
    """ONNX wrapper for policies with TensorDict observations."""

    def __init__(self, env, actor: torch.nn.Module, verbose=False):
        super().__init__()
        assert not actor.is_recurrent, "The actor is recurrent, which is not supported for this ONNX export"
        self.actor = copy.deepcopy(actor).cpu().eval()
        self.verbose = verbose
        obs = env.get_observations().detach().cpu()
        self.input_names = resolve_policy_observation_groups(actor, obs)
        self.input_shapes = {name: tuple(obs[name].shape[1:]) for name in self.input_names}

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        from tensordict import TensorDict

        batch_size = inputs[0].shape[0]
        obs = TensorDict({name: value for name, value in zip(self.input_names, inputs)}, batch_size=[batch_size])
        output = self.actor(obs)
        actions = output["actions"] if isinstance(output, dict) else output
        input_guard = sum(value.reshape(batch_size, -1).sum(dim=1, keepdim=True) for value in inputs)
        return actions + input_guard * 0.0

    def export(self, path, filename) -> str:
        self.to("cpu")
        dummy_inputs = tuple(torch.zeros((1, *self.input_shapes[name])) for name in self.input_names)
        onnx_path = os.path.join(path, filename)
        torch.onnx.export(
            self,
            dummy_inputs,
            onnx_path,
            export_params=True,
            opset_version=17,
            verbose=self.verbose,
            input_names=self.input_names,
            output_names=["actions"],
            dynamic_axes={name: {0: "batch"} for name in self.input_names} | {"actions": {0: "batch"}},
        )
        return onnx_path


def collect_observation_terms_metadata(base_env, observation_groups: list[str]) -> dict:
    obs_manager = base_env.observation_manager
    active_terms = obs_manager.active_terms
    term_dims = obs_manager.group_obs_term_dim
    concatenate = obs_manager.group_obs_concatenate
    term_cfgs = getattr(obs_manager, "_group_obs_term_cfgs", {})

    out = {}
    for group_name in observation_groups:
        names = list(active_terms.get(group_name, []))
        dims = list(term_dims.get(group_name, []))
        cfgs = list(term_cfgs.get(group_name, []))
        terms = []
        for idx, name in enumerate(names):
            shape = list(dims[idx]) if idx < len(dims) else []
            cfg = cfgs[idx] if idx < len(cfgs) else None
            history_length = int(getattr(cfg, "history_length", 0)) if cfg is not None else 0
            flatten_history_dim = bool(getattr(cfg, "flatten_history_dim", True)) if cfg is not None else True
            base_shape = list(shape)
            if history_length > 0:
                if flatten_history_dim and len(shape) == 1 and shape[0] % history_length == 0:
                    base_shape = [shape[0] // history_length]
                elif not flatten_history_dim and len(shape) >= 1 and shape[0] == history_length:
                    base_shape = shape[1:]
            terms.append(
                {
                    "name": name,
                    "shape": shape,
                    "base_shape": base_shape,
                    "history_length": history_length,
                    "flatten_history_dim": flatten_history_dim,
                }
            )
        out[group_name] = {
            "concatenate_terms": bool(concatenate.get(group_name, True)),
            "terms": terms,
        }
    return out


def _metadata_to_jsonable(value):
    """Convert common numpy/torch values into JSON-serializable Python values."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is available in normal IsaacLab runs.
        np = None

    if torch.is_tensor(value):
        value = value.detach().cpu()
        return value.tolist() if value.ndim > 0 else value.item()
    if isinstance(value, torch.Size):
        return list(value)
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(_metadata_to_jsonable(k)): _metadata_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_metadata_to_jsonable(item) for item in value]
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return float(value)
    return value


def append_onnx_metadata(onnx_path: str, metadata: dict) -> None:
    """Append JSON-friendly metadata to an ONNX file."""
    model = onnx.load(onnx_path)
    existing = {entry.key: entry for entry in model.metadata_props}
    for key, value in metadata.items():
        entry = existing.get(key)
        if entry is None:
            entry = model.metadata_props.add()
            entry.key = key
        value = _metadata_to_jsonable(value)
        entry.value = json.dumps(value) if isinstance(value, (dict, list, tuple)) else str(value)
    onnx.save(model, onnx_path)


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    # TODO modify the attach_onnx_metadata function
    onnx_path = os.path.join(path, filename)
    metadata = {
        "run_path": run_path,
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": env.scene["robot"].data.default_joint_pos_nominal.cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": env.observation_manager.active_terms["policy"],
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
