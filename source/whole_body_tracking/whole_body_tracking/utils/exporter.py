# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import json
import numbers
import os
import torch
from typing import TYPE_CHECKING, cast

import onnx
from rsl_rl.models.mlp_model import MLPModel

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
else:
    ManagerBasedRLEnv = object


def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor: MLPModel,
    path: str,
    filename="policy.onnx",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(env, actor, verbose)
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


# class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
#     # OnnxExporter can merge multiple NNs through defining the forward function in the class
#     def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
#         super().__init__(actor_critic, normalizer, verbose)
#         cmd = env.command_manager.get_term("motion")
#         # Support both single-motion (MotionCommand/MotionCommandSonic) and multi-motion (MultiMotionCommand)
#         if hasattr(cmd, "motion"):
#             self.time_step_total = cmd.motion.time_step_total
#         elif hasattr(cmd, "dataloader"):
#             self.time_step_total = cmd.dataloader.time_step_total
#         else:
#             self.time_step_total = 0

#     def forward(self, x):
#         # 只返回actions输出
#         return self.actor(self.normalizer(x))

#     def _get_actor_obs_dim(self):
#         """Observation dim for ONNX export: support Sequential actor or custom (e.g. Actor_FSQVAE)."""
#         if hasattr(self.actor, "num_actor_obs"):
#             return self.actor.num_actor_obs
#         return self.actor[0].in_features

#     def export(self, path, filename):
#         self.to("cpu")
#         obs = torch.zeros(1, self._get_actor_obs_dim())
#         torch.onnx.export(
#             self,
#             obs,
#             os.path.join(path, filename),
#             export_params=True,
#             opset_version=11,
#             verbose=self.verbose,
#             input_names=["obs"],
#             output_names=["actions"],
#             dynamic_axes={},
#         )


class _OnnxMotionPolicyExporter(torch.nn.Module):
    # OnnxExporter can merge multiple NNs through defining the forward function in the class
    # more details can refer to isaaclab_rl.rsl_rl.exporter._OnnxPolicyExporter in your IsaacLab dir
    def __init__(self, env: ManagerBasedRLEnv, actor: MLPModel, verbose=False):
        # the actor details can be seen in rsl_rl.models.mlp_model.MLPModel in your conda environment
        super().__init__()
        assert not actor.is_recurrent, "The actor is recurrent, which is not supported for this ONNX export"
        self.actor = actor
        self.normalizer = actor.obs_normalizer
        self.mlp = actor.mlp
        self.verbose = verbose

    def forward(self, x):
        # the path from observation to actions
        return self.mlp(self.normalizer(x))

    def _get_actor_obs_dim(self) -> int:
        """Observation dim for ONNX export: support Sequential actor or custom (e.g. Actor_FSQVAE)."""
        dim = getattr(self.actor, "obs_dim", None)
        if dim is None:
            dim = self.actor.mlp[0].in_features
        # Some upstream types are loosely annotated; coerce to a concrete int for torch.zeros().
        if isinstance(dim, torch.Tensor):
            return int(dim.item())
        if isinstance(dim, torch.nn.Module):
            raise TypeError(f"Invalid obs_dim type: {type(dim)!r}")
        return int(cast(int, dim))

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros((1, self._get_actor_obs_dim()))
        torch.onnx.export(
            self,
            (obs,),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )


class _GroupedOnnxMotionPolicyExporter(torch.nn.Module):
    """ONNX wrapper for ActorModel/MyModel policies with TensorDict observations."""

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
    onnx_path = os.path.join(path, filename)

    # IsaacLab terms sometimes store scalars as plain floats; normalize to a JSON/CSV-friendly value.
    joint_pos_term = env.action_manager.get_term("joint_pos")
    action_scale = getattr(joint_pos_term, "_scale", None)
    if isinstance(action_scale, torch.Tensor):
        action_scale_value = action_scale[0].cpu().tolist() if action_scale.ndim > 0 else action_scale.item()
    elif isinstance(action_scale, (list, tuple)):
        action_scale_value = action_scale[0] if len(action_scale) > 0 else 0.0
    elif action_scale is None:
        action_scale_value = 0.0
    else:
        action_scale_value = float(action_scale)

    metadata = {
        "run_path": run_path,
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": env.scene["robot"].data.default_joint_pos_nominal.cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": env.observation_manager.active_terms["policy"],
        "action_scale": action_scale_value,
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
