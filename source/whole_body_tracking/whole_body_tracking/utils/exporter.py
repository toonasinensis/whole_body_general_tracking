# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
from typing import cast

import onnx

from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.models.mlp_model import MLPModel

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
