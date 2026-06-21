from __future__ import annotations

import whole_body_tracking.tasks.tracking.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import obs_modules as obs


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        motion_joint_pos = ObsTerm(
            func=obs.motion_joint_pos, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        motion_joint_vel = ObsTerm(
            func=obs.motion_joint_vel, params={"command_name": "motion"}, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        motion_anchor_lin_vel_b = ObsTerm(
            func=obs.motion_anchor_lin_vel_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        motion_anchor_ang_vel_b = ObsTerm(
            func=obs.motion_anchor_ang_vel_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        motion_anchor_project_gravity = ObsTerm(
            func=obs.motion_anchor_project_gravity,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        motion_anchor_pos_z = ObsTerm(
            func=obs.motion_anchor_pos_z, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        motion_anchor_ori_b = ObsTerm(func=obs.motion_anchor_ori_b, params={"command_name": "motion"})

        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 10
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PropCfg(ObsGroup):
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 10
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class RbtCmdMfCfg(ObsGroup):
        motion_joint_pos_multi_future = ObsTerm(
            func=obs.motion_joint_pos_mf, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        motion_joint_vel_multi_future = ObsTerm(
            func=obs.motion_joint_vel_mf, params={"command_name": "motion"}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        motion_anchor_ori_b_multi_future = ObsTerm(
            func=obs.motion_anchor_ori_b_mf, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ZRbtCmdMfCfg(ObsGroup):
        motion_joint_pos_multi_future = ObsTerm(
            func=obs.motion_joint_pos_mf, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        motion_joint_vel_multi_future = ObsTerm(
            func=obs.motion_joint_vel_mf, params={"command_name": "motion"}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        motion_anchor_ori_b_multi_future = ObsTerm(
            func=obs.motion_anchor_ori_b_mf, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        motion_anchor_z_multi_future = ObsTerm(
            func=obs.motion_anchor_z_mf, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class SmplCmdMfCfg(ObsGroup):
        smpl_joints_local_multi_future = ObsTerm(
            func=obs.smpl_joints_local_multi_future,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        smpl_root_quat_w_dif_l_multi_future = ObsTerm(
            func=obs.smpl_root_quat_w_dif_l_multi_future,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=obs.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=obs.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=obs.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=obs.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

    @configclass
    class AmpCfg(ObsGroup):
        body_pos_b = ObsTerm(
            func=obs.amp_robot_body_pos_b,
            params={"asset_name": "robot", "anchor_body_name": "", "body_names": ()},
        )
        body_ori_b = ObsTerm(
            func=obs.amp_robot_body_ori_b,
            params={"asset_name": "robot", "anchor_body_name": "", "body_names": ()},
        )
        body_lin_vel_b = ObsTerm(
            func=obs.amp_robot_body_lin_vel_b,
            params={"asset_name": "robot", "anchor_body_name": "", "body_names": ()},
        )
        body_ang_vel_b = ObsTerm(
            func=obs.amp_robot_body_ang_vel_b,
            params={"asset_name": "robot", "anchor_body_name": "", "body_names": ()},
        )

        def __post_init__(self):
            self.history_length = 1
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()
    rbt_cmd_mf: RbtCmdMfCfg = RbtCmdMfCfg()
    zrbt_cmd_mf: ZRbtCmdMfCfg = ZRbtCmdMfCfg()
    smpl_cmd_mf: SmplCmdMfCfg = SmplCmdMfCfg()
    prop: PropCfg = PropCfg()
    amp: AmpCfg | None = None


__all__ = ["ObservationsCfg"]
