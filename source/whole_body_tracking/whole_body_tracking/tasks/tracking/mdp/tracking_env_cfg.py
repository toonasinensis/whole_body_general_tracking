from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from .cfg_actions import ActionsCfg
from .cfg_commands import CommandsCfg
from .cfg_curriculum import CurriculumCfg
from .cfg_events import EventCfg
from .cfg_observations import ObservationsCfg
from .cfg_rewards import RewardsCfg
from .cfg_scene import MySceneCfg
from .cfg_terminations import TerminationsCfg


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 10.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


__all__ = [
    "ActionsCfg",
    "CommandsCfg",
    "CurriculumCfg",
    "EventCfg",
    "ObservationsCfg",
    "RewardsCfg",
    "MySceneCfg",
    "TerminationsCfg",
    "TrackingEnvCfg",
]
