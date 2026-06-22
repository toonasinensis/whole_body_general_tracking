import gymnasium as gym

from . import agents, flat_env_cfg, heading_env_cfg, mixed_terrain_env_cfg, paired_terrain_env_cfg, velocity_amp_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="TR-G1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="FM-G1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatFMPPORunnerCfg",
    },
)


gym.register(
    id="AMP-G1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatAMPEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatAMPRunnerCfg",
    },
)

gym.register(
    id="RLBC-G1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatAMPEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRLBCDRunnerCfg",
    },
)

gym.register(
    id="TerrainPairMulti-G1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": paired_terrain_env_cfg.G1PairedTerrainEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_paired_terrain_cfg:G1PairedTerrainHeightScanRunnerCfg",
    },
)

gym.register(
    id="TerrainPairMixed-G1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": mixed_terrain_env_cfg.G1MixedFlatMeshEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MixedFlatMeshRunnerCfg",
    },
)

gym.register(
    id="HeadingWalkAMP-G1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": heading_env_cfg.G1HeadingWalkAMPEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_heading_cfg:G1HeadingWalkAMPRunnerCfg",
    },
)

gym.register(
    id="HeadingWalkAMP-Modal-G1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": velocity_amp_env_cfg.G1VelocityFlatAMPModalEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_velocity_amp_cfg:G1VelocityFlatAMPModalRunnerCfg",
    },
)

gym.register(
    id="Distill-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distill_cfg:RslRlDistillRunnerCfg",
    },
)

gym.register(
    id="LatentDistill-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_latent_distill_cfg:G1FlatLatentDistillRunnerCfg",
    },
)

gym.register(
    id="MLPDistill-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_latent_distill_cfg:G1FlatMLPDistillRunnerCfg",
    },
)

gym.register(
    id="LatentResidual-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_latent_residual_cfg:G1FlatLatentResidualPPORunnerCfg",
    },
)
