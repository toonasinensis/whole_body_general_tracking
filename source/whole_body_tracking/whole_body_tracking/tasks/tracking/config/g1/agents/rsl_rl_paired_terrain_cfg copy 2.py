from pathlib import Path

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from whole_body_tracking.tasks.tracking.config.g1.velocity_amp_env_cfg import G1_AMP_ANCHOR_BODY_NAME, G1_AMP_BODY_NAMES

from .rsl_rl_ppo_cfg import CriticCfg, MyPpoAlgorithmCfg

_WBT_ROOT = Path(__file__).resolve().parents[8]


G1_AMP_MOTION_DIR = str(_WBT_ROOT / "data/mesh_flatwbc_model151000_z10cm/tracking_npz_data_amp")
G1_AMP_MOTION_DIR = str(_WBT_ROOT / "/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/data/g1_amp/WalkandRun_isaaclab")


@configclass
class PairedTerrainActorShellCfg:
    class_name: str = "ActorModel"
    distribution_cfg: dict = {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
    }
    # backbone: dict = {
    #     "class_name": "MyMLPModel",
    #     "hidden_dims": [1024, 512, 256],
    #     "activation": "swish",
    #     "obs_normalization": True,
    # }

    backbone: dict = {
        "class_name": "MyModel",
        "main_encoder": "encoder_g1",
        "encoder_mask_group": "aux_mask",
        # "aux_loss_mask_group": "aux_mask",
        "encoder": {
            "encoder_g1": {
                "encoder_groups": ["wbc_cmd"],
                "hidden_dims": [512, 256],
                "activation": "swish",
            },
            "encoder_smpl": {
                "encoder_groups": ["velcommand"],  # 输入的 encoder_groups
                "hidden_dims": [512, 256],  # ecoder MLP 隐层
                "activation": "swish",
            },
        },
        # "loss": {
        #     "token": 1.0,  # token loss 权重: MSE(encoder(token), token_target) * weight，0.0 = 关闭
        #     "recon": 0.010,
        #     "re_encode": (
        #         1.0
        #     ),
        # },
        "fsq": {"num_fsq_levels": 32, "fsq_level_list": 32, "max_num_tokens": 2},
        "latent_dim": 128,
        "decoder": {
            "action_decoder": {
                "decoder_groups": ["prop", "terrain"],
                "hidden_dims": [1024, 512, 256],
                "activation": "swish",
                "detach_latent": True,
                "outputs": ["actions"],
            },
            # "g1_kin_decoder": {
            #     "decoder_groups": [],
            #     "hidden_dims": [ 1024, 512, 256],
            #     "activation": "swish",
            #     "detach_latent": False,
            #     "outputs": ["wbc_cmd"],
            # },
        },
        "activation": "swish",
        "obs_normalization": True,
    }


@configclass
class PairedTerrainModalityFusionActorCfg:
    class_name: str = "ActorModel"
    distribution_cfg: dict = {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
    }
    backbone: dict = {
        "class_name": "ModalityFusionTransformerModel",
        "d_model": 256,
        "num_layers": 2,
        "num_heads": 2,
        "dim_feedforward": 512,
        "dropout": 0.0,
        "activation": "relu",
        "transformer_activation": "relu",
        "transformer_norm_first": False,
        "token_layout": "separate",
        "use_type_embedding": True,
        "use_pos_embedding": False,
        "obs_normalization": True,
        "token_specs": [
            {"name": "prop", "group": "prop"},
            # {"name": "terrain", "group": "terrain"},
            # {"name": "wbc", "group": "wbc_cmd"},
            {"name": "vel", "group": "velcommand"},
        ],
        # "modality_mask_specs": {
        #     # WBC command is valid in tracking/WBC domains.
        #     "wbc": {"group": "aux_mask", "index": 0, "threshold": 0.5},
        #     # Velocity command is valid in flat_velocity / velocity_terrain domains.
        #     "vel": {"group": "vel_task_mask", "index": 0, "threshold": 0.5},
        # },
        "token_hidden_dims": [512, 256],
        "head_hidden_dims": [512, 256],
        "head_activation": "relu",
    }


@configclass
class CriticC2fg:
    class_name: str = "MLPModel"
    hidden_dims: list = [1024, 512, 256]
    activation: str = "swish"
    obs_normalization: bool = True
    distribution_cfg: dict = None


@configclass
class ModalityFusionCriticCfg:
    class_name: str = "MLPModel"
    hidden_dims: list = [1024, 512]
    activation: str = "relu"
    obs_normalization: bool = True
    distribution_cfg: dict = None


@configclass
class G1PairedTerrainHeightScanRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000000000000000
    save_interval = 500
    experiment_name = "g1_paired_terrain_height_scan"
    empirical_normalization = True

    algorithm = MyPpoAlgorithmCfg()
    actor = PairedTerrainActorShellCfg()
    critic = CriticC2fg()

    obs_groups = {
        "actor": ["prop", "velcommand", "wbc_cmd", "vel_task_mask", "aux_mask", "terrain"],
        "critic": [
            "critic",
            "velcommand",
            "wbc_cmd",
            "vel_task_mask",
            "terrain",
        ],
    }


@configclass
class G1MixedTerrainHeightScanAMPRunnerCfg(G1PairedTerrainHeightScanRunnerCfg):
    experiment_name = "g1_new_walk"

    algorithm = MyPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        plugins=[
            {
                "class_name": "AMPPlugin",
                "amp_reward_coef": 0.2,
                "amp_motion_files": G1_AMP_MOTION_DIR,
                "amp_task_reward_lerp": 0.9,
                "amp_discr_hidden_dims": [1024, 512, 512],
                "amp_body_names": G1_AMP_BODY_NAMES,
                "amp_anchor_name": G1_AMP_ANCHOR_BODY_NAME,
                "amp_mask_group": "amp_mask",
                "amp_use_height_scan": False,
            }
        ],
    )


@configclass
class G1MixedTerrainHeightScanAMPModalityFusionRunnerCfg(G1MixedTerrainHeightScanAMPRunnerCfg):
    experiment_name = "amp_walk"
    run_name = "g1_new_walk"

    actor = PairedTerrainModalityFusionActorCfg()
    critic = ModalityFusionCriticCfg()

    algorithm = MyPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        min_policy_std=[0.05] * 29,
        plugins=[
            {
                "class_name": "AMPPlugin",
                "amp_reward_coef": 0.5,
                "amp_motion_files": G1_AMP_MOTION_DIR,
                "amp_task_reward_lerp": 0.9,
                "amp_discr_hidden_dims": [1024, 512],
                "amp_body_names": G1_AMP_BODY_NAMES,
                "amp_anchor_name": G1_AMP_ANCHOR_BODY_NAME,
                "amp_mask_group": "amp_mask",
                "amp_use_height_scan": False,
            }
        ],
    )
