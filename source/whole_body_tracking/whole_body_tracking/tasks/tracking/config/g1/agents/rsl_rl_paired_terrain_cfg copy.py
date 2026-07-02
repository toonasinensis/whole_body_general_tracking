from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from whole_body_tracking.tasks.tracking.config.g1.velocity_amp_env_cfg import G1_AMP_ANCHOR_BODY_NAME, G1_AMP_BODY_NAMES

from .rsl_rl_ppo_cfg import CriticCfg, MyPpoAlgorithmCfg
from .rsl_rl_velocity_amp_cfg import G1_AMP_MOTION_DIR


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
        # "encoder_mask_group": "aux_mask",
        # "aux_loss_mask_group": "aux_mask",
        "encoder": {
            "encoder_g1": {
                "encoder_groups": ["rbt_cmd_mf", "velcommand"],
                "hidden_dims": [1024, 512, 512],
                "activation": "swish",
            },
            "encoder_smpl": {
                "encoder_groups": ["velcommand"],  # 输入的 encoder_groups
                "hidden_dims": [1024, 512, 512],  # ecoder MLP 隐层
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
                "hidden_dims": [2048, 1024, 512, 512],
                "activation": "swish",
                "detach_latent": True,
                "outputs": ["actions"],
            },
            "g1_kin_decoder": {
                "decoder_groups": [],
                "hidden_dims": [4096, 2048, 1024, 512, 256],
                "activation": "swish",
                "detach_latent": False,
                "outputs": ["rbt_cmd_mf"],
            },
        },
        "activation": "swish",
        "obs_normalization": True,
    }


@configclass
class G1PairedTerrainHeightScanRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000000000000000
    save_interval = 500
    experiment_name = "g1_paired_terrain_height_scan"
    empirical_normalization = True

    algorithm = MyPpoAlgorithmCfg()
    actor = PairedTerrainActorShellCfg()
    critic = CriticCfg()

    # obs_groups = {
    #     "actor": ["prop", "velcommand", "terrain"],
    #     "critic": ["critic", "terrain"],
    # }

    obs_groups = {
        "actor": ["prop", "rbt_cmd_mf", "smpl_cmd_mf", "terrain", "velcommand", "aux_mask"],
        "critic": ["critic", "terrain"],
    }


@configclass
class G1MixedTerrainHeightScanAMPRunnerCfg(G1PairedTerrainHeightScanRunnerCfg):
    experiment_name = "g1_mixed_terrain_height_scan_amp"

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
                "amp_reward_coef": 0.3,
                "amp_motion_files": G1_AMP_MOTION_DIR,
                "amp_task_reward_lerp": 0.9,
                "amp_discr_hidden_dims": [1024, 512, 512],
                "amp_body_names": G1_AMP_BODY_NAMES,
                "amp_anchor_name": G1_AMP_ANCHOR_BODY_NAME,
                "amp_mask_group": "amp_mask",
                "amp_use_height_scan": True,
            }
        ],
    )
