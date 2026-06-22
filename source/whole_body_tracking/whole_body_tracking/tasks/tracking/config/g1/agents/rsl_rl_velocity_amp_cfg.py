from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import MyPpoAlgorithmCfg
from whole_body_tracking.tasks.tracking.config.g1.velocity_amp_env_cfg import G1_AMP_ANCHOR_BODY_NAME, G1_AMP_BODY_NAMES

G1_LAFAN_WALK_AMP_MOTION_DIR = "/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/data/g1_amp/LAFAN_WALK"


@configclass
class VelocityModalTransformerActorCfg:
    class_name: str = "ActorModel"
    distribution_cfg: dict = {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
    }
    backbone: dict = {
        "class_name": "MyMLPModel",
        "hidden_dims": [1024, 512, 256],
        "activation": "swish",
        "obs_normalization": True,
    }

    # backbone: dict = {
    #     "class_name": "ModalityFusionTransformerModel",
    #     "d_model": 64,
    #     "token_layout": "concat",
    #     "num_layers": 4,
    #     "num_heads": 2,
    #     "dim_feedforward": 512,
    #     "activation": "relu",
    #     "transformer_activation": "relu",
    #     "transformer_norm_first": False,
    #     "use_type_embedding": False,
    #     "use_pos_embedding": False,
    #     "obs_normalization": True,
    #     "token_specs": [
    #         {"name": "prop", "group": "prop"},
    #         {"name": "lin_vel", "group": "base_velocity_cmd", "slice": [0, 2]},
    #         {"name": "yaw_rate", "group": "base_velocity_cmd", "slice": [2, 3]},
    #     ],
    #     "token_hidden_dims": [256, 128],
    #     "head_hidden_dims": [1024, 512],
    #     "head_activation": "relu",
    # }


@configclass
class VelocityModalCriticCfg:
    class_name: str = "MLPModel"
    hidden_dims: list = [1024, 512]
    activation: str = "relu"
    obs_normalization: bool = True
    distribution_cfg: dict = None


@configclass
class G1VelocityFlatAMPModalRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "g1_velocity_flat_amp_modal"
    run_name = "isaaclab_flat_velocity_amp_modal"
    empirical_normalization = True

    actor = VelocityModalTransformerActorCfg()
    critic = VelocityModalCriticCfg()

    obs_groups = {
        "actor": ["prop", "base_velocity_cmd"],
        "critic": ["policy"],
    }

    algorithm = MyPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        min_policy_std=0.055,
        plugins=[
            {
                "class_name": "AMPPlugin",
                "amp_reward_coef": 0.05,
                "amp_motion_files": G1_LAFAN_WALK_AMP_MOTION_DIR,
                "amp_task_reward_lerp": 0.9,
                "amp_discr_hidden_dims": [1024, 512],
                "amp_replay_buffer_size": 100000,
                "amp_body_names": G1_AMP_BODY_NAMES,
                "amp_anchor_name": G1_AMP_ANCHOR_BODY_NAME,
            }
        ],
    )
