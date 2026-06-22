from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import CriticCfg, MyPpoAlgorithmCfg
from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1_AMP_ANCHOR_BODY_NAME, G1_AMP_BODY_NAMES

G1_LAFAN_WALK_AMP_MOTION_DIR = "/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/data/g1_amp/LAFAN_WALK"


@configclass
class HeadingTransformerActorCfg:
    class_name: str = "ActorModel"
    distribution_cfg: dict = {
        "class_name": "GaussianDistribution",
        "init_std": 0.25,
        "std_type": "scalar",
    }
    backbone: dict = {
        "class_name": "MyMLPModel",
        "hidden_dims": [1024, 512, 256],
        "activation": "swish",
        "obs_normalization": True,
    }
    # backbone: dict = {
    #     "class_name": "GoalConditionedTransformerModel",
    #     "d_model": 256,
    #     "num_layers": 4,
    #     "num_heads": 4,
    #     "mlp_ratio": 4,
    #     "activation": "gelu",
    #     "pool": "flatten",
    #     "obs_normalization": True,
    #     "prop_group": "prop",
    #     "command_group": "heading_cmd",
    #     "tracking_tokens": [],
    #     "head_hidden_dims": [512, 256],
    #     "action_head_init_scale": 0.01,
    # }


@configclass
class HeadingModalTransformerActorCfg:
    class_name: str = "ActorModel"
    distribution_cfg: dict = {
        "class_name": "GaussianDistribution",
        "init_std": 0.055,
        "std_type": "scalar",
    }
    backbone: dict = {
        "class_name": "ModalityFusionTransformerModel",
        "d_model": 64,
        "token_layout": "concat",
        "num_layers": 4,
        "num_heads": 2,
        "dim_feedforward": 512,
        "activation": "relu",
        "transformer_activation": "relu",
        "transformer_norm_first": False,
        "use_type_embedding": False,
        "use_pos_embedding": False,
        "obs_normalization": True,
        "token_specs": [
            {"name": "prop", "group": "prop"},
            {"name": "heading", "group": "heading_cmd", "slice": [0, 2]},
            {"name": "facing", "group": "heading_cmd", "slice": [2, 4]},
            {"name": "speed", "group": "heading_cmd", "slice": [4, 5]},
        ],
        "token_hidden_dims": [256, 128],
        "head_hidden_dims": [1024, 512],
        "head_activation": "relu",
    }


@configclass
class G1HeadingWalkAMPRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000000000000000
    save_interval = 500
    experiment_name = "g1_heading_walk_amp"
    empirical_normalization = True

    actor = HeadingTransformerActorCfg()
    critic = CriticCfg()

    obs_groups = {
        "actor": ["prop", "heading_cmd"],
        "critic": ["critic", "heading_cmd"],
    }

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
                "amp_reward_coef": 0.05,
                "amp_motion_files": G1_LAFAN_WALK_AMP_MOTION_DIR,
                "amp_task_reward_lerp": 0.9,
                "amp_discr_hidden_dims": [1024, 512, 256],
                "amp_replay_buffer_size": 200000,
                "amp_body_names": G1_AMP_BODY_NAMES,
                "amp_anchor_name": G1_AMP_ANCHOR_BODY_NAME,
            }
        ],
    )


@configclass
class G1HeadingWalkAMPModalRunnerCfg(G1HeadingWalkAMPRunnerCfg):
    experiment_name = "g1_heading_walk_amp_modal"
    run_name = "modal_transformer_default_reset"

    actor = HeadingModalTransformerActorCfg()

    obs_groups = {
        "actor": ["prop", "heading_cmd"],
        "critic": ["critic", "heading_cmd"],
    }

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
                "amp_reward_coef": 0.05,
                "amp_motion_files": G1_LAFAN_WALK_AMP_MOTION_DIR,
                "amp_task_reward_lerp": 0.9,
                "amp_discr_hidden_dims": [1024, 1024, 512, 256],
                "amp_replay_buffer_size": 200000,
                "amp_body_names": G1_AMP_BODY_NAMES,
                "amp_anchor_name": G1_AMP_ANCHOR_BODY_NAME,
            }
        ],
    )
