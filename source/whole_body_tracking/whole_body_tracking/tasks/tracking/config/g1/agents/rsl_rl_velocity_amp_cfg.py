from pathlib import Path

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import MyPpoAlgorithmCfg
from whole_body_tracking.tasks.tracking.config.g1.velocity_amp_env_cfg import G1_AMP_ANCHOR_BODY_NAME, G1_AMP_BODY_NAMES

_WBT_ROOT = Path(__file__).resolve().parents[8]
G1_AMP_MOTION_DIR = str(_WBT_ROOT / "/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/data/g1_amp/WalkandRun_isaaclab")
# Keep the legacy constant name because mixed/heading AMP configs import it.


@configclass
class VelocityMLPActorCfg:
    class_name: str = "ActorModel"
    distribution_cfg: dict = {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
    }
    backbone: dict = {
        "class_name": "MyMLPModel",
        "hidden_dims": [512, 256],
        "activation": "swish",
        "obs_normalization": True,
    }


@configclass
class VelocityModalityFusionTransformerActorCfg:
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
        "use_type_embedding": True,
        "use_pos_embedding": False,
        "obs_normalization": True,
        "token_specs": [
            {"name": "character_state", "group": ["prop"]},  # , "base_velocity_cmd"]},
            {"name": "target_goal", "group": "base_velocity_cmd"},
        ],
        "token_hidden_dims": [512, 256],
        "head_hidden_dims": [512, 256],
        "head_activation": "relu",
    }


@configclass
class VelocityCriticCfg:
    class_name: str = "MLPModel"
    hidden_dims: list = [512, 256]
    activation: str = "relu"
    obs_normalization: bool = True
    distribution_cfg: dict = None


@configclass
class G1VelocityFlatAMPMLPRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000000
    save_interval = 200
    experiment_name = "MLP"
    run_name = "rsl_rl_velocity_amp_cfg"
    empirical_normalization = True

    actor = VelocityMLPActorCfg()
    critic = VelocityCriticCfg()

    obs_groups = {
        "actor": ["prop", "velcommand"],
        "critic": ["policy", "velcommand"],
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
                "amp_reward_coef": 0.2,
                "amp_motion_files": G1_AMP_MOTION_DIR,
                "amp_task_reward_lerp": 0.9,
                "amp_discr_hidden_dims": [512, 256],
                "amp_replay_buffer_size": 100000,
                "amp_body_names": G1_AMP_BODY_NAMES,
                "amp_anchor_name": G1_AMP_ANCHOR_BODY_NAME,
                "amp_use_height_scan": False,
            }
        ],
    )


@configclass
class G1VelocityFlatAMPModalityFusionRunnerCfg(G1VelocityFlatAMPMLPRunnerCfg):
    experiment_name = "amp_walk"
    run_name = "velocity_modal_transformer_1_token"

    actor = VelocityModalityFusionTransformerActorCfg()
    critic = VelocityCriticCfg()

    obs_groups = {
        "actor": ["prop", "velcommand"],
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
                "amp_reward_coef": 0.2,
                "amp_motion_files": G1_AMP_MOTION_DIR,
                "amp_task_reward_lerp": 0.9,
                "amp_discr_hidden_dims": [1024, 512],
                "amp_replay_buffer_size": 100000,
                "amp_body_names": G1_AMP_BODY_NAMES,
                "amp_anchor_name": G1_AMP_ANCHOR_BODY_NAME,
                "amp_use_height_scan": False,
            }
        ],
    )
