import os

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg  # noqa: F401

from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1_AMP_ANCHOR_BODY_NAME, G1_AMP_BODY_NAMES

G1_AMP_MOTION_DIR = os.path.normpath(
    "/home/lianwenkang/workspace/whole_body_general_tracking/dataset_txt/lafan/amp"
)
G1_RLBC_TEACHER_CHECKPOINT_PATH = os.path.normpath(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../../../../../../../logs/rsl_rl/g1_amp/2026-05-30_16-31-18/model_100500.pt",
        )
    )
)
G1_RLBC_TEACHER_ACTOR_CFG = {
    "class_name": "ActorModel",
    "distribution_cfg": {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
    },
    "backbone": {
        "class_name": "MyMLPModel",
        "hidden_dims": [1024, 512, 256],
        "activation": "swish",
        "obs_normalization": True,
    },
}


@configclass
class ActorCfg:
    class_name: str = "MLPModel"
    hidden_dims: list = [4096, 2048, 1024, 512, 256]
    activation: str = "swish"
    obs_normalization: bool = True
    distribution_cfg: dict = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}


@configclass
class CriticCfg:
    class_name: str = "MLPModel"
    hidden_dims: list = [4096, 2048, 1024, 512, 256]
    activation: str = "swish"
    obs_normalization: bool = True
    distribution_cfg: dict = None


# @configclass
# class ActorShellCfg:
#     class_name: str = "ActorModel"
#     distribution_cfg: dict = {
#         "class_name": "GaussianDistribution",
#         "init_std": 1.0,
#         "std_type": "scalar",
#     }
#     backbone: dict = {
#         "class_name": "MyMLPModel",
#         "hidden_dims": [4096, 2048, 1024, 512, 256],
#         "activation": "swish",
#         "obs_normalization": True,
#     }


@configclass
class ActorShellCfg:
    class_name: str = "ActorModel"
    distribution_cfg: dict = {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
    }
    backbone: dict = {
        "class_name": "FSQModel",
        "main_encoder": "encoder_g1",
        "encoder": {
            "encoder_g1": {
                "encoder_groups": ["rbt_cmd_mf"],  # 输入的 encoder_groups
                "hidden_dims": [2048, 1024, 512, 512],  # ecoder MLP 隐层
                "activation": "swish",
            },
            # "encoder_smpl": {
            #     "encoder_groups": ["smpl_cmd_mf"],  # 输入的 encoder_groups
            #     "hidden_dims": [2048, 1024, 512, 512],  # ecoder MLP 隐层
            #     "activation": "swish",
            # },
        },
        "loss": {
            # "token": 1.0,  # token loss 权重: MSE(encoder(token), token_target) * weight，0.0 = 关闭
            # # "re_encode": (
            # #     1.0
            # # ),  # 循环 loss 权重: MSE(encoder(smpl_recon.detach()), encoder(token).detach()) * weight，0.0 = 关闭
            "recon": (
                0.010
            ),  # 重建 loss 权重: MSE(encoder(smpl_recon.detach()), encoder(token).detach()) * weight，0.0 = 关闭
        },
        "fsq": {"num_fsq_levels": 32, "fsq_level_list": 32, "max_num_tokens": 2},
        "latent_dim": 128,  # = num_fsq_levels(32) × max_num_tokens(2)
        "decoder": {
            "action_decoder": {  # action decoder只负责输出action,输入latent和prop
                "decoder_groups": [
                    "prop"
                ],  # decoder 输入中哪些是 passthrough 的原始 obs（默认为 []，即全 latent 输入）
                "hidden_dims": [2048, 1024, 512, 512],  # decoder MLP 隐层
                "activation": "swish",
                "detach_latent": True,  # True = decoder 梯度不回传 encoder
                "outputs": [
                    "actions"
                ],  # True = decoder 输出 action，False = decoder 只做辅助任务（如重建），actor output 直接来自 latent
            },
            "g1_kin_decoder": {  # g1_kin_decoder 负责重建 rbt_cmd_mf
                "decoder_groups": [],  # decoder 输入中哪些是 passthrough 的原始 obs（默认为 []，即全 latent 输入）
                "hidden_dims": [4096, 2048, 1024, 512, 256],  # decoder MLP 隐层
                "activation": "swish",
                "detach_latent": False,  # False = recon loss 梯度穿过 FSQ STE 回传 encoder
                "outputs": [
                    "rbt_cmd_mf"
                ],  # True = decoder 输出 action，False = decoder 只做辅助任务（如重建），actor output 直接来自 latent
                # 循环 loss 权重: MSE(encoder_g1(recon.detach()), z_g1.detach()) * weight，0.0 = 关闭
            },
        },
        "activation": "swish",
        "obs_normalization": True,
    }


@configclass
class MyPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.005
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    plugins: list = []


@configclass
class G1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000000000000000
    save_interval = 500
    experiment_name = "g1_flat"
    empirical_normalization = True

    algorithm = RslRlPpoAlgorithmCfg(
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
    )
    actor = ActorCfg()

    critic = CriticCfg()

    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
    }


@configclass
class G1FlatFMPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000000000000000
    save_interval = 500
    experiment_name = "g1_flat"
    empirical_normalization = True

    algorithm = MyPpoAlgorithmCfg()

    actor = ActorShellCfg()

    critic = CriticCfg()

    obs_groups = {
        "actor": ["prop", "rbt_cmd_mf", "smpl_cmd_mf"],
        "critic": ["critic"],
    }


@configclass
class xwlActorShellCfg:
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


@configclass
class xwlCriticCfg:
    class_name: str = "MLPModel"
    hidden_dims: list = [1024, 512, 256]
    activation: str = "swish"
    obs_normalization: bool = True
    distribution_cfg: dict = None


@configclass
class G1FlatAMPRunnerCfg(G1FlatFMPPORunnerCfg):
    experiment_name = "g1_amp"

    actor = ActorShellCfg()
    critic = CriticCfg()
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
                "amp_reward_coef": 0.1,
                "amp_motion_files": G1_AMP_MOTION_DIR,
                "amp_task_reward_lerp": 0.75,
                "amp_discr_hidden_dims": [1024, 1024, 512, 256],
                "amp_replay_buffer_size": 200000,
                "amp_body_names": G1_AMP_BODY_NAMES,
                "amp_anchor_name": G1_AMP_ANCHOR_BODY_NAME,
                "min_normalized_std": [0.05] * 29,
            }
        ],
    )


@configclass
class G1FlatRLBCDRunnerCfg(G1FlatAMPRunnerCfg):
    experiment_name = "g1_rlbc"

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
            # {
            #     "class_name": "AMPPlugin",
            #     "amp_reward_coef": 0.1,
            #     "amp_motion_files": G1_AMP_MOTION_DIR,
            #     "amp_task_reward_lerp": 0.75,
            #     "amp_discr_hidden_dims": [1024, 1024, 512, 256],
            #     "amp_replay_buffer_size": 200000,
            #     "amp_body_names": G1_AMP_BODY_NAMES,
            #     "amp_anchor_name": G1_AMP_ANCHOR_BODY_NAME,
            #     "min_normalized_std": [0.05] * 29,
            # },
            {
                "class_name": "TeacherKLPlugin",
                "teacher_checkpoint_path": G1_RLBC_TEACHER_CHECKPOINT_PATH,
                "teacher_actor": G1_RLBC_TEACHER_ACTOR_CFG,
                "teacher_obs_groups": {"actor": ["prop", "rbt_cmd_mf", "smpl_cmd_mf"]},
                "teacher_obs_aliases": {"rbt_cmd_mf": "zrbt_cmd_mf"},
                "start_loss_coef": 0.5,
                "end_loss_coef": 0.0,
                "end_step": 2000,
                "kl_direction": "student_teacher",
            },
        ],
    )
