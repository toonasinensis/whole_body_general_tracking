from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg  # noqa: F401


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
        "class_name": "MyModel",
        "main_encoder": "encoder_g1",
        "encoder": {
            "encoder_g1": {
                "encoder_groups": ["rbt_cmd_mf"],  # 输入的 encoder_groups
                "hidden_dims": [2048, 1024, 512, 512],  # ecoder MLP 隐层
                "activation": "swish",
            },
            "encoder_smpl": {
                "encoder_groups": ["smpl_cmd_mf"],  # 输入的 encoder_groups
                "hidden_dims": [2048, 1024, 512, 512],  # ecoder MLP 隐层
                "activation": "swish",
            },
        },
        "loss": {
            "token": 1.0,  # token loss 权重: MSE(encoder(token), token_target) * weight，0.0 = 关闭
            "re_encode": (
                1.0
            ),  # 循环 loss 权重: MSE(encoder(smpl_recon.detach()), encoder(token).detach()) * weight，0.0 = 关闭
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
    save_interval = 100
    experiment_name = "g1_flat"
    empirical_normalization = True

    algorithm = MyPpoAlgorithmCfg()

    actor = ActorShellCfg()

    critic = CriticCfg()

    obs_groups = {
        "actor": ["prop", "rbt_cmd_mf", "smpl_cmd_mf"],
        "critic": ["critic"],
    }
