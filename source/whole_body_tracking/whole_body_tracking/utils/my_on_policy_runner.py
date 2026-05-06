import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        policy_path = path.split("model")[0]
        filename = policy_path.split("/")[-2] + ".onnx"
        # export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
        # run_path = os.path.basename(os.path.normpath(policy_path))
        # attach_onnx_metadata(self.env.unwrapped, run_path, path=policy_path, filename=filename)


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        # rsl_rl calls writer.save_model → wandb.save(pt). Disable that when env asks, still torch.save locally.
        if (
            os.environ.get("WANDB_LOG_CHECKPOINTS", "1").strip().lower() in ("0", "false", "no", "off")
            and getattr(self, "logger_type", None) == "wandb"
            and getattr(self, "writer", None) is not None
        ):
            real_save_model = self.writer.save_model
            self.writer.save_model = lambda *a, **k: None
            super().save(path, infos)
            self.writer.save_model = real_save_model
        else:
            super().save(path, infos)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if (
            os.environ.get("WANDB_LOG_GIT_FILES", "1").strip().lower() in ("0", "false", "no", "off")
            and str(self.cfg.get("logger", "")).lower() == "wandb"
        ):
            from rsl_rl.utils import wandb_utils

            real_save_file = wandb_utils.WandbSummaryWriter.save_file
            wandb_utils.WandbSummaryWriter.save_file = lambda self_w, path, iteration=None: None
            out = super().learn(num_learning_iterations, init_at_random_ep_len)
            wandb_utils.WandbSummaryWriter.save_file = real_save_file
            return out
        return super().learn(num_learning_iterations, init_at_random_ep_len)
