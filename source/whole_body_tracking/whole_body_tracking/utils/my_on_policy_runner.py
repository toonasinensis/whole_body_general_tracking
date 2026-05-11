import os

from rsl_rl.env import VecEnv
from rsl_rl.runners import DeparseOnPolicyRunner
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str | None = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        # rsl_rl calls writer.save_model --> wandb.save(pt). 
        # Disable that when env asks, still torch.save locally.
        if (
            os.environ.get("WANDB_LOG_CHECKPOINTS", "1").strip().lower() in ("0", "false", "no", "off")
            and getattr(self, "logger_type", None) == "wandb"
            and getattr(self, "writer", None) is not None
        ):  # avoid uploading pt model to wandb storage
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
        ):  # avoid uploading pt model to wandb storage
            from rsl_rl.utils import wandb_utils
            real_save_file = wandb_utils.WandbSummaryWriter.save_file
            wandb_utils.WandbSummaryWriter.save_file = lambda self_w, path, iteration=None: None
            out = super().learn(num_learning_iterations, init_at_random_ep_len)
            wandb_utils.WandbSummaryWriter.save_file = real_save_file
            return out
        return super().learn(num_learning_iterations, init_at_random_ep_len)


class MotionDeparseOnPolicyRunner(DeparseOnPolicyRunner):
    """Deparse runner with motion-training logging controls."""

    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu", registry_name: str | None = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos: dict | None = None) -> None:
        if os.environ.get("WANDB_LOG_CHECKPOINTS", "1").strip().lower() in ("0", "false", "no", "off"):
            real_save_model = self.logger.save_model
            self.logger.save_model = lambda *args, **kwargs: None
            try:
                super().save(path, infos)
            finally:
                self.logger.save_model = real_save_model
        else:
            super().save(path, infos)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if (
            os.environ.get("WANDB_LOG_GIT_FILES", "1").strip().lower() in ("0", "false", "no", "off")
            and str(self.cfg.get("logger", "")).lower() == "wandb"
        ):
            from rsl_rl.utils import wandb_utils

            real_save_file = wandb_utils.WandbSummaryWriter.save_file
            wandb_utils.WandbSummaryWriter.save_file = lambda self_w, path: None
            try:
                return super().learn(num_learning_iterations, init_at_random_ep_len)
            finally:
                wandb_utils.WandbSummaryWriter.save_file = real_save_file
        return super().learn(num_learning_iterations, init_at_random_ep_len)
