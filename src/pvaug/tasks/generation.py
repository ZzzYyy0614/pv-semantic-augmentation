from .base import ResearchTask
from ..engine import StepResult
from ..models.sketch_ldm import ExponentialMovingAverage


class DiffusionTask(ResearchTask):
    kind = "sketch_ldm"
    artifact = "last.pt"

    def __init__(self, model, config):
        self.model = model
        self.ema = ExponentialMovingAverage(model, config.ema_decay)

    def train_step(self, batch, epoch):
        loss = self.model.loss(batch["image"], batch["mask"], batch["label"])
        return StepResult(loss, {"noise_loss": loss.detach()}, len(batch["label"]))

    def after_optimizer_step(self):
        self.ema.update(self.model)

    def state_dict(self):
        return {"ema": self.ema.state_dict()}

    def load_state_dict(self, state):
        self.ema.load_state_dict(state["ema"])


class AutoencoderTask(ResearchTask):
    """Optional VQ bootstrap objective, separate from the original GAN training."""

    kind = "autoencoder"
    artifact = "first_stage.pt"

    def __init__(self, model):
        self.model = model

    def train_step(self, batch, epoch):
        loss = self.model.reconstruction_loss(batch["image"] * 2 - 1)
        return StepResult(loss, {"reconstruction_vq_loss": loss.detach()}, len(batch["label"]))
