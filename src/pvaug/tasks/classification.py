import torch
from .base import ResearchTask
from ..engine import StepResult
from ..losses import semantic_loss, linear_alpha


class ClassificationTask(ResearchTask):
    """CE and frozen-teacher semantic supervision share one classification task."""

    kind = "classifier"
    artifact = "best.pt"

    def __init__(self, model, config, teacher=None, validator=None):
        self.model, self.config = model, config
        self.teacher, self.validator = teacher, validator
        if teacher is not None:
            teacher.requires_grad_(False).eval()

    def train_step(self, batch, epoch):
        logits = self.model(batch["image"], batch["edge"])
        with torch.no_grad():
            teacher_logits = (
                self.teacher(batch["image"], batch["edge"]) if self.teacher is not None else None
            )
        alpha = linear_alpha(
            epoch, self.config.epochs, self.config.alpha_start, self.config.alpha_end
        )
        loss, metrics = semantic_loss(
            logits, batch["label"], teacher_logits, self.config.temperature, alpha
        )
        metrics["accuracy"] = (logits.argmax(1) == batch["label"]).float().mean().detach()
        return StepResult(loss, metrics, len(batch["label"]))

    def validate(self, device):
        return self.validator(self.model, device) if self.validator else None

    def score(self, train, validation):
        return validation["accuracy"] if validation else -train["loss"]

    def epoch_metadata(self, epoch):
        return {
            "alpha": linear_alpha(
                epoch, self.config.epochs, self.config.alpha_start, self.config.alpha_end
            )
            if self.teacher
            else 0.0
        }
