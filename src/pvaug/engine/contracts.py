"""The engine knows tensors and lifecycle events; it does not know EL methods."""

from dataclasses import dataclass, field
from typing import Any, Protocol
import torch
from torch import nn


@dataclass
class StepResult:
    loss: torch.Tensor
    metrics: dict[str, float | torch.Tensor]
    size: int


@dataclass
class TrainState:
    epoch: int = -1
    step: int = 0
    best: float = -float("inf")
    improved: bool = False
    skipped_updates: int = 0
    record: dict[str, Any] = field(default_factory=dict)


class Task(Protocol):
    """Minimal contract for a trainable research method."""

    model: nn.Module
    kind: str
    artifact: str

    def train_step(self, batch: dict, epoch: int) -> StepResult: ...
    def validate(self, device: torch.device) -> dict | None: ...
    def score(self, train: dict, validation: dict | None) -> float: ...
    def after_optimizer_step(self) -> None: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
    def epoch_metadata(self, epoch: int) -> dict: ...


class Callback:
    """Override the events needed by an extension. Callbacks execute in registration order."""

    def on_fit_start(self, trainer):
        pass

    def on_optimizer_step(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        pass

    def on_fit_end(self, trainer):
        pass

    def on_exception(self, trainer, exception):
        pass
