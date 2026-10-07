from dataclasses import replace
import torch
from torch import nn
import pytest
from pvaug.config import TrainConfig, EngineConfig
from pvaug.engine import Trainer, Callback, StepResult
from pvaug.tasks import ResearchTask
from pvaug.utils import load_checkpoint


class RegressionTask(ResearchTask):
    kind = "regression-test"
    artifact = "last.pt"

    def __init__(self, model):
        self.model = model

    def train_step(self, batch, epoch):
        loss = (self.model(batch["x"]) - batch["y"]).square().mean()
        return StepResult(loss, {}, len(batch["x"]))


def test_sample_weighted_accumulation_matches_large_batch_with_partial_group(tmp_path):
    torch.manual_seed(7)
    x, y = torch.randn(7, 3), torch.randn(7, 1)
    config = TrainConfig(
        output=str(tmp_path / "micro"),
        epochs=1,
        lr=0.01,
        momentum=0,
        weight_decay=0,
        engine=EngineConfig(accumulation_steps=3),
    )
    micro, large = nn.Linear(3, 1), nn.Linear(3, 1)
    large.load_state_dict(micro.state_dict())
    # 2+2+2 and then 1; actual final group size must control gradient normalization.
    batches = [{"x": x[i : i + 2], "y": y[i : i + 2]} for i in range(0, 7, 2)]
    combined = [{"x": x[:6], "y": y[:6]}, {"x": x[6:], "y": y[6:]}]
    Trainer(RegressionTask(micro), config, lambda epoch: batches, "cpu").fit()
    Trainer(
        RegressionTask(large),
        replace(config, output=str(tmp_path / "large"), engine=EngineConfig()),
        lambda epoch: combined,
        "cpu",
    ).fit()
    for a, b in zip(micro.parameters(), large.parameters()):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)
    assert load_checkpoint(tmp_path / "micro/last.pt")["step"] == 2


def test_callbacks_precision_clipping_and_optimizer_step_limit(tmp_path):
    events = []

    class Trace(Callback):
        def on_fit_start(self, trainer):
            events.append("start")

        def on_optimizer_step(self, trainer):
            events.append("step")

        def on_epoch_end(self, trainer):
            events.append("epoch")

        def on_fit_end(self, trainer):
            events.append("end")

    config = TrainConfig(
        output=str(tmp_path / "bf16"),
        epochs=3,
        max_steps=1,
        optimizer="adamw",
        engine=EngineConfig(precision="bf16", accumulation_steps=2, clip_grad_norm=0.1),
    )
    model = nn.Linear(3, 1)
    batches = [{"x": torch.ones(2, 3), "y": torch.ones(2, 1)}] * 3
    Trainer(RegressionTask(model), config, lambda epoch: batches, "cpu", callbacks=[Trace()]).fit()
    assert events == ["start", "step", "epoch", "end"]
    checkpoint = load_checkpoint(tmp_path / "bf16/last.pt")
    assert checkpoint["step"] == 1 and checkpoint["format_version"] == 2
    assert "environment.json" in {p.name for p in (tmp_path / "bf16").iterdir()}


def test_exception_callback_and_no_invalid_checkpoint(tmp_path):
    class Invalid(RegressionTask):
        def train_step(self, batch, epoch):
            return StepResult(self.model(batch["x"]).sum() * float("nan"), {}, 2)

    caught = []

    class Observer(Callback):
        def on_exception(self, trainer, exception):
            caught.append(type(exception))

    config = TrainConfig(output=str(tmp_path / "invalid"), epochs=1)
    engine = Trainer(
        Invalid(nn.Linear(3, 1)),
        config,
        lambda epoch: [{"x": torch.ones(2, 3)}],
        "cpu",
        callbacks=[Observer()],
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        engine.fit()
    assert caught == [RuntimeError] and not (tmp_path / "invalid/last.pt").exists()


def test_fp16_rejects_cpu_before_training(tmp_path):
    config = TrainConfig(output=str(tmp_path), engine=EngineConfig(precision="fp16"))
    with pytest.raises(ValueError, match="requires CUDA"):
        Trainer(RegressionTask(nn.Linear(3, 1)), config, lambda epoch: [], "cpu")
