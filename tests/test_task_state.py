from dataclasses import replace
import shutil
import torch
from torch import nn
from pvaug.config import TrainConfig
from pvaug.engine import Trainer
from pvaug.tasks import ClassificationTask, DiffusionTask
from pvaug.models.sketch_ldm import SketchLDM
from pvaug.utils import load_checkpoint, seed_everything
from test_methods import tiny_ldm


def test_semantic_task_keeps_teacher_parameters_and_mode_frozen(tmp_path):
    class SmallClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.BatchNorm1d(3)
            self.head = nn.Linear(3, 5)

        def forward(self, image, edge=None):
            return self.head(self.norm(image))

    teacher, student = SmallClassifier(), SmallClassifier()
    initial = {key: value.clone() for key, value in teacher.state_dict().items()}
    config = TrainConfig(output=str(tmp_path / "semantic"), epochs=1)
    batch = {
        "image": torch.rand(4, 3),
        "edge": torch.zeros(4, 1),
        "label": torch.tensor([0, 1, 2, 3]),
    }
    Trainer(
        ClassificationTask(student, config, teacher), config, lambda epoch: [batch], "cpu"
    ).fit()
    assert not teacher.training and all(
        parameter.grad is None for parameter in teacher.parameters()
    )
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    for key, value in initial.items():
        assert torch.equal(value, teacher.state_dict()[key]), key


def test_diffusion_resume_restores_ema_optimizer_and_rng(tmp_path):
    config = tiny_ldm(output=str(tmp_path / "full"), epochs=2, device="cpu")
    seed_everything(19)
    batch = {
        "image": torch.rand(2, 3, 32, 32),
        "mask": torch.rand(2, 1, 32, 32),
        "label": torch.tensor([0, 1]),
    }
    seed_everything(22)
    task = DiffusionTask(SketchLDM(config), config)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    from pvaug.utils import atomic_save

    def save(value, path):
        atomic_save(value, path)
        if value["epoch"] == 0:
            shutil.copyfile(path, snapshot / path.name)

    Trainer(task, config, lambda epoch: [batch], "cpu", save=save).fit()
    resumed_config = replace(config, output=str(tmp_path / "resumed"))
    seed_everything(999)
    resumed_task = DiffusionTask(SketchLDM(resumed_config), resumed_config)
    Trainer(resumed_task, resumed_config, lambda epoch: [batch], "cpu").fit(snapshot / "last.pt")
    full = load_checkpoint(tmp_path / "full/last.pt")
    resumed = load_checkpoint(tmp_path / "resumed/last.pt")
    assert full["step"] == resumed["step"] == 2
    for key in full["model"]:
        assert torch.equal(full["model"][key], resumed["model"][key]), key
    assert full["ema"]["updates"] == resumed["ema"]["updates"]
    for key in full["ema"]["shadow"]:
        assert torch.equal(full["ema"]["shadow"][key], resumed["ema"]["shadow"][key]), key
