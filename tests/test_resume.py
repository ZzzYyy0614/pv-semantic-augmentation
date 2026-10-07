from dataclasses import replace
import shutil
import torch
from pvaug.config import TrainConfig, ModelConfig
from pvaug.demo import make_demo
from pvaug.utils import load_checkpoint
import pvaug.train as training


def test_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    manifest = make_demo(tmp_path / "data")
    config = TrainConfig(
        manifest=str(manifest),
        output=str(tmp_path / "full"),
        model=ModelConfig(fusion=False),
        image_size=32,
        epochs=2,
        batch_size=5,
        device="cpu",
        seed=14,
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    save = training.atomic_save

    def capture_first_epoch(value, path):
        save(value, path)
        if value["epoch"] == 0:
            shutil.copyfile(path, snapshot / path.name)

    monkeypatch.setattr(training, "atomic_save", capture_first_epoch)
    training.train_classifier(config)
    monkeypatch.setattr(training, "atomic_save", save)
    resumed = replace(config, output=str(tmp_path / "resumed"))
    training.train_classifier(resumed, resume=snapshot / "last.pt")
    full = load_checkpoint(tmp_path / "full/last.pt")
    partial = load_checkpoint(tmp_path / "resumed/last.pt")
    assert full["step"] == partial["step"]
    assert full["epoch"] == partial["epoch"] == 1
    for key, value in full["model"].items():
        assert torch.equal(value, partial["model"][key]), key
