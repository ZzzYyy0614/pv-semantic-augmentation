from dataclasses import replace
import torch
from pvaug.config import CLASSES
from pvaug.demo import make_demo
from pvaug.data import Sample, audit_samples, prepare_folders, read_manifest
from pvaug.generation import train_autoencoder
from pvaug.models.sketch_ldm import VQFirstStage
from test_methods import tiny_ldm
from PIL import Image


def test_autoencoder_bootstrap_saves_loadable_first_stage(tmp_path):
    manifest = make_demo(tmp_path / "data")
    config = replace(
        tiny_ldm(),
        manifest=str(manifest),
        output=str(tmp_path / "ae"),
        epochs=1,
        batch_size=2,
        max_steps=1,
        device="cpu",
    )
    checkpoint = train_autoencoder(config)
    stage = VQFirstStage(config)
    stage.load_pretrained(checkpoint)
    image = torch.rand(1, 3, 32, 32)
    result = stage.decode(stage.encode(image))
    assert result.shape == image.shape and torch.isfinite(result).all()


def test_prepare_preserves_existing_test_and_pairs_edges_by_filename(tmp_path):
    source = tmp_path / "raw"
    train = source / "train" / CLASSES[0]
    test = source / "test" / CLASSES[0]
    edge = source / "train_sobeledge" / CLASSES[0]
    for folder in [train, test, edge]:
        folder.mkdir(parents=True)
    for i in range(5):
        Image.new("RGB", (16, 16), (10 + i, 20, 30)).save(train / f"{i}.png")
        Image.new("L", (16, 16), i).save(edge / f"{i}.png")
    heldout = test / "heldout.png"
    Image.new("RGB", (16, 16), (50, 20, 30)).save(heldout)
    manifest = tmp_path / "real.csv"
    prepare_folders(source, manifest, CLASSES, val_fraction=0.2)
    samples = read_manifest(manifest, CLASSES)
    assert next(s for s in samples if s.image == heldout).split == "test"
    assert sum(s.split == "val" for s in samples) == 1
    assert all(s.edge.name == s.image.name for s in samples if s.edge)
    assert audit_samples(samples)["samples"] == 6


def test_generator_provenance_excludes_validation(tmp_path):
    # A generated pool with valid domains is still rejected if its generator saw holdout data.
    import json
    import pytest
    from pvaug.config import TrainConfig, ModelConfig
    from pvaug.data import write_manifest
    from pvaug.train import train_classifier
    from pvaug.utils import sha256

    manifest = make_demo(tmp_path / "data")
    real = read_manifest(manifest, CLASSES)
    generated = []
    for i, sample in enumerate(s for s in real if s.split == "train" and s.label != "defect_free"):
        path = tmp_path / "fake" / f"{i}.png"
        path.parent.mkdir(exist_ok=True)
        Image.new("RGB", (16, 16), (200 + i, 100, 50)).save(path)
        generated.append(Sample(path, sample.label, "train", "generated"))
    fake_manifest = tmp_path / "fake" / "generated.csv"
    write_manifest(fake_manifest, generated)
    (fake_manifest.parent / "generation_metadata.json").write_text(
        json.dumps(
            {
                "training_audit": {
                    "real_train_hashes": [sha256(next(s.image for s in real if s.split == "val"))]
                }
            }
        ),
        encoding="utf-8",
    )
    config = TrainConfig(
        manifest=str(manifest),
        generated_manifest=str(fake_manifest),
        output=str(tmp_path / "student"),
        model=ModelConfig(fusion=False),
        image_size=32,
        epochs=1,
    )
    with pytest.raises(ValueError, match="Generator used images outside"):
        train_classifier(config)
