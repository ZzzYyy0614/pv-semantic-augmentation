import csv
from dataclasses import replace
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from .config import TrainConfig, config_dict, write_json
from .data import read_manifest, audit_samples, select_generated, make_loader
from .metrics import classification_metrics
from .models import DefectClassifier
from .utils import (
    seed_everything,
    resolve_device,
    atomic_save,
    load_checkpoint,
)


def model_from_checkpoint(path, device="cpu"):
    checkpoint = load_checkpoint(path)
    if checkpoint.get("kind") != "classifier":
        raise ValueError(
            "Expected a pvaug classifier checkpoint; convert raw legacy weights with --init"
        )
    config = TrainConfig(**checkpoint["config"])
    model_config = replace(config.model, pretrained=False)
    model = DefectClassifier(model_config, len(config.classes), config.normalize)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device), config, checkpoint


def initialize_weights(model, path):
    checkpoint = load_checkpoint(path)
    if checkpoint.get("kind") == "classifier":
        state = checkpoint["model"]
    else:
        state = checkpoint.get("state_dict", checkpoint)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        if model.config.variant == "recovered" and not any(k.startswith("core.") for k in state):
            state = {"core." + k: v for k, v in state.items()}
        # Wrapper buffers are deterministic preprocessing constants, absent in original weights.
        for key, value in model.state_dict().items():
            if not key.startswith("core."):
                state.setdefault(key, value)
    model.load_state_dict(state, strict=True)


def convert_legacy_classifier(config, legacy_path, output, training_manifest=None):
    model = DefectClassifier(
        replace(config.model, pretrained=False), len(config.classes), config.normalize
    )
    initialize_weights(model, legacy_path)
    audit = None
    if training_manifest:
        samples = read_manifest(training_manifest, config.classes)
        if any(s.domain != "real" for s in samples):
            raise ValueError("Original teacher training manifest must contain only real data")
        audit = audit_samples(samples)
    checkpoint = {
        "kind": "classifier",
        "format_version": 1,
        "model": model.state_dict(),
        "config": config_dict(config),
        "audit": audit,
        "legacy_conversion": True,
        "training_domains": ["real"] if audit else ["unknown"],
    }
    atomic_save(checkpoint, output)
    return {
        "output": str(output),
        "classes": config.classes,
        "training_provenance": "user-supplied original split" if audit else "unknown",
    }


@torch.no_grad()
def evaluate_model(model, loader, classes, device):
    model.eval()
    targets, predictions, rows, losses = [], [], [], []
    for batch in loader:
        images, edges = batch["image"].to(device), batch["edge"].to(device)
        labels = batch["label"].to(device)
        logits = model(images, edges)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite evaluation logits")
        probs = logits.softmax(1).cpu()
        predicted = logits.argmax(1).cpu().tolist()
        true = labels.cpu().tolist()
        targets.extend(true)
        predictions.extend(predicted)
        losses.extend(F.cross_entropy(logits, labels, reduction="none").cpu().tolist())
        for path, target, pred, prob in zip(batch["path"], true, predicted, probs.tolist()):
            rows.append(
                {
                    "image": path,
                    "target": classes[target],
                    "prediction": classes[pred],
                    **{f"p_{name}": p for name, p in zip(classes, prob)},
                }
            )
    metrics = classification_metrics(targets, predictions, classes)
    metrics["loss"] = sum(losses) / len(losses)
    return metrics, rows


def write_predictions(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_classifier(config, resume=None, init=None):
    seed_everything(config.seed)
    device = resolve_device(config.device)
    real = read_manifest(config.manifest, config.classes)
    if any(s.domain != "real" for s in real):
        raise ValueError("The real manifest must contain real samples only")
    generated = []
    if config.generated_manifest:
        pool = read_manifest(config.generated_manifest, config.classes)
        if any(s.domain != "generated" for s in pool):
            raise ValueError("Generated manifest must contain only generated/train rows")
        generated = select_generated(
            [s for s in real if s.split == "train"],
            pool,
            config.generated_ratio,
            config.classes,
            config.seed,
        )
    audit = audit_samples(real + generated)
    generation_provenance = None
    if config.generated_manifest:
        metadata_path = Path(config.generated_manifest).parent / "generation_metadata.json"
        if metadata_path.is_file():
            generation_provenance = json.loads(metadata_path.read_text(encoding="utf-8"))
            generator_audit = generation_provenance.get("training_audit")
            if generator_audit and not set(generator_audit["real_train_hashes"]) <= set(
                audit["real_train_hashes"]
            ):
                raise ValueError(
                    "Generator used images outside the classifier's real training split"
                )
    train = [s for s in real if s.split == "train"] + generated
    val = [s for s in real if s.split == "val"]
    if config.resample_defects:
        train += [s for s in train if s.domain == "real" and s.label != "defect_free"]
    if len(train) < 2:
        raise ValueError("Need at least 2 training images")
    output = Path(config.output)
    if (output / "last.pt").exists() and resume is None:
        raise FileExistsError(f"{output}/last.pt exists; use --resume or a new --output")
    output.mkdir(parents=True, exist_ok=True)
    model = DefectClassifier(config.model, len(config.classes), config.normalize).to(device)
    if init:
        initialize_weights(model, init)
    teacher = None
    if config.teacher:
        teacher, teacher_config, teacher_ckpt = model_from_checkpoint(config.teacher, device)
        if (
            teacher_config.classes != config.classes
            or teacher_config.image_size != config.image_size
        ):
            raise ValueError("Teacher classes/order and input size must match student")
        if teacher_ckpt["training_domains"] != ["real"]:
            raise ValueError("Semantic estimator must be pretrained on real training data")
        if teacher_ckpt["audit"]["real_train_hashes"] != audit["real_train_hashes"]:
            raise ValueError("Teacher was trained on a different real split")
        teacher.requires_grad_(False).eval()
    from .engine import Trainer
    from .tasks import ClassificationTask

    validator = (
        (
            lambda current, dev: evaluate_model(
                current, make_loader(val, config), config.classes, dev
            )[0]
        )
        if val
        else None
    )
    task = ClassificationTask(model, config, teacher, validator)
    engine = Trainer(
        task,
        config,
        lambda epoch: make_loader(train, config, True, epoch),
        device,
        audit=audit,
        metadata={
            "generation_provenance": generation_provenance,
            "training_domains": sorted({sample.domain for sample in train}),
        },
        save=atomic_save,
    )
    return engine.fit(resume)


def evaluate_checkpoint(checkpoint, manifest, output, split="test", device="auto", batch_size=24):
    device = resolve_device(device)
    model, config, ckpt = model_from_checkpoint(checkpoint, device)
    samples = read_manifest(manifest, config.classes)
    report = audit_samples(samples)
    selected = [s for s in samples if s.split == split]
    if split != "train" and any(s.domain != "real" for s in selected):
        raise ValueError("Evaluation holdout must be real")
    if split != "train":
        from .utils import sha256

        training = set((ckpt.get("audit") or {}).get("real_train_hashes", []))
        if any(sha256(s.image) in training for s in selected):
            raise ValueError("Evaluation images overlap checkpoint training images")
    config = replace(config, batch_size=batch_size, conventional_augmentation=False)
    metrics, rows = evaluate_model(model, make_loader(selected, config), config.classes, device)
    metrics["training_provenance"] = "recorded" if ckpt.get("audit") else "unknown legacy split"
    write_json(Path(output) / "metrics.json", metrics)
    write_json(Path(output) / "audit.json", report)
    write_predictions(Path(output) / "predictions.csv", rows)
    return metrics
