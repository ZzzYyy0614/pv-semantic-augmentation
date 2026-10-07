import csv
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from .config import write_json
from .data import ELDataset, Sample, EXTENSIONS
from .train import model_from_checkpoint, write_predictions
from .utils import resolve_device


def predict(checkpoint, source, output, device="auto", gradcam=False):
    device = resolve_device(device)
    model, config, _ = model_from_checkpoint(checkpoint, device)
    model.eval()
    source = Path(source)
    files = (
        sorted(p for p in source.rglob("*") if p.suffix.lower() in EXTENSIONS)
        if source.is_dir()
        else [source]
    )
    if not files:
        raise ValueError("No input images")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, file in enumerate(files):
        sample = ELDataset(
            [Sample(file, config.classes[0], "test")], config.classes, config.image_size
        )[0]
        image, edge = sample["image"][None].to(device), sample["edge"][None].to(device)
        if gradcam:
            model.zero_grad(set_to_none=True)
            features = model.forward_features(image, edge)
            features.retain_grad()
            logits = model.classify_features(features)
            target = int(logits.argmax(1))
            logits[0, target].backward()
            weights = features.grad.mean((2, 3), keepdim=True)
            cam = (weights * features).sum(1, keepdim=True).relu().detach()
            cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
            cam = cam[0, 0].cpu().numpy()
            cam /= max(float(cam.max()), 1e-8)
            heat = np.stack([np.ones_like(cam), cam, np.zeros_like(cam)], axis=-1)
            rgb = sample["image"].permute(1, 2, 0).numpy()
            blend = rgb * (1 - 0.5 * cam[..., None]) + heat * 0.5 * cam[..., None]
            Image.fromarray((blend.clip(0, 1) * 255).astype(np.uint8)).save(
                output / f"{index:04d}_{file.stem}_cam.png"
            )
        else:
            with torch.no_grad():
                logits = model(image, edge)
        probabilities = logits.detach().softmax(1)[0].cpu().tolist()
        target = int(logits.argmax(1))
        rows.append(
            {
                "image": str(file.resolve()),
                "prediction": config.classes[target],
                "confidence": probabilities[target],
                **{f"p_{name}": p for name, p in zip(config.classes, probabilities)},
            }
        )
    write_predictions(output / "predictions.csv", rows)
    return rows


def benchmark(checkpoint, device="auto", warmup=10, iterations=100):
    if warmup < 0 or iterations < 1:
        raise ValueError("Invalid benchmark iterations")
    device = resolve_device(device)
    model, config, _ = model_from_checkpoint(checkpoint, device)
    model.eval()
    image = torch.rand(1, 3, config.image_size, config.image_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(image)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            model(image)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "fps": iterations / elapsed,
        "latency_ms": elapsed * 1000 / iterations,
        "batch_size": 1,
        "image_size": config.image_size,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "iterations": iterations,
        "scope": "Model forward including online Sobel; excludes decoding and resizing",
    }


def confidence_histogram(predictions_csv, output, bins=10):
    if bins < 1:
        raise ValueError("bins must be positive")
    values = {}
    with Path(predictions_csv).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            label = row.get("target")
            if not label:
                raise ValueError("Use eval predictions.csv with ground truth target labels")
            values.setdefault(label, []).append(float(row[f"p_{label}"]))
    result = {
        name: {
            "counts": np.histogram(scores, bins=bins, range=(0, 1))[0].tolist(),
            "edges": np.linspace(0, 1, bins + 1).tolist(),
            "samples": len(scores),
        }
        for name, scores in values.items()
    }
    write_json(output, result)
    return result
