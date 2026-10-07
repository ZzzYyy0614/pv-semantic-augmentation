import csv
from pathlib import Path
import torch
from PIL import Image
from torchvision.transforms import functional as TF, InterpolationMode
from .config import LDMConfig, config_dict, write_json
from .data import read_manifest, audit_samples, Sample, write_manifest, make_loader
from .models.sketch_ldm import SketchLDM, VQFirstStage, ExponentialMovingAverage
from .utils import (
    seed_everything,
    resolve_device,
    atomic_save,
    load_checkpoint,
)


def generation_data(config):
    classes = list(dict.fromkeys(["defect_free"] + config.classes))
    samples = read_manifest(config.manifest, classes)
    if any(s.domain != "real" for s in samples):
        raise ValueError("Generator and autoencoder may train only on real samples")
    audit = audit_samples(samples)
    selected = [s for s in samples if s.split == "train" and s.label in config.classes]
    if not selected:
        raise ValueError("No defect training images")
    return selected, audit


def train_generator(config, resume=None):
    seed_everything(config.seed)
    device = resolve_device(config.device)
    samples, audit = generation_data(config)
    if any(s.mask is None for s in samples):
        raise ValueError("Every diffusion training image needs a manually annotated mask/sketch")
    output = Path(config.output)
    if (output / "last.pt").exists() and not resume:
        raise FileExistsError(f"{output}/last.pt exists; use --resume or a new output")
    model = SketchLDM(config, load_first_stage=not (resume or config.initialize)).to(device)
    if config.initialize:
        model.load_legacy(config.initialize)
    from .engine import Trainer
    from .tasks import DiffusionTask

    task = DiffusionTask(model, config)
    engine = Trainer(
        task,
        config,
        lambda epoch: make_loader(samples, config, True, epoch, generation=True),
        device,
        audit=audit,
    )
    return engine.fit(resume)


def train_autoencoder(config, resume=None):
    """Optional L1 + VQ bootstrap; the archived perceptual/GAN objective is documented separately."""
    from .engine import Trainer
    from .tasks import AutoencoderTask

    seed_everything(config.seed)
    device = resolve_device(config.device)
    samples, audit = generation_data(config)
    model = VQFirstStage(config).to(device)
    engine = Trainer(
        AutoencoderTask(model),
        config,
        lambda epoch: make_loader(samples, config, True, epoch),
        device,
        audit=audit,
    )
    return engine.fit(resume)


def convert_legacy_generator(config, legacy_path, output):
    """Strict architecture conversion; class ordering must be supplied in the config."""
    model = SketchLDM(config, load_first_stage=False)
    model.load_legacy(legacy_path)
    # Legacy sampling uses the U-Net EMA. Map its flattened buffer names when present.
    raw = load_checkpoint(legacy_path)
    state = raw.get("state_dict", raw)
    used_ema = 0
    with torch.no_grad():
        for name, parameter in model.unet.named_parameters():
            key = "model_ema." + ("diffusion_model." + name).replace(".", "")
            if key in state:
                parameter.copy_(state[key])
                used_ema += 1
    atomic_save(
        {
            "kind": "sketch_ldm",
            "format_version": 1,
            "model": model.state_dict(),
            "config": config_dict(config),
            "legacy_conversion": True,
            "legacy_ema_parameters": used_ema,
            "audit": None,
            "ema": None,
        },
        output,
    )
    return {
        "output": str(output),
        "ema_parameters": used_ema,
        "provenance": "Legacy training split is unknown; check original train/test lists.",
    }


def generate_samples(
    checkpoint, sketches_csv, output, per_class=200, steps=50, seed=42, device="auto", eta=0.0
):
    if per_class < 1:
        raise ValueError("per_class must be positive")
    raw = load_checkpoint(checkpoint)
    if raw.get("kind") != "sketch_ldm":
        raise ValueError("Convert legacy generator weights before sampling")
    config = LDMConfig(**raw["config"])
    device = resolve_device(device)
    model = SketchLDM(config, load_first_stage=False).to(device).eval()
    model.load_state_dict(raw["model"], strict=True)
    ema = ExponentialMovingAverage(model)
    if raw.get("ema"):
        ema.load_state_dict(raw["ema"])
    sketches_csv = Path(sketches_csv).resolve()
    sketches = {name: [] for name in config.classes}
    with sketches_csv.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["label"] not in sketches:
                raise ValueError(f"Unknown sketch label: {row['label']}")
            sketches[row["label"]].append((sketches_csv.parent / row["mask"]).resolve())
    for name, paths in sketches.items():
        if not paths or any(not p.is_file() for p in paths):
            raise ValueError(f"Need existing sketch masks for {name}")
    output = Path(output).resolve()
    if (output / "generated.csv").exists():
        raise FileExistsError("Generated output already exists; choose a new directory")
    samples = []
    with ema.apply(model), torch.no_grad():
        for class_id, name in enumerate(config.classes):
            folder = output / "images" / name
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(per_class):
                path = sketches[name][index % len(sketches[name])]
                with Image.open(path) as image:
                    mask = image.convert("L")
                    side = min(mask.size)
                    mask = TF.center_crop(mask, [side, side])
                    mask = TF.to_tensor(
                        TF.resize(
                            mask, [config.image_size] * 2, interpolation=InterpolationMode.BICUBIC
                        )
                    )
                image_seed = seed + class_id * per_class + index
                image = model.sample(
                    mask[None].to(device),
                    torch.tensor([class_id], device=device),
                    steps,
                    image_seed,
                    eta,
                )[0].cpu()
                file = folder / f"{index:05d}.png"
                TF.to_pil_image(image).save(file)
                # The sketch may have a different native size; it remains in generation provenance.
                samples.append(Sample(file, name, "train", "generated"))
    write_manifest(output / "generated.csv", samples)
    write_json(
        output / "generation_metadata.json",
        {
            "checkpoint": str(Path(checkpoint).resolve()),
            "seed": seed,
            "steps": steps,
            "eta": eta,
            "per_class": per_class,
            "classes": config.classes,
            "smoke_random_first_stage": config.smoke_random_first_stage,
            "training_audit": raw.get("audit"),
            "legacy_conversion": raw.get("legacy_conversion", False),
            "sketches": {k: [str(p) for p in v] for k, v in sketches.items()},
        },
    )
    print(f"Generated {len(samples)} images: {output / 'generated.csv'}", flush=True)
    return output / "generated.csv"
