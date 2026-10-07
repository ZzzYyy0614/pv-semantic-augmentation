"""Deterministic toy images for software tests, never evidence of paper accuracy."""

from dataclasses import replace
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .config import (
    CLASSES,
    TrainConfig,
    LDMConfig,
    ModelConfig,
    EngineConfig,
    config_dict,
    write_json,
)
from .data import Sample, write_manifest


def make_demo(output, size=64, seed=123):
    output = Path(output).resolve()
    if (output / "real.csv").exists():
        raise FileExistsError("Demo dataset exists; choose a new directory")
    rng = np.random.default_rng(seed)
    samples, sketches = [], []
    for class_index, name in enumerate(CLASSES):
        for split, number in [("train", 2), ("val", 1), ("test", 1)]:
            for index in range(number):
                canvas = Image.fromarray(
                    rng.normal(140, 8, (size, size)).clip(0, 255).astype(np.uint8)
                )
                mask = Image.new("L", (size, size), 0)
                draw, region = ImageDraw.Draw(canvas), ImageDraw.Draw(mask)
                for x in [size // 3, 2 * size // 3]:
                    draw.line((x, 0, x, size), fill=80, width=1)
                offset = int(rng.integers(-size // 8, size // 8))
                if name == "micro_crack":
                    points = [
                        (size // 4, size // 4),
                        (size // 2 + offset, size // 2),
                        (3 * size // 4, 3 * size // 4),
                    ]
                    draw.line(points, fill=20, width=2)
                    region.line(points, fill=255, width=3)
                elif name == "busbar_corrosion":
                    box = (size // 3 - 2, size // 5, size // 3 + 3, 4 * size // 5)
                    draw.rectangle(box, fill=35)
                    region.rectangle(box, fill=255)
                elif name == "break":
                    points = [(0, size // 2), (size // 2 + offset, size), (0, size)]
                    draw.polygon(points, fill=25)
                    region.polygon(points, fill=255)
                elif name == "black_core":
                    box = (size // 3 + offset, size // 3, 2 * size // 3 + offset, 2 * size // 3)
                    draw.ellipse(box, fill=20)
                    region.ellipse(box, fill=255)
                image_path = output / "images" / split / name / f"{index:02d}.png"
                mask_path = output / "masks" / split / name / f"{index:02d}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                canvas.convert("RGB").save(image_path)
                mask.save(mask_path)
                samples.append(
                    Sample(
                        image_path,
                        name,
                        split,
                        mask=mask_path,
                        group=f"{class_index}-{split}-{index}",
                    )
                )
                if split == "train" and index == 0 and name != "defect_free":
                    sketches.append(
                        {"mask": mask_path.relative_to(output).as_posix(), "label": name}
                    )
    write_manifest(output / "real.csv", samples)
    with (output / "sketches.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["mask", "label"])
        writer.writeheader()
        writer.writerows(sketches)
    return output / "real.csv"


def smoke(output="runs/smoke"):
    import torch
    from .orchestration import Pipeline, Stage
    from .inference import predict

    torch.set_num_threads(2)
    output = Path(output).resolve()
    manifest = make_demo(output / "data")
    teacher_config = TrainConfig(
        manifest=str(manifest),
        output=str(output / "estimator"),
        model=ModelConfig(variant="paper"),
        engine=EngineConfig(clip_grad_norm=1.0),
        image_size=32,
        batch_size=2,
        epochs=2,
        max_steps=2,
        device="cpu",
    )
    generator_config = LDMConfig(
        manifest=str(manifest),
        output=str(output / "generator"),
        first_stage_checkpoint=None,
        smoke_random_first_stage=True,
        first_stage_channels=32,
        first_stage_mult=[1, 2],
        first_stage_res_blocks=1,
        image_size=32,
        model_channels=32,
        channel_mult=[1, 2],
        num_res_blocks=1,
        num_heads=4,
        attention_resolutions=[1, 2],
        cross_attention_dim=32,
        batch_size=2,
        epochs=2,
        timesteps=10,
        max_steps=2,
        device="cpu",
    )
    root = output / "configs"
    write_json(root / "estimator.json", config_dict(teacher_config))
    write_json(root / "generator.json", config_dict(generator_config))
    write_json(
        root / "recognizer.json",
        config_dict(replace(teacher_config, output=str(output / "recognizer"))),
    )
    pipeline = Pipeline(
        "cpu-method-smoke",
        output,
        [
            Stage("estimator", "classification", "estimator.json"),
            Stage("generator", "diffusion", "generator.json"),
            Stage(
                "synthetic",
                "sampling",
                inputs={
                    "checkpoint": "@generator.checkpoint",
                    "sketches": str(output / "data/sketches.csv"),
                },
                options={"per_class": 2, "steps": 2, "device": "cpu"},
            ),
            Stage(
                "recognizer",
                "classification",
                "recognizer.json",
                inputs={
                    "teacher": "@estimator.checkpoint",
                    "generated_manifest": "@synthetic.manifest",
                },
            ),
            Stage(
                "evaluation",
                "evaluation",
                inputs={"checkpoint": "@recognizer.checkpoint", "manifest": str(manifest)},
                options={"device": "cpu", "batch_size": 2},
            ),
        ],
        root,
    )
    write_json(
        root / "workflow.json",
        {
            "schema_version": 1,
            "name": pipeline.name,
            "output": str(output),
            "stages": [config_dict(stage) for stage in pipeline.stages],
        },
    )
    state = pipeline.run()
    student = output / "recognizer/best.pt"
    predict(
        student,
        output / "data/images/test/micro_crack/00.png",
        output / "prediction",
        device="cpu",
        gradcam=True,
    )
    result = {
        "status": "passed",
        "purpose": "Software smoke test on toy images; not paper results",
        "teacher": str(output / "estimator/best.pt"),
        "generator": str(output / "generator/last.pt"),
        "student": str(student),
        "pipeline": str(output / "pipeline.json"),
        "workflow": str(root / "workflow.json"),
        "completed_stages": list(state["stages"]),
        "test_samples": 5,
    }
    write_json(output / "smoke_report.json", result)
    return result
