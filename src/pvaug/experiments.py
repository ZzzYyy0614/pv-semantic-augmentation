from dataclasses import replace
from pathlib import Path
from .config import ModelConfig, config_dict, write_json


def experiment_plan(base, output, suite="components", seeds=(42,)):
    configs = []
    for seed in seeds:
        if suite == "components":
            settings = [
                ("baseline", False, False, False),
                ("fusion_rgb", True, False, False),
                ("edge", True, True, False),
                ("direct_generated", True, True, False),
                ("semantic_real_only", True, True, True),
                ("semantic_generated", True, True, True),
            ]
            for name, fusion, prior, kd in settings:
                config = replace(
                    base,
                    seed=seed,
                    model=replace(base.model, fusion=fusion, edge_prior=prior),
                    teacher=base.teacher if kd else None,
                    generated_manifest=base.generated_manifest
                    if name in {"direct_generated", "semantic_generated"}
                    else None,
                )
                configs.append((f"{name}_seed{seed}", config))
        elif suite == "temperature":
            configs.extend(
                (f"temperature{t}_seed{seed}", replace(base, seed=seed, temperature=t))
                for t in [1.0, 5.0, 10.0, 15.0, 20.0, 25.0]
            )
        elif suite == "ratio":
            configs.extend(
                (f"ratio{r}_seed{seed}", replace(base, seed=seed, generated_ratio=r))
                for r in [0.0, 0.5, 1.0, 2.0]
            )
        elif suite == "backbones":
            configs.extend(
                (
                    f"{b}_seed{seed}",
                    replace(base, seed=seed, model=ModelConfig(backbone=b, variant="paper")),
                )
                for b in ["resnet18", "convnext_tiny", "swin_t"]
            )
        else:
            raise ValueError("Unknown experiment suite")
    paths = []
    for name, config in configs:
        config = replace(config, output=str(Path(output) / name))
        if config.teacher is None and ("semantic" in name or suite != "components"):
            raise ValueError("A pretrained teacher path is required for this suite")
        path = Path(output) / "configs" / f"{name}.json"
        write_json(path, config_dict(config))
        paths.append(path)
    return paths
