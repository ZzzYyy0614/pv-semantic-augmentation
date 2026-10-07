"""Deterministic YAML composition with typed validation and explicit override paths."""

from dataclasses import fields
from pathlib import Path
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException
from ..config import TrainConfig, LDMConfig, ModelConfig, EngineConfig

TASKS = {"classification": TrainConfig, "diffusion": LDMConfig, "autoencoder": LDMConfig}
SECTIONS = {
    "schema_version",
    "task",
    "data",
    "model",
    "optimizer",
    "trainer",
    "semantic",
    "experiment",
}


def compose(path, overrides=(), _stack=()):
    """Defaults are merged in order, then this file, then command-line dot overrides."""
    path = Path(path).resolve()
    if path in _stack:
        raise ValueError(
            "Configuration inheritance cycle: " + " -> ".join(map(str, (*_stack, path)))
        )
    current = OmegaConf.load(path)
    if not OmegaConf.is_dict(current):
        raise ValueError("Configuration root must be a mapping")
    defaults = current.pop("defaults", [])
    if not isinstance(defaults, (list, tuple)) and not OmegaConf.is_list(defaults):
        raise ValueError("defaults must be a list of relative configuration paths")
    merged = OmegaConf.create({})
    for parent in defaults:
        if not isinstance(parent, str):
            raise ValueError("Each defaults entry must be a configuration path")
        merged = OmegaConf.merge(merged, compose(path.parent / parent, _stack=(*_stack, path)))
    merged = OmegaConf.merge(merged, current)
    for override in overrides:
        key, separator, _ = override.partition("=")
        if not separator:
            raise ValueError(f"Override must be key=value: {override}")
        cursor = OmegaConf.to_container(merged, resolve=False)
        for part in key.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                raise ValueError(f"Unknown override path: {key}")
            cursor = cursor[part]
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist([override]))
    # Resolve only after composition, so parent interpolations can reference child values.
    return merged


def _check_keys(mapping, allowed, context):
    if not isinstance(mapping, dict):
        raise ValueError(f"{context} must be a mapping")
    unknown = set(mapping) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {context} fields: {sorted(unknown)}")


def _construct(cls, values):
    try:
        typed = OmegaConf.merge(OmegaConf.structured(cls), OmegaConf.create(values))
        config = OmegaConf.to_object(typed)
    except OmegaConfBaseException as error:
        raise ValueError(f"Invalid typed configuration: {error}") from error
    if isinstance(config, TrainConfig) and config.model.variant == "paper":
        from ..models.backbones import BACKBONES
        from ..models.fusion import FUSIONS

        if config.model.backbone not in BACKBONES.names:
            raise ValueError(f"Unknown registered backbone: {config.model.backbone}")
        if config.model.fusion_kind not in FUSIONS.names:
            raise ValueError(f"Unknown registered fusion: {config.model.fusion_kind}")
    return config


def to_execution_config(document, kind=None):
    """The hierarchical public schema compiles to small, checkpoint-stable dataclasses."""
    if "schema_version" not in document:
        cls = kind or TrainConfig
        return _construct(cls, document)
    _check_keys(document, SECTIONS, "root")
    if document["schema_version"] != 2:
        raise ValueError("Only schema_version=2 is supported")
    task = document.get("task")
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    cls = TASKS[task]
    if kind is not None and cls is not kind:
        raise ValueError(f"Config task {task} cannot be used as {kind.__name__}")
    flat = {}
    groups = {
        "data": {
            "manifest",
            "generated_manifest",
            "classes",
            "image_size",
            "normalize",
            "generated_ratio",
            "conventional_augmentation",
            "resample_defects",
        },
        "optimizer": {"name", "lr", "momentum", "weight_decay", "schedule"},
        "trainer": {"epochs", "batch_size", "workers", "device", "max_steps", "engine"},
        "semantic": {"teacher", "temperature", "alpha_start", "alpha_end"},
        "experiment": {"output", "seed"},
    }
    aliases = {"name": "optimizer", "schedule": "lr_schedule"}
    allowed = {item.name for item in fields(cls)}
    for section, keys in groups.items():
        values = document.get(section, {})
        _check_keys(values, keys, section)
        for key, value in values.items():
            target = aliases.get(key, key)
            if target not in allowed:
                raise ValueError(f"{section}.{key} does not apply to {task}")
            flat[target] = value
    model = document.get("model", {})
    if cls is TrainConfig:
        _check_keys(model, {item.name for item in fields(ModelConfig)}, "model")
        flat["model"] = model
    else:
        sections = {
            "first_stage": {
                "first_stage_checkpoint",
                "smoke_random_first_stage",
                "first_stage_channels",
                "first_stage_mult",
                "first_stage_res_blocks",
                "latent_channels",
                "n_embed",
                "scale_factor",
            },
            "denoiser": {
                "model_channels",
                "channel_mult",
                "num_res_blocks",
                "num_heads",
                "attention_resolutions",
                "gradient_checkpointing",
            },
            "conditioning": {"cross_attention", "cross_attention_dim"},
            "diffusion": {
                "timesteps",
                "beta_schedule",
                "beta_start",
                "beta_end",
                "loss_type",
                "ema_decay",
            },
            "initialization": {"initialize"},
        }
        _check_keys(model, sections, "model")
        for section, keys in sections.items():
            values = model.get(section, {})
            _check_keys(values, keys, "model." + section)
            flat.update(values)
    if "engine" in flat:
        _check_keys(flat["engine"], {item.name for item in fields(EngineConfig)}, "trainer.engine")
    return _construct(cls, flat)


def resolve_config(path, overrides=()):
    composed = compose(path, overrides)
    return OmegaConf.to_container(composed, resolve=True, throw_on_missing=True)


def load_config(path, kind=None, overrides=()):
    return to_execution_config(resolve_config(path, overrides), kind)
