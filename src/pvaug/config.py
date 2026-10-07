"""Typed execution configuration, including compatibility with v1 checkpoints."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

CLASSES = ["defect_free", "micro_crack", "busbar_corrosion", "break", "black_core"]


@dataclass
class EngineConfig:
    precision: str = "fp32"
    accumulation_steps: int = 1
    clip_grad_norm: float | None = None
    checkpoint_every: int = 1

    def __post_init__(self):
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        if self.accumulation_steps < 1 or self.checkpoint_every < 1:
            raise ValueError("Accumulation and checkpoint intervals must be positive")
        if self.clip_grad_norm is not None and self.clip_grad_norm <= 0:
            raise ValueError("clip_grad_norm must be positive")


@dataclass
class ModelConfig:
    backbone: str = "resnet18"
    variant: str = "recovered"
    edge_prior: bool = True
    fusion: bool = True
    pretrained: bool = False
    gate_reduction: int = 4
    edge_width_ratio: float = 1.0
    fusion_kind: str = "gated"

    def __post_init__(self):
        if not self.backbone:
            raise ValueError("Backbone name cannot be empty")
        if self.variant not in {"recovered", "paper"}:
            raise ValueError("variant must be recovered or paper")
        if self.variant == "recovered" and self.backbone not in {"resnet18", "convnext_tiny"}:
            raise ValueError("Recovered adapter supports ResNet/ConvNeXt; use variant=paper")
        if self.variant == "recovered" and self.pretrained:
            raise ValueError("Use --init to load recovered weights; torchvision weights differ")
        if self.gate_reduction < 1 or self.edge_width_ratio <= 0:
            raise ValueError("Gate reduction and edge width ratio must be positive")
        if not self.fusion_kind:
            raise ValueError("Fusion name cannot be empty")
        if self.variant == "recovered" and self.fusion_kind != "gated":
            raise ValueError("Alternative fusion strategies require variant=paper")


@dataclass
class TrainConfig:
    manifest: str = "data/real.csv"
    generated_manifest: str | None = None
    output: str = "runs/teacher"
    teacher: str | None = None
    classes: list[str] = field(default_factory=lambda: CLASSES.copy())
    model: ModelConfig = field(default_factory=ModelConfig)
    image_size: int = 224
    normalize: bool = True
    epochs: int = 100
    batch_size: int = 24
    lr: float = 5e-4
    momentum: float = 0.9
    weight_decay: float = 5e-2
    lr_schedule: str = "constant"
    temperature: float = 15.0
    alpha_start: float = 0.7
    alpha_end: float = 0.3
    generated_ratio: float = 1.0
    conventional_augmentation: bool = False
    resample_defects: bool = False
    seed: int = 42
    workers: int = 0
    device: str = "auto"
    max_steps: int | None = None
    optimizer: str = "sgd"
    engine: EngineConfig = field(default_factory=EngineConfig)

    def __post_init__(self):
        if isinstance(self.engine, dict):
            self.engine = EngineConfig(**self.engine)
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if len(self.classes) < 2 or len(set(self.classes)) != len(self.classes):
            raise ValueError("Need at least two distinct class names")
        if self.image_size < 32 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("Invalid image size, epochs, or batch size")
        if self.lr <= 0 or self.weight_decay < 0 or not 0 <= self.momentum < 1:
            raise ValueError("Invalid optimizer settings")
        if self.temperature <= 0 or not 0 <= self.alpha_start <= 1 or not 0 <= self.alpha_end <= 1:
            raise ValueError("Invalid distillation settings")
        if self.generated_ratio < 0 or self.workers < 0:
            raise ValueError("Invalid generated ratio or workers")
        if self.lr_schedule not in {"constant", "cosine"}:
            raise ValueError("lr_schedule must be constant or cosine")
        if self.optimizer not in {"sgd", "adam", "adamw"}:
            raise ValueError("Unsupported optimizer")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive")


@dataclass
class LDMConfig:
    manifest: str = "data/real.csv"
    output: str = "runs/sketch_ldm"
    classes: list[str] = field(default_factory=lambda: CLASSES[1:].copy())
    first_stage_checkpoint: str | None = "weights/vqvae.ckpt"
    smoke_random_first_stage: bool = False
    first_stage_channels: int = 128
    first_stage_mult: list[int] = field(default_factory=lambda: [1, 2, 4])
    first_stage_res_blocks: int = 2
    latent_channels: int = 3
    n_embed: int = 256
    image_size: int = 256
    batch_size: int = 4
    lr: float = 1e-6
    epochs: int = 200
    timesteps: int = 1000
    beta_schedule: str = "scaled_linear"
    beta_start: float = 0.0015
    beta_end: float = 0.0205
    model_channels: int = 256
    channel_mult: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    num_res_blocks: int = 2
    num_heads: int = 8
    attention_resolutions: list[int] = field(default_factory=lambda: [8, 4, 2])
    cross_attention_dim: int = 512
    cross_attention: bool = True
    loss_type: str = "l2"
    scale_factor: float = 1.0
    gradient_checkpointing: bool = False
    ema_decay: float = 0.9999
    initialize: str | None = None
    seed: int = 42
    workers: int = 0
    device: str = "auto"
    max_steps: int | None = None
    weight_decay: float = 0.0
    optimizer: str = "adam"
    lr_schedule: str = "constant"
    engine: EngineConfig = field(default_factory=EngineConfig)

    def __post_init__(self):
        if isinstance(self.engine, dict):
            self.engine = EngineConfig(**self.engine)
        if self.optimizer not in {"sgd", "adam", "adamw"} or self.weight_decay < 0:
            raise ValueError("Invalid optimizer")
        if self.lr_schedule not in {"constant", "cosine"}:
            raise ValueError("Invalid learning rate schedule")
        if self.image_size < 32 or self.image_size % 8:
            raise ValueError("Diffusion image size must be >=32 and divisible by 8")
        if self.epochs < 1 or self.batch_size < 1 or self.lr <= 0 or self.timesteps < 2:
            raise ValueError("Invalid diffusion training parameters")
        if self.model_channels < 32 or self.model_channels % 32:
            raise ValueError("model_channels must be divisible by 32")
        if self.first_stage_channels < 32 or self.first_stage_channels % 32:
            raise ValueError("first_stage_channels must be divisible by 32")
        divisor = 2 ** (len(self.first_stage_mult) + len(self.channel_mult) - 2)
        if self.image_size % divisor:
            raise ValueError(f"Image size must be divisible by total downsampling {divisor}")
        if len(self.classes) < 2 or len(set(self.classes)) != len(self.classes):
            raise ValueError("Invalid classes")
        if self.loss_type not in {"l1", "l2"} or not 0 <= self.ema_decay < 1:
            raise ValueError("Invalid loss or EMA")
        if not 0 < self.beta_start < self.beta_end < 1 or self.workers < 0:
            raise ValueError("Invalid scheduler or workers")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive")


def read_config(path, kind=TrainConfig):
    from .configuration import load_config

    return load_config(path, kind=kind)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def config_dict(config):
    return asdict(config)
