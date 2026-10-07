"""Four-stage backbone contracts hide architecture-specific tensor layouts."""

from dataclasses import dataclass
from torch import nn
from torchvision import models
from ..registry import Registry


@dataclass(frozen=True)
class FeatureSpec:
    channels: tuple[int, ...]
    reductions: tuple[int, ...] = (4, 8, 16, 32)
    layout: str = "NCHW"

    def __post_init__(self):
        if len(self.channels) != 4 or len(self.reductions) != 4:
            raise ValueError("Native backbones must expose four feature stages")
        if (
            min(self.channels) < 1
            or min(self.reductions) < 1
            or self.layout not in {"NCHW", "NHWC"}
        ):
            raise ValueError("Invalid backbone feature specification")

    def spatial(self, feature):
        return feature.permute(0, 3, 1, 2).contiguous() if self.layout == "NHWC" else feature

    def native(self, feature):
        return feature.permute(0, 2, 3, 1).contiguous() if self.layout == "NHWC" else feature


@dataclass
class BackboneParts:
    stem: nn.Module
    stages: nn.ModuleList
    norm: nn.Module
    spec: FeatureSpec
    norm_position: str = "none"

    def __post_init__(self):
        if len(self.stages) != 4 or self.norm_position not in {"none", "before_pool", "after_pool"}:
            raise ValueError("Invalid backbone stage or normalization contract")


BACKBONES = Registry[BackboneParts]("backbone")


@BACKBONES.register("resnet18")
def resnet18(pretrained=False):
    base = models.resnet18(weights="DEFAULT" if pretrained else None)
    return BackboneParts(
        nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool),
        nn.ModuleList([base.layer1, base.layer2, base.layer3, base.layer4]),
        nn.Identity(),
        FeatureSpec((64, 128, 256, 512)),
    )


def _hierarchical_parts(base, layout, norm):
    stages = nn.ModuleList(
        [
            base.features[1],
            nn.Sequential(base.features[2], base.features[3]),
            nn.Sequential(base.features[4], base.features[5]),
            nn.Sequential(base.features[6], base.features[7]),
        ]
    )
    return BackboneParts(
        base.features[0],
        stages,
        norm,
        FeatureSpec((96, 192, 384, 768), layout=layout),
        norm_position="before_pool" if layout == "NHWC" else "after_pool",
    )


@BACKBONES.register("convnext_tiny")
def convnext_tiny(pretrained=False):
    base = models.convnext_tiny(weights="DEFAULT" if pretrained else None)
    return _hierarchical_parts(base, "NCHW", base.classifier[0])


@BACKBONES.register("swin_t")
def swin_t(pretrained=False):
    base = models.swin_t(weights="DEFAULT" if pretrained else None)
    return _hierarchical_parts(base, "NHWC", base.norm)
