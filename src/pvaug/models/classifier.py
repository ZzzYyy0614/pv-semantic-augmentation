import torch
from torch import nn
import torch.nn.functional as F
from .edge import EdgeBlock, SobelPrior
from ..config import ModelConfig
from .backbones import BACKBONES
from .fusion import FUSIONS
from .compatibility import build_recovered, recovered_features


class DefectClassifier(nn.Module):
    """Three fusion points between four backbone stages, following Fig. 6."""

    def __init__(self, config: ModelConfig, num_classes=5, normalize=True):
        super().__init__()
        self.config = config
        self.normalize = normalize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.sobel = SobelPrior()
        if config.variant == "recovered":
            self.core = build_recovered(config, num_classes)
            return
        parts = BACKBONES.build(config.backbone, pretrained=config.pretrained)
        self.stem, self.stages, self.norm = parts.stem, parts.stages, parts.norm
        self.feature_spec = parts.spec
        self.norm_position = parts.norm_position
        self.head = nn.Linear(parts.spec.channels[-1], num_classes)
        channels, stem_stride = parts.spec.channels, parts.spec.reductions[0]
        if config.fusion:
            widths = [max(8, int(c * config.edge_width_ratio)) for c in channels[:3]]
            self.edge_stem = EdgeBlock(1, widths[0], stem_stride, stem_stride, 0)
            self.edge_blocks = nn.ModuleList(
                [
                    EdgeBlock(widths[0], widths[0], 1),
                    EdgeBlock(widths[0], widths[1]),
                    EdgeBlock(widths[1], widths[2]),
                ]
            )
            self.fusions = nn.ModuleList(
                [
                    FUSIONS.build(
                        config.fusion_kind,
                        channels=c,
                        edge_channels=e,
                        reduction=config.gate_reduction,
                    )
                    for c, e in zip(channels, widths)
                ]
            )

    def forward_features(self, image, edge=None):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected Bx3xHxW RGB in [0,1]")
        normalized = (image - self.mean) / self.std if self.normalize else image
        if self.config.variant == "recovered":
            core = self.core
            if self.config.fusion:
                edge = self.sobel(image) if edge is None else edge
                if not self.config.edge_prior:
                    edge = image.mean(1, keepdim=True)
                edge = (edge - 0.485) / 0.229 if self.normalize else edge
            return recovered_features(core, self.config, normalized, edge)
        x = self.stem(normalized)
        if self.config.fusion:
            prior = (
                (self.sobel(image) if edge is None else edge)
                if self.config.edge_prior
                else image.mean(1, keepdim=True)
            )
            edge = self.edge_stem(prior)
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index < 3 and self.config.fusion:
                edge = self.edge_blocks[index](edge)
                spatial = self.feature_spec.spatial(x)
                if edge.shape[-2:] != spatial.shape[-2:]:
                    edge = F.interpolate(
                        edge, size=spatial.shape[-2:], mode="bilinear", align_corners=False
                    )
                x = self.feature_spec.native(spatial + self.fusions[index](spatial, edge))
        if self.norm_position == "before_pool":
            x = self.norm(x)
        return self.feature_spec.spatial(x)

    def classify_features(self, features):
        if self.config.variant == "recovered":
            if self.config.backbone == "resnet18":
                return self.core.fc(self.core.avg_pool(features).flatten(1))
            return self.core.head(self.core.norm(features.mean([-2, -1])))
        pooled = F.adaptive_avg_pool2d(features, 1)
        if self.norm_position == "after_pool":
            pooled = self.norm(pooled)
        return self.head(pooled.flatten(1))

    def forward(self, image, edge=None):
        return self.classify_features(self.forward_features(image, edge))
