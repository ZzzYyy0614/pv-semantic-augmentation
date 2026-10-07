"""Fusion strategies expose a shared (visual, prior) -> feature contract."""

import torch
from torch import nn
from .edge import GateFusionBlock
from ..registry import Registry

FUSIONS = Registry[nn.Module]("fusion")


@FUSIONS.register("gated")
def gated(channels, edge_channels, reduction=4):
    return GateFusionBlock(channels, edge_channels, reduction)


class AdditiveFusion(nn.Module):
    def __init__(self, channels, edge_channels):
        super().__init__()
        self.project = nn.Conv2d(edge_channels, channels, 1, bias=False)

    def forward(self, visual, prior):
        return visual + self.project(prior)


@FUSIONS.register("additive")
def additive(channels, edge_channels, reduction=4):
    return AdditiveFusion(channels, edge_channels)


class ConcatenationFusion(nn.Module):
    def __init__(self, channels, edge_channels):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(channels + edge_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, visual, prior):
        return self.project(torch.cat([visual, prior], dim=1))


@FUSIONS.register("concat")
def concatenate(channels, edge_channels, reduction=4):
    return ConcatenationFusion(channels, edge_channels)
