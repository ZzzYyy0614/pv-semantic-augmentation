import torch
from torch import nn
import torch.nn.functional as F


class SobelPrior(nn.Module):
    """Fixed Sobel magnitude on unnormalized [0,1] RGB; per-image max normalization."""

    def __init__(self):
        super().__init__()
        kernels = torch.tensor(
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]],
            dtype=torch.float32,
        )
        self.register_buffer("kernels", kernels.unsqueeze(1))
        self.register_buffer("rgb_weights", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

    def forward(self, image):
        gray = (image * self.rgb_weights).sum(1, keepdim=True)
        mode = "reflect" if min(gray.shape[-2:]) > 1 else "replicate"
        gradients = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode=mode), self.kernels)
        # Clamp the magnitude rather than add epsilon, preserving zero edges on flat images.
        magnitude = gradients.square().sum(1, keepdim=True).sqrt()
        return magnitude / magnitude.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)


class EdgeBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=2, kernel_size=3, padding=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class GateFusionBlock(nn.Module):
    """Two spatial maps: f = sigmoid(g1)*backbone + sigmoid(g2)*edge (Fig. 6)."""

    def __init__(self, channels, edge_channels=None, reduction=4):
        super().__init__()
        edge_channels = edge_channels or channels
        self.project = (
            nn.Identity()
            if edge_channels == channels
            else nn.Conv2d(edge_channels, channels, 1, bias=False)
        )
        hidden = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, 2, 1), nn.Sigmoid()
        )

    def forward(self, backbone, edge):
        edge = self.project(edge)
        if edge.shape[-2:] != backbone.shape[-2:]:
            edge = F.interpolate(
                edge, size=backbone.shape[-2:], mode="bilinear", align_corners=False
            )
        w_backbone, w_edge = self.gate(torch.cat([backbone, edge], dim=1)).split(1, dim=1)
        return w_backbone * backbone + w_edge * edge
