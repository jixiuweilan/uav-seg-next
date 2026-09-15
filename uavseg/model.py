"""Approved compact U-Net, imported only by execution-side model code."""

import torch
from torch import nn
from torch.nn import functional as F

from .common import AuditError
from .contracts import MODEL_ID


def block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.GroupNorm(8, out_channels), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.GroupNorm(8, out_channels), nn.ReLU())


class CompactUNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths = (16, 32, 64, 128)
        self.encoder = nn.ModuleList(block(a, b) for a, b in zip((3, *widths[:-1]), widths))
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = block(128, 256)
        self.decoder = nn.ModuleList(block(a + b, b) for a, b in
                                     zip((256, 128, 64, 32), reversed(widths)))
        self.head = nn.Conv2d(16, 8, 1)

    def forward(self, value):
        if (value.ndim != 4 or value.shape[0] < 1 or value.shape[1] != 3 or
                min(value.shape[-2:]) < 16 or not value.is_floating_point()):
            raise AuditError('模型输入须为 N×3×H×W 浮点张量，H、W 至少16')
        skips = []
        for stage in self.encoder:
            value = stage(value)
            skips.append(value)
            value = self.pool(value)
        value = self.bottleneck(value)
        for stage, skip in zip(self.decoder, reversed(skips)):
            value = F.interpolate(value, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            value = stage(torch.cat((value, skip), dim=1))
        return self.head(value)
