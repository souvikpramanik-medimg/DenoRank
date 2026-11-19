"""Downstream segmenter.

DenoRank only needs a differentiable binary segmenter, so the segmenter is a plug-in:
a plain U-Net ships as the default, and any ``nn.Module`` mapping (B,3,H,W) -> (B,1,H,W)
logits can be registered with :func:`register_segmenter`.
"""
from typing import Callable, Dict

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Standard 4-level U-Net returning raw logits."""

    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 32):
        super().__init__()
        self.e1, self.e2 = DoubleConv(in_ch, base), DoubleConv(base, base * 2)
        self.e3, self.e4 = DoubleConv(base * 2, base * 4), DoubleConv(base * 4, base * 8)
        self.bottleneck = DoubleConv(base * 8, base * 16)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2); self.d4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2); self.d3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2); self.d2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2); self.d1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.d4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.d3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


_REGISTRY: Dict[str, Callable[..., nn.Module]] = {"unet": lambda base=32: UNet(base=base)}


def register_segmenter(name: str, builder: Callable[..., nn.Module]) -> None:
    """Register a custom segmenter: ``builder(base=int) -> nn.Module``."""
    _REGISTRY[name.lower()] = builder


def build_segmenter(name: str = "unet", base: int = 32) -> nn.Module:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown segmenter '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key](base=base)
