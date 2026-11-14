"""Compact CNNs for native 64x64 multispectral input (baselines B3/B6/B7 and the PG-ASST encoder).

Common interface for every model in this project:
  model.embed(x)  -> (B, D) latent vector (used by the physics auxiliary head)
  model.head(z)   -> logits
  model(x)        -> logits
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SE(nn.Module):
    """Squeeze-and-Excitation over channels (Hu et al.)."""

    def __init__(self, ch: int, reduction: int = 4, min_hidden: int = 4):
        super().__init__()
        h = max(ch // reduction, min_hidden)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch, h), nn.ReLU(inplace=True),
                                nn.Linear(h, ch), nn.Sigmoid())
        self.last_weights: torch.Tensor | None = None

    def forward(self, x):
        w = self.fc(x)
        self.last_weights = w.detach()
        return x * w.view(x.shape[0], -1, 1, 1)


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, se: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.se = SE(cout) if se else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.short = nn.Identity()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        return self.act(out + self.short(x))


class SmallResNet(nn.Module):
    """ResNet-style CNN for 64x64 patches.

    Stem: 3x3 conv stride 2 -> 32x32; stages at 32, 16, 8, 4 px (strides 2, 4, 8, 16). The stride-8 stage
    (8x8 grid) feeds the Transformer (decision D-007). Stages double channels and halve resolution.
    ``in_se`` adds a soft per-image SE gate over the INPUT bands (baseline B7, weights map to wavelengths);
    ``se`` adds SE on feature channels in every block (baseline B6, weights do not map to wavelengths).
    """

    def __init__(self, in_ch: int = 12, num_classes: int = 10, widths=(32, 64, 128, 256), blocks=(1, 1, 1, 1),
                 stem_stride: int = 2, dropout: float = 0.2, se: bool = False, in_se: bool = False):
        super().__init__()
        self.in_se = SE(in_ch, reduction=2, min_hidden=4) if in_se else nn.Identity()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, widths[0], 3, stem_stride, 1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        stages, cin = [], widths[0]
        for i, (w, n) in enumerate(zip(widths, blocks)):
            layers = []
            for j in range(n):
                stride = 2 if (j == 0 and i > 0) else 1
                layers.append(BasicBlock(cin, w, stride, se))
                cin = w
            stages.append(nn.Sequential(*layers))
        self.stages = nn.ModuleList(stages)
        self.out_dim = cin
        self.stage_dims = list(widths)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(cin, num_classes))

    def feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(self.in_se(x))
        maps = []
        for s in self.stages:
            x = s(x)
            maps.append(x)
        return maps

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.feature_maps(x)[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


class GatedClassifier(nn.Module):
    """Band gate in front of an encoder (any model with embed/head)."""

    def __init__(self, gate: nn.Module, encoder: nn.Module):
        super().__init__()
        self.gate = gate
        self.encoder = encoder

    def embed(self, x):
        return self.encoder.embed(self.gate(x))

    def head(self, z):
        return self.encoder.head(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.gate(x))
