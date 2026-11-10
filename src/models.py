"""
Learned surface-area correctors.

Both models predict y = log(A_true / A_mc_grey), the log-ratio between the
reference area and the greyscale marching-cubes area. The final estimate is
A_pred = A_mc_grey * exp(y_pred). Predicting a dimensionless correction makes
the target independent of object size, and frames the network as a
bias-correction layer on top of the classical mesh measurement.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_bn(cin: int, cout: int) -> list[nn.Module]:
    return [nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True)]


class CNN3D(nn.Module):
    """5-block 3D CNN encoder + regression head.

    Block 1 has one conv (cheap at full resolution); blocks 2-5 have two.
    Each block ends in 2x max-pooling, so a 64^3 input reaches 2^3 and a 32^3
    input reaches 1^3. Global average and max pooling make the head
    independent of the input size.
    """

    def __init__(self, channels=(16, 32, 64, 128, 256), dropout: float = 0.2):
        super().__init__()
        layers, cin = [], 1
        for i, c in enumerate(channels):
            layers += conv_bn(cin, c)
            if i > 0:
                layers += conv_bn(c, c)
            layers.append(nn.MaxPool3d(2))
            cin = c
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(2 * cin, 128), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x):
        h = self.encoder(x)
        h = torch.cat([h.mean(dim=(2, 3, 4)), h.amax(dim=(2, 3, 4))], 1)
        return self.head(h).squeeze(1)


class PointNet(nn.Module):
    """PointNet regressor: shared MLP (64, 64, 64, 128, 1024), global max-pool,
    MLP head 512 -> 128 -> 1. Rotation robustness comes from random SO(3)
    augmentation instead of a T-Net (the target is rotation invariant)."""

    def __init__(self, widths=(64, 64, 64, 128, 1024), dropout: float = 0.3):
        super().__init__()
        layers, cin = [], 3
        for w in widths:
            layers += [nn.Conv1d(cin, w, 1, bias=False), nn.BatchNorm1d(w), nn.ReLU(inplace=True)]
            cin = w
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(cin, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(512, 128), nn.BatchNorm1d(128),
                                  nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, pts):            # pts: (B, n, 3)
        h = self.mlp(pts.transpose(1, 2))
        return self.head(h.amax(dim=2)).squeeze(1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
