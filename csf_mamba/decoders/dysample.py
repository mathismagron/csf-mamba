"""DySample : upsampling appris par échantillonnage dynamique (ChessMamba).

Beaucoup plus léger qu'un ConvTranspose : un petit générateur d'offsets déplace
une grille d'échantillonnage régulière, puis grid_sample interpole. Ici la
variante « style LP » (offsets bornés par un scope appris) ×2.
"""

import torch
from torch import nn


class DySample(nn.Module):
    def __init__(self, channels: int, scale: int = 2, groups: int = 4):
        super().__init__()
        self.scale = scale
        self.groups = groups
        self.offset = nn.Conv2d(channels, 2 * groups * scale * scale, kernel_size=1)
        self.scope = nn.Conv2d(channels, 2 * groups * scale * scale, kernel_size=1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        nn.init.constant_(self.scope.bias, 0.0)

    @staticmethod
    def _base_grid(h: int, w: int, device, dtype) -> torch.Tensor:
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device, dtype=dtype),
            torch.linspace(-1, 1, w, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack((xs, ys), dim=-1)  # (H, W, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        s = self.scale
        offset = (self.offset(x) * self.scope(x).sigmoid()) * 0.5  # (B, 2·G·s², H, W)

        offset = offset.view(b, self.groups, 2, s, s, h, w)
        offset = offset.permute(0, 1, 5, 3, 6, 4, 2).reshape(b, self.groups, h * s, w * s, 2)

        base = self._base_grid(h * s, w * s, x.device, x.dtype)
        grid = base.view(1, 1, h * s, w * s, 2) + offset
        grid = grid.reshape(b * self.groups, h * s, w * s, 2)

        xg = x.view(b, self.groups, c // self.groups, h, w).reshape(
            b * self.groups, c // self.groups, h, w
        )
        out = nn.functional.grid_sample(
            xg, grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        return out.view(b, c, h * s, w * s)
