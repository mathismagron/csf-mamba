"""Décodeur binaire (BCD) : lieu de la contribution.

À chaque stage, le C²S²-Block a déjà fusionné la paire bi-temporelle en Z_i. Le
décodeur remonte la pyramide (DySample + depthwise), produit la carte de
changement finale Y_BCD, et expose les cartes intermédiaires {CM_i} qui guident
le décodeur sémantique via la CGA.
"""

import torch
from torch import nn

from .dysample import DySample


class UpBlock(nn.Module):
    """Fusion top-down : upsample le grossier, ajoute le fin, raffine (depthwise)."""

    def __init__(self, in_channels: int, skip_channels: int):
        super().__init__()
        self.up = DySample(in_channels)
        self.reduce = nn.Conv2d(in_channels, skip_channels, kernel_size=1)
        self.refine = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, kernel_size=3, padding=1, groups=skip_channels),
            nn.GroupNorm(1, skip_channels),
            nn.GELU(),
        )

    def forward(self, coarse: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.refine(self.reduce(self.up(coarse)) + skip)


class BinaryDecoder(nn.Module):
    """{Z_i} (fin -> grossier) -> Y_BCD (B,2,H,W) et {CM_i} par stage."""

    def __init__(self, channels: tuple[int, ...] = (96, 192, 384, 768), num_change: int = 2):
        super().__init__()
        self.ups = nn.ModuleList(
            UpBlock(channels[i + 1], channels[i]) for i in reversed(range(len(channels) - 1))
        )
        # Une tête de changement par stage : sert Y_BCD (stage le plus fin) et {CM_i}.
        self.change_heads = nn.ModuleList(
            nn.Conv2d(c, num_change, kernel_size=1) for c in channels
        )
        self.final_up = DySample(channels[0])
        self.num_change = num_change

    def forward(self, feats: list[torch.Tensor]) -> dict:
        current = feats[-1]
        decoded = [current]
        for up, skip in zip(self.ups, reversed(feats[:-1])):
            current = up(current, skip)
            decoded.append(current)

        decoded = decoded[::-1]  # [stage1(fin) ... stage4(grossier)]
        change_maps = [head(feat) for head, feat in zip(self.change_heads, decoded)]

        # Y_BCD : on remonte le stage le plus fin (stride 4) à pleine résolution
        # via DySample appris, puis tête de changement à échelle 1.
        y_bcd = self.change_heads[0](self.final_up(self.final_up(decoded[0])))
        return {"y_bcd": y_bcd, "change_maps": change_maps, "features": decoded}
