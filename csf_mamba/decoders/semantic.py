"""Décodeur sémantique SCD : un seul décodeur à poids partagés.

Économie de paramètres (÷2 vs deux décodeurs indépendants, cf. §5) : le même
décodeur traite T1 puis T2, distingués par un embedding temporel τ₁/τ₂ ajouté
aux features d'entrée. Chaque stage est guidé par la carte de changement CM_i du
décodeur binaire via une CGA résiduelle X·(1+σ(CM)).
"""

import torch
from torch import nn

from ..modules.fusion import ResidualCGA
from .dysample import DySample


class SharedSemanticDecoder(nn.Module):
    def __init__(
        self,
        channels: tuple[int, ...] = (96, 192, 384, 768),
        num_classes: int = 9,
        num_change: int = 2,
    ):
        super().__init__()
        self.channels = channels
        # Embedding temporel : un biais appris par canal et par date, par stage.
        self.tau = nn.ParameterList(
            nn.Parameter(torch.zeros(2, c)) for c in channels
        )
        self.cga = nn.ModuleList(ResidualCGA(num_change) for _ in channels)

        self.ups = nn.ModuleList()
        for i in reversed(range(len(channels) - 1)):
            self.ups.append(_SemUp(channels[i + 1], channels[i]))

        self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        self.final_up = DySample(channels[0])

    def _inject(self, feats: list[torch.Tensor], change_maps: list[torch.Tensor], date: int):
        out = []
        for stage, (feat, cm, tau, cga) in enumerate(zip(feats, change_maps, self.tau, self.cga)):
            biased = feat + tau[date].view(1, -1, 1, 1)
            out.append(cga(biased, cm))
        return out

    def _decode_one(self, feats, change_maps, date):
        guided = self._inject(feats, change_maps, date)
        current = guided[-1]
        for up, skip in zip(self.ups, reversed(guided[:-1])):
            current = up(current, skip)
        logits = self.head(self.final_up(self.final_up(current)))
        return logits, current  # logits pleine résolution + feature fine (pour L_sc)

    def forward(self, feats: list[torch.Tensor], change_maps: list[torch.Tensor]) -> dict:
        logits_t1, feat_t1 = self._decode_one(feats, change_maps, date=0)
        logits_t2, feat_t2 = self._decode_one(feats, change_maps, date=1)
        return {
            "sem_t1": logits_t1, "sem_t2": logits_t2,
            "feat_t1": feat_t1, "feat_t2": feat_t2,
        }


class _SemUp(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int):
        super().__init__()
        self.up = DySample(in_channels)
        self.reduce = nn.Conv2d(in_channels, skip_channels, kernel_size=1)
        self.refine = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, kernel_size=3, padding=1, groups=skip_channels),
            nn.GroupNorm(1, skip_channels),
            nn.GELU(),
        )

    def forward(self, coarse, skip):
        return self.refine(self.reduce(self.up(coarse)) + skip)
