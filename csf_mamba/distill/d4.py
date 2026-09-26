"""Groupe diédral D4 : convention UNIQUE partagée par le cache du professeur et l'élève.

Élément g = (hflip, k) : flip horizontal (axe W) optionnel, PUIS
`torch.rot90(·, k, dims=(H, W))`. Index de cache = 4·hflip + k. C'est la convention
avec laquelle le cache `perascd_second_train` a été écrit le 26 septembre (son
`meta.json` la recopie dans `d4.elements_hflip_then_rot90k`) : ne pas la modifier.
"""

import torch

D4 = [(h, k) for h in (0, 1) for k in range(4)]   # index = 4*h + k


def apply_d4(x: torch.Tensor, h: int, k: int) -> torch.Tensor:
    """x : (..., H, W). Flip horizontal optionnel, puis k quarts de tour."""
    if h:
        x = torch.flip(x, dims=[-1])
    return torch.rot90(x, k, dims=[-2, -1]) if k else x


def canonical_d4(h: int, v: int, k: int) -> int:
    """Index de cache de « hflip^h, puis vflip^v, puis rot90^k » — l'ordre de
    `csf_mamba.datasets.transforms` (RandomFlip puis RandomRot90). Déterminé par
    une sonde asymétrique plutôt que par une table écrite à la main."""
    probe = torch.arange(9.0).view(3, 3)
    x = probe
    if h:
        x = torch.flip(x, dims=[1])
    if v:
        x = torch.flip(x, dims=[0])
    if k:
        x = torch.rot90(x, k, dims=[0, 1])
    for i, (hh, kk) in enumerate(D4):
        if torch.equal(apply_d4(probe, hh, kk), x):
            return i
    raise AssertionError("élément de D4 introuvable")
