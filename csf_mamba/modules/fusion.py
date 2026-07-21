"""Briques de fusion à zéro (ou quasi zéro) paramètre : FFT2 et CGA.

Ce sont les contributions de Mamba-FCS qui ne coûtent rien en budget.
"""

import torch
from torch import nn


class FFTBranch(nn.Module):
    """Injection spatio-fréquentielle : [Z, |Ff1 − Ff2|] -> conv 1x1 -> C.

    La FFT n'a aucun poids appris ; seule la conv 1x1 de fusion (2C -> C) compte.
    Réservée aux stages haute résolution (1–2), cf. §5 du plan.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.fuse = nn.Conv2d(2 * channels, channels, kernel_size=1)

    @staticmethod
    def log_amplitude(x: torch.Tensor) -> torch.Tensor:
        """Log-amplitude du spectre 2D, canal par canal. Sortie réelle."""
        spectrum = torch.fft.fft2(x.float(), norm="ortho")
        return torch.log1p(spectrum.abs()).to(x.dtype)

    def forward(self, z: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        delta = (self.log_amplitude(f1) - self.log_amplitude(f2)).abs()
        return self.fuse(torch.cat((z, delta), dim=1))


class ResidualCGA(nn.Module):
    """Change-Guided Attention résiduelle : X̂ = X · (1 + σ(CM)).

    Le gate résiduel est imposé par le format Hi-UCD (sémantique pleine-scène,
    cf. §5.1) : un gate multiplicatif X⊙σ(CM) annulerait le signal dans les
    régions inchangées, qui doivent pourtant être classées correctement.
    """

    def __init__(self, change_channels: int = 2):
        super().__init__()
        self.to_gate = nn.Conv2d(change_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, change_map: torch.Tensor) -> torch.Tensor:
        if change_map.shape[-2:] != x.shape[-2:]:
            change_map = nn.functional.interpolate(
                change_map, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        return x * (1 + torch.sigmoid(self.to_gate(change_map)))
