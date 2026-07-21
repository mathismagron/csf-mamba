"""MCA-SF : pré-agrégation locale multi-dilatée co-conçue avec le damier.

À d=1, le noyau 3x3 n'a de poids qu'au centre et aux quatre coins — exactement
les voisins de *même phase* dans le damier. Les dilatations plus grandes
utilisent un noyau plein.
"""

import torch
from torch import nn

# Noyau centre-coins : les 4-voisins (croix) sont de phase opposée, on les exclut.
_CENTER_CORNERS = torch.tensor(
    [[1.0, 0.0, 1.0],
     [0.0, 1.0, 0.0],
     [1.0, 0.0, 1.0]]
)


class _MaskedDWConv(nn.Conv2d):
    """Depthwise 3x3 dilatée, avec masque optionnel figé sur le noyau."""

    def __init__(self, channels: int, dilation: int, masked: bool):
        super().__init__(
            channels, channels, kernel_size=3, padding=dilation,
            dilation=dilation, groups=channels, bias=False,
        )
        mask = _CENTER_CORNERS if masked else torch.ones(3, 3)
        self.register_buffer("kernel_mask", mask.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(x, self.weight * self.kernel_mask, self.bias)


class MCASF(nn.Module):
    """Agrégation locale : somme de branches depthwise dilatées, puis 1x1.

    Spatial préservé : (B,C,H,W) -> (B,C,H,W).
    """

    def __init__(self, channels: int, dilations: tuple[int, ...] = (1, 3, 5)):
        super().__init__()
        self.branches = nn.ModuleList(
            _MaskedDWConv(channels, d, masked=(d == 1)) for d in dilations
        )
        self.norm = nn.GroupNorm(1, channels)
        self.project = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        aggregated = sum(branch(x) for branch in self.branches)
        return x + self.project(self.act(self.norm(aggregated)))
