"""C²S²-Block : la fusion bi-temporelle, cœur de la contribution.

Le bloc consomme *directement* la paire (X^T1, X^T2) et la remplace par un
tenseur fusionné : il n'y a pas de concaténation 5C séparée en amont comme dans
Mamba-FCS (cf. §7 du plan).

Deux variantes concurrentes, jamais mariées (cf. §6) :
  core="chess" -> damier + MCA-SF + S6 standard   (défaut, prouvé en multi-classe)
  core="l1"    -> récurrence CSSM-L1 sur flux propres F1/F2  (ablation)
"""

import torch
from torch import nn

from .chessboard import composites, deinterleave, interleave
from .cssm import CSSML1
from .mca_sf import MCASF
from .ssm import SSMScan


class C2S2Block(nn.Module):
    def __init__(
        self,
        channels: int,
        core: str = "chess",
        d_state: int = 16,
        backend: str = "auto",
        distance: str = "l1",
    ):
        super().__init__()
        if core not in {"chess", "l1"}:
            raise ValueError(f"core inconnu : {core!r} (attendu 'chess' ou 'l1')")
        self.core = core
        self.norm = nn.LayerNorm(channels)

        if core == "chess":
            self.local = MCASF(channels)
            self.scan = SSMScan(channels, d_state=d_state, backend=backend)
        else:
            self.scan = CSSML1(channels, d_state=d_state, distance=distance)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) x2 -> Z (B,C,H,W)."""
        height, width = f1.shape[-2:]

        if self.core == "l1":
            y1, y2 = self.scan(f1, f2)
            return y1 + y2

        xa, xb = composites(f1, f2)
        xa, xb = self.local(xa), self.local(xb)

        seq = interleave(xa, xb)
        seq = self.scan(self.norm(seq)) + seq
        ya, yb = deinterleave(seq, height, width)
        return ya + yb
