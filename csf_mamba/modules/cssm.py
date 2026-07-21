"""Récurrence CSSM change-aware (variante d'ablation, cf. §6 du plan).

    h_t      = A·h_{t−1} + d(B'_t·z^post_t , B_t·z^pre_t)
    y^pre_t  = C_t·h_t  + D_t·z^pre_t
    y^post_t = C'_t·h_t + D'_t·z^post_t

Le changement pilote l'état caché au lieu d'être calculé après le SSM. Les
projections B/B' sont asymétriques : le pré passe *toujours* par B, le post
*toujours* par B'. C'est cette asymétrie qui porte la sensibilité au changement,
et c'est pourquoi elle est incompatible avec le damier (qui permute les dates
une case sur deux) — les deux variantes concourent, elles ne fusionnent pas.

⚠️ DEUX POINTS À CONFIRMER sur le dépôt CSSM (github.com/Elman295/CSSM), cf.
prérequis §12.2-5 du plan, avant de tirer une conclusion de l'ablation :
  1. L'axe de réduction de la distance. Ici : réduction sur la dimension d'état
     N -> un scalaire par canal, diffusé sur N. Cohérent pour l1/l2/cosinus,
     mais leur code fait peut-être un |·| élément par élément (qui garderait N).
  2. La normalisation d'entrée exacte (RMSNorm ?).

PyTorch pur, donc aucun risque kernel — mais scan séquentiel, donc lent (§11-3).
"""

import torch
from torch import nn


def _distance(post: torch.Tensor, pre: torch.Tensor, kind: str) -> torch.Tensor:
    """(B, d, N) x2 -> (B, d, 1). Ablation CSSM Table 3 : L1 > L2 > cosinus."""
    if kind == "l1":
        return (post - pre).abs().sum(dim=-1, keepdim=True)
    if kind == "l2":
        return (post - pre).pow(2).sum(dim=-1, keepdim=True).sqrt()
    if kind == "cosine":
        return 1 - nn.functional.cosine_similarity(post, pre, dim=-1).unsqueeze(-1)
    if kind == "chebyshev":
        return (post - pre).abs().amax(dim=-1, keepdim=True)
    raise ValueError(f"distance inconnue : {kind!r}")


class CSSML1(nn.Module):
    """Récurrence bi-temporelle sur flux propres. (B,C,H,W) x2 -> (B,C,H,W) x2."""

    def __init__(self, channels: int, d_state: int = 16, distance: str = "l1"):
        super().__init__()
        self.d_state = d_state
        self.distance = distance

        self.norm = nn.GroupNorm(1, channels)
        # Projections asymétriques : B pour le pré, B' pour le post.
        self.B_pre = nn.Linear(channels, d_state, bias=False)
        self.B_post = nn.Linear(channels, d_state, bias=False)
        self.C_pre = nn.Linear(channels, d_state, bias=False)
        self.C_post = nn.Linear(channels, d_state, bias=False)
        self.dt = nn.Linear(channels, channels, bias=True)

        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(channels, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D_pre = nn.Parameter(torch.ones(channels))
        self.D_post = nn.Parameter(torch.ones(channels))

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x).flatten(2).transpose(1, 2)  # (B, L, C)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = f1.shape
        z_pre, z_post = self._flatten(f1), self._flatten(f2)
        length = h * w

        dt = nn.functional.softplus(self.dt(z_pre + z_post))     # (B, L, C)
        a_mat = -torch.exp(self.A_log)                           # (C, N)
        da = torch.exp(dt.unsqueeze(-1) * a_mat)                 # (B, L, C, N)

        # Contributions asymétriques : B·z^pre et B'·z^post.
        bz_pre = self.B_pre(z_pre).unsqueeze(2) * dt.unsqueeze(-1)
        bz_post = self.B_post(z_post).unsqueeze(2) * dt.unsqueeze(-1)

        state = torch.zeros(b, c, self.d_state, device=f1.device, dtype=f1.dtype)
        out_pre, out_post = [], []
        for t in range(length):
            innovation = _distance(bz_post[:, t], bz_pre[:, t], self.distance)
            state = da[:, t] * state + innovation
            out_pre.append(torch.einsum("bdn,bn->bd", state, self.C_pre(z_pre[:, t])))
            out_post.append(torch.einsum("bdn,bn->bd", state, self.C_post(z_post[:, t])))

        y_pre = torch.stack(out_pre, dim=1) + self.D_pre * z_pre
        y_post = torch.stack(out_post, dim=1) + self.D_post * z_post

        reshape = lambda y: y.transpose(1, 2).reshape(b, c, h, w)  # noqa: E731
        return reshape(y_pre), reshape(y_post)
