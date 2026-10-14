"""Entrelacement damier + parcours serpent (ChessMamba).

Zero paramètre. Construit deux composites complémentaires à partir de la paire
bi-temporelle, puis les sérialise en une séquence 1D unique où chaque token est
entouré de voisins de phase opposée.
"""

import torch


def chessboard_mask(height: int, width: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Masque damier M de forme (1, 1, H, W) : 1 sur les cases (i+j) pair."""
    rows = torch.arange(height, device=device).view(-1, 1)
    cols = torch.arange(width, device=device).view(1, -1)
    return ((rows + cols) % 2 == 0).to(dtype).view(1, 1, height, width)


def composites(f1: torch.Tensor, f2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """X^a = M⊙F1 + (1−M)⊙F2, X^b complémentaire. Entrées/sorties (B,C,H,W)."""
    mask = chessboard_mask(f1.shape[-2], f1.shape[-1], f1.device, f1.dtype)
    return mask * f1 + (1 - mask) * f2, mask * f2 + (1 - mask) * f1


def _snake(x: torch.Tensor) -> torch.Tensor:
    """Inverse une ligne sur deux le long de la largeur : (B,C,H,W) -> (B,C,H,W)."""
    x = x.clone()
    x[:, :, 1::2] = x[:, :, 1::2].flip(-1)
    return x


def interleave(xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
    """(B,C,H,W) x2 -> séquence (B, L, C) avec L = H·2W, ordre serpent.

    Les composites alternent le long de la largeur, donc deux tokens voisins
    dans la séquence proviennent toujours de phases opposées.
    """
    b, c, h, w = xa.shape
    woven = torch.stack((xa, xb), dim=-1).reshape(b, c, h, 2 * w)
    return _snake(woven).flatten(2).transpose(1, 2).contiguous()


def deinterleave(seq: torch.Tensor, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse exact de `interleave` : (B, L, C) -> deux tenseurs (B,C,H,W)."""
    b, length, c = seq.shape
    assert length == height * 2 * width, f"séquence {length} ≠ H·2W = {height * 2 * width}"
    woven = _snake(seq.transpose(1, 2).reshape(b, c, height, 2 * width))
    unwoven = woven.reshape(b, c, height, width, 2)
    return unwoven[..., 0], unwoven[..., 1]
