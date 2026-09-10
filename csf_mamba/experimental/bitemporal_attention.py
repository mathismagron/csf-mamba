"""Attention bi-temporelle jointe — brique hybride Mamba/Transformer.

POURQUOI CETTE BRIQUE, ET PAS UNE AUTRE
=======================================

Trois mesures du projet dictent la conception, plutôt qu'une intuition :

1. **Le goulot est la localisation du changement, pas la sémantique.** Diagnostic
   du 14 août : IoU du changement 0,577, mais 0,864 de justesse sémantique *à
   l'intérieur* des zones détectées. Et `SeK = κ·exp(IoU_fg)/e` : tout point de
   SeK passe par l'IoU. Une brique qui n'améliore pas la localisation ne sert à
   rien ici.

2. **La machinerie SSM ne pèse que 4 % du calcul**, contre 82 % pour les
   convolutions (comptage du 10 septembre). Le raisonnement « global » du modèle
   est donc bien plus mince que son étiquette « Mamba » ne le suggère.

3. **Un scan SSM est ordonné.** Deux positions éloignées dans l'ordre de balayage
   n'interagissent qu'à travers un état compressé. Or détecter un changement, c'est
   **apparier** une position à T1 avec la même zone à T2 — une opération par paires,
   sans ordre. C'est exactement ce que fait l'attention, et la littérature de la
   détection de changement le dit sans détour : *« change detection tasks are cross
   attention driven because the nature of the work is to compare bi-temporal
   information »* (RCDT, 2022).

CHOIX DE CONCEPTION
===================

**Séquence jointe plutôt que self + cross séparées.** On concatène les jetons des
deux dates en une seule séquence de 2·H·W et on laisse une attention unique opérer
dessus. Elle apprend seule à mélanger dans une date et entre les dates, là où deux
blocs séparés imposeraient la répartition — et coûteraient près du double.

**Aux stages profonds seulement.** L'attention est quadratique en nombre de jetons.
Au stage 4 d'une entrée 512², la grille fait 16×16 : 512 jetons pour les deux dates
réunies, un coût négligeable. Au stage 1 (128×128), ce serait 32 768 jetons et le
calcul exploserait. MambaVision (CVPR 2025) arrive à la même conclusion par une
autre route : placer quelques blocs d'attention **aux derniers stages** est ce qui
rapporte le plus, et le hybride y est plus rapide que le Mamba pur comme que le
ViT pur.

**Encodage de position par convolution depthwise**, et non par une table apprise.
Une table est liée à une résolution : elle casserait au changement de taille de
crop (256 en juillet, 512 aujourd'hui). La convolution 3×3 depthwise est
indifférente à la résolution — c'est le *conditional positional encoding* de CPVT.
Sans position, l'attention serait invariante par permutation et ne pourrait pas
localiser quoi que ce soit.

**Embedding de date**, comme le τ du décodeur sémantique : sans lui, les deux
moitiés de la séquence seraient indiscernables et « comparer les dates » n'aurait
pas de sens.

**⚠️ LayerScale initialisé à ~0.** C'est le choix le plus important du fichier. Les
deux résidus sont multipliés par un gamma appris, initialisé à 1e-5 : **à
l'initialisation, le bloc est l'identité**, et le modèle hybride produit
exactement les sorties du modèle de référence. Il ne peut donc pas partir plus bas
que la ligne de base ; il apprend s'il le veut à se servir de l'attention. Sans
cela, on injecterait du bruit dans un modèle qui marche, et un résultat négatif ne
dirait pas si l'idée est mauvaise ou si l'initialisation l'a tuée.
"""

import torch
import torch.nn.functional as F
from torch import nn


class _JointAttentionBlock(nn.Module):
    """Un bloc Transformer pré-normalisé sur la séquence jointe des deux dates."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float,
                 layer_scale: float, dropout: float):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} non divisible par num_heads {num_heads}")
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )
        # Voir la docstring du module : à gamma ~ 0, le bloc est l'identité.
        self.gamma1 = nn.Parameter(torch.full((dim,), layer_scale))
        self.gamma2 = nn.Parameter(torch.full((dim,), layer_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, C)
        b, n, c = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)   # (B, heads, N, dh)
        # scaled_dot_product_attention bascule sur FlashAttention quand elle peut :
        # même résultat, mémoire linéaire au lieu de quadratique.
        a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).reshape(b, n, c)
        x = x + self.gamma1 * self.proj(a)
        x = x + self.gamma2 * self.mlp(self.norm2(x))
        return x


class BiTemporalAttention(nn.Module):
    """Attention jointe sur les cartes de traits des deux dates, à un stage donné.

    Entrée : deux tenseurs (B, C, H, W). Sortie : deux tenseurs de même forme.
    """

    def __init__(self, dim: int, depth: int = 2, num_heads: int = 8,
                 mlp_ratio: float = 2.0, layer_scale: float = 1e-5,
                 dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        # Position conditionnelle : depthwise 3x3, indifférente à la résolution.
        self.pos = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        nn.init.zeros_(self.pos.weight)
        nn.init.zeros_(self.pos.bias)
        # Embedding de date, même rôle que le τ du décodeur sémantique.
        self.date = nn.Parameter(torch.zeros(2, dim))
        self.blocks = nn.ModuleList(
            _JointAttentionBlock(dim, num_heads, mlp_ratio, layer_scale, dropout)
            for _ in range(depth)
        )

    def _tokens(self, f: torch.Tensor, date: int) -> torch.Tensor:
        f = f + self.pos(f)                       # position conditionnelle
        b, c, h, w = f.shape
        t = f.flatten(2).transpose(1, 2)          # (B, H·W, C)
        return t + self.date[date].view(1, 1, -1)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor):
        b, c, h, w = f1.shape
        if f2.shape != f1.shape:
            raise ValueError(f"formes différentes : {f1.shape} et {f2.shape}")
        x = torch.cat([self._tokens(f1, 0), self._tokens(f2, 1)], dim=1)
        for blk in self.blocks:
            x = blk(x)
        n = h * w
        out1 = x[:, :n].transpose(1, 2).reshape(b, c, h, w)
        out2 = x[:, n:].transpose(1, 2).reshape(b, c, h, w)
        return out1, out2
