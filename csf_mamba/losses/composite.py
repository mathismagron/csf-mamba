"""Perte composite : L = CE_BCD + ½(CE_T1 + CE_T2) + λ_sek·SeK + λ_sc·L_sc

SeK et L_sc coûtent zéro paramètre. Deux points établis en reproduisant Mamba-FCS :

1. La SeK-loss de Mamba-FCS **inclut déjà le mIoU en interne** (elle combine soft
   kappa et mIoU pondéré-fréquence, puis `-log`). Leur training n'a donc PAS de
   terme mIoU séparé — un seul terme SeK, poids 0,5. On suit cette formulation
   (le `λ₁·mIoU` distinct du plan v2 était redondant).

2. Elle ne construit PAS de carte SCD « from-to » unique : elle opère sur les deux
   branches sémantiques restreintes aux zones changées par le `change_mask`
   (dérivé du canal changement). Voir `sek_mambafcs.py`.

L_sc comble un trou réel : la SeK-loss ne supervise que l'intérieur des zones
changées ; rien ne supervise la cohérence sémantique sur les pixels inchangés —
majoritaires sur Hi-UCD (sémantique pleine-scène). L_sc s'en charge.

La SeK-loss est un portage VERBATIM de Mamba-FCS, validé numériquement
(`tests/test_sek_port.py`) pour garantir la comparabilité (§12.1).
"""

import torch
from torch import nn

from .sek_mambafcs import SeKLossMambaFCS

IGNORE_INDEX = 255


def semantic_consistency_loss(
    feat_t1: torch.Tensor, feat_t2: torch.Tensor, unchanged: torch.Tensor
) -> torch.Tensor:
    """L_sc (AtrousMamba/Bi-SRNet) : cosinus poussé vers 1 là où rien n'a changé,
    vers 0 ailleurs. Sur Hi-UCD le masque vient directement du canal 3.

    `unchanged` : booléen (B, H, W), à pleine résolution. Il est sous-échantillonné
    à la résolution des features (nearest, pour rester binaire). Zéro paramètre.
    """
    cosine = nn.functional.cosine_similarity(feat_t1, feat_t2, dim=1)  # (B, h, w)
    if unchanged.shape[-2:] != cosine.shape[-2:]:
        unchanged = nn.functional.interpolate(
            unchanged.unsqueeze(1).float(), size=cosine.shape[-2:], mode="nearest"
        ).squeeze(1)
    unchanged = unchanged.to(cosine.dtype)
    n_un = unchanged.sum().clamp_min(1.0)
    n_ch = (1 - unchanged).sum().clamp_min(1.0)

    pull = ((1 - cosine) * unchanged).sum() / n_un
    push = (cosine.clamp_min(0) * (1 - unchanged)).sum() / n_ch
    return pull + push


def dice_loss_change(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Dice sur la classe 'changement' (indice 1) du BCD. Robuste au déséquilibre :
    optimise le recouvrement prédiction/vérité, pas la justesse pixel par pixel.
    """
    prob = logits.softmax(dim=1)[:, 1]                 # P(changement), (B,H,W)
    valid = (target != IGNORE_INDEX).float()
    tgt = (target == 1).float() * valid
    prob = prob * valid
    intersection = (prob * tgt).sum()
    denom = prob.sum() + tgt.sum()
    return 1 - (2 * intersection + eps) / (denom + eps)


class CSFMambaLoss(nn.Module):
    """Assemble les cinq termes. `scd_target` est la carte SCD (0 = no-change)."""

    def __init__(
        self,
        num_semantic_classes: int,
        lambda_sek: float = 0.5,
        lambda_sc: float = 0.1,
        lambda_dice: float = 1.0,
        sek_non_change_class: int = 0,
        bcd_change_weight: float = 1.0,
    ):
        super().__init__()
        self.num_semantic_classes = num_semantic_classes
        self.lambda_sek = lambda_sek
        self.lambda_sc = lambda_sc
        self.lambda_dice = lambda_dice
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        # BCD pondéré : la classe 1 (changement) est rare -> on la sur-pondère
        # pour éviter que le modèle collapse vers « aucun changement » (SeK=0).
        bcd_w = torch.tensor([1.0, float(bcd_change_weight)])
        self.ce_bcd = nn.CrossEntropyLoss(weight=bcd_w, ignore_index=IGNORE_INDEX)
        self.sek = SeKLossMambaFCS(
            num_classes=num_semantic_classes, non_change_class=sek_non_change_class
        )

    def forward(self, outputs: dict, targets: dict, apply_sek: bool = True) -> dict:
        loss_bcd = self.ce_bcd(outputs["bcd"], targets["change"])
        loss_sem = 0.5 * (
            self.ce(outputs["sem_t1"], targets["sem_t1"])
            + self.ce(outputs["sem_t2"], targets["sem_t2"])
        )

        terms = {"ce_bcd": loss_bcd, "ce_sem": loss_sem}

        # Dice sur le changement : complète la CE pondérée contre le déséquilibre.
        if self.lambda_dice > 0:
            terms["dice"] = self.lambda_dice * dice_loss_change(outputs["bcd"], targets["change"])

        # SeK (verbatim Mamba-FCS) : pilotée par le change_mask, opère sur les deux
        # branches sémantiques. `apply_sek=False` pendant le warmup (la sémantique
        # doit d'abord apprendre avant que SeK, qui la suppose correcte, aide).
        if apply_sek:
            change_mask = (targets["change"] != 0).float()
            terms["sek"] = self.lambda_sek * self.sek(
                outputs["sem_t1"], outputs["sem_t2"],
                targets["sem_t1"], targets["sem_t2"], change_mask,
            )

        if "feat_t1" in outputs and "unchanged" in targets:
            terms["sc"] = self.lambda_sc * semantic_consistency_loss(
                outputs["feat_t1"], outputs["feat_t2"], targets["unchanged"]
            )

        terms["total"] = sum(terms.values())
        return terms
