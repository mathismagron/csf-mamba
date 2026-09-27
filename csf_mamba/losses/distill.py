"""Pertes de distillation des sorties (étapes 2 et 3 de documentation/distillation.md).

Cible : sorties du professeur lues dans le cache (`kd_teacher`, 15 canaux à la
résolution native de son décodeur), suréchantillonnées en bilinéaire vers la sortie
de l'élève — exactement l'opération finale du professeur.

* Changement (étape 2). Logit élève d = bcd[:,1] − bcd[:,0] (softmax à 2 classes ⇔
  sigmoïde de d). BCE à cibles douces σ(z_t / T), × T². Tous les pixels : la carte
  binaire est supervisée partout sur SECOND.
* Sémantique (étape 3). KL(q_t ‖ p_s) sur les classes réelles **1..6** seulement :
  le canal 0 du professeur n'a jamais été une cible de sa CE (ignore_index=0),
  l'inclure transmettrait du bruit. Softmax renormalisée sur 1..6, température T, × T².
  Masque : `changed` = changé selon la vérité OU selon le professeur ; `all` = toute
  l'image (y compris les zones non-changées, sans étiquette sémantique).

Les valeurs rapportées sont des divergences (≥ 0, nulles quand l'élève reproduit le
professeur) : on retranche l'entropie du professeur à la BCE, ce qui ne change pas
le gradient mais rend le terme lisible dans les logs.
"""

import torch
import torch.nn.functional as F
from torch import nn

EPS = 1e-8


CHANGE_TARGETS = ("teacher", "gt", "gt_smooth")


class DistillLoss(nn.Module):
    """`change_target` sert aux **contrôles** du terme de changement (même perte, même λ,
    même pipeline de données, cache lu mais cible remplacée) :

    * ``teacher``   la KD proprement dite : σ(z_t / T) ;
    * ``gt``        la vérité terrain binaire : mesure ce que rapporte le seul fait
                    d'ajouter une BCE de poids λ sur le logit de changement ;
    * ``gt_smooth`` la vérité moyennée à la résolution native du professeur puis
                    suréchantillonnée comme lui : mêmes bords adoucis que la cible du
                    professeur, sans son savoir. Écart teacher − gt_smooth = ce que le
                    professeur apporte au-delà du lissage des contours.
    """

    def __init__(self, lambda_change: float = 0.0, t_change: float = 1.0,
                 lambda_sem: float = 0.0, t_sem: float = 2.0, sem_mask: str = "changed",
                 change_target: str = "teacher"):
        super().__init__()
        if sem_mask not in ("changed", "all"):
            raise ValueError(sem_mask)
        if change_target not in CHANGE_TARGETS:
            raise ValueError(change_target)
        self.lambda_change, self.t_change = lambda_change, t_change
        self.lambda_sem, self.t_sem, self.sem_mask = lambda_sem, t_sem, sem_mask
        self.change_target = change_target

    def _change_target(self, tz, teacher, change_gt, size):
        if self.change_target == "teacher":
            return torch.sigmoid(tz[:, 14] / self.t_change)
        gt = change_gt.float()
        if gt.shape[-2:] != size:
            gt = F.interpolate(gt[:, None], size=size, mode="nearest")[:, 0]
        if self.change_target == "gt":
            return gt
        native = teacher.shape[-2:]
        low = F.adaptive_avg_pool2d(gt[:, None], native)
        return F.interpolate(low, size=size, mode="bilinear", align_corners=False)[:, 0].clamp(0, 1)

    @property
    def active(self) -> bool:
        return self.lambda_change > 0 or self.lambda_sem > 0

    def forward(self, outputs: dict, teacher: torch.Tensor, change_gt: torch.Tensor) -> dict:
        size = outputs["bcd"].shape[-2:]
        tz = F.interpolate(teacher.float(), size=size, mode="bilinear", align_corners=False)
        terms = {}
        if self.lambda_change > 0:
            T = self.t_change
            d = (outputs["bcd"][:, 1] - outputs["bcd"][:, 0]) / T
            p_t = self._change_target(tz, teacher, change_gt, size)
            bce = F.binary_cross_entropy_with_logits(d, p_t)
            h_t = -(p_t * torch.log(p_t + EPS) + (1 - p_t) * torch.log(1 - p_t + EPS)).mean()
            terms["kd_change"] = self.lambda_change * T * T * (bce - h_t.detach())
        if self.lambda_sem > 0:
            T = self.t_sem
            if self.sem_mask == "all":
                mask = torch.ones_like(tz[:, 14])
            else:
                mask = ((change_gt == 1) | (tz[:, 14] > 0)).float()
            denom = mask.sum().clamp(min=1.0)
            total = 0.0
            for key, sl in (("sem_t1", slice(1, 7)), ("sem_t2", slice(8, 14))):
                logq = F.log_softmax(tz[:, sl] / T, dim=1)
                logp = F.log_softmax(outputs[key][:, 1:7] / T, dim=1)
                kl = (logq.exp() * (logq - logp)).sum(1)
                total = total + (kl * mask).sum() / denom
            terms["kd_sem"] = self.lambda_sem * T * T * total / 2
        return terms
