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


class DistillLoss(nn.Module):
    def __init__(self, lambda_change: float = 0.0, t_change: float = 1.0,
                 lambda_sem: float = 0.0, t_sem: float = 2.0, sem_mask: str = "changed"):
        super().__init__()
        if sem_mask not in ("changed", "all"):
            raise ValueError(sem_mask)
        self.lambda_change, self.t_change = lambda_change, t_change
        self.lambda_sem, self.t_sem, self.sem_mask = lambda_sem, t_sem, sem_mask

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
            p_t = torch.sigmoid(tz[:, 14] / T)
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
