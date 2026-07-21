"""SeK-loss — portage VERBATIM de Mamba-FCS (§12.1 du plan).

Source : third_party/MambaFCS/changedetection/utils_func/loss.py::SeK_Loss
Repris tel quel pour garantir la comparabilité stricte baseline ↔ modèle proposé.
Toute modification casserait la comparaison — ne pas « améliorer » ce fichier ;
tout ajustement doit passer par un wrapper ou une sous-classe.

Enseignement clé du portage : Mamba-FCS ne construit PAS une carte SCD « from-to »
unique. La loss opère directement sur les DEUX branches sémantiques (pred_t1,
pred_t2) restreintes aux régions changées par `change_mask`, calcule un soft-kappa
+ mIoU pondéré par fréquence par date, et les moyenne. La carte from-to n'existe
que côté *évaluation* (argmax, cf. SCDD_eval), pas côté loss.

Convention d'index (celle de Mamba-FCS) : le canal `non_change_class` (0 par
défaut) est exclu de la matrice de confusion ; les labels valent 255 là où on
ignore (one-hot protégé ci-dessous contre l'out-of-range).
"""

import torch
import torch.nn.functional as F
from torch import nn


class SeKLossMambaFCS(nn.Module):
    def __init__(self, num_classes, non_change_class=0, beta=1.5, gamma=0.5, eps=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.non_change = non_change_class
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred_t1, pred_t2, label_t1, label_t2, change_mask):
        """
        pred_t1, pred_t2 : (B, C, H, W) logits.
        label_t1, label_t2 : (B, H, W) labels entiers (255 = ignore).
        change_mask : (B, H, W) binaire (1 = changé).
        """
        B, C, H, W = pred_t1.shape
        device = pred_t1.device

        change_mask = change_mask.unsqueeze(1)  # B,1,H,W
        valid_classes = [c for c in range(C) if c != self.non_change]

        prob_t1 = F.softmax(pred_t1, dim=1) * change_mask
        prob_t2 = F.softmax(pred_t2, dim=1) * change_mask

        prob_t1 = prob_t1.permute(0, 2, 3, 1).reshape(-1, C)
        prob_t2 = prob_t2.permute(0, 2, 3, 1).reshape(-1, C)
        label_t1 = label_t1.reshape(-1)
        label_t2 = label_t2.reshape(-1)
        mask = change_mask.reshape(-1).bool()

        prob_t1 = prob_t1[mask]
        prob_t2 = prob_t2[mask]
        label_t1 = label_t1[mask]
        label_t2 = label_t2[mask]

        if prob_t1.size(0) == 0:
            return torch.tensor(0.0).to(device)

        # Garde-fou (ajout hors-verbatim, sans effet sur les valeurs valides) :
        # les labels ignore (255) restant après le masque changement casseraient
        # F.one_hot(., C). On les exclut explicitement.
        valid_t1 = label_t1 < C
        valid_t2 = label_t2 < C
        prob_t1, label_t1 = prob_t1[valid_t1], label_t1[valid_t1]
        prob_t2, label_t2 = prob_t2[valid_t2], label_t2[valid_t2]
        if prob_t1.size(0) == 0 or prob_t2.size(0) == 0:
            return torch.tensor(0.0).to(device)

        def compute_kappa(probs, labels):
            oh_labels = F.one_hot(labels, C).float()
            conf_matrix = torch.matmul(probs.T, oh_labels)
            conf_matrix = conf_matrix[valid_classes][:, valid_classes]
            total = conf_matrix.sum()
            po = torch.diag(conf_matrix).sum() / total
            row_sum = conf_matrix.sum(dim=1)
            col_sum = conf_matrix.sum(dim=0)
            pe = torch.sum(row_sum * col_sum) / (total ** 2)
            return (po - pe) / (1 - pe + self.eps)

        kappa_t1 = compute_kappa(prob_t1, label_t1)
        kappa_t2 = compute_kappa(prob_t2, label_t2)
        kappa = (kappa_t1 + kappa_t2) / 2

        def compute_iou(probs, labels):
            oh_labels = F.one_hot(labels, C).float()
            intersection = (probs * oh_labels).sum(dim=0)[valid_classes]
            union = probs.sum(dim=0)[valid_classes] + oh_labels.sum(dim=0)[valid_classes] - intersection
            freq = oh_labels.sum(dim=0)[valid_classes]
            weights = 1 / torch.log(freq + 1 + self.eps)
            weights = weights / weights.sum()
            return (intersection / (union + self.eps) * weights).sum()

        iou_t1 = compute_iou(prob_t1, label_t1)
        iou_t2 = compute_iou(prob_t2, label_t2)
        miou = (iou_t1 + iou_t2) / 2

        sek_value = kappa * torch.exp(self.beta * miou)
        log_sek = (sek_value + self.eps).log()
        log_miou = (miou + 1e-6).log()
        loss = -log_sek - self.gamma * log_miou
        return torch.clamp(loss, min=0.0)
