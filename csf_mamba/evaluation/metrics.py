"""Métriques SCD : SeK / Fscd / mIoU / OA.

Maths portées VERBATIM de ChangeMamba
(third_party/ChangeMamba/changedetection/evaluation/scd.py) pour que nos chiffres
soient directement comparables à la littérature. `fast_hist`, `cal_kappa` et le
calcul SeK/Fscd sont repris tels quels ; seule l'enveloppe (accumulation par
histogramme au lieu de stocker toutes les cartes, gestion de l'ignore) est à nous
— nécessaire pour tenir les ~40k images de Hi-UCD.

Convention (alignée sur la convention A du dataset) : la carte SCD par date vaut
0 = no-change, 1..N = classe sémantique dans les régions changées. Notre index 0
sémantique réservé coïncide donc exactement avec le no-change de l'éval.

SeK = κ_n0 · exp(IoU_fg) / e   (Yang et al., 2022, tel qu'implémenté par ChangeMamba)
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

IGNORE_INDEX = 255


@dataclass(frozen=True)
class SCDMetrics:
    sek: float
    fscd: float
    miou: float
    oa: float
    kappa: float


def fast_hist(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """VERBATIM ChangeMamba : histogramme n×n, a=pred, b=label, filtré 0<=a<n."""
    k = (a >= 0) & (a < n) & (b >= 0) & (b < n)
    return np.bincount(n * a[k].astype(int) + b[k].astype(int), minlength=n ** 2).reshape(n, n)


def cal_kappa(hist: np.ndarray) -> float:
    """VERBATIM ChangeMamba."""
    if hist.sum() == 0:
        return 0.0
    po = np.diag(hist).sum() / hist.sum()
    pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
    if pe == 1:
        return 0.0
    return float((po - pe) / (1 - pe))


def metrics_from_hist(hist: np.ndarray) -> SCDMetrics:
    """Calcul SeK/Fscd/mIoU/OA à partir d'un histogramme SCD accumulé.

    Corps repris de SCDD_eval_all (ChangeMamba), plus OA = accord global.
    """
    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()

    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)

    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist) + 1e-10)
    iou_fg = iu[1]
    iou_mean = (iu[0] + iu[1]) / 2
    sek = (kappa_n0 * math.exp(iou_fg)) / math.e

    pixel_sum = hist.sum()
    change_pred_sum = pixel_sum - hist.sum(1)[0].sum()
    change_label_sum = pixel_sum - hist.sum(0)[0].sum()
    sc_tp = np.diag(hist[1:, 1:]).sum()
    sc_precision = sc_tp / (change_pred_sum + 1e-10)
    sc_recall = sc_tp / (change_label_sum + 1e-10)
    fscd = (
        2 * sc_precision * sc_recall / (sc_precision + sc_recall)
        if sc_precision > 0 and sc_recall > 0 else 0.0
    )
    oa = float(np.diag(hist).sum() / (pixel_sum + 1e-10))
    return SCDMetrics(sek=float(sek), fscd=float(fscd), miou=float(iou_mean),
                      oa=oa, kappa=float(kappa_n0))


class SCDEvaluator:
    """Accumulateur d'histogramme SCD sur tout un split (les deux dates sommées)."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.hist = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)

    @staticmethod
    def _scd_map(sem_argmax: torch.Tensor, change_pred: torch.Tensor) -> torch.Tensor:
        """Carte SCD par date : classe sémantique, 0 (no-change) hors changement."""
        return sem_argmax * (change_pred != 0)

    def add(self, outputs: dict, targets: dict):
        """Accumule un batch. Attend les logits sem_t1/sem_t2/bcd et les cibles."""
        change_pred = outputs["bcd"].argmax(1)               # (B,H,W)
        sem_a = outputs["sem_t1"].argmax(1)
        sem_b = outputs["sem_t2"].argmax(1)

        pred_a = self._scd_map(sem_a, change_pred).cpu().numpy()
        pred_b = self._scd_map(sem_b, change_pred).cpu().numpy()

        # Vérité : sémantique masquée par le changement RÉEL ; 0 = no-change.
        gt_change = (targets["change"] != 0)                 # bool, ignore=255 -> True mais filtré
        gt_a = (targets["sem_t1"] * gt_change).cpu().numpy()
        gt_b = (targets["sem_t2"] * gt_change).cpu().numpy()

        # Pixels valides : sémantique GT non-ignore et changement GT non-ignore.
        valid = ((targets["sem_t1"] != IGNORE_INDEX)
                 & (targets["sem_t2"] != IGNORE_INDEX)
                 & (targets["change"] != IGNORE_INDEX)).cpu().numpy()

        for pred, gt in ((pred_a, gt_a), (pred_b, gt_b)):
            self.hist += fast_hist(pred[valid], gt[valid], self.num_classes)

    def compute(self) -> SCDMetrics:
        return metrics_from_hist(self.hist)
