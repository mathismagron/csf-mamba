"""Validation des métriques SCD contre l'implémentation ChangeMamba.

Comme pour la SeK-loss, on n'importe pas le module d'éval (imports fragiles) : on
extrait les fonctions `fast_hist`, `cal_kappa`, `get_hist`, `SCDD_eval_all` du
fichier source et on les exécute isolément, puis on compare à notre calcul par
histogramme sur des cartes SCD aléatoires identiques.
"""

import ast
import math
from pathlib import Path

import numpy as np
from scipy import stats

from csf_mamba.evaluation.metrics import fast_hist, metrics_from_hist

_ORIG = (
    Path(__file__).resolve().parents[1]
    / "third_party" / "ChangeMamba" / "changedetection" / "evaluation" / "scd.py"
)
_NEEDED = {"fast_hist", "get_hist", "cal_kappa", "SCDD_eval_all"}


def _load_original_eval():
    source = _ORIG.read_text()
    tree = ast.parse(source)
    ns = {"np": np, "math": math, "stats": stats}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _NEEDED:
            exec(ast.get_source_segment(source, node), ns)  # noqa: S102 — test only
    return ns


def test_metrics_match_changemamba():
    if not _ORIG.exists():
        print(f"⊘ original absent ({_ORIG}) — cloner ChangeMamba pour valider")
        return

    orig = _load_original_eval()
    rng = np.random.default_rng(0)
    num_class = 10
    preds = [rng.integers(0, num_class, (32, 32)) for _ in range(6)]
    labels = [rng.integers(0, num_class, (32, 32)) for _ in range(6)]

    ref_kappa, ref_fscd, ref_miou, ref_sek = orig["SCDD_eval_all"](preds, labels, num_class)

    hist = np.zeros((num_class, num_class))
    for p, l in zip(preds, labels):
        hist += fast_hist(p, l, num_class)
    ours = metrics_from_hist(hist)

    assert np.isclose(ours.sek, ref_sek, atol=1e-6), f"SeK {ours.sek} vs {ref_sek}"
    assert np.isclose(ours.fscd, ref_fscd, atol=1e-6), f"Fscd {ours.fscd} vs {ref_fscd}"
    assert np.isclose(ours.miou, ref_miou, atol=1e-6), f"mIoU {ours.miou} vs {ref_miou}"
    assert np.isclose(ours.kappa, ref_kappa, atol=1e-6), f"kappa {ours.kappa} vs {ref_kappa}"
    print(f"✓ métriques identiques à ChangeMamba "
          f"(SeK={ours.sek:.4f} Fscd={ours.fscd:.4f} mIoU={ours.miou:.4f})")


if __name__ == "__main__":
    test_metrics_match_changemamba()
