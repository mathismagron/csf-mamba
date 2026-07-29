"""Garde-fou : une prédiction PARFAITE doit donner des métriques parfaites.

Ce test existe parce qu'un bug l'a violé : le masque de validité exigeait une
sémantique annotée sur TOUS les pixels, ce qui supprimait la population « non
changé » sur les datasets à sémantique change-only (SECOND). Une prédiction
parfaite y donnait alors mIoU = 0,50.

Les deux conventions de labellisation sont couvertes :
  - « pleine scène » (Hi-UCD) : sémantique annotée partout
  - « change-only » (SECOND)  : sémantique annotée uniquement sur le changement
"""

import torch

from csf_mamba.evaluation.metrics import IGNORE_INDEX, SCDEvaluator

B, C, H, W = 2, 7, 64, 64


def _perfect_outputs(sem, changed, num_classes):
    """Logits qui prédisent exactement `sem` et `changed`."""
    logits_sem = torch.zeros(B, num_classes, H, W)
    logits_bcd = torch.zeros(B, 2, H, W)
    for b in range(B):
        for cl in range(num_classes):
            logits_sem[b, cl][sem[b] == cl] = 10.0
        logits_bcd[b, 1][changed[b]] = 10.0
        logits_bcd[b, 0][~changed[b]] = 10.0
    return {"bcd": logits_bcd, "sem_t1": logits_sem, "sem_t2": logits_sem}


def _run(sem, changed):
    outputs = _perfect_outputs(sem, changed, C)
    targets = {"change": changed.long(), "sem_t1": sem, "sem_t2": sem}
    ev = SCDEvaluator(num_classes=C)
    ev.add(outputs, targets)
    return ev, ev.compute()


def test_perfect_prediction_change_only_labels():
    """Convention SECOND : sémantique annotée UNIQUEMENT sur le changement."""
    torch.manual_seed(0)
    changed = torch.rand(B, H, W) > 0.75
    sem = torch.where(changed, torch.randint(1, C, (B, H, W)),
                      torch.full((B, H, W), IGNORE_INDEX))
    ev, m = _run(sem, changed)

    assert ev.hist[0, 0] > 0, "les pixels inchangés doivent être comptés (label 0)"
    for name, value in (("SeK", m.sek), ("Fscd", m.fscd), ("mIoU", m.miou),
                        ("OA", m.oa), ("kappa", m.kappa)):
        assert abs(value - 1.0) < 1e-6, f"{name} = {value}, attendu 1.0"
    print("✓ change-only (SECOND) : prédiction parfaite -> toutes métriques à 1.0")


def test_perfect_prediction_full_scene_labels():
    """Convention Hi-UCD : sémantique annotée partout (avec quelques non-annotés)."""
    torch.manual_seed(1)
    changed = torch.rand(B, H, W) > 0.9
    sem = torch.randint(1, C, (B, H, W))
    sem[torch.rand(B, H, W) > 0.95] = IGNORE_INDEX   # quelques pixels non annotés
    ev, m = _run(sem, changed)

    assert ev.hist[0, 0] > 0, "les pixels inchangés doivent être comptés (label 0)"
    for name, value in (("SeK", m.sek), ("Fscd", m.fscd), ("mIoU", m.miou),
                        ("OA", m.oa), ("kappa", m.kappa)):
        assert abs(value - 1.0) < 1e-6, f"{name} = {value}, attendu 1.0"
    print("✓ pleine scène (Hi-UCD) : prédiction parfaite -> toutes métriques à 1.0")


if __name__ == "__main__":
    test_perfect_prediction_change_only_labels()
    test_perfect_prediction_full_scene_labels()
