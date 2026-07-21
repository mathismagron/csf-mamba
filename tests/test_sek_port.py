"""Validation numérique : notre portage SeK == l'original Mamba-FCS.

L'original (third_party/MambaFCS/.../loss.py) a des imports module lourds et
fragiles (scipy, torchvision, imports `MambaFCS.*`). On n'importe donc pas le
module : on extrait la *source de la classe* `SeK_Loss` et on l'exécute dans un
espace de noms isolé (torch/nn/F seulement), puis on compare, sur des tenseurs
aléatoires identiques, à notre portage `SeKLossMambaFCS`.

Si ce test passe, notre SeK-loss est numériquement identique à la baseline —
condition du §12.1 pour que les tableaux comparatifs aient un sens.
"""

import ast
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from csf_mamba.losses.sek_mambafcs import SeKLossMambaFCS

_ORIG = (
    Path(__file__).resolve().parents[1]
    / "third_party" / "MambaFCS" / "changedetection" / "utils_func" / "loss.py"
)


def _load_original_sek_class():
    """Extrait la classe SeK_Loss du fichier original et l'exécute isolément."""
    source = _ORIG.read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SeK_Loss":
            class_src = ast.get_source_segment(source, node)
            namespace = {"torch": torch, "nn": nn, "F": F}
            exec(class_src, namespace)  # noqa: S102 — source de confiance, test only
            return namespace["SeK_Loss"]
    raise RuntimeError("classe SeK_Loss introuvable dans l'original")


def test_port_matches_original():
    if not _ORIG.exists():
        print(f"⊘ original absent ({_ORIG}) — cloner MambaFCS pour valider")
        return

    original_cls = _load_original_sek_class()
    torch.manual_seed(0)

    B, C, H, W = 2, 7, 16, 16          # 7 classes (comme SECOND : 0 + 6 réelles)
    pred_t1 = torch.randn(B, C, H, W)
    pred_t2 = torch.randn(B, C, H, W)
    label_t1 = torch.randint(0, C, (B, H, W))
    label_t2 = torch.randint(0, C, (B, H, W))
    change_mask = (torch.rand(B, H, W) > 0.3).float()

    ours = SeKLossMambaFCS(num_classes=C)(pred_t1, pred_t2, label_t1, label_t2, change_mask)
    orig = original_cls(num_classes=C)(pred_t1, pred_t2, label_t1, label_t2, change_mask)

    assert torch.allclose(ours, orig, atol=1e-6), f"portage != original : {ours} vs {orig}"
    print(f"✓ portage SeK numériquement identique (loss = {ours.item():.6f})")


if __name__ == "__main__":
    test_port_matches_original()
