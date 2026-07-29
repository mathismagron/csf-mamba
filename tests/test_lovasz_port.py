"""Validation numérique : notre portage Lovász == l'original de Mamba-FCS.

Même approche que pour la SeK-loss : on n'importe pas le module original (ses
imports sont lourds), on en extrait les fonctions et on les exécute isolément.
"""

import ast
from itertools import filterfalse as ifilterfalse
from pathlib import Path

import torch
import torch.nn.functional as F

from csf_mamba.losses.lovasz import lovasz_softmax

_ORIG = (
    Path(__file__).resolve().parents[1]
    / "third_party" / "MambaFCS" / "changedetection" / "utils_func" / "lovasz_loss.py"
)
_NEEDED = {"lovasz_grad", "lovasz_softmax", "lovasz_softmax_flat",
           "flatten_probas", "mean", "isnan"}


def _load_original():
    source = _ORIG.read_text()
    tree = ast.parse(source)
    ns = {"torch": torch, "F": F, "ifilterfalse": ifilterfalse,
          "Variable": lambda x: x}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _NEEDED:
            src = ast.get_source_segment(source, node)
            src = src.replace("classes is 'present'", "classes == 'present'")
            exec(src, ns)  # noqa: S102 — source de confiance, test only
    return ns


def test_lovasz_matches_original():
    if not _ORIG.exists():
        print(f"⊘ original absent ({_ORIG}) — cloner MambaFCS pour valider")
        return

    orig = _load_original()
    torch.manual_seed(0)
    B, C, H, W = 2, 7, 24, 24
    probas = torch.randn(B, C, H, W).softmax(dim=1)
    labels = torch.randint(0, C, (B, H, W))
    labels[torch.rand(B, H, W) > 0.9] = 255            # quelques pixels ignorés

    ours = lovasz_softmax(probas, labels, ignore=255)
    ref = orig["lovasz_softmax"](probas, labels, ignore=255)
    assert torch.allclose(ours, ref, atol=1e-6), f"portage != original : {ours} vs {ref}"

    # cas binaire (celui du BCD)
    probas2 = torch.randn(B, 2, H, W).softmax(dim=1)
    labels2 = torch.randint(0, 2, (B, H, W))
    ours2 = lovasz_softmax(probas2, labels2, ignore=255)
    ref2 = orig["lovasz_softmax"](probas2, labels2, ignore=255)
    assert torch.allclose(ours2, ref2, atol=1e-6), f"binaire : {ours2} vs {ref2}"

    print(f"✓ portage Lovász numériquement identique "
          f"(multi-classe {ours.item():.6f} | binaire {ours2.item():.6f})")


if __name__ == "__main__":
    test_lovasz_matches_original()
