"""Compte les GMACs et les paramètres du modèle.

⚠️ Deux pièges, tous deux traités ici :

1. **fvcore compte des MACs, pas des FLOPs.** Son API s'appelle `flop_count` mais
   elle compte les multiplications-accumulations. Le nombre renvoyé est donc bien
   des **GMACs** (1 MAC ≈ 2 FLOPs). C'est la convention des tableaux d'efficience
   en vision, celle qu'utilisent ChangeMamba et Mamba-FCS.

2. **Les opérations SSM ne sont pas comptées par défaut.** Le scan sélectif est
   une `torch.autograd.Function` personnalisée : sans handler explicite, fvcore
   l'ignore **sans avertissement** et le total est massivement sous-estimé. On
   réutilise les handlers de ChangeMamba (`selective_scan_flop_jit`), les mêmes
   que ceux ayant servi à produire leurs chiffres publiés — donc comparables.

Les opérations non reconnues sont affichées : ce qui n'est pas compté est visible.

    python -m scripts.count_gmacs --encoder vmamba_mini --size 512

Nécessite un GPU (le kernel selective_scan ne tourne pas sur CPU).
"""

import argparse
import sys
from pathlib import Path

import torch
from fvcore.nn import flop_count, parameter_count

from csf_mamba.datasets import DATASETS
from csf_mamba.model import CSFMamba

_CHANGEMAMBA = Path(__file__).resolve().parents[1] / "third_party" / "ChangeMamba"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="second", choices=sorted(DATASETS))
    p.add_argument("--encoder", default="vmamba_mini",
                   choices=["conv", "vmamba_mini", "vmamba_tiny"])
    p.add_argument("--core", default="chess", choices=["chess", "l1"])
    p.add_argument("--size", type=int, default=512, help="Côté de l'image d'entrée.")
    return p.parse_args()


def supported_ops():
    """Handlers pour les opérations que fvcore ne sait pas compter seul."""
    if str(_CHANGEMAMBA) not in sys.path:
        sys.path.insert(0, str(_CHANGEMAMBA))
    from changedetection.models.vmamba import selective_scan_flop_jit

    return {
        # Activations / opérations élémentaires : ignorées comme relu l'est.
        "aten::silu": None,
        "aten::neg": None,
        "aten::exp": None,
        "aten::flip": None,
        # Scan sélectif : handlers de ChangeMamba, pour des chiffres comparables.
        "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanNRow": selective_scan_flop_jit,
    }


@torch.no_grad()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("Un GPU est requis (le kernel selective_scan ne tourne pas sur CPU).")

    _, num_classes = DATASETS[args.dataset]
    model = CSFMamba(num_semantic_classes=num_classes,
                     encoder=args.encoder, core=args.core, backend="mamba").cuda().eval()

    s = args.size
    inputs = (torch.randn(1, 3, s, s).cuda(), torch.randn(1, 3, s, s).cuda())

    params = parameter_count(model)[""]
    print(f"modèle construit ({params / 1e6:.2f} M) — traçage fvcore en cours "
          f"(lent sur les modèles SSM, plusieurs minutes)...", flush=True)
    gmacs, unsupported = flop_count(model=model, inputs=inputs,
                                    supported_ops=supported_ops())
    total = sum(gmacs.values())

    print(f"\n===== {args.encoder} / {args.core} / {args.dataset} — entrée {s}x{s} =====")
    print(f"  Paramètres : {params / 1e6:8.2f} M")
    print(f"  GMACs      : {total:8.2f}   (pour UNE paire d'images)")
    print(f"  GFLOPs     : {2 * total:8.2f}   (≈ 2 x GMACs)")

    print("\n  Répartition par type d'opération :")
    for op, v in sorted(gmacs.items(), key=lambda kv: -kv[1]):
        if v >= 0.01:
            print(f"    {op:24s} {v:8.2f} GMACs ({100 * v / total:4.1f} %)")

    if unsupported:
        print("\n  ⚠️ opérations NON comptées (nombre d'occurrences) :")
        for op, n in sorted(unsupported.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {op:40s} {n}")
        print("  (vérifier qu'aucune n'est un scan SSM : ce serait une sous-estimation)")


if __name__ == "__main__":
    main()
