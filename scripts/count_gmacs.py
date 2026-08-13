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
import time
from pathlib import Path

_T0 = time.monotonic()


def log(msg: str) -> None:
    """Trace horodatée, écrite avant tout import lourd.

    Le job 190707 a été tué après 5 h sans produire une seule ligne Python : la
    bannière du shell s'affichait, puis plus rien. Impossible de savoir si le
    temps partait dans les imports, l'initialisation CUDA ou le traçage fvcore.
    On instrumente chaque étape plutôt que de deviner.
    """
    print(f"[{time.monotonic() - _T0:7.1f} s] {msg}", flush=True)


log("processus python démarré")
import torch
log(f"torch {torch.__version__} importé")
from fvcore.nn import flop_count, parameter_count
from fvcore.nn.jit_handles import einsum_flop_jit, get_shape
log("fvcore importé")

from csf_mamba.datasets import DATASETS
from csf_mamba.model import CSFMamba
log("csf_mamba importé")

_CHANGEMAMBA = Path(__file__).resolve().parents[1] / "third_party" / "ChangeMamba"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="second", choices=sorted(DATASETS))
    p.add_argument("--encoder", default="vmamba_mini",
                   choices=["conv", "vmamba_mini", "vmamba_tiny"])
    p.add_argument("--core", default="chess", choices=["chess", "l1"])
    p.add_argument("--decoder-refine", default="dw", choices=["dw", "full"])
    # Mêmes ablations que train.py : sans elles, on mesurerait le modèle complet
    # en croyant mesurer la variante — l'export d'une variable que le script
    # ignore ne produit aucune erreur, juste un chiffre faux.
    p.add_argument("--fusion", default="c2s2", choices=["c2s2", "concat"])
    p.add_argument("--no-cga", dest="cga", action="store_false")
    p.add_argument("--no-mcasf", dest="mcasf", action="store_false")
    p.add_argument("--upsample", default="dysample", choices=["dysample", "bilinear"])
    p.add_argument("--size", type=int, default=512, help="Côté de l'image d'entrée.")
    return p.parse_args()


def _einsum_flop(inputs, outputs):
    """einsum, compatible PyTorch récent.

    Le handler de fvcore impose `assert len(inputs) == 2` (équation, tenseurs),
    mais PyTorch trace désormais `aten::einsum` avec un troisième argument (le
    `path` d'optimisation). On tronque aux deux premiers et on délègue : le calcul
    reste celui de fvcore, seule l'assertion est contournée.
    """
    return einsum_flop_jit(inputs[:2], outputs)


def _mamba_inner_flop(inputs, outputs):
    """`mamba_inner_fn` de mamba_ssm — le kernel fusionné de NOS blocs C²S².

    Sans ce handler, fvcore ne compte RIEN pour les quatre C²S² : le kernel fusionne
    conv1d, x_proj, dt_proj, le scan sélectif et out_proj dans une seule
    `autograd.Function` opaque, et les sous-modules apparaissent « never called ».
    L'omission portait sur ~20 % du total.

    Signature : mamba_inner_fn(xz, conv1d_w, conv1d_b, x_proj_w, delta_proj_w,
                               out_proj_w, ...)
    """
    xz = get_shape(inputs[0])                    # (B, 2*d_inner, L)
    conv1d_w = get_shape(inputs[1])              # (d_inner, 1, d_conv)
    x_proj_w = get_shape(inputs[3])              # (dt_rank + 2*d_state, d_inner)
    delta_proj_w = get_shape(inputs[4])          # (d_inner, dt_rank)
    out_proj_w = get_shape(inputs[5])            # (d_model, d_inner)

    b, l = xz[0], xz[2]
    d_inner, d_conv = conv1d_w[0], conv1d_w[2]
    dt_rank = delta_proj_w[1]
    d_state = (x_proj_w[0] - dt_rank) // 2
    d_model = out_proj_w[0]

    macs = b * l * (
        d_inner * d_conv                     # convolution causale 1D
        + d_inner * x_proj_w[0]              # projection x -> (dt, B, C)
        + dt_rank * d_inner                  # projection de delta
        + 9 * d_inner * d_state              # scan sélectif (convention ChangeMamba)
        + d_inner * d_model                  # projection de sortie
    )
    return macs


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
        # einsum : contourne l'incompatibilité fvcore / PyTorch récent.
        "aten::einsum": _einsum_flop,
        # Kernel fusionné de mamba_ssm : sans lui, les C²S² comptent zéro.
        "prim::PythonOp.MambaInnerFn": _mamba_inner_flop,
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
    log(f"CUDA disponible : {torch.cuda.get_device_name(0)}")

    _, num_classes = DATASETS[args.dataset]
    model = CSFMamba(num_semantic_classes=num_classes,
                     encoder=args.encoder, core=args.core, backend="mamba",
                     decoder_refine=args.decoder_refine, fusion=args.fusion,
                     cga=args.cga, mcasf=args.mcasf, upsample=args.upsample)
    log("modèle instancié sur CPU")

    # Garde-fou : vérifier que le modèle CONSTRUIT correspond à ce qui est demandé.
    # Un paramètre accepté par argparse mais oublié dans la construction produit un
    # chiffre faux sous une bannière rassurante — c'est arrivé deux fois. On préfère
    # un plantage à un nombre plausible et faux.
    attendu = {"c2s2": "C2S2Block", "concat": "ConcatFusion"}[args.fusion]
    obtenu = type(model.c2s2[0]).__name__
    if obtenu != attendu:
        raise SystemExit(f"⛔ --fusion {args.fusion} demandé mais le modèle contient "
                         f"{obtenu} au lieu de {attendu}")
    ventilation = {nom: sum(q.numel() for q in mod.parameters())
                   for nom, mod in model.named_children()}
    ventilation["total"] = sum(q.numel() for q in model.parameters())
    log(f"modèle vérifié — c2s2 = {obtenu} | {ventilation}")
    model = model.cuda().eval()
    log("modèle transféré sur GPU")

    s = args.size
    inputs = (torch.randn(1, 3, s, s).cuda(), torch.randn(1, 3, s, s).cuda())

    params = parameter_count(model)[""]
    log(f"paramètres comptés : {params / 1e6:.2f} M — traçage fvcore en cours "
        f"(lent sur les modèles SSM)")
    ops = supported_ops()
    log("handlers fvcore prêts (ChangeMamba importé)")
    gmacs, unsupported = flop_count(model=model, inputs=inputs, supported_ops=ops)
    log("traçage terminé")
    total = sum(gmacs.values())

    variantes = [f"fusion={args.fusion}", f"décodeur {args.decoder_refine}",
                 f"up={args.upsample}"]
    if not args.cga:
        variantes.append("SANS CGA")
    if not args.mcasf:
        variantes.append("SANS MCA-SF")
    print(f"\n===== {args.encoder} / {args.core} / {' / '.join(variantes)}"
          f" / {args.dataset} — entrée {s}x{s} =====")
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
        for op in unsupported:
            if "SelectiveScan" in op or "MambaInner" in op:
                print(f"  ⛔ {op} NON compté -> le total est FAUX")


if __name__ == "__main__":
    main()
