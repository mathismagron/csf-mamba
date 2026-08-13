"""Audit du comptage de GMACs — validation de bout en bout.

Le chiffre de GMACs est l'argument central du projet : il ne peut pas reposer sur
« on a ajouté des handlers ». Ce script répond à quatre questions.

**1. Notre chaîne reproduit-elle un chiffre publié ?** C'est le test décisif. On
applique NOS handlers au modèle de ChangeMamba, construit depuis LEUR code, et on
compare aux valeurs de leur table de complexité — 21,51 M / 73,42 GMACs pour
MambaSCD-Tiny, 89,99 M / 211,55 pour Base. Si notre pipeline retrouve leurs
chiffres sur leur modèle, il n'y a plus de doute de convention ni de handler pour
le nôtre. Si elle ne les retrouve pas, le tableau d'efficience est à revoir.

**2. Qu'est-ce qui n'est PAS compté ?** fvcore ignore silencieusement ce qu'il ne
sait pas traiter. On liste la totalité des opérations non comptées, pas seulement
les dix premières, et on distingue celles qui ne portent aucun MAC (permutations,
activations, génération de grille) de celles qui en portent réellement.

**3. Un module à paramètres a-t-il échappé au traçage ?** Un sous-module jamais
appelé pendant le trace ne contribue rien au total, ce qui est soit normal
(`drop_path` en inférence) soit une sous-estimation massive — c'est exactement
ainsi que les quatre C²S² comptaient zéro avant l'ajout du handler `MambaInnerFn`.

**4. Le coût de la FFT.** `aten::fft_fft2` n'est pas compté par fvcore. On le
borne analytiquement pour pouvoir écrire une phrase exacte plutôt que de l'ignorer.

    python -m scripts.audit_gmacs                    # audit complet
    python -m scripts.audit_gmacs --skip-changemamba # sans le test décisif

Nécessite un GPU.
"""

import argparse
import math
import sys
from pathlib import Path

import torch
from fvcore.nn import flop_count, parameter_count

from csf_mamba.datasets import DATASETS
from csf_mamba.model import CSFMamba
from scripts.count_gmacs import supported_ops

_CHANGEMAMBA = Path(__file__).resolve().parents[1] / "third_party" / "ChangeMamba"

# Table de complexité publiée par ChangeMamba, entrée 512x512 bi-temporelle.
# Ce sont NOS cibles de validation : leur reproduction valide toute la chaîne.
PUBLIE = {
    "vssm1/vssm_tiny_224_0229flex.yaml": ("MambaSCD-Tiny", 21.51, 73.42),
    "vssm1/vssm_small_224.yaml": ("MambaSCD-Small", 54.28, 146.70),
    "vssm1/vssm_base_224.yaml": ("MambaSCD-Base", 89.99, 211.55),
}

# Opérations sans aucune multiplication-accumulation : les ignorer est correct,
# et c'est aussi ce que fait la référence à laquelle on se compare.
SANS_MAC = {
    "aten::add", "aten::add_", "aten::sub", "aten::rsub", "aten::neg",
    "aten::mul", "aten::mul_", "aten::div", "aten::clone", "aten::copy_",
    "aten::gelu", "aten::silu", "aten::relu", "aten::sigmoid", "aten::softmax",
    "aten::exp", "aten::log1p", "aten::abs", "aten::sqrt", "aten::pow",
    "aten::flip", "aten::cat", "aten::stack", "aten::permute", "aten::transpose",
    "aten::view", "aten::reshape", "aten::contiguous", "aten::slice",
    "aten::unsqueeze", "aten::squeeze", "aten::expand", "aten::chunk",
    "aten::linspace", "aten::meshgrid", "aten::arange", "aten::zeros",
    "aten::ones", "aten::full", "aten::clamp", "aten::max", "aten::min",
    "aten::mean", "aten::sum", "aten::softplus", "aten::pad", "aten::interpolate",
    "aten::upsample_bilinear2d", "aten::upsample_nearest2d", "aten::to",
    "aten::type_as", "aten::detach", "aten::split", "aten::select",
    # Réorganisation des parcours 4 directions de VMamba : pure permutation.
    "prim::PythonOp.CrossScan", "prim::PythonOp.CrossMerge",
}

# Opérations qui portent DES MACS et dont l'absence serait une sous-estimation.
PORTE_DES_MACS = {"aten::fft_fft2", "aten::fft_ifft2", "aten::fft_rfft2",
                  "aten::bmm", "aten::mm", "aten::addmm", "aten::conv1d",
                  "aten::conv2d", "aten::linear", "aten::einsum", "aten::matmul"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--dataset", default="second", choices=sorted(DATASETS))
    p.add_argument("--encoder", default="vmamba_mini")
    p.add_argument("--fusion", default="c2s2", choices=["c2s2", "concat"])
    p.add_argument("--skip-changemamba", action="store_true")
    p.add_argument("--params-only", action="store_true",
                   help="Contrôle préalable SANS GPU ni traçage : vérifie que le "
                        "dépôt tiers est là, que les trois configs se construisent, "
                        "et que leurs comptes de paramètres correspondent aux "
                        "valeurs publiées. Quelques secondes une fois torch chargé.")
    return p.parse_args()


def controle_prealable():
    """Vérifie tout ce qui peut l'être sans GPU, avant d'engager une heure de calcul.

    Le compte de paramètres est déjà une validation forte : s'il correspond aux
    valeurs publiées pour les trois variantes, c'est que le modèle construit est
    bien le leur — et la comparaison de GMACs qui suivra portera sur le bon objet.
    """
    print("=" * 78)
    print("CONTRÔLE PRÉALABLE — sans GPU")
    print("=" * 78)
    if not _CHANGEMAMBA.is_dir():
        sys.exit(f"⛔ dépôt tiers absent : {_CHANGEMAMBA}\n"
                 "   le récupérer avec scripts/setup_third_party.sh")
    print(f"  ✅ dépôt tiers présent : {_CHANGEMAMBA}")

    cfg_dir = _CHANGEMAMBA / "changedetection" / "configs"
    for cfg in PUBLIE:
        chemin = cfg_dir / cfg
        etat = "✅" if chemin.is_file() else "⛔ ABSENT"
        print(f"  {etat} {cfg}")

    sys.path.insert(0, str(_CHANGEMAMBA))
    from scripts.evaluate_changemamba import build_model

    print(f"\n  {'modèle':<18} {'params publiés':>15} {'construits':>12} {'écart':>9}")
    ok = True
    for cfg, (nom, p_pub, _) in PUBLIE.items():
        try:
            m = build_model(cfg)          # construction seule : pas de CUDA requis
            n = sum(x.numel() for x in m.parameters()) / 1e6
            ecart = 100 * (n - p_pub) / p_pub
            verdict = "✅" if abs(ecart) < 1 else "⛔"
            ok &= abs(ecart) < 1
            print(f"  {nom:<18} {p_pub:>15.2f} {n:>12.2f} {ecart:>8.2f}% {verdict}")
            del m
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  {nom:<18} ⛔ ÉCHEC : {type(e).__name__}: {e}")

    print("\n  " + ("✅ tout est en place — l'audit complet peut être lancé."
                    if ok else
                    "⛔ à corriger AVANT de lancer l'audit : les comptes de "
                    "paramètres ne correspondent pas,\n     donc le modèle "
                    "construit n'est pas celui dont les GMACs sont publiés."))
    return ok


def trace(model, size, titre):
    """Trace un modèle et renvoie (GMACs, params en M, non comptés, jamais appelés)."""
    inputs = (torch.randn(1, 3, size, size).cuda(), torch.randn(1, 3, size, size).cuda())
    params = parameter_count(model)[""]
    print(f"\n  traçage de {titre} ({params / 1e6:.2f} M)...", flush=True)
    gmacs, non_comptes = flop_count(model=model, inputs=inputs, supported_ops=supported_ops())
    return sum(gmacs.values()), params / 1e6, dict(non_comptes), gmacs


def modules_jamais_appeles(model, size):
    """Modules PORTEURS DE PARAMÈTRES qui ne reçoivent aucun gradient au trace.

    Un module à paramètres qui n'apparaît pas dans le graphe ne contribue rien au
    total. C'est normal pour `drop_path` en inférence, alarmant sinon.
    """
    vus = set()
    hooks = [m.register_forward_hook(lambda mod, *_: vus.add(id(mod)))
             for m in model.modules()]
    with torch.no_grad():
        model(torch.randn(1, 3, size, size).cuda(), torch.randn(1, 3, size, size).cuda())
    for h in hooks:
        h.remove()
    manquants = []
    for nom, mod in model.named_modules():
        if id(mod) in vus or not nom:
            continue
        n_p = sum(p.numel() for p in mod.parameters(recurse=False))
        if n_p:
            manquants.append((nom, n_p))
    return manquants


def cout_fft(channels, size, stages):
    """Borne haute du coût des FFT2, que fvcore ne compte pas.

    Une FFT2 sur (C,H,W) coûte ~ C·H·W·log2(H·W) multiplications complexes, soit
    ~4x en multiplications réelles. On majore volontairement : l'idée est de
    pouvoir écrire « au plus X GMACs non comptés », pas de raffiner.
    """
    total = 0.0
    for i in stages:
        h = w = size // (4 * 2 ** i)
        total += 4 * channels[i] * h * w * math.log2(max(2, h * w))
    return total / 1e9


def main():
    args = parse_args()
    if args.params_only:
        sys.exit(0 if controle_prealable() else 1)
    if not torch.cuda.is_available():
        sys.exit("Un GPU est requis (kernel selective_scan).")

    print("=" * 78)
    print("AUDIT 1 — notre chaîne reproduit-elle les chiffres publiés ?")
    print("=" * 78)
    if args.skip_changemamba:
        print("  (ignoré à la demande)")
    else:
        sys.path.insert(0, str(_CHANGEMAMBA))
        from scripts.evaluate_changemamba import build_model
        print(f"\n  {'modèle':<18} {'params publiés':>15} {'mesurés':>10} "
              f"{'GMACs publiés':>14} {'mesurés':>10} {'écart':>8}")
        for cfg, (nom, p_pub, g_pub) in PUBLIE.items():
            try:
                m = build_model(cfg).cuda().eval()
                g, p, _, _ = trace(m, args.size, nom)
                ecart = 100 * (g - g_pub) / g_pub
                verdict = "✅" if abs(ecart) < 2 else "⛔"
                print(f"  {nom:<18} {p_pub:>15.2f} {p:>10.2f} "
                      f"{g_pub:>14.2f} {g:>10.2f} {ecart:>7.1f}% {verdict}")
                del m
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001 — on veut voir l'échec, pas planter
                print(f"  {nom:<18} ÉCHEC : {type(e).__name__}: {e}")
        print("\n  Un écart inférieur à 2 % valide la convention ET les handlers :")
        print("  notre pipeline retrouve leurs chiffres sur leur modèle.")

    print("\n" + "=" * 78)
    print("AUDIT 2, 3, 4 — notre modèle")
    print("=" * 78)
    _, num_classes = DATASETS[args.dataset]
    model = CSFMamba(num_semantic_classes=num_classes, encoder=args.encoder,
                     backend="mamba", fusion=args.fusion).cuda().eval()
    total, params, non_comptes, par_op = trace(model, args.size, f"CSF-Mamba {args.fusion}")

    print(f"\n  Paramètres : {params:.2f} M     GMACs : {total:.2f}")
    print("\n  Répartition :")
    for op, v in sorted(par_op.items(), key=lambda kv: -kv[1]):
        if v >= 0.005:
            print(f"    {op:28s} {v:8.3f}  ({100 * v / total:5.1f} %)")

    print("\n  --- AUDIT 2 : opérations NON comptées (TOUTES) ---")
    suspects = []
    for op, n in sorted(non_comptes.items(), key=lambda kv: -kv[1]):
        if op in SANS_MAC:
            statut = "sans MAC"
        elif op in PORTE_DES_MACS:
            statut = "⛔ PORTE DES MACS"
            suspects.append(op)
        else:
            statut = "⚠️ à qualifier"
            suspects.append(op)
        print(f"    {op:40s} {n:>5}   {statut}")
    if not suspects:
        print("    -> aucune opération non comptée ne porte de MAC.")

    print("\n  --- AUDIT 3 : modules à paramètres jamais appelés ---")
    manquants = modules_jamais_appeles(model, args.size)
    if not manquants:
        print("    -> aucun. Tous les modules paramétrés sont dans le graphe.")
    for nom, n_p in manquants:
        print(f"    {nom:50s} {n_p:>10,} paramètres")

    print("\n  --- AUDIT 4 : coût des FFT non comptées ---")
    if "aten::fft_fft2" in non_comptes:
        borne = cout_fft(model.channels, args.size, sorted(model.fft_stages))
        print(f"    {non_comptes['aten::fft_fft2']} FFT2 sur les stages "
              f"{sorted(model.fft_stages)}")
        print(f"    borne haute : {borne:.3f} GMACs, soit {100 * borne / total:.2f} % du total")
        print(f"    total majoré : {total + borne:.2f} GMACs")
    else:
        print("    -> aucune FFT dans ce modèle.")

    print("\n" + "=" * 78)
    print("Conclusion à reporter : le chiffre est comparable à la littérature si")
    print("l'audit 1 passe, exhaustif au sens des MACs si les audits 2 et 3 sont")
    print("vides, et majoré par l'audit 4 pour les opérations hors convention.")


if __name__ == "__main__":
    main()
