"""Évalue un checkpoint ChangeMamba publié avec NOTRE code de métriques.

Pourquoi : c'est la comparaison la plus rigoureuse possible avec le SOTA. Même
split officiel, mêmes images, même code de métriques des deux côtés — plus aucun
doute de protocole ne subsiste. Le tableau d'efficience du README compare
aujourd'hui nos chiffres à ceux qu'ils publient, obtenus par leur propre code.

Bénéfice secondaire, non négligeable : charger le checkpoint donne son **compte de
paramètres exact**. Le journal a longtemps porté « ~37 M » pour MambaSCD-Tiny, une
estimation par analogie jamais vérifiée, contredite par leur table de complexité
qui annonce 21,51 M. Ce script tranche.

    python -m scripts.evaluate_changemamba \\
        --checkpoint $SCRATCH/pretrained_weight/MambaSCD_Tiny_SECOND_SeK_0.2208.pth

⚠️ **La normalisation d'entrée est le piège de cet exercice.** ChangeMamba
normalise avec les statistiques ImageNet sur l'échelle 0-255
(`mean=[123.675, 116.28, 103.53]`, `std=[58.395, 57.12, 57.375]`), là où notre
dataloader renvoie des images dans [0, 1]. Alimenter leur modèle avec nos tenseurs
produirait des prédictions dégradées et un SeK artificiellement bas — que nous
publierions en croyant avoir mesuré leur modèle. On applique donc leur
normalisation, tout en gardant NOS labels, NOTRE split et NOTRE évaluateur.

Le script compare le résultat au SeK annoncé dans le nom du fichier et refuse de
présenter le chiffre comme exploitable en cas d'écart important.
"""

import argparse
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from csf_mamba.datasets import DATASETS
from csf_mamba.evaluation.metrics import SCDEvaluator

_CHANGEMAMBA = Path(__file__).resolve().parents[1] / "third_party" / "ChangeMamba"

# Statistiques de normalisation de ChangeMamba (changedetection/datasets/imutils.py).
IMAGENET_MEAN = [123.675, 116.28, 103.53]
IMAGENET_STD = [58.395, 57.12, 57.375]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--config", default="vssm1/vssm_tiny_224_0229flex.yaml",
                   help="Config VSSM, relative à changedetection/configs/.")
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


def build_model(cfg_name: str):
    """Instancie MambaSCD exactement comme leur SCDTrainer.build_model."""
    if str(_CHANGEMAMBA) not in sys.path:
        sys.path.insert(0, str(_CHANGEMAMBA))
    from changedetection.configs.config import get_config
    from changedetection.models.ChangeMambaSCD import ChangeMambaSCD
    from changedetection.script.script_utils import get_vssm_kwargs

    cfg_path = _CHANGEMAMBA / "changedetection" / "configs" / cfg_name
    if not cfg_path.is_file():
        raise SystemExit(f"config introuvable : {cfg_path}")

    ns = argparse.Namespace(cfg=str(cfg_path), opts=None, batch_size=None, data_path=None,
                            zip=None, cache_mode=None, pretrained=None,
                            encoder_pretrained_path=None, model_checkpoint_path=None,
                            resume=None, resume_training_path=None, accumulation_steps=None,
                            use_checkpoint=None, disable_amp=None, output=None, tag=None,
                            enable_amp=None, optim=None, memory_limit_rate=None,
                            fused_layernorm=None, fused_window_process=None, amp_opt_level=None,
                            throughput=None, traincost=None)
    cfg = get_config(ns)
    # output_clf=7 : 6 classes réelles + l'index 0 réservé — notre convention A,
    # native chez eux aussi. Aucun décalage d'indices, les prédictions sont
    # directement comparables à nos labels.
    return ChangeMambaSCD(output_cd=2, output_clf=7, pretrained=None, **get_vssm_kwargs(cfg))


def load_weights(model, path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model", "state_dict", "net"):
        if isinstance(ckpt, dict) and key in ckpt:
            ckpt = ckpt[key]
            break
    ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"  poids chargés : {len(ckpt)} tenseurs | manquants {len(missing)} | "
          f"inattendus {len(unexpected)}")
    if missing or unexpected:
        # Un chargement partiel donnerait un modèle à moitié aléatoire et un SeK
        # faux. On refuse plutôt que de produire un chiffre trompeur.
        print("  premiers manquants  :", missing[:5])
        print("  premiers inattendus :", unexpected[:5])
        raise SystemExit("⛔ le state_dict ne correspond pas exactement à ce modèle — "
                         "vérifier --config (tiny / small / base) avant d'interpréter "
                         "le moindre chiffre.")


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        sys.exit("Un GPU est requis (le kernel selective_scan ne tourne pas sur CPU).")

    model = build_model(args.config)
    load_weights(model, args.checkpoint)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  paramètres : {n_params:,} ({n_params / 1e6:.2f} M)")
    model = model.to(device).eval()

    dataset_cls, num_classes = DATASETS["second"]
    ds = dataset_cls(args.data_root, split=args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"  {len(ds)} paires dans le split '{args.split}'")

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    evaluator = SCDEvaluator(num_classes=num_classes)

    for i, batch in enumerate(loader):
        # Notre dataloader renvoie [0, 1] : on remonte en 0-255 avant d'appliquer
        # LEUR normalisation. Exact, à la précision flottante près.
        t1 = (batch["img_t1"].to(device) * 255.0 - mean) / std
        t2 = (batch["img_t2"].to(device) * 255.0 - mean) / std

        out_cd, out_t1, out_t2 = model(t1, t2)
        preds = {"change": out_cd.argmax(1),
                 "sem_t1": out_t1.argmax(1), "sem_t2": out_t2.argmax(1)}
        targets = {k: batch[k].to(device) for k in ("change", "sem_t1", "sem_t2")}
        evaluator.update(preds, targets)
        if i % 50 == 0:
            print(f"    lot {i}/{len(loader)}", flush=True)

    m = evaluator.compute()
    print(f"\n===== MambaSCD ({Path(args.checkpoint).name}) — notre code de métriques =====")
    print(f"  Paramètres : {n_params / 1e6:8.2f} M")
    for name in ("sek", "fscd", "miou", "oa", "kappa"):
        print(f"  {name:<10} : {getattr(m, name):8.4f}")

    # Contrôle : le SeK publié figure dans le nom du fichier (…_SeK_0.2208.pth).
    published = re.search(r"SeK[_-]?(\d\.\d+)", Path(args.checkpoint).name)
    if published:
        ref = float(published.group(1))
        gap = m.sek - ref
        print(f"\n  SeK annoncé par ChangeMamba : {ref:.4f}   |   mesuré ici : {m.sek:.4f}"
              f"   |   écart : {gap:+.4f}")
        if abs(gap) > 0.01:
            print("  ⛔ écart trop grand pour être du bruit de protocole. Ne PAS reprendre")
            print("     ce chiffre : chercher d'abord la cause (normalisation, config du")
            print("     backbone, split, redimensionnement) — un modèle mal alimenté")
            print("     produit un SeK bas qu'on prendrait à tort pour sa performance.")
        else:
            print("  ✅ concordance : notre chaîne d'évaluation reproduit leur chiffre,")
            print("     ce qui valide À LA FOIS notre code de métriques et la comparaison.")


if __name__ == "__main__":
    main()
