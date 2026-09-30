"""Vérifie LandsatSCD512 (version PerASCD) fichier par fichier, avant tout entraînement.

    python -m scripts.check_landsat_perascd --data-root $SCRATCH/LandsatSCD512 \
        --out $SCRATCH/csf-distill/checks/landsat_perascd_check.json

Contrôles (tous bloquants sauf mention) :
- comptes par split = 1431 / 477 / 477, aucun nom commun entre splits ;
- mêmes noms dans im1, im2, label1, label2 ;
- images RGB 512×512, labels mono-canal 512×512, indices 0..4 ;
- **label1 > 0 ⇔ label2 > 0** : le changement est défini par la seule date 1 (comme
  PerASCD) ; si les deux dates ne s'accordent pas, cette définition serait fausse ;
- aucune transition d'une classe vers elle-même (label1 == label2 > 0) — le jeu ne
  compte que 10 types de transitions, toutes entre classes différentes ;
- relevé (non bloquant) : taux de changement, pixels par classe, table des
  transitions observées (à comparer à MAP_A / MAP_B), pixels blancs hors emprise.
Sans torch ni GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from csf_mamba.datasets.landsat_perascd import (  # noqa: E402
    CLASS_NAMES, NUM_SEMANTIC_CLASSES, SPLIT_SIZES,
)

# Transitions publiées (MCDlabel_to_SCDlabel.py) : (classe T1, classe T2), sans le 0.
MAP_A = [0, 1, 1, 2, 2, 2, 3, 3, 4, 4]
MAP_B = [0, 2, 3, 1, 3, 4, 1, 2, 1, 2]
PUBLISHED = {(a, b) for a, b in zip(MAP_A[1:], MAP_B[1:])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    errors, report = [], {"root": str(args.data_root), "splits": {}}
    names = {}
    for split, expected in SPLIT_SIZES.items():
        d = args.data_root / split
        sets = {f: {p.name for p in (d / f).glob("*.png")} for f in ("im1", "im2", "label1", "label2")}
        ids = sorted(sets["im1"])
        names[split] = set(ids)
        if len(ids) != expected:
            errors.append(f"{split} : {len(ids)} paires, {expected} attendues")
        for f, s in sets.items():
            if s != sets["im1"]:
                errors.append(f"{split}/{f} : {len(s ^ sets['im1'])} noms différents de im1")
        n = np.zeros(NUM_SEMANTIC_CLASSES, np.int64)
        trans = np.zeros((NUM_SEMANTIC_CLASSES,) * 2, np.int64)
        changed = total = disagree = white = 0
        bad = []
        for i, name in enumerate(ids):
            im1 = np.asarray(Image.open(d / "im1" / name))
            im2 = np.asarray(Image.open(d / "im2" / name))
            l1 = np.asarray(Image.open(d / "label1" / name))
            l2 = np.asarray(Image.open(d / "label2" / name))
            if im1.shape != (512, 512, 3) or im2.shape != (512, 512, 3):
                bad.append(f"{name} image {im1.shape} {im2.shape}")
            if l1.shape != (512, 512) or l2.shape != (512, 512):
                bad.append(f"{name} label {l1.shape} {l2.shape}")
                continue
            if max(l1.max(), l2.max()) >= NUM_SEMANTIC_CLASSES:
                bad.append(f"{name} indice {max(l1.max(), l2.max())}")
                continue
            disagree += int(((l1 > 0) != (l2 > 0)).sum())
            changed += int((l1 > 0).sum())
            total += l1.size
            white += int((np.all(im1 == 255, axis=2) & np.all(im2 == 255, axis=2)).sum())
            n += np.bincount(l1.ravel(), minlength=NUM_SEMANTIC_CLASSES)
            np.add.at(trans, (l1.ravel(), l2.ravel()), 1)
            if (i + 1) % 500 == 0:
                print(f"   {split} {i + 1}/{len(ids)}", flush=True)
        if bad:
            errors.append(f"{split} : {len(bad)} fichiers non conformes (ex. {bad[:3]})")
        if disagree:
            errors.append(f"{split} : {disagree} pixels où label1 > 0 et label2 > 0 divergent")
        self_trans = int(sum(trans[k, k] for k in range(1, NUM_SEMANTIC_CLASSES)))
        if self_trans:
            errors.append(f"{split} : {self_trans} pixels de transition d'une classe vers elle-même")
        observed = {(a, b) for a in range(1, 5) for b in range(1, 5) if trans[a, b] > 0}
        report["splits"][split] = {
            "pairs": len(ids), "change_rate": changed / max(total, 1),
            "pixels_T1_by_class": dict(zip(CLASS_NAMES, n.tolist())),
            "transitions": {f"{CLASS_NAMES[a]}->{CLASS_NAMES[b]}": int(trans[a, b])
                            for a, b in sorted(observed)},
            "transitions_not_published": sorted(map(list, observed - PUBLISHED)),
            "published_not_observed": sorted(map(list, PUBLISHED - observed)),
            "white_both_dates_px": white,
        }
        print(f"== {split} : {len(ids)} paires, changement {100 * changed / max(total, 1):.2f} %, "
              f"{len(observed)} transitions observées", flush=True)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        common = names[a] & names[b]
        if common:
            errors.append(f"{len(common)} noms communs {a}/{b} (ex. {sorted(common)[:3]})")
    report["errors"] = errors
    report["pass"] = not errors
    print(json.dumps({k: v for k, v in report.items() if k != "splits"}, indent=1, ensure_ascii=False))
    print(json.dumps({s: {k: r[k] for k in ("change_rate", "transitions_not_published",
                                            "published_not_observed")}
                      for s, r in report["splits"].items()}, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
