"""Vérifie, sans se fier aux noms de fichiers, si le split de LandsatSCD512 fuit.

    python -m scripts.verify_landsat_leakage --data-root $SCRATCH/LandsatSCD512 \
        --out $SCRATCH/csf-distill/checks/landsat_leakage.json

Quatre contrôles indépendants, du plus brut au plus parlant :

A. **Doublons exacts d'images** (hash des pixels, aucun nom utilisé) : pour chaque paire
   de val/test, ses images im1 et im2 existent-elles, pixel pour pixel, quelque part dans
   l'entraînement (en im1 ou en im2 d'une autre paire) ?
B. **Hypothèse « indice _k = emplacement »** : regroupe les images par (k, année) d'après
   les noms, puis vérifie sur les pixels qu'un groupe ne contient qu'une image distincte,
   et que deux k différents à la même année donnent des images différentes.
C. **Cohérence des labels** : à (k, année) fixés, les pixels annotés dans plusieurs paires
   portent-ils la même classe ?
D. **Oracle par simple recherche** (le plus parlant) : on prédit le test SANS modèle, en
   recopiant les classes connues dans l'ENTRAÎNEMENT pour le même (k, année) ; changement
   prédit là où les deux classes sont connues et diffèrent, « sans changement » ailleurs.
   SeK de cet oracle, calculée avec la métrique du projet. Une valeur élevée signifie que
   le test est en grande partie résoluble par mémoire, sans généralisation.

Le script ne lit que les PNG ; sans torch ni GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from csf_mamba.evaluation.metrics import fast_hist, metrics_from_hist  # noqa: E402

NAME = re.compile(r"^From(\d{4})To(\d{4})_(\d+)\.png$")
NC = 5


def _md5(arr: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    root = args.data_root

    pairs = []                      # (split, name, year1, year2, k)
    for split in ("train", "val", "test"):
        for p in sorted((root / split / "im1").glob("*.png")):
            m = NAME.match(p.name)
            if not m:
                raise SystemExit(f"nom inattendu : {split}/{p.name}")
            pairs.append((split, p.name, m[1], m[2], m[3]))
    print(f"{len(pairs)} paires", flush=True)

    # --- lecture unique : hash des images, labels gardés pour C et D
    img_hash = {}                   # (split, name, 'im1'|'im2') -> md5
    by_k_year = defaultdict(set)    # (k, year) -> {md5}
    train_hashes = set()
    labels = {}                     # (split, name) -> (l1, l2)
    for i, (split, name, a, b, k) in enumerate(pairs):
        for f, year in (("im1", a), ("im2", b)):
            h = _md5(np.asarray(Image.open(root / split / f / name)))
            img_hash[(split, name, f)] = h
            by_k_year[(k, year)].add(h)
            if split == "train":
                train_hashes.add(h)
        labels[(split, name)] = (np.asarray(Image.open(root / split / "label1" / name)),
                                 np.asarray(Image.open(root / split / "label2" / name)))
        if (i + 1) % 500 == 0:
            print(f"   lu {i + 1}/{len(pairs)}", flush=True)

    report = {"n_pairs": len(pairs),
              "n_locations_by_name": len({p[4] for p in pairs}),
              "n_distinct_images": len(set(img_hash.values()))}

    # --- A. doublons exacts, sans noms
    A = {}
    for split in ("val", "test"):
        sp = [p for p in pairs if p[0] == split]
        one = sum((img_hash[(split, p[1], "im1")] in train_hashes) or
                  (img_hash[(split, p[1], "im2")] in train_hashes) for p in sp)
        both = sum((img_hash[(split, p[1], "im1")] in train_hashes) and
                   (img_hash[(split, p[1], "im2")] in train_hashes) for p in sp)
        A[split] = {"pairs": len(sp), "at_least_one_image_in_train": one, "both_images_in_train": both}
    report["A_exact_image_duplicates"] = A

    # --- B. (k, année) = une seule image ? années égales, k différents = images différentes ?
    sizes = [len(v) for v in by_k_year.values()]
    year_groups = defaultdict(list)
    for (k, y), hs in by_k_year.items():
        year_groups[y].append(hs)
    cross_k_shared = 0              # années où une même image apparaît sous deux k différents
    for g in year_groups.values():
        counts = defaultdict(int)
        for hs in g:
            for h in hs:
                counts[h] += 1
        cross_k_shared += any(c > 1 for c in counts.values())
    report["B_location_hypothesis"] = {
        "k_year_groups": len(by_k_year),
        "groups_with_exactly_one_distinct_image": sum(s == 1 for s in sizes),
        "max_distinct_images_in_a_group": max(sizes),
        "years_where_two_k_share_an_image": int(cross_k_shared),
    }

    # --- C. cohérence des labels à (k, année) fixés, et cartes de classes connues (train seul)
    H = W = None
    known_train = {}                # (k, year) -> carte uint8 des classes connues (0 = inconnu)
    conflicts = agree = 0
    seen_all = {}
    for split, name, a, b, k in pairs:
        l1, l2 = labels[(split, name)]
        H, W = l1.shape
        for lab, year in ((l1, a), (l2, b)):
            key = (k, year)
            cur = seen_all.setdefault(key, np.zeros_like(lab))
            both = (cur > 0) & (lab > 0)
            conflicts += int((cur[both] != lab[both]).sum()); agree += int((cur[both] == lab[both]).sum())
            cur[(cur == 0) & (lab > 0)] = lab[(cur == 0) & (lab > 0)]
            if split == "train":
                kt = known_train.setdefault(key, np.zeros_like(lab))
                kt[(kt == 0) & (lab > 0)] = lab[(kt == 0) & (lab > 0)]
    report["C_label_consistency"] = {"overlapping_annotated_pixels": agree + conflicts,
                                     "conflicting": conflicts,
                                     "agreement": agree / max(agree + conflicts, 1)}

    # --- D. oracle de recherche sur le test (aucun modèle)
    hist = np.zeros((NC, NC))
    cov_num = cov_den = 0
    for split, name, a, b, k in pairs:
        if split != "test":
            continue
        l1, l2 = labels[(split, name)]
        ca = known_train.get((k, a), np.zeros((H, W), np.uint8))
        cb = known_train.get((k, b), np.zeros((H, W), np.uint8))
        known = (ca > 0) & (cb > 0)
        change = known & (ca != cb)
        pa, pb = ca * change, cb * change
        gt_change = l1 > 0
        hist += fast_hist(pa.ravel(), (l1 * gt_change).ravel(), NC)
        hist += fast_hist(pb.ravel(), (l2 * gt_change).ravel(), NC)
        cov_num += int((known & gt_change).sum()); cov_den += int(gt_change.sum())
    m = metrics_from_hist(hist)
    report["D_lookup_oracle_test"] = {
        "sek": m.sek, "fscd": m.fscd, "miou": m.miou, "oa": m.oa,
        "changed_test_pixels_with_both_classes_known_from_train": cov_num / max(cov_den, 1),
        "note": "aucun modèle : classes recopiées de l'entraînement au même (k, année)",
    }

    print(json.dumps(report, indent=1, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
