"""Découpage de LandsatSCD512 disjoint par EMPLACEMENT : 4 plis de validation croisée.

    python -m scripts.make_landsat_location_folds --data-root <LandsatSCD512> \
        --out splits/LandsatSCD_loc

Pourquoi. Les 2 385 paires de LandsatSCD512 ne couvrent que 12 emplacements (indice `_k`
de `From<a>To<b>_<k>.png`), chacun vu à ~28 années : 333 images distinctes en tout. Tout
split aléatoire met les mêmes images en train et en test, et un oracle sans modèle y
obtient SeK 0,78 (`scripts/verify_landsat_leakage.py`). Ici, un emplacement n'appartient
qu'à un split par pli.

Schéma : 4 groupes de test de 3 emplacements. Pour chaque pli, 1 emplacement de val pris
parmi les 9 restants, et 8 emplacements en entraînement. Toutes les paires d'un
emplacement suivent cet emplacement. Les paires sont désignées par `<split>/<nom>`, le
dossier d'origine de l'archive, ce qui permet de lire les images sans recopier le jeu.

Choix des groupes, déterministe et sans regarder aucun modèle : énumération exhaustive des
15 400 partitions de 12 emplacements en 4 groupes de 3. On retient celle qui minimise la
somme, sur les groupes, de la divergence de Jensen-Shannon entre la distribution des
transitions du groupe et la distribution globale, plus l'écart relatif de taux de
changement. Égalités départagées par l'ordre lexicographique. L'emplacement de val de
chaque pli est, parmi les 9 hors test, le plus proche de la distribution des 9 (même
critère), avec la contrainte que les 4 val soient distincts.

Sorties : `fold{0..3}/{train,val,test}.txt` et `README.json` (statistiques par
emplacement, composition des plis, critère). Le script vérifie qu'aucun emplacement ne
traverse deux splits.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

NAME = re.compile(r"^From(\d{4})To(\d{4})_(\d+)\.png$")
# Transitions publiées (MCDlabel_to_SCDlabel.py), sans le 0.
MAP_A = [1, 1, 2, 2, 2, 3, 3, 4, 4]
MAP_B = [2, 3, 1, 3, 4, 1, 2, 1, 2]
TRANS = list(zip(MAP_A, MAP_B))


def location_stats(root: Path):
    pairs = defaultdict(list)
    trans = defaultdict(lambda: np.zeros(len(TRANS)))
    changed = defaultdict(int)
    total = defaultdict(int)
    for split in ("train", "val", "test"):
        for p in sorted((root / split / "im1").glob("*.png")):
            m = NAME.match(p.name)
            if not m:
                raise SystemExit(f"nom inattendu : {split}/{p.name}")
            k = m[3]
            pairs[k].append(f"{split}/{p.name}")
            l1 = np.asarray(Image.open(root / split / "label1" / p.name))
            l2 = np.asarray(Image.open(root / split / "label2" / p.name))
            ch = l1 > 0
            changed[k] += int(ch.sum()); total[k] += ch.size
            for t, (a, b) in enumerate(TRANS):
                trans[k][t] += int(((l1 == a) & (l2 == b)).sum())
    return pairs, trans, changed, total


def js(p, q):
    p = p / p.sum(); q = q / q.sum(); m = (p + q) / 2
    def f(x, y):
        nz = x > 0
        return float(np.sum(x[nz] * np.log(x[nz] / y[nz])))
    return 0.5 * f(p, m) + 0.5 * f(q, m)


def score_group(locs, trans, changed, total, ref_t, ref_rate):
    t = sum(trans[k] for k in locs)
    rate = sum(changed[k] for k in locs) / sum(total[k] for k in locs)
    return js(t, ref_t) + abs(rate - ref_rate) / ref_rate


def partitions(items):
    """Partitions de `items` (12) en groupes non ordonnés de 3."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for pair in itertools.combinations(rest, 2):
        group = (first,) + pair
        remaining = [x for x in rest if x not in pair]
        for tail in partitions(remaining):
            yield [group] + tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    pairs, trans, changed, total = location_stats(args.data_root)
    locs = sorted(pairs)
    if len(locs) != 12:
        raise SystemExit(f"{len(locs)} emplacements, 12 attendus")
    ref_t = sum(trans.values())
    ref_rate = sum(changed.values()) / sum(total.values())

    best = None
    n_part = 0
    for part in partitions(locs):
        n_part += 1
        s = sum(score_group(g, trans, changed, total, ref_t, ref_rate) for g in part)
        key = (round(s, 12), part)
        if best is None or key < best:
            best = key
    score, groups = best

    folds, used_val = [], set()
    for f, test in enumerate(groups):
        rest = [k for k in locs if k not in test]
        rt = sum(trans[k] for k in rest)
        rr = sum(changed[k] for k in rest) / sum(total[k] for k in rest)
        cands = sorted((score_group((k,), trans, changed, total, rt, rr), k) for k in rest if k not in used_val)
        val = cands[0][1]
        used_val.add(val)
        train = [k for k in rest if k != val]
        folds.append({"test": list(test), "val": [val], "train": train})

    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for f, fold in enumerate(folds):
        d = args.out / f"fold{f}"
        d.mkdir(exist_ok=True)
        seen = {}
        for split in ("train", "val", "test"):
            for k in fold[split]:
                assert k not in seen, f"emplacement {k} dans {seen.get(k)} et {split}"
                seen[k] = split
            ids = sorted(i for k in fold[split] for i in pairs[k])
            (d / f"{split}.txt").write_text("\n".join(ids) + "\n")
            fold[f"n_{split}"] = len(ids)
        assert sorted(seen) == locs
        summary.append(fold)
        print(f"fold{f} : test {fold['test']} ({fold['n_test']}) | val {fold['val']} ({fold['n_val']}) | "
              f"train {len(fold['train'])} emplacements ({fold['n_train']})")

    readme = {
        "source": "LandsatSCD512.zip (HF SathShen/PerASCD-datasets), sha256 "
                  "9dc42679859d0ec5fe02045055063205cdad50fbe002910b9372f0ad96e1a0c0",
        "why": "random splits share all 12 locations (333 distinct images); see scripts/verify_landsat_leakage.py",
        "criterion": "exhaustive search over all partitions into 4 groups of 3; minimise sum of JS(transitions) "
                     "+ |change rate - global| / global; val = closest single location among the 9 others, distinct across folds",
        "n_partitions_searched": n_part, "best_score": score,
        "global_change_rate": ref_rate,
        "locations": {k: {"pairs": len(pairs[k]), "change_rate": changed[k] / total[k],
                          "transitions": dict(zip([f"{a}->{b}" for a, b in TRANS], trans[k].astype(int).tolist()))}
                      for k in locs},
        "folds": summary,
    }
    (args.out / "README.json").write_text(json.dumps(readme, indent=1))
    print(f"{n_part} partitions examinées ; score retenu {score:.4f}")


if __name__ == "__main__":
    main()
