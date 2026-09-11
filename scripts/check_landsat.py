"""Vérifie un dump Landsat-SCD AVANT d'y lancer le moindre entraînement.

Pourquoi ce script existe. Le dataloader a été écrit d'après le code de
Mamba-FCS, pas d'après les fichiers : personne n'a encore regardé le dump. Or le
même projet a déjà porté pendant deux semaines de **faux noms de classes** sur
SECOND, parce que l'ordre d'énumération de l'article ne correspondait pas aux
indices réels. Un contrôle qui coûte deux minutes évite un entraînement de 12 h
sur des cibles mal interprétées — ou pire, un chiffre publié avec de mauvaises
étiquettes.

Ne dépend ni de torch ni du GPU : numpy et PIL suffisent, donc il tourne sur un
nœud de connexion.

    python -m scripts.check_landsat --data-root $SCRATCH/Landsat-SCD
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

# ⚠️ La littérature annonce 2 425 paires au total, « 1 908 pour l'entraînement et
# 477 pour le test ». Or 1 908 + 477 = 2 385, pas 2 425 : il manque 40 paires. La
# lecture la plus probable est que 1 908 désigne le seul `train_list.txt` et que
# les 40 restantes forment `val_list.txt`. On vérifie donc durement le TOTAL, qui
# est la taille du jeu et ne prête pas à interprétation, et on se contente de
# rapporter la répartition sans la trancher.
ATTENDU = {"paires": 2425, "train": 1908, "test": 477, "taille": 416, "classes": 5}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--echantillon", type=int, default=200,
                   help="Nombre de tuiles inspectées en profondeur.")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root)
    pbs = []

    def verif(ok, msg, detail=""):
        print(f"  {'✓' if ok else '⛔'} {msg}{(' — ' + detail) if detail else ''}")
        if not ok:
            pbs.append(msg)

    print(f"Racine : {root}\n")
    print("== 1. Arborescence ==")
    for d in ("A", "B", "labelA", "labelB"):
        verif((root / d).is_dir(), f"dossier {d}/")
    if pbs:
        print("\nArborescence incomplète — voir la docstring de "
              "csf_mamba/datasets/landsat_scd.py pour le format attendu.")
        return 1

    print("\n== 2. Listes officielles ==")
    listes = {}
    for nom in ("train", "val", "test"):
        f = root / f"{nom}_list.txt"
        if f.is_file():
            listes[nom] = [l.strip() for l in f.read_text().splitlines() if l.strip()]
            print(f"  ✓ {nom}_list.txt : {len(listes[nom])} entrées")
        else:
            print(f"  ⚠️ {nom}_list.txt absent")
    tot = sum(len(v) for v in listes.values())
    verif(tot == ATTENDU["paires"], f"total des listes = {ATTENDU['paires']}", f"trouvé {tot}")

    n_train, n_val = len(listes.get("train", [])), len(listes.get("val", []))
    n_test = len(listes.get("test", []))
    print(f"  répartition : train {n_train} | val {n_val} | test {n_test}")
    print(f"  la littérature annonce 1 908 en entraînement et 477 en test, mais")
    print(f"  1 908 + 477 = 2 385 alors que le jeu en compte 2 425 : les 40 paires")
    print(f"  manquantes sont vraisemblablement la validation. À confronter au dump.")
    if n_test and n_test != ATTENDU["test"]:
        print(f"  ⚠️ test = {n_test} au lieu des {ATTENDU['test']} annoncés — "
              f"la comparaison à Mamba-FCS ne porterait pas sur le même split.")
        pbs.append("taille du split de test")
    print(f"  -> Mamba-FCS entraîne sur train + val, soit {n_train + n_val} paires ici.")

    noms = [n for v in listes.values() for n in v]
    if not noms:
        noms = sorted(p.name for p in (root / "A").glob("*"))
        print(f"  (aucune liste : repli sur le glob de A/, {len(noms)} fichiers)")

    print("\n== 3. Les quatre dossiers contiennent-ils les mêmes identifiants ? ==")
    for d in ("A", "B", "labelA", "labelB"):
        manquants = [n for n in noms[:args.echantillon] if not (root / d / n).is_file()]
        verif(not manquants, f"{d}/ contient l'échantillon",
              f"{len(manquants)} manquants, ex. {manquants[:3]}" if manquants else "")

    print(f"\n== 4. Inspection de {min(args.echantillon, len(noms))} tuiles ==")
    tailles, indices, ndim_labels = Counter(), Counter(), Counter()
    frac_chg, desaccords, n_lus = [], 0, 0
    for n in noms[:args.echantillon]:
        try:
            img = Image.open(root / "A" / n)
            a = np.asarray(Image.open(root / "labelA" / n))
            b = np.asarray(Image.open(root / "labelB" / n))
        except Exception as e:
            print(f"  ⛔ lecture impossible pour {n} : {e}")
            pbs.append("lecture")
            break
        n_lus += 1
        tailles[img.size] += 1
        ndim_labels[a.ndim] += 1
        if a.ndim == 2:
            indices.update(np.unique(a).tolist())
            indices.update(np.unique(b).tolist())
            chg = (a > 0) | (b > 0)
            frac_chg.append(float(chg.mean()))
            # Si la sémantique n'est annotée QUE dans les zones changées, les deux
            # cartes doivent porter une classe aux mêmes pixels. Un désaccord
            # signifierait que le OU de Mamba-FCS n'est pas anodin.
            desaccords += int(((a > 0) != (b > 0)).sum())

    print(f"  tailles d'image : {dict(tailles)}")
    verif(all(t == (ATTENDU['taille'], ATTENDU['taille']) for t in tailles),
          f"toutes les tuiles font {ATTENDU['taille']}x{ATTENDU['taille']}")
    print(f"  dimensions des labels : {dict(ndim_labels)}")
    verif(set(ndim_labels) == {2}, "labels mono-canal (cartes d'indices, pas RGB)",
          "un dump RGB demanderait une table couleur -> indice")

    if indices:
        vus = sorted(indices)
        print(f"  indices présents dans labelA/labelB : {vus}")
        verif(max(vus) < ATTENDU["classes"],
              f"indices < {ATTENDU['classes']} (0 réservé + 4 classes réelles)",
              f"maximum trouvé : {max(vus)}")
        verif(0 in vus, "l'index 0 est présent (zones sans changement)")

    if frac_chg:
        m = float(np.mean(frac_chg))
        print(f"  fraction de pixels changés : {m*100:.2f} % en moyenne "
              f"(min {min(frac_chg)*100:.1f} %, max {max(frac_chg)*100:.1f} %)")
        print(f"  -> pour mémoire : SECOND 20,1 %, Hi-UCD 1,4 %")
        if m < 0.03:
            print("  ⚠️ taux très faible : la compensation de déséquilibre sera")
            print("     probablement NÉCESSAIRE ici, comme sur Hi-UCD. Ne PAS")
            print("     reprendre le réglage de SECOND, où elle est nuisible.")

    if n_lus:
        print(f"  pixels où une seule des deux dates porte une classe : {desaccords}")
        if desaccords == 0:
            print("     -> les deux cartes s'accordent : le OU de Mamba-FCS est sans effet")
        else:
            print("     -> le OU n'est PAS anodin ; on le conserve, c'est leur convention")

    print("\n== 5. Une version colorée existe-t-elle ? ==")
    colores = [p.name for p in root.iterdir()
               if p.is_dir() and ("color" in p.name.lower() or "rgb" in p.name.lower())]
    if colores:
        print(f"  ✓ dossiers trouvés : {colores}")
        print("    -> permettrait de VÉRIFIER l'ordre des noms de classes,")
        print("       comme scripts/check_second_classes l'a fait pour SECOND.")
    else:
        print("  ⚠️ aucun dossier coloré. L'ordre des noms de classes")
        print("     ('farmland, desert, buildings, water') reste NON VÉRIFIÉ.")
        print("     Aucune métrique n'en dépend, mais une figure ou un rapport si.")

    print("\n" + "=" * 70)
    if pbs:
        print(f"⛔ {len(pbs)} problème(s) : " + " | ".join(pbs))
        print("   Ne PAS lancer d'entraînement avant de les avoir compris.")
        return 1
    print("✓ Dump conforme au format attendu. L'entraînement peut être lancé.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
