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
import re
import sys
from collections import Counter
from pathlib import Path

# Les noms observés dans le dump portent des suffixes d'augmentation HORS LIGNE :
#   From1990To1993_01.png            tuile source
#   From1990To1993_01CropResize0.png recadrage
#   From1990To1993_01ZheDang1.png    occlusion (遮挡)
#   From1990To1993_01rotate180.png   rotation
# La tuile d'origine se termine par « _<numéro> » ; tout ce qui suit est un
# suffixe de variante.
BASE = re.compile(r"^(.*_\d+)(.*)$")

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


def _decrire_images(dossier: Path, n=3):
    """Mode, taille et valeurs d'un échantillon d'images d'un dossier."""
    fichiers = sorted(p for p in dossier.iterdir() if p.is_file())[:n]
    for f in fichiers:
        try:
            img = Image.open(f)
        except Exception as e:
            print(f"      {f.name} : illisible ({e})")
            continue
        arr = np.asarray(img)
        print(f"      {f.name}")
        print(f"        mode={img.mode} taille={img.size} shape={arr.shape} dtype={arr.dtype}")
        if arr.ndim == 2:
            vals = np.unique(arr)
            print(f"        {len(vals)} valeurs distinctes : {vals[:20].tolist()}"
                  f"{' …' if len(vals) > 20 else ''}")
        elif arr.ndim == 3:
            couleurs = np.unique(arr.reshape(-1, arr.shape[2]), axis=0)
            print(f"        {len(couleurs)} couleurs distinctes : "
                  f"{[tuple(int(x) for x in c) for c in couleurs[:8]]}"
                  f"{' …' if len(couleurs) > 8 else ''}")
        if img.mode == "P":
            pal = img.getpalette() or []
            entrees = [tuple(pal[i * 3:i * 3 + 3]) for i in range(min(len(pal) // 3, 12))]
            print(f"        palette (12 premières entrées) : {entrees}")
            print("        -> image À PALETTE : les VALEURS sont des indices, la")
            print("           couleur n'est qu'un affichage. np.asarray rend bien")
            print("           les indices, pas du RGB.")


def explorer(root: Path):
    """Rapporte ce que contient RÉELLEMENT le dump, quand il ne suit pas le format.

    Le dump brut de figshare contient `A/`, `B/` et un unique `label/`, là où le
    code de Mamba-FCS attend `labelA/`, `labelB/` et trois listes de splits. Ils
    ont donc prétraité le jeu sans publier ce prétraitement. Pour écrire un
    dataloader juste il faut d'abord savoir ce que `label/` encode :

    * une carte sémantique par date fusionnée — peu probable avec un seul dossier ;
    * une carte de TRANSITION « from-to » — les 10 types de changement annoncés le
      suggèrent, et il faudrait alors la décoder en deux cartes sémantiques ;
    * une image à palette dont les indices sont les classes.

    Ce rapport tranche à partir des fichiers, pas d'une hypothèse.
    """
    print("\n" + "=" * 70)
    print("EXPLORATION — le dump ne suit pas le format attendu, voici son contenu")
    print("=" * 70)

    print("\n-- entrées à la racine --")
    for p in sorted(root.iterdir()):
        if p.is_dir():
            fichiers = [f for f in p.iterdir() if f.is_file()]
            exts = Counter(f.suffix.lower() for f in fichiers)
            print(f"  {p.name + '/':<16} {len(fichiers):>6} fichiers   {dict(exts)}")
        else:
            print(f"  {p.name:<16} {p.stat().st_size:>6} octets  (fichier)")

    print("\n-- listes de splits --")
    listes = sorted(root.rglob("*.txt"))
    if listes:
        for f in listes:
            n = len([l for l in f.read_text().splitlines() if l.strip()])
            print(f"  {f.relative_to(root)} : {n} entrées")
    else:
        print("  ⚠️ AUCUN fichier .txt : le dump ne fournit pas les splits officiels.")
        print("     Mamba-FCS en utilise trois (train/val/test). Sans eux, notre")
        print("     comparaison à leur chiffre ne porterait pas sur le même découpage.")

    for nom in ("label", "labelA", "labelB", "A", "B"):
        d = root / nom
        if d.is_dir():
            print(f"\n-- contenu de {nom}/ --")
            _decrire_images(d)

    print("\n" + "=" * 70)
    print("À DÉCIDER À PARTIR DE CE RAPPORT :")
    print("  1. Que code `label/` — sémantique par date, ou transition « from-to » ?")
    print("  2. Comment obtenir les splits de Mamba-FCS, sans quoi la comparaison")
    print("     à leur chiffre ne serait pas appariée.")
    print("Ne rien écrire dans le dataloader avant d'avoir répondu aux deux.")


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
        # Le dump brut de figshare ne suit PAS la structure qu'attend le code
        # de Mamba-FCS : ils l'ont prétraité sans publier le prétraitement.
        # Plutôt que de refuser, on explore et on rapporte ce qui est
        # réellement là — seule façon de décider quoi écrire ensuite.
        explorer(root)
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

    print("\n== 5. ⚠️ Variantes augmentées et fuite entre splits ==")
    if len(listes) >= 2 and "test" in listes:
        def base(nom):
            m = BASE.match(Path(nom).stem)
            return (m.group(1), m.group(2)) if m else (Path(nom).stem, "")

        suffixes = Counter()
        bases = {}
        for split, noms_l in listes.items():
            bases[split] = set()
            for n in noms_l:
                b, s = base(n)
                bases[split].add(b)
                suffixes[s or "(tuile source)"] += 1

        print("  suffixes rencontrés :")
        for s, c in suffixes.most_common(12):
            print(f"    {s:<24} {c:>6}")
        n_src = suffixes.get("(tuile source)", 0)
        tot_f = sum(suffixes.values())
        if tot_f and n_src < tot_f:
            print(f"  -> {n_src} tuiles sources pour {tot_f} fichiers : le jeu contient")
            print(f"     des variantes AUGMENTÉES HORS LIGNE (rotations, recadrages,")
            print(f"     occlusions). Deux conséquences à ne pas manquer.")

        entrainement = bases.get("train", set()) | bases.get("val", set())
        fuite = entrainement & bases["test"]
        print(f"\n  tuiles sources distinctes : entraînement {len(entrainement)}, "
              f"test {len(bases['test'])}")
        if fuite:
            print(f"  ⛔ FUITE : {len(fuite)} tuiles sources apparaissent des DEUX côtés")
            print(f"     du split. Exemples : {sorted(fuite)[:3]}")
            print(f"     Le test contient donc des variantes d'images d'entraînement.")
            print(f"     Tout chiffre publié sur ce jeu — le nôtre comme celui de")
            print(f"     Mamba-FCS — en est affecté. À SIGNALER, pas à corriger")
            print(f"     unilatéralement : changer de split romprait la comparaison.")
            pbs.append("fuite entre splits")
        else:
            print("  ✓ aucune tuile source partagée entre entraînement et test")

    print("\n== 6. Une version colorée existe-t-elle ? ==")
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
