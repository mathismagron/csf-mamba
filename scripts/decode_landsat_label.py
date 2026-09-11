"""Retrouver la table des transitions de Landsat-SCD depuis les données.

LE PROBLÈME
===========
Le dump ne contient qu'un dossier `label/`, et sa description figshare dit :
« each "from-to" change type is a separate class representing land-cover
transitions ». Les valeurs 0..9 encodent donc des TRANSITIONS, pas la sémantique
par date. Or tout le pipeline du projet — loss, métriques SeK, décodeurs —
travaille sur deux cartes sémantiques, une par date.

Il faut donc décoder `v -> (classe à T1, classe à T2)`. La table n'est publiée ni
dans les métadonnées figshare, ni dans le dépôt de Mamba-FCS, qui a prétraité le
jeu sans publier son prétraitement.

LA MÉTHODE
==========
Elle se déduit des données, et sa prédiction est falsifiable.

Si la valeur `v` code « classe X → classe Y », alors les pixels portant `v`
montrent l'aspect de X dans l'image A et celui de Y dans l'image B. Donc :

* deux valeurs partageant la même classe de DÉPART ont des couleurs moyennes
  proches **dans A** ;
* deux valeurs partageant la même classe d'ARRIVÉE ont des couleurs moyennes
  proches **dans B** ;
* avec 4 classes de terrain, les couleurs moyennes doivent former **4 groupes
  dans A et 4 dans B** — pas 9, pas 2.

Si l'on n'observe pas cette structure, l'hypothèse « from-to » est fausse et il
ne faut pas décoder. C'est le point : la méthode peut échouer bruyamment.

Le jeu s'y prête particulièrement — désert, terres agricoles, bâti et eau ont des
signatures colorimétriques très séparées sur du Landsat 30 m.

    python -m scripts.decode_landsat_label --data-root $SCRATCH/Landsat-SCD
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from csf_mamba.datasets.landsat_scd import apparier_dossiers


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--tuiles", type=int, default=400,
                   help="Nombre de tuiles échantillonnées régulièrement.")
    p.add_argument("--classes", type=int, default=4,
                   help="Nombre de classes de terrain attendu.")
    p.add_argument("--seuil", type=float, default=18.0,
                   help="Distance RGB en deçà de laquelle deux valeurs sont "
                        "réputées partager une classe.")
    return p.parse_args()


def _groupes(moyennes: dict, seuil: float):
    """Regroupe des couleurs moyennes par proximité — agglomération simple.

    Pas de dépendance à scikit-learn : le problème est minuscule (une dizaine de
    points en dimension 3) et un regroupement par lien simple suffit.
    """
    cles = sorted(moyennes)
    groupes = []
    for k in cles:
        place = False
        for g in groupes:
            if any(np.linalg.norm(moyennes[k] - moyennes[o]) < seuil for o in g):
                g.append(k); place = True; break
        if not place:
            groupes.append([k])
    # Fusion des groupes devenus proches après agglomération.
    fusionne = True
    while fusionne:
        fusionne = False
        for i in range(len(groupes)):
            for j in range(i + 1, len(groupes)):
                if any(np.linalg.norm(moyennes[a] - moyennes[b]) < seuil
                       for a in groupes[i] for b in groupes[j]):
                    groupes[i] += groupes.pop(j); fusionne = True; break
            if fusionne:
                break
    return groupes


def main():
    args = parse_args()
    root = Path(args.data_root)
    # ⚠️ Apparier sur le nom en minuscules : le dump mélange `ZheDang` et
    # `Zhedang` entre dossiers, et Linux distingue la casse.
    try:
        cles, rapport = apparier_dossiers(root)
    except ValueError as e:
        print(f"⛔ {e}")
        return 1
    exclus = {d: len(v) for d, v in rapport["exclus"].items() if v}
    if exclus:
        print(f"⚠️ noms présents dans un dossier mais pas dans les autres : {exclus}")
        for d, v in rapport["exclus"].items():
            if v:
                print(f"   {d}/ : ex. {v[:3]}")
    pas = max(1, len(cles) // args.tuiles)
    echantillon = cles[::pas][:args.tuiles]
    print(f"Échantillon : {len(echantillon)} tuiles sur {len(cles)} appariées\n")
    nom_reel = rapport["par_dossier"]

    somme_a, somme_b, compte = {}, {}, {}
    for cle in echantillon:
        lab = np.asarray(Image.open(root / "label" / nom_reel["label"][cle]))
        a = np.asarray(Image.open(root / "A" / nom_reel["A"][cle]).convert("RGB"),
                       dtype=np.float64)
        b = np.asarray(Image.open(root / "B" / nom_reel["B"][cle]).convert("RGB"),
                       dtype=np.float64)
        if a.shape[:2] != lab.shape:
            print(f"⛔ formes incohérentes pour {cle} : {a.shape[:2]} vs {lab.shape}")
            return 1
        for v in np.unique(lab).tolist():
            m = lab == v
            n = int(m.sum())
            somme_a[v] = somme_a.get(v, 0.0) + a[m].sum(axis=0)
            somme_b[v] = somme_b.get(v, 0.0) + b[m].sum(axis=0)
            compte[v] = compte.get(v, 0) + n

    total = sum(compte.values())
    moy_a = {v: somme_a[v] / compte[v] for v in compte}
    moy_b = {v: somme_b[v] / compte[v] for v in compte}

    print(f"{'valeur':>7} {'% pixels':>10}   {'couleur moyenne dans A':>24}   "
          f"{'couleur moyenne dans B':>24}   {'écart A->B':>10}")
    for v in sorted(compte):
        da = tuple(round(x) for x in moy_a[v])
        db = tuple(round(x) for x in moy_b[v])
        ecart = np.linalg.norm(moy_a[v] - moy_b[v])
        print(f"{v:>7} {compte[v] / total * 100:>9.2f} %   {str(da):>24}   "
              f"{str(db):>24}   {ecart:>10.1f}")

    # La valeur 0 (« sans changement ») mélange toutes les classes : on l'exclut
    # du regroupement, mais on s'en sert comme contrôle plus bas.
    vals = sorted(v for v in compte if v != 0)

    # ⚠️ Regrouper A et B ENSEMBLE, et non séparément. Une classe de terrain a la
    # même apparence qu'elle soit la classe de départ d'une transition ou son
    # arrivée : « désert » doit recevoir le même indice dans les deux cas. Les
    # regrouper séparément donne deux numérotations indépendantes, et fabrique de
    # fausses transitions vers elles-mêmes — c'est ce qu'un test sur une table
    # connue a révélé.
    pool = {}
    for v in vals:
        pool[("A", v)] = moy_a[v]
        pool[("B", v)] = moy_b[v]
    groupes = _groupes(pool, args.seuil)
    classe = {k: i for i, g in enumerate(groupes, 1) for k in g}

    print(f"\n== Regroupement conjoint des couleurs A et B (seuil {args.seuil}) ==")
    for i, g in enumerate(groupes, 1):
        centre = np.mean([pool[k] for k in g], axis=0)
        membres = sorted(f"{d}{v}" for d, v in g)
        print(f"  classe {i} : couleur {tuple(round(x) for x in centre)}  <- {membres}")

    ok = len(groupes) == args.classes
    print(f"\n== Verdict ==")
    if not ok:
        print(f"  ⛔ {len(groupes)} classes détectées au lieu de {args.classes}.")
        print(f"     Deux possibilités :")
        print(f"     - le seuil est mal choisi : réessayer avec --seuil 10, 25, 35 ;")
        print(f"     - l'hypothèse « from-to » est fausse pour ce dump.")
        print(f"     NE PAS décoder tant que la structure n'apparaît pas.")
        return 1

    print(f"  ✓ {args.classes} classes de terrain retrouvées : la structure")
    print(f"    « from-to » est CONFIRMÉE par les données.\n")
    print(f"    {'valeur':>7} -> {'classe T1':>10} {'classe T2':>10}")
    diag = []
    table = {}
    for v in vals:
        d1, d2 = classe[("A", v)], classe[("B", v)]
        table[v] = (d1, d2)
        print(f"    {v:>7} -> {d1:>10} {d2:>10}")
        if d1 == d2:
            diag.append(v)

    if diag:
        print(f"\n  ⚠️ les valeurs {diag} ont MÊME classe de départ et d'arrivée.")
        print(f"     Une transition vers soi-même n'a pas de sens : le seuil est")
        print(f"     probablement trop grand, ou ces valeurs ne sont pas des")
        print(f"     transitions. NE PAS utiliser cette table telle quelle.")
        ok = False
    else:
        print(f"\n  ✓ aucune transition vers elle-même : la table est cohérente.")

    ecart0 = np.linalg.norm(moy_a[0] - moy_b[0])
    ecarts = [np.linalg.norm(moy_a[v] - moy_b[v]) for v in vals]
    print(f"\n  Contrôle indépendant — la valeur 0 désigne « sans changement », son")
    print(f"  écart A->B doit donc être PLUS FAIBLE que celui des transitions :")
    print(f"    valeur 0     : {ecart0:.1f}")
    print(f"    transitions  : {min(ecarts):.1f} à {max(ecarts):.1f} (médiane "
          f"{float(np.median(ecarts)):.1f})")
    if ecart0 < min(ecarts):
        print(f"    ✓ contrôle passé : 0 est bien la classe « sans changement ».")
    else:
        print(f"    ⚠️ contrôle NON passé : 0 n'est peut-être pas « sans changement »,")
        print(f"       ou l'échantillon est trop petit. À élucider avant de décoder.")
        ok = False

    if ok:
        print(f"\n  TABLE À REPORTER dans csf_mamba/datasets/landsat_scd.py :")
        print(f"    TRANSITIONS = {table}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
