"""Agrège les runs par configuration : moyenne, écart-type, comparaison au témoin.

Plusieurs graines par configuration sont la seule façon de savoir si un écart est
réel. Le plancher de bruit historique (±0,004 de SeK) venait d'un unique réplicat
accidentel — un couple de valeurs, sans écart-type. Or nos deux dernières
conclusions (décodeur élargi, backbone tiny en crops 512) reposaient sur des
écarts de 0,005, à peine au-dessus de ce seuil mal établi.

Les runs sont regroupés par nom de configuration : le suffixe de graine `-sN` ou
`_sN` est retiré, tout ce qui reste identifie la configuration. Ainsi
`second_mini_chess_crop512-s1`, `-s2`, `-s3` et `second_mini_chess_crop512`
forment un seul groupe de 4 graines.

    python -m scripts.aggregate_seeds $SCRATCH/csf-mamba-runs/second_mini_chess_*
    python -m scripts.aggregate_seeds --ref crop512 $SCRATCH/csf-mamba-runs/*

Deux SeK sont rapportés, et l'écart entre eux est instructif :

* **meilleur** — le maximum sur les 100 époques, ce que retient `best.pt`. C'est
  la pratique de la littérature, mais c'est un maximum sur une série bruitée :
  il est optimiste, et d'autant plus qu'une configuration est instable. Sur
  SECOND il est en outre sélectionné sur le split de test, faute de split de
  validation — une fuite légère, à mentionner dans le rapport.
* **final** — la dernière époque. Insensible au bruit de sélection, c'est la
  mesure honnête pour comparer deux schedules de learning rate : un LR constant
  produit des courbes plus agitées en fin d'entraînement, donc un maximum
  mécaniquement plus flatteur sans que le modèle soit meilleur.

Aucune dépendance lourde : stdlib seule, donc exécutable sur le nœud de connexion.
"""

import argparse
import csv
import math
import re
import statistics
from pathlib import Path

SEED_SUFFIX = re.compile(r"[-_]s\d+$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="Dossiers de runs (contenant metrics.csv).")
    p.add_argument("--ref", default=None,
                   help="Fragment identifiant la configuration témoin, pour la "
                        "colonne d'écart. Par défaut : le meilleur SeK moyen.")
    p.add_argument("--metric", default="sek", choices=["sek", "fscd", "miou", "oa", "kappa"])
    p.add_argument("--min-epochs", type=int, default=0,
                   help="Ignore les runs ayant écrit moins de N époques. Un run "
                        "encore en cours a un `metrics.csv` partiel : son maximum "
                        "est bas et tire tout son groupe vers le bas.")
    return p.parse_args()


def config_name(run_dir: Path) -> str:
    return SEED_SUFFIX.sub("", run_dir.name)


def read_run(run_dir: Path, metric: str):
    """-> (meilleure valeur, époque du pic, valeur finale, IoU du changement) ou None."""
    csv_path = run_dir / "metrics.csv"
    if not csv_path.is_file():
        return None
    rows = [r for r in csv.DictReader(csv_path.open()) if r.get(metric)]
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r[metric]))
    last = rows[-1]
    # IoU du changement reconstruit : SeK = kappa * exp(IoU) / e.
    sek, kappa = float(best["sek"]), float(best["kappa"])
    iou = 1 + math.log(sek / kappa) if sek > 0 and kappa > 0 else float("nan")
    return float(best[metric]), int(best["epoch"]), float(last[metric]), iou, len(rows)


# Seuils bilatéraux à 95 % de la loi de Student, par degré de liberté.
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36,
       8: 2.31, 9: 2.26, 10: 2.23, 12: 2.18, 15: 2.13, 20: 2.09, 30: 2.04}


def t_threshold(df: float) -> float:
    """Seuil à 95 % pour un df quelconque, arrondi conservativement."""
    if df < 1:
        return float("inf")
    keys = sorted(T95)
    for k in keys:
        if df <= k:
            return T95[k]
    return 1.96


def welch(a: dict, b: dict):
    """t de Welch entre deux groupes, et s'il franchit le seuil.

    Ne suppose PAS des variances égales : SE = √(s₁²/n₁ + s₂²/n₂), avec le degré
    de liberté de Welch-Satterthwaite. C'est le test conservateur, et le seul
    valide quand un groupe est nettement plus dispersé que l'autre.
    """
    (ma, sa), na = a["best"], a["n"]
    (mb, sb), nb = b["best"], b["n"]
    if na < 2 or nb < 2:
        return None, False
    va, vb = sa ** 2 / na, sb ** 2 / nb
    if va + vb == 0:
        return None, False
    t = (ma - mb) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    return t, abs(t) > t_threshold(df)


def mean_sd(values):
    """Moyenne et écart-type d'échantillon (n-1), en ignorant les NaN.

    L'IoU du changement vaut NaN quand SeK ou kappa est négatif — ce qui arrive
    sur un run encore en cours, dont le `metrics.csv` ne contient que les
    premières époques. `statistics.stdev` lève alors une AttributeError obscure
    (il passe par des Fraction, où NaN n'entre pas). On filtre en amont : un
    tableau partiel doit rester lisible, l'agrégation servant justement à suivre
    des lots en cours d'exécution.
    """
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    m = statistics.fmean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
    return m, sd


def iou_txt(v: float) -> str:
    return "—" if math.isnan(v) else f"{v:.4f}"


def fmt(m, sd):
    return f"{m:.4f} ± {sd:.4f}" if not math.isnan(sd) else f"{m:.4f}    —   "


def main():
    args = parse_args()
    groups: dict[str, list] = {}
    for path in args.runs:
        d = Path(path)
        if not d.is_dir():
            continue
        r = read_run(d, args.metric)
        if r is None:
            print(f"  (ignoré, pas de metrics.csv exploitable : {d.name})")
            continue
        if r[4] < args.min_epochs:
            print(f"  (ignoré, {r[4]} époques < --min-epochs {args.min_epochs} : {d.name})")
            continue
        groups.setdefault(config_name(d), []).append((r, d.name))

    if not groups:
        raise SystemExit("aucun run exploitable")

    stats = {}
    for name, entries in groups.items():
        runs = [r for r, _ in entries]
        best_m, best_sd = mean_sd([r[0] for r in runs])
        last_m, last_sd = mean_sd([r[2] for r in runs])
        iou_m, _ = mean_sd([r[3] for r in runs])
        stats[name] = dict(n=len(runs), best=(best_m, best_sd), last=(last_m, last_sd),
                           iou=iou_m, epochs=[r[1] for r in runs],
                           lengths=[(r[4], nom) for r, nom in entries])

    # Témoin : celui demandé, sinon le meilleur en moyenne.
    if args.ref:
        # Un nom exact l'emporte : « crop512 » ne doit pas être ambigu du seul
        # fait que « crop512-constlr » existe aussi.
        matches = [k for k in stats if k == args.ref] or [k for k in stats if args.ref in k]
        if len(matches) != 1:
            raise SystemExit(f"--ref '{args.ref}' correspond à {matches or 'rien'} ; "
                             "il faut qu'il désigne exactement une configuration")
        ref = matches[0]
    else:
        ref = max(stats, key=lambda k: stats[k]["best"][0])

    # Écart-type intra-configuration, mis en commun sur tous les groupes d'au
    # moins 2 graines. Un groupe seul en donne une estimation très instable ; les
    # mettre en commun suppose une variance comparable, raisonnable entre
    # configurations proches.
    #
    # On met en commun les VARIANCES, pondérées par les degrés de liberté :
    #
    #     σ² = Σ (nᵢ−1)·sᵢ²  /  Σ (nᵢ−1)
    #
    # Une moyenne non pondérée ne serait exacte que si tous les groupes avaient la
    # même taille — ce qui cesse d'être vrai dès qu'on mêle des groupes de 3 et de
    # 4 graines. Et ce sont bien les variances qui s'additionnent, jamais les
    # écarts-types.
    pooled = [(s_["best"][1], s_["n"]) for s_ in stats.values()
              if not math.isnan(s_["best"][1])]
    num = sum((n - 1) * sd ** 2 for sd, n in pooled)
    den = sum(n - 1 for _, n in pooled)
    sigma = math.sqrt(num / den) if den else float("nan")

    # Comparer deux MOYENNES demande l'erreur-type de la différence,
    # SE = σ·√(1/n₁ + 1/n₂), et non σ seul. Diviser par σ surestime la certitude :
    # face au témoin à 4 graines, σ vaut 0,0089 mais SE vaut 0,0100 pour un run
    # unique et 0,0063 entre deux groupes de 4. On rapporte donc t = Δ/SE, et le
    # seuil est celui de Student au degré de liberté du σ mis en commun.
    df = den
    t_crit = t_threshold(df)

    print(f"\nMétrique : {args.metric}   |   témoin : {ref}")
    if not math.isnan(sigma):
        print(f"Écart-type mis en commun sur {len(pooled)} configuration(s) : "
              f"σ = {sigma:.4f}   (l'ancien plancher supposé était 0,0040)")
        print(f"Seuil de significativité à 95 % : |t| > {t_crit:.2f}  (df = {df})")
    print()
    print(f"| {'configuration':<34} | n | {'meilleur':^18} | IoU chgt | {'Δ':^9} | "
          f"{'t comm.':^8} | {'t Welch':^8} | {'verdict':^12} |")
    print("|" + "-" * 36 + "|---|" + "-" * 20 + "|----------|" + "-" * 11 + "|"
          + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 14 + "|")

    for name in sorted(stats, key=lambda k: -stats[k]["best"][0]):
        s = stats[name]
        delta = s["best"][0] - stats[ref]["best"][0]
        if name == ref:
            print(f"| {name:<34} | {s['n']} | {fmt(*s['best']):^18} | {iou_txt(s['iou']):>8} | "
                  f"{'—':^9} | {'—':^8} | {'—':^8} | {'témoin':^12} |")
            continue
        se_p = sigma * math.sqrt(1 / s["n"] + 1 / stats[ref]["n"])
        t_p = delta / se_p
        ok_p = abs(t_p) > t_crit

        # Welch : chaque groupe apporte SA propre variabilité, sans supposer les
        # variances égales. Indispensable ici — le témoin a été jusqu'à cinq fois
        # plus dispersé que les autres groupes, et la mise en commun le diluait.
        # Non calculable à une seule graine, faute d'écart-type.
        t_w, ok_w = welch(s, stats[ref])
        if t_w is None:
            verdict, tw_txt = ("appuyé" if ok_p else "non établi"), "  n=1"
        else:
            tw_txt = f"{t_w:+.2f}"
            verdict = ("ÉTABLI" if ok_p and ok_w else
                       "partiel" if ok_p or ok_w else "non établi")
        print(f"| {name:<34} | {s['n']} | {fmt(*s['best']):^18} | {iou_txt(s['iou']):>8} | "
              f"{delta:+9.4f} | {t_p:^+8.2f} | {tw_txt:^8} | {verdict:^12} |")

    print("\nÉpoques des pics par configuration :")
    for name in sorted(stats):
        print(f"  {name:<40} {sorted(stats[name]['epochs'])}")

    singles = [n for n, s in stats.items() if s["n"] == 1]
    if singles:
        print("\n⚠️ une seule graine, donc aucun écart-type — l'écart au témoin "
              "n'y est pas interprétable :")
        for n in singles:
            print(f"    {n}")
    # Un run encore en cours a un metrics.csv tronqué : son maximum est bas et
    # tire la moyenne de son groupe vers le bas, ce qui ferait conclure à tort
    # qu'une ablation dégrade. On refuse de laisser passer ça en silence.
    suspects = []
    for name, s_ in stats.items():
        longueurs = [n for n, _ in s_["lengths"]]
        if longueurs and min(longueurs) < 0.8 * max(longueurs):
            suspects.append((name, s_["lengths"]))
    if suspects:
        print("\n⛔ GROUPES HÉTÉROGÈNES — des runs ont bien moins d'époques que")
        print("   leurs voisins : ils sont vraisemblablement encore en cours, et")
        print("   leur maximum partiel FAUSSE la moyenne du groupe.")
        for name, longueurs in suspects:
            print(f"   {name}")
            for n_ep, nom in sorted(longueurs):
                print(f"       {n_ep:>4} époques   {nom}")
        print("   -> relancer avec --min-epochs (p. ex. 95 pour des runs de 100).")

    print("\nVerdicts : ÉTABLI = les deux tests concordent. « partiel » = seule la")
    print("variance mise en commun conclut, pas Welch — il manque des graines, le")
    print("résultat est prometteur mais pas acquis. « appuyé » = groupe à une seule")
    print("graine, Welch incalculable, donc jamais concluant à lui seul.")
    print("\n« non établi » ne veut pas dire « identique » : cela veut dire que ce")
    print("nombre de graines ne permet pas de les distinguer. Pour détecter un effet")
    print("de taille Δ entre deux configurations, il faut environ 8σ²/Δ² graines")
    print("de chaque côté :")
    if not math.isnan(sigma):
        for d in (0.005, 0.010, 0.015, 0.020):
            print(f"    Δ = {d:.3f}  ->  {math.ceil(8 * sigma ** 2 / d ** 2):3d} graines par configuration")


if __name__ == "__main__":
    main()
