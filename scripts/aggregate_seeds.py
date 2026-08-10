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
    return float(best[metric]), int(best["epoch"]), float(last[metric]), iou


def mean_sd(values):
    """Moyenne et écart-type d'échantillon (n-1). L'écart-type n'a pas de sens à n=1."""
    m = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else float("nan")
    return m, sd


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
        groups.setdefault(config_name(d), []).append(r)

    if not groups:
        raise SystemExit("aucun run exploitable")

    stats = {}
    for name, runs in groups.items():
        best_m, best_sd = mean_sd([r[0] for r in runs])
        last_m, last_sd = mean_sd([r[2] for r in runs])
        iou_m, _ = mean_sd([r[3] for r in runs])
        stats[name] = dict(n=len(runs), best=(best_m, best_sd), last=(last_m, last_sd),
                           iou=iou_m, epochs=[r[1] for r in runs])

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

    # L'écart-type intra-configuration le plus fiable : mis en commun sur tous les
    # groupes ayant au moins 2 graines. Un groupe seul en donne une estimation
    # très instable ; les mettre en commun suppose une variance comparable, ce qui
    # est raisonnable entre configurations proches.
    pooled = [s["best"][1] for s in stats.values() if not math.isnan(s["best"][1])]
    sigma = math.sqrt(statistics.fmean([s ** 2 for s in pooled])) if pooled else float("nan")

    # Comparer deux MOYENNES demande l'erreur-type de la différence,
    # SE = σ·√(1/n₁ + 1/n₂), et non σ seul. Diviser par σ surestime la certitude :
    # face au témoin à 4 graines, σ vaut 0,0089 mais SE vaut 0,0100 pour un run
    # unique et 0,0063 entre deux groupes de 4. On rapporte donc t = Δ/SE, et le
    # seuil est celui de Student au degré de liberté du σ mis en commun.
    df = sum(s_["n"] - 1 for s_ in stats.values() if s_["n"] > 1)
    t_crit = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45,
              7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}.get(df, 2.10 if df > 10 else float("inf"))

    print(f"\nMétrique : {args.metric}   |   témoin : {ref}")
    if not math.isnan(sigma):
        print(f"Écart-type mis en commun sur {len(pooled)} configuration(s) : "
              f"σ = {sigma:.4f}   (l'ancien plancher supposé était 0,0040)")
        print(f"Seuil de significativité à 95 % : |t| > {t_crit:.2f}  (df = {df})")
    print()
    print(f"| {'configuration':<38} | n | {'meilleur':^18} | {'final':^18} | IoU chgt | "
          f"{'écart au témoin':^28} |")
    print("|" + "-" * 40 + "|---|" + "-" * 20 + "|" + "-" * 20 + "|----------|"
          + "-" * 30 + "|")

    for name in sorted(stats, key=lambda k: -stats[k]["best"][0]):
        s = stats[name]
        delta = s["best"][0] - stats[ref]["best"][0]
        if name == ref:
            verdict = "— (témoin)"
        elif math.isnan(sigma) or sigma == 0:
            verdict = f"{delta:+.4f}  (σ inconnu)"
        else:
            se = sigma * math.sqrt(1 / s["n"] + 1 / stats[ref]["n"])
            t = delta / se
            label = "ÉTABLI" if abs(t) > t_crit else "non établi"
            verdict = f"{delta:+.4f}  t={t:+.2f}  {label}"
        print(f"| {name:<38} | {s['n']} | {fmt(*s['best']):^18} | "
              f"{fmt(*s['last']):^18} | {s['iou']:8.4f} | {verdict:<28} |")

    print("\nÉpoques des pics par configuration :")
    for name in sorted(stats):
        print(f"  {name:<40} {sorted(stats[name]['epochs'])}")

    singles = [n for n, s in stats.items() if s["n"] == 1]
    if singles:
        print("\n⚠️ une seule graine, donc aucun écart-type — l'écart au témoin "
              "n'y est pas interprétable :")
        for n in singles:
            print(f"    {n}")
    print("\n« non établi » ne veut pas dire « identique » : cela veut dire que ce")
    print("nombre de graines ne permet pas de les distinguer. Pour détecter un effet")
    print("de taille Δ entre deux configurations, il faut environ 8σ²/Δ² graines")
    print("de chaque côté :")
    if not math.isnan(sigma):
        for d in (0.005, 0.010, 0.015, 0.020):
            print(f"    Δ = {d:.3f}  ->  {math.ceil(8 * sigma ** 2 / d ** 2):3d} graines par configuration")


if __name__ == "__main__":
    main()
