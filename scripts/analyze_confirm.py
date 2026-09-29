"""Analyse de la confirmation sur test (étape 5, MODE=confirm) contre la baseline sans KD.

Baseline : les 7 graines de référence (`second_mini_chess_crop512-lean-augswap-ema-s<seed>`,
200 époques, LR constant) **tronquées aux 120 premières époques** — avec un LR constant,
un run de 120 époques est exactement le préfixe d'un run de 200 (même code à l'octet
près, vérifié le 27 sept.). Bras KD : dossiers `confirm_<tag>-s<seed>/`.

Convention du projet : max SeK test (sélection d'époque sur le test) **et** SeK de la
dernière époque, qui ne sélectionne rien. Pour chaque bras et chaque métrique :

- moyenne ± écart-type, écart à la baseline ;
- test de Welch contre les 7 graines, test apparié sur les graines communes ;
- IC 95 % de l'écart (Welch) ;
- fraction de l'écart professeur − élève comblée, (S_KD − S_base) / (S_prof − S_base),
  avec son IC (professeur fixe : checkpoint publié, une seule mesure).

    python scripts/analyze_confirm.py logs/confirm/runs --baseline-dir logs/baseline120 \\
        --out logs/confirm/analyse
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE_RE = re.compile(r"^second_mini_chess_crop512-lean-augswap-ema-s(?P<seed>\d+)$")
KD_RE = re.compile(r"^confirm_(?P<tag>.+)-s(?P<seed>\d+)$")
PT = 100.0
# SeK test du professeur publié sous NOTRE protocole (étape 0, job 4026049, fp32).
TEACHER_SEK = 26.1079


def summarise(path: Path, epochs: int) -> dict:
    m = pd.read_csv(path).drop_duplicates("epoch", keep="last").sort_values("epoch")
    m = m[m.epoch < epochs].reset_index(drop=True)
    i = int(m.sek.idxmax())
    return {"n_epochs": len(m), "complete": bool(len(m) == epochs),
            "max_sek": m.sek[i] * PT, "epoch_max": int(m.epoch[i]),
            "fscd_at_max": m.fscd[i] * PT, "miou_at_max": m.miou[i] * PT,
            "final_sek": m.sek.iloc[-1] * PT, "final_fscd": m.fscd.iloc[-1] * PT,
            "final_miou": m.miou.iloc[-1] * PT}


def collect(root: Path, regex, epochs: int, tag: str | None = None) -> list[dict]:
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        mt = regex.match(d.name)
        if not mt or not (d / "metrics.csv").exists():
            continue
        r = summarise(d / "metrics.csv", epochs)
        r.update(run=d.name, seed=int(mt["seed"]), tag=tag or mt["tag"])
        rows.append(r)
    return rows


def compare(kd: np.ndarray, base: np.ndarray, kd_by_seed: dict, base_by_seed: dict,
            teacher: float) -> dict:
    d = kd.mean() - base.mean()
    w = stats.ttest_ind(kd, base, equal_var=False)
    # IC de Welch de la différence.
    v1, v2 = kd.var(ddof=1) / len(kd), base.var(ddof=1) / len(base)
    df = (v1 + v2) ** 2 / (v1 ** 2 / (len(kd) - 1) + v2 ** 2 / (len(base) - 1))
    half = stats.t.ppf(0.975, df) * np.sqrt(v1 + v2)
    common = sorted(set(kd_by_seed) & set(base_by_seed))
    diffs = np.array([kd_by_seed[s] - base_by_seed[s] for s in common])
    paired_p = stats.ttest_rel([kd_by_seed[s] for s in common],
                               [base_by_seed[s] for s in common]).pvalue if len(common) > 1 else np.nan
    gap = teacher - base.mean()
    return {"kd_mean": kd.mean(), "kd_sd": kd.std(ddof=1), "n_kd": len(kd),
            "base_mean": base.mean(), "base_sd": base.std(ddof=1), "n_base": len(base),
            "delta": d, "ci95_lo": d - half, "ci95_hi": d + half, "welch_p": float(w.pvalue),
            "paired_seeds": ",".join(map(str, common)),
            "paired_delta_mean": diffs.mean() if len(diffs) else np.nan,
            "paired_delta_sd": diffs.std(ddof=1) if len(diffs) > 1 else np.nan,
            "paired_p": float(paired_p),
            "gap_teacher_minus_base": gap,
            "gap_closed": d / gap, "gap_closed_lo": (d - half) / gap, "gap_closed_hi": (d + half) / gap}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs_dir", type=Path, help="dossier des confirm_<tag>-s<seed>/")
    ap.add_argument("--baseline-dir", type=Path, required=True,
                    help="dossier des second_mini_chess_crop512-lean-augswap-ema-s<seed>/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--teacher-sek", type=float, default=TEACHER_SEK)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    base = collect(args.baseline_dir, BASE_RE, args.epochs, tag="baseline")
    kd = collect(args.runs_dir, KD_RE, args.epochs)
    if len(base) < 2 or not kd:
        raise SystemExit(f"baseline : {len(base)} runs, KD : {len(kd)} runs")
    runs = pd.DataFrame(base + kd).sort_values(["tag", "seed"])
    runs.to_csv(args.out / "runs.csv", index=False, float_format="%.4f")
    incomplete = runs.loc[~runs.complete, ["run", "n_epochs"]].to_dict("records")

    rows = []
    b = runs[runs.tag == "baseline"]
    for tag in sorted(set(runs.tag) - {"baseline"}):
        k = runs[runs.tag == tag]
        for metric in ("max_sek", "final_sek", "fscd_at_max", "miou_at_max"):
            teacher = args.teacher_sek if metric in ("max_sek", "final_sek") else np.nan
            r = compare(k[metric].to_numpy(), b[metric].to_numpy(),
                        dict(zip(k.seed, k[metric])), dict(zip(b.seed, b[metric])), teacher)
            r.update(tag=tag, metric=metric)
            rows.append(r)
    res = pd.DataFrame(rows)
    front = ["tag", "metric", "n_kd", "kd_mean", "kd_sd", "n_base", "base_mean", "base_sd",
             "delta", "ci95_lo", "ci95_hi", "welch_p", "paired_delta_mean", "paired_p",
             "gap_closed", "gap_closed_lo", "gap_closed_hi"]
    res = res[front + [c for c in res.columns if c not in front]]
    res.to_csv(args.out / "confirm.csv", index=False, float_format="%.6g")   # p-valeurs petites : pas d'arrondi à 4 décimales
    (args.out / "summary.json").write_text(json.dumps({
        "epochs": args.epochs, "teacher_sek": args.teacher_sek, "incomplete_runs": incomplete,
        "results": json.loads(res.to_json(orient="records"))}, indent=2, ensure_ascii=False))

    pd.set_option("display.width", 220)
    print(runs[["run", "tag", "seed", "max_sek", "epoch_max", "final_sek", "fscd_at_max",
                "miou_at_max"]].round(3).to_string(index=False))
    print()
    print(res[front].round(3).to_string(index=False))
    if incomplete:
        print("runs incomplets :", incomplete)


if __name__ == "__main__":
    main()
