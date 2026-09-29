"""Analyse d'une vague de criblage KD (MODE=screen, sélection sur la val de 297 paires).

Lit les dossiers `screen_<tag>-s<seed>/` rapatriés de
`$SCRATCH/csf-distill/runs/` (metrics.csv, timing.csv, config.txt) et produit :

- `runs.csv`     une ligne par run : max SeK val (+ époque, Fscd/mIoU à cette
                 époque), SeK finale, moyenne des 10 dernières époques, durées ;
- `arms.csv`     une ligne par bras (tag) : moyennes ± écart-types sur les graines,
                 écart au témoin en points de SeK, test de Welch et test à variance
                 poolée (sur tous les bras), critère de réussite du plan (Δ ≥ +0,36 pt
                 sur le max) ;
- `curves.png`   SeK val par époque (moyenne des graines, graines en trait fin) et
                 écart au témoin par époque ;
- `summary.json` le tout, plus les contrôles (configs identiques hors seed / KD,
                 runs complets).

    python scripts/analyze_screen.py logs/screen_w1/runs --out logs/screen_w1/analyse
    python scripts/analyze_screen.py logs/screen_w{1,2,3}/runs --compare-to kdchg-l8 --out logs/screen_w3/analyse_all

Rien ici ne lit le test : c'est le criblage (documentation/distillation.md §3).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RUN_RE = re.compile(r"^screen_(?P<tag>.+)-s(?P<seed>\d+)$")
# Champs de config.txt qui ont le droit de différer d'un run à l'autre.
FREE_KEYS = {"seed", "kd_chg", "t_chg", "kd_sem", "t_sem", "mask", "tgt", "kd_feat", "feat_stages", "teacher"}
CRITERION_PT = 0.36          # plan §4 : Δ ≥ +0,36 pt contre le témoin, sur val
PT = 100.0                   # SeK en points


def parse_config(path: Path) -> dict[str, str]:
    """`k=v` séparés par des espaces ; `batch=2 x accum=4` -> accum gardé à part."""
    txt = path.read_text().replace(" x ", " ")
    out = {}
    for tok in txt.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def load_run(d: Path, epochs: int) -> tuple[dict, pd.DataFrame]:
    m = pd.read_csv(d / "metrics.csv")
    # Une reprise après une coupure entre l'écriture du CSV et celle de last.pt
    # rejoue une époque : on garde sa dernière occurrence.
    m = m.drop_duplicates("epoch", keep="last").sort_values("epoch").reset_index(drop=True)
    info = {"run": d.name, "n_epochs": int(len(m)),
            "complete": bool(len(m) == epochs and m.epoch.iloc[-1] == epochs - 1)}
    i = int(m.sek.idxmax())
    info.update(
        max_sek=m.sek[i], epoch_max=int(m.epoch[i]),
        fscd_at_max=m.fscd[i], miou_at_max=m.miou[i],
        max_fscd=m.fscd.max(),
        final_sek=m.sek.iloc[-1], final_fscd=m.fscd.iloc[-1],
        last10_sek=m.sek.iloc[-10:].mean(),
    )
    t = d / "timing.csv"
    if t.exists():
        tm = pd.read_csv(t).drop_duplicates("epoch", keep="last")
        tm = tm[tm.epoch > 0]          # époque 0 : compilation / préchauffage
        info.update(train_s_med=tm.train_s.median(), val_s_med=tm.val_s.median(),
                    peak_gb=tm.peak_train_gb.max(),
                    train_h_total=(tm.train_s.sum() + tm.val_s.sum()) / 3600)
    return info, m


def welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    r = stats.ttest_ind(a, b, equal_var=False)
    return float(r.statistic), float(r.pvalue)


def pooled_test(groups: dict[str, np.ndarray], a: str, b: str):
    """t sur la différence a − b avec la variance intra-bras poolée sur TOUS les bras
    (df = N − k) : avec 2 graines par bras, c'est la seule estimation du bruit
    un peu stable."""
    ss = sum(((g - g.mean()) ** 2).sum() for g in groups.values() if len(g) > 1)
    df = sum(len(g) - 1 for g in groups.values() if len(g) > 1)
    if df == 0:
        return np.nan, np.nan, np.nan, 0
    sp = np.sqrt(ss / df)
    ga, gb = groups[a], groups[b]
    se = sp * np.sqrt(1 / len(ga) + 1 / len(gb))
    t = (ga.mean() - gb.mean()) / se
    p = 2 * stats.t.sf(abs(t), df)
    return float(t), float(p), float(sp), int(df)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs_dir", type=Path, nargs="+",
                    help="un ou plusieurs dossiers contenant des screen_<tag>-s<seed>/ (vagues)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--witness", default="witness", help="tag du témoin sans KD")
    ap.add_argument("--compare-to", default=None,
                    help="tag d'un second repère (p. ex. kdchg-l8) : incréments appariés par graine "
                         "et test à variance poolée contre lui -> increments.csv")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    infos, curves, configs = [], {}, {}
    dirs = sorted((p for r in args.runs_dir for p in r.iterdir() if p.is_dir()), key=lambda p: p.name)
    dup = sorted({d.name for d in dirs if [e.name for e in dirs].count(d.name) > 1})
    if dup:
        raise SystemExit(f"runs présents dans plusieurs dossiers : {dup}")
    for d in dirs:
        mt = RUN_RE.match(d.name)
        if not mt or mt["tag"] == "smoke" or not (d / "metrics.csv").exists():
            continue
        info, m = load_run(d, args.epochs)
        info.update(tag=mt["tag"], seed=int(mt["seed"]))
        infos.append(info)
        curves[d.name] = m
        if (d / "config.txt").exists():
            configs[d.name] = parse_config(d / "config.txt")
    if not infos:
        raise SystemExit(f"aucun run screen_* dans {args.runs_dir}")
    runs = pd.DataFrame(infos).sort_values(["tag", "seed"])
    runs.to_csv(args.out / "runs.csv", index=False)

    # Contrôle : tout ce qui n'est pas seed / KD doit être identique.
    cfg_issues = []
    if configs:
        ref_name, ref = next(iter(configs.items()))
        for name, c in configs.items():
            for k in set(ref) | set(c):
                if k in FREE_KEYS:
                    continue
                if ref.get(k) != c.get(k):
                    cfg_issues.append(f"{name}: {k}={c.get(k)} (≠ {ref_name}: {ref.get(k)})")
    incomplete = runs.loc[~runs.complete, ["run", "n_epochs"]].to_dict("records")

    tags = list(runs.tag.unique())
    if args.witness not in tags:
        raise SystemExit(f"témoin '{args.witness}' absent (tags : {tags})")
    order = [args.witness] + sorted(t for t in tags if t != args.witness)

    metrics = ["max_sek", "final_sek", "last10_sek", "max_fscd"]
    arms = []
    for t in order:
        r = runs[runs.tag == t]
        row = {"tag": t, "n": len(r), "seeds": ",".join(map(str, r.seed))}
        for k in metrics:
            row[f"{k}_mean"] = r[k].mean() * PT
            row[f"{k}_sd"] = r[k].std(ddof=1) * PT if len(r) > 1 else np.nan
        # Max de la courbe moyenne (moins biaisé que la moyenne des max).
        mean_curve = pd.concat([curves[n].set_index("epoch").sek for n in r.run], axis=1).mean(1)
        row["meancurve_max"] = mean_curve.max() * PT
        row["meancurve_epoch"] = int(mean_curve.idxmax())
        for k in ["train_s_med", "val_s_med", "peak_gb", "train_h_total"]:
            if k in r:
                row[k] = r[k].mean()
        arms.append(row)
    arms = pd.DataFrame(arms)

    # Écarts au témoin, sur chaque métrique.
    for k in metrics:
        groups = {t: runs.loc[runs.tag == t, k].to_numpy() * PT for t in order}
        w = groups[args.witness]
        for i, t in enumerate(order):
            g = groups[t]
            arms.loc[i, f"{k}_delta"] = g.mean() - w.mean()
            if t == args.witness:
                continue
            arms.loc[i, f"{k}_welch_p"] = welch(g, w)[1]
            _, p, sp, df = pooled_test(groups, t, args.witness)
            arms.loc[i, f"{k}_pooled_p"] = p
            arms.loc[i, f"{k}_pooled_sd"] = sp
            arms.loc[i, f"{k}_pooled_df"] = df
    arms["criterion_met"] = [None if t == args.witness else bool(d >= CRITERION_PT)
                             for t, d in zip(arms.tag, arms.max_sek_delta)]
    arms.to_csv(args.out / "arms.csv", index=False, float_format="%.4f")

    # Incréments contre un second repère (étape 3 : + KD sémantique vs KD du changement seul).
    incr = None
    if args.compare_to:
        if args.compare_to not in tags:
            raise SystemExit(f"repère '{args.compare_to}' absent (tags : {tags})")
        rows = []
        ref = runs[runs.tag == args.compare_to].set_index("seed")
        for t in order:
            if t in (args.witness, args.compare_to):
                continue
            x = runs[runs.tag == t].set_index("seed")
            row = {"tag": t, "vs": args.compare_to}
            for k in ("max_sek", "final_sek", "last10_sek"):
                groups = {u: runs.loc[runs.tag == u, k].to_numpy() * PT for u in order}
                row[f"{k}_incr"] = groups[t].mean() - groups[args.compare_to].mean()
                _, p, _, _ = pooled_test(groups, t, args.compare_to)
                row[f"{k}_pooled_p"] = p
                common = sorted(set(x.index) & set(ref.index))
                row[f"{k}_per_seed"] = " ; ".join(
                    f"s{s}:{(x.loc[s, k] - ref.loc[s, k]) * PT:+.2f}" for s in common)
            rows.append(row)
        incr = pd.DataFrame(rows)
        incr.to_csv(args.out / "increments.csv", index=False, float_format="%.4f")

    # Figure de travail (pas une figure d'article).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    pal = plt.cm.tab10.colors if len(order) <= 10 else plt.cm.tab20.colors
    colors = {t: pal[i % len(pal)] for i, t in enumerate(order)}
    wc = pd.concat([curves[n].set_index("epoch").sek for n in runs[runs.tag == args.witness].run],
                   axis=1).mean(1) * PT
    for t in order:
        names = list(runs[runs.tag == t].run)
        cs = pd.concat([curves[n].set_index("epoch").sek for n in names], axis=1, keys=names) * PT
        for n in names:
            a1.plot(cs.index, cs[n], color=colors[t], lw=0.5, alpha=0.35)
        a1.plot(cs.index, cs.mean(1), color=colors[t], lw=1.6, label=f"{t} (n={len(names)})")
        if t != args.witness:
            d = (cs.mean(1) - wc).rolling(9, center=True, min_periods=1).mean()
            a2.plot(d.index, d, color=colors[t], lw=1.6, label=t)
    a1.set(xlabel="époque", ylabel="SeK val (pt)", title="Criblage : SeK sur la val (297 paires)")
    lo = np.nanpercentile(wc.to_numpy()[10:], 1) - 2 if len(wc) > 10 else None
    if lo is not None:
        a1.set_ylim(bottom=lo)
    a1.legend(fontsize=8)
    a2.axhline(0, color="k", lw=0.8)
    a2.axhline(CRITERION_PT, color="grey", ls="--", lw=0.8, label=f"critère +{CRITERION_PT} pt")
    a2.set(xlabel="époque", ylabel="Δ SeK val vs témoin (pt, moyenne glissante 9 ép.)",
           title="Écart au témoin par époque")
    a2.legend(fontsize=8)
    fig.savefig(args.out / "curves.png", dpi=150)

    summary = {
        "runs_dir": [str(r) for r in args.runs_dir], "compare_to": args.compare_to, "epochs": args.epochs, "witness": args.witness,
        "n_runs": len(runs), "incomplete_runs": incomplete, "config_issues": cfg_issues,
        "criterion_pt": CRITERION_PT,
        "arms": json.loads(arms.to_json(orient="records")),
        "increments": json.loads(incr.to_json(orient="records")) if incr is not None else None,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    pd.set_option("display.width", 200)
    cols = ["tag", "n", "max_sek_mean", "max_sek_sd", "max_sek_delta", "max_sek_welch_p",
            "max_sek_pooled_p", "final_sek_mean", "final_sek_delta", "final_sek_pooled_p",
            "meancurve_max", "meancurve_epoch", "criterion_met"]
    print(arms[[c for c in cols if c in arms]].round(3).to_string(index=False))
    if incr is not None:
        print(f"\nincréments contre {args.compare_to} :")
        print(incr.round(3).to_string(index=False))
    if incomplete:
        print("runs incomplets :", incomplete)
    if cfg_issues:
        print("⚠️ configs différentes :", *cfg_issues, sep="\n  ")


if __name__ == "__main__":
    main()
