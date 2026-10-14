"""Revision wave A (documentation/revision_vague_a.md): paired analysis of the new arms.

Inputs (logs/ is git-ignored; outputs in paper/tables/ are the versioned record):
    logs/revision_a/runs/*            new runs (metrics.csv, metrics_test.csv, config.txt)
    logs/revision_a/cache/*/meta.json student-as-teacher caches
    paper/tables/second_runs.csv      existing SECOND arms (120 epochs)
    logs/landsat_loc/analyse/landsat_loc_kd_runs.csv   existing Landsat base / KD runs
    logs/landsat_loc/landsat_f?_*-s?  raw logs, used only to check that the selection rule
                                      below reproduces the existing table

Selection rules (identical to the paper):
    SECOND  : max test SeK over the 120 epochs, and test SeK at the final epoch;
    Landsat : test SeK at the epoch of best fold-validation SeK (first occurrence), and
              test SeK at the final epoch.
Statistics (fixed in advance in revision_vague_a.md §4):
    SECOND  : paired by seed, t interval over seeds;
    Landsat : fold-level means of the paired differences (n = 3 folds, t interval, df = 2),
              plus the sign count over fold x seed pairs (no test over those pairs).

    python paper/scripts/analyze_revision_a.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs"
RA = LOGS / "revision_a"
OUT = ROOT / "paper" / "tables"
FOLDS = (1, 2, 3)


def ci(x, level=0.95):
    x = np.asarray(x, float)
    m = x.mean()
    if len(x) < 2:
        return m, np.nan, np.nan, np.nan
    se = stats.sem(x)
    lo, hi = stats.t.interval(level, len(x) - 1, m, se)
    return m, lo, hi, stats.ttest_1samp(x, 0).pvalue


def second_run(d: Path, epochs=120):
    m = pd.read_csv(d / "metrics.csv")
    assert len(m) == epochs, (d, len(m))
    i = int(m.sek.values.argmax())
    return dict(sek_max=m.sek[i] * 100, epoch_max=i, fscd_at_max=m.fscd[i] * 100, miou_at_max=m.miou[i] * 100,
                sek_final=m.sek.iloc[-1] * 100, final_fscd=m.fscd.iloc[-1] * 100, final_miou=m.miou.iloc[-1] * 100)


def landsat_run(d: Path, epochs=100):
    v, t = pd.read_csv(d / "metrics.csv"), pd.read_csv(d / "metrics_test.csv")
    assert len(v) == len(t) == epochs, (d, len(v), len(t))
    i = int(v.sek.values.argmax())
    return dict(val_max=v.sek[i] * 100, ep_best_val=i, test_at_best_val=t.sek[i] * 100, fscd=t.fscd[i] * 100,
                miou=t.miou[i] * 100, test_final=t.sek.iloc[-1] * 100, test_max=t.sek.max() * 100)


def check_landsat_rule(existing: pd.DataFrame) -> float:
    """Max abs deviation between this rule on raw logs and the existing table (base/kd, folds 0-3)."""
    dev = []
    for _, r in existing.iterrows():
        tag = {"base": "base", "kd": "kdlogits"}[r.arm]
        d = LOGS / "landsat_loc" / f"landsat_f{int(r.fold)}_{tag}-s{int(r.seed)}"
        if (d / "metrics_test.csv").is_file():
            x = landsat_run(d)
            dev.append(abs(x["test_at_best_val"] - r.test_at_best_val))
            dev.append(abs(x["test_final"] - r.test_final))
    assert dev, "no raw Landsat logs found for the rule check"
    return max(dev)


def main():
    # ------------------------------------------------------------------ new runs
    rows = []
    for d in sorted((RA / "runs").iterdir()):
        name, seed = d.name.rsplit("-s", 1)
        conf = (d / "config.txt").read_text().strip()
        if name.startswith("confirm_"):
            arm = {"confirm_selfkd": "self_kd", "confirm_nokd-cos": "no_kd_cosine"}[name]
            rows.append(dict(dataset="SECOND", fold=np.nan, arm=arm, seed=int(seed), run=d.name, config=conf,
                             **second_run(d)))
        else:
            f, tag = name.split("_", 2)[1][1:], name.split("_", 2)[2]
            arm = {"selfkd": "self_kd", "ctl-gt": "gt_only_control"}[tag]
            eps = json.loads((d / "kd_control_eps.json").read_text())["eps"] if tag == "ctl-gt" else np.nan
            rows.append(dict(dataset="Landsat-SCD", fold=int(f), arm=arm, seed=int(seed), run=d.name, config=conf,
                             sem_eps=eps, **landsat_run(d)))
    new = pd.DataFrame(rows)

    # sanity: configurations are what the plan says
    for _, r in new.iterrows():
        c = r.config
        if r.arm == "self_kd":
            assert "teacher=student" in c and "kd_chg=8" in c and "mask=all" in c, c
        if r.arm == "gt_only_control":
            assert "tgt=gt_smooth" in c and "sem_tgt=gt_ls" in c and "teacher=perascd" in c, c
        if r.arm == "no_kd_cosine":
            assert "lr=cosine" in c and "kd_chg=0" in c and "kd_sem=0" in c, c

    # ------------------------------------------------------------------ existing arms
    sr = pd.read_csv(OUT / "second_runs.csv")
    sr = sr[sr.epochs == 120]
    lr = pd.read_csv(LOGS / "landsat_loc/analyse/landsat_loc_kd_runs.csv")
    rule_dev = check_landsat_rule(lr)
    assert rule_dev < 2e-3, f"Landsat selection rule does not reproduce the table: {rule_dev}"
    lr = lr.assign(arm=lr.arm.map({"base": "no_kd", "kd": "kd_full_vitg"}), dataset="Landsat-SCD")

    sec = pd.concat([sr[["arm", "seed", "sek_max", "sek_final", "fscd_at_max", "miou_at_max"]],
                     new[new.dataset == "SECOND"][["arm", "seed", "sek_max", "sek_final", "fscd_at_max", "miou_at_max"]]])
    lan = pd.concat([lr[["arm", "fold", "seed", "test_at_best_val", "test_final", "fscd", "miou"]],
                     new[new.dataset == "Landsat-SCD"][["arm", "fold", "seed", "test_at_best_val", "test_final",
                                                        "fscd", "miou"]]])
    lan = lan[lan.fold.isin(FOLDS)]

    # self-teacher scores (the student's own undistilled seed-1 run)
    self_teacher = {"SECOND": float(sr[(sr.arm == "no_kd") & (sr.seed == 1)].sek_max.iloc[0])}
    for f in FOLDS:
        self_teacher[f"fold{f}"] = float(lr[(lr.arm == "no_kd") & (lr.fold == f) & (lr.seed == 1)].test_at_best_val.iloc[0])

    # ------------------------------------------------------------------ paired comparisons
    st = []
    sp = sec.pivot_table(index="seed", columns="arm", values=["sek_max", "sek_final", "fscd_at_max", "miou_at_max"])
    for a, b, name in [("self_kd", "no_kd", "self_kd_minus_no_kd"),
                       ("kd_full_vitg", "self_kd", "kd_vitg_minus_self_kd"),
                       ("kd_full_vitg", "no_kd", "kd_vitg_minus_no_kd_same_seeds"),
                       ("no_kd_cosine", "no_kd", "no_kd_cosine_minus_constant"),
                       ("kd_full_vitg", "no_kd_cosine", "kd_vitg_minus_no_kd_cosine")]:
        for m in ["sek_max", "sek_final", "fscd_at_max", "miou_at_max"]:
            d = (sp[(m, a)] - sp[(m, b)]).dropna()
            mm, lo, hi, p = ci(d)
            st.append(dict(dataset="SECOND", comparison=name, metric=m, unit="seed", n=len(d),
                           seeds=",".join(str(int(s)) for s in d.index), mean=mm, lo=lo, hi=hi, p=p,
                           n_pairs=len(d), n_positive=int((d > 0).sum()), min=d.min(), max=d.max()))
    s_self = sec[sec.arm == "self_kd"]
    for m in ["sek_max", "sek_final"]:
        mm, lo, hi, p = ci(s_self[m] - self_teacher["SECOND"])
        st.append(dict(dataset="SECOND", comparison="self_kd_minus_its_teacher", metric=m, unit="seed",
                       n=len(s_self), mean=mm, lo=lo, hi=hi, p=p, n_pairs=len(s_self),
                       n_positive=int((s_self[m] > self_teacher["SECOND"]).sum())))

    lp = lan.pivot_table(index=["fold", "seed"], columns="arm", values=["test_at_best_val", "test_final", "fscd", "miou"])
    for a, b, name in [("self_kd", "no_kd", "self_kd_minus_no_kd"),
                       ("kd_full_vitg", "self_kd", "kd_vitg_minus_self_kd"),
                       ("gt_only_control", "no_kd", "gt_control_minus_no_kd"),
                       ("kd_full_vitg", "gt_only_control", "kd_vitg_minus_gt_control"),
                       ("self_kd", "gt_only_control", "self_kd_minus_gt_control")]:
        for m in ["test_at_best_val", "test_final", "fscd", "miou"]:
            d = (lp[(m, a)] - lp[(m, b)]).dropna()
            fold_means = d.groupby(level="fold").mean()
            mm, lo, hi, p = ci(fold_means)
            st.append(dict(dataset="Landsat-SCD folds 1-3", comparison=name, metric=m, unit="fold", n=len(fold_means),
                           seeds=";".join(f"f{int(f)}s{int(s)}" for f, s in d.index), mean=mm, lo=lo, hi=hi, p=p,
                           n_pairs=len(d), n_positive=int((d > 0).sum()), min=d.min(), max=d.max(),
                           fold_means=";".join(f"{v:.3f}" for v in fold_means.values)))
    for m in ["test_at_best_val", "test_final"]:
        dd = []
        for f in FOLDS:
            x = lan[(lan.arm == "self_kd") & (lan.fold == f)][m]
            dd.append((x - self_teacher[f"fold{f}"]).mean())
        mm, lo, hi, p = ci(dd)
        st.append(dict(dataset="Landsat-SCD folds 1-3", comparison="self_kd_minus_its_teacher", metric=m, unit="fold",
                       n=3, mean=mm, lo=lo, hi=hi, p=p, fold_means=";".join(f"{v:.3f}" for v in dd)))
    stats_df = pd.DataFrame(st)

    # ------------------------------------------------------------------ arm summaries
    arms = []
    for arm, g in sec.groupby("arm"):
        arms.append(dict(dataset="SECOND", fold="", arm=arm, n=len(g), seeds=",".join(map(str, sorted(g.seed))),
                         sek_max_mean=g.sek_max.mean(), sek_max_sd=g.sek_max.std(ddof=1),
                         sek_final_mean=g.sek_final.mean(), sek_final_sd=g.sek_final.std(ddof=1)))
    for (f, arm), g in lan.groupby(["fold", "arm"]):
        arms.append(dict(dataset="Landsat-SCD", fold=int(f), arm=arm, n=len(g), seeds=",".join(map(str, sorted(g.seed))),
                         test_at_best_val_mean=g.test_at_best_val.mean(), test_at_best_val_sd=g.test_at_best_val.std(ddof=1),
                         test_final_mean=g.test_final.mean(), test_final_sd=g.test_final.std(ddof=1)))
    arms_df = pd.DataFrame(arms)

    # ------------------------------------------------------------------ caches
    caches = []
    for d in sorted((RA / "cache").iterdir()):
        m = json.loads((d / "meta.json").read_text())
        c = m["checks"]
        ag = list(c["change_agreement_vs_identity"].values())
        caches.append(dict(cache=d.name, n_pairs=m["n_pairs"], checkpoint=m["checkpoint"]["path"],
                           train_sek_full=c["train_sek_full_res_identity"] * 100,
                           train_sek_cache=c["train_sek_cache_reupsampled"] * 100,
                           train_fscd_full=c["train_fscd_full_res_identity"] * 100,
                           train_fscd_cache=c["train_fscd_cache_reupsampled"] * 100,
                           d4_change_agreement_min=min(ag), d4_change_agreement_mean=float(np.mean(ag))))
    caches_df = pd.DataFrame(caches)
    eps = new[new.arm == "gt_only_control"].groupby("fold").sem_eps.agg(["min", "max"]).reset_index()
    assert (eps["min"] == eps["max"]).all(), eps

    # cost
    hours = []
    for d in sorted((RA / "runs").iterdir()):
        t = pd.read_csv(d / "timing.csv")
        hours.append(dict(run=d.name, a100_h=sum(t[c].sum() for c in ("train_s", "val_s") if c in t) / 3600))
    cost = pd.DataFrame(hours)

    new.drop(columns=["config"]).to_csv(OUT / "revision_a_runs.csv", index=False)
    stats_df.to_csv(OUT / "revision_a_stats.csv", index=False)
    arms_df.to_csv(OUT / "revision_a_arms.csv", index=False)
    caches_df.to_csv(OUT / "revision_a_caches.csv", index=False)
    pd.DataFrame([dict(key=k, value=v) for k, v in self_teacher.items()]
                 + [dict(key=f"eps_fold{int(r.fold)}", value=r["min"]) for _, r in eps.iterrows()]
                 + [dict(key="landsat_rule_max_abs_dev", value=rule_dev),
                    dict(key="a100_h_runs_total", value=cost.a100_h.sum())]).to_csv(OUT / "revision_a_misc.csv", index=False)
    cost.to_csv(OUT / "revision_a_cost.csv", index=False)
    print(f"rule check {rule_dev:.2e}; {len(new)} runs; {len(stats_df)} comparisons; {cost.a100_h.sum():.1f} A100-h")


if __name__ == "__main__":
    main()
