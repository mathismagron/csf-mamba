"""Build the canonical result tables of the paper from the raw run logs.

Every number quoted in the manuscript must come from a CSV written here
(paper/tables/). Run from the repository root:

    python paper/scripts/build_tables.py

Sources (all local, retrieved from Narval):
  logs/baseline_augswap_ema/   7 baseline seeds, 200 epochs (SECOND)
  logs/confirm/analyse/        KD confirmation on the SECOND test (ViT-G teacher)
  logs/confirm_vmb/analyse/    same recipe, VMamba-B teacher
  logs/e200_landsat/analyse/   200-epoch extension of the full-KD arm (SECOND)
                               and the 250-epoch Landsat random-split baseline
  logs/landsat_e600/analyse/   600-epoch Landsat random-split baseline, leakage list
  logs/screen_w4/analyse_all/  KD screening on the SECOND internal validation split
  logs/landsat_loc/            location-disjoint Landsat folds (baselines, teachers, KD)
  logs/final_tables/           efficiency and training-cost measurements
  logs/*.json                  teacher reproduction and efficiency jobs
Units: SeK, Fscd, mIoU, OA in percentage points.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs"
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

TEACHER_VITG_SEK = 26.1079   # published checkpoint, re-evaluated under our protocol (job 4026049)
TEACHER_VMB_SEK = 25.314     # published checkpoint, ImageNet normalisation (job 4265829)


def ci_mean(x, level=0.95):
    x = np.asarray(x, float)
    m, se = x.mean(), stats.sem(x)
    lo, hi = stats.t.interval(level, len(x) - 1, m, se)
    return m, lo, hi


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    lo, hi = d - stats.t.ppf(0.975, df) * se, d + stats.t.ppf(0.975, df) * se
    p = stats.ttest_ind(a, b, equal_var=False).pvalue
    return d, lo, hi, p


# ----------------------------------------------------------------- SECOND runs
def second_runs() -> pd.DataFrame:
    r1 = pd.read_csv(LOGS / "confirm/analyse/runs.csv")
    r2 = pd.read_csv(LOGS / "confirm_vmb/analyse/runs.csv")
    r = pd.concat([r1, r2]).drop_duplicates(subset=["run"]).reset_index(drop=True)
    r = r.rename(columns={"max_sek": "sek_max", "final_sek": "sek_final"})
    arm = {"baseline": "no_kd", "kdchg-l8": "kd_change_vitg", "kdlogits": "kd_full_vitg",
           "kdlogits-vmb": "kd_full_vmambab"}
    r["arm"] = r.tag.map(arm)
    assert r.arm.notna().all(), r.tag.unique()
    assert (r.n_epochs == 120).all() and r.complete.all()
    r["epochs"] = 120
    cols = ["arm", "seed", "epochs", "sek_max", "epoch_max", "fscd_at_max", "miou_at_max",
            "sek_final", "final_fscd", "final_miou", "run"]
    r = r[cols]
    # 200-epoch extension (post hoc, declared as such) and the 200-epoch baseline
    e = pd.read_csv(LOGS / "e200_landsat/analyse/kdlogits_e200_runs.csv")
    ext = pd.DataFrame({"arm": "kd_full_vitg", "seed": e.seed, "epochs": 200,
                        "sek_max": e.max200, "epoch_max": e.ep_max200, "sek_final": e.final200,
                        "run": [f"confirm_kdlogits-e200-s{s}" for s in e.seed]})
    base = []
    for d in sorted((LOGS / "baseline_augswap_ema").iterdir()):
        m = pd.read_csv(d / "metrics.csv")
        assert len(m) == 200, d
        i = int(m.sek.values.argmax())
        base.append(dict(arm="no_kd", seed=int(d.name.rsplit("-s", 1)[1]), epochs=200,
                         sek_max=m.sek[i] * 100, epoch_max=i, fscd_at_max=m.fscd[i] * 100,
                         miou_at_max=m.miou[i] * 100, sek_final=m.sek.iloc[-1] * 100,
                         final_fscd=m.fscd.iloc[-1] * 100, final_miou=m.miou.iloc[-1] * 100,
                         run=d.name))
    out = pd.concat([r, ext, pd.DataFrame(base)], ignore_index=True)
    return out


def second_arms(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, ep), g in runs.groupby(["arm", "epochs"], sort=False):
        b = runs[(runs.arm == "no_kd") & (runs.epochs == ep)]
        row = dict(arm=arm, epochs=ep, n=len(g))
        for m in ["sek_max", "sek_final", "fscd_at_max", "miou_at_max"]:
            if g[m].notna().all():
                row[f"{m}_mean"], row[f"{m}_sd"] = g[m].mean(), g[m].std(ddof=1)
        if arm != "no_kd":
            for m in ["sek_max", "sek_final"]:
                d, lo, hi, p = welch(g[m], b[m])
                row.update({f"{m}_gain": d, f"{m}_gain_lo": lo, f"{m}_gain_hi": hi, f"{m}_gain_welch_p": p})
                pb = b.set_index("seed")[m]; pk = g.set_index("seed")[m]
                common = sorted(set(pb.index) & set(pk.index))
                dd = pk.loc[common] - pb.loc[common]
                row[f"{m}_gain_paired_mean"] = dd.mean()
                row[f"{m}_gain_paired_p"] = stats.ttest_1samp(dd, 0).pvalue
            T = TEACHER_VMB_SEK if arm.endswith("vmambab") else TEACHER_VITG_SEK
            row["teacher_sek"] = T
            for m in ["sek_max", "sek_final"]:
                mm, lo, hi = ci_mean(g[m] - T)
                row.update({f"{m}_minus_teacher": mm, f"{m}_minus_teacher_lo": lo,
                            f"{m}_minus_teacher_hi": hi,
                            f"{m}_minus_teacher_p": stats.ttest_1samp(g[m] - T, 0).pvalue})
                gap = T - b[m].mean()
                row[f"{m}_gap_closed"] = (g[m].mean() - b[m].mean()) / gap
        rows.append(row)
    return pd.DataFrame(rows)


def second_pairwise(runs: pd.DataFrame) -> pd.DataFrame:
    """Paired-by-seed comparisons between KD arms at 120 epochs."""
    r = runs[runs.epochs == 120]
    piv = r.pivot_table(index="seed", columns="arm", values=["sek_max", "sek_final", "fscd_at_max", "miou_at_max"])
    rows = []
    for a, b, name in [("kd_full_vitg", "kd_change_vitg", "semantic_term_increment"),
                       ("kd_full_vitg", "kd_full_vmambab", "vitg_minus_vmambab"),
                       ("kd_full_vmambab", "kd_change_vitg", "vmambab_full_minus_vitg_change")]:
        for m in ["sek_max", "sek_final", "fscd_at_max", "miou_at_max"]:
            d = (piv[(m, a)] - piv[(m, b)]).dropna()
            mm, lo, hi = ci_mean(d)
            rows.append(dict(comparison=name, metric=m, n_pairs=len(d), mean=mm, lo=lo, hi=hi,
                             p=stats.ttest_1samp(d, 0).pvalue, n_positive=int((d > 0).sum()),
                             min=d.min(), max=d.max()))
    # 200-epoch extension vs the same seeds at 120 epochs
    k = runs[runs.arm == "kd_full_vitg"].pivot_table(index="seed", columns="epochs", values=["sek_max", "sek_final"])
    for m in ["sek_max", "sek_final"]:
        d = k[(m, 200)] - k[(m, 120)]
        mm, lo, hi = ci_mean(d)
        rows.append(dict(comparison="kd_full_vitg_e200_minus_e120", metric=m, n_pairs=len(d), mean=mm, lo=lo,
                         hi=hi, p=stats.ttest_1samp(d, 0).pvalue, n_positive=int((d > 0).sum()),
                         min=d.min(), max=d.max()))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- teachers
def teachers() -> pd.DataFrame:
    s0 = json.load(open(LOGS / "step_0_4026049.json"))
    vb = json.load(open(LOGS / "confirm_vmb/checks/step_0_vmambab_4265829.json"))
    rows = []
    for name, d, norm in [("PerASCD ViT-G/16 (PerA) + ViT-Adapter + CG-Decoder", s0, "PerA"),
                          ("VMamba-B + CG-Decoder", vb, "ImageNet")]:
        rows.append(dict(teacher=name, params_M=d["params_M"], normalisation=norm,
                         published=json.dumps(d.get("published")), results=json.dumps(d.get("results"))[:3000],
                         flops=json.dumps(d.get("flops"))[:800], job=d.get("step")))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- screening (SECOND val)
def screening():
    a = pd.read_csv(LOGS / "screen_w4/analyse_all/arms.csv")
    r = pd.read_csv(LOGS / "screen_w4/analyse_all/runs.csv")
    for c in ["max_sek", "final_sek", "last10_sek", "fscd_at_max", "miou_at_max", "max_fscd", "final_fscd"]:
        if c in r and r[c].max() <= 1.0:
            r[c] = r[c] * 100
    inc = pd.concat([pd.read_csv(LOGS / f"screen_w{w}/analyse_all/increments.csv").assign(wave=w) for w in (3, 4)])
    return a, r, inc


# ----------------------------------------------------------------- Landsat
def landsat():
    A = LOGS / "landsat_loc/analyse"
    out = {}
    for f in ["landsat_loc_kd_runs", "landsat_loc_kd_folds", "landsat_loc_kd_stats",
              "landsat_loc_teachers", "landsat_loc_base_folds"]:
        out[f] = pd.read_csv(A / f"{f}.csv")
    rnd = []
    e250 = pd.read_csv(LOGS / "e200_landsat/analyse/landsat_base_runs.csv")
    e600 = pd.read_csv(LOGS / "landsat_e600/analyse/landsat_base_e600_runs.csv")
    rnd.append(dict(setting="student, no KD, random pair split, 250 epochs", n=len(e250),
                    sek_test_mean=e250.test_at_best_val.mean(), sek_test_sd=e250.test_at_best_val.std(ddof=1)))
    rnd.append(dict(setting="student, no KD, random pair split, 600 epochs", n=len(e600),
                    sek_test_mean=e600.test_at_best_val.mean(), sek_test_sd=e600.test_at_best_val.std(ddof=1)))
    rnd.append(dict(setting="lookup oracle without learning (verify_landsat_leakage.py, job JSON)", n=1,
                    sek_test_mean=77.87, sek_test_sd=np.nan))
    rnd.append(dict(setting="PerASCD, published (arXiv 2602.13780, random split)", n=1,
                    sek_test_mean=65.21, sek_test_sd=np.nan))
    out["landsat_random_split"] = pd.DataFrame(rnd)
    out["landsat_e600_slopes"] = pd.read_csv(LOGS / "landsat_e600/analyse/landsat_base_e600_slopes.csv")
    lk = pd.read_csv(LOGS / "landsat_e600/analyse/landsat_split_leakage.csv")
    out["landsat_leakage_pairs"] = lk
    lk["both"] = lk.t1_image_in_train & lk.t2_image_in_train
    out["landsat_leakage_summary"] = lk.groupby("split").agg(
        n_pairs=("pair", "size"), both_images_in_train=("both", "sum"),
        n_locations=("location", "nunique")).reset_index()
    return out


# ----------------------------------------------------------------- gain vs teacher advantage
def gain_vs_advantage(arms, lf, lruns):
    rows = []
    base = arms[(arms.arm == "no_kd") & (arms.epochs == 120)].iloc[0]
    for arm, T, lab in [("kd_full_vitg", TEACHER_VITG_SEK, "SECOND, PerASCD ViT-G"),
                        ("kd_full_vmambab", TEACHER_VMB_SEK, "SECOND, VMamba-B")]:
        k = arms[(arms.arm == arm) & (arms.epochs == 120)].iloc[0]
        se = np.hypot(k.sek_max_sd / np.sqrt(k.n), base.sek_max_sd / np.sqrt(base.n))
        rows.append(dict(config=lab, dataset="SECOND", selection="max test SeK over epochs",
                         teacher_sek=T, student_sek=base.sek_max_mean, kd_sek=k.sek_max_mean,
                         teacher_advantage=T - base.sek_max_mean, kd_gain=k.sek_max_mean - base.sek_max_mean,
                         kd_gain_se=se, kd_minus_teacher=k.sek_max_mean - T, gate_pass=True))
    for _, f in lf.iterrows():
        d = lruns[(lruns.fold == f.fold)].pivot_table(index="seed", columns="arm", values="test_at_best_val")
        dd = d.kd - d.base
        rows.append(dict(config=f"Landsat-SCD fold {int(f.fold)}", dataset="Landsat-SCD",
                         selection="test SeK at best val epoch", teacher_sek=f.teacher_test,
                         student_sek=f.base_test, kd_sek=f.kd_test, teacher_advantage=f.teacher_minus_base,
                         kd_gain=dd.mean(), kd_gain_se=dd.std(ddof=1) / np.sqrt(len(dd)),
                         kd_minus_teacher=f.kd_minus_teacher, gate_pass=bool(f.gate_pass)))
    g = pd.DataFrame(rows)
    r, p = stats.pearsonr(g.teacher_advantage, g.kd_gain)
    return g, r, p


def _hours(paths, cols=("train_s", "val_s")):
    h = []
    for p in paths:
        t = pd.read_csv(p)
        h.append(sum(t[c].sum() for c in cols if c in t) / 3600)
    return h


def compute_cost() -> pd.DataFrame:
    """A100-hours per study. SECOND rows come from sacct (journal, 29-30 Sep); Landsat rows are
    summed from timing.csv (training + per-epoch evaluation; job start-up and staging excluded)."""
    L = LOGS
    rows = [dict(study="SECOND: KD screening, 3 logit waves (26 runs)", runs=26, a100_h=126.8, source="sacct"),
            dict(study="SECOND: feature-KD screening (6 runs)", runs=6, a100_h=66.4, source="sacct"),
            dict(study="SECOND: test confirmation, ViT-G teacher (10 runs)", runs=10, a100_h=66.0, source="sacct"),
            dict(study="SECOND: end-to-end trial", runs=1, a100_h=0.5, source="sacct"),
            dict(study="SECOND: VMamba-B teacher study (5 runs + teacher eval/cache)", runs=5, a100_h=32.6, source="sacct"),
            dict(study="SECOND: 200-epoch extension (5 resumes)", runs=5, a100_h=21.7, source="sacct")]
    e250 = _hours(sorted((L / "e200_landsat").glob("landsat_base-s?/timing.csv")))
    e600 = _hours(sorted((L / "landsat_e600").glob("landsat_base-e600-s?/timing.csv")))
    pilot = _hours([L / "landsat_loc/runs/landsat_f0_pilot-s1/timing.csv"])
    base = _hours(sorted((L / "landsat_loc").glob("landsat_f?_base-s?/timing.csv")))
    kd = _hours(sorted((L / "landsat_loc").glob("landsat_f?_kdlogits-s?/timing.csv")))
    tea = _hours(sorted((L / "landsat_loc/teachers").glob("landsat_vitg_f?/timing.csv")), ("train_s", "eval_s"))
    rows += [dict(study="Landsat random split: student baseline, 600 epochs (5 runs, incl. first 250)", runs=5,
                  a100_h=sum(e600), source="timing.csv"),
             dict(study="Landsat fold 0: 600-epoch budget pilot (stopped at epoch 477)", runs=1, a100_h=sum(pilot), source="timing.csv"),
             dict(study="Landsat folds: student baselines (12 runs)", runs=12, a100_h=sum(base), source="timing.csv"),
             dict(study="Landsat folds: PerASCD ViT-G teachers (4 runs)", runs=4, a100_h=sum(tea), source="timing.csv"),
             dict(study="Landsat folds: KD students (12 runs)", runs=12, a100_h=sum(kd), source="timing.csv")]
    df = pd.DataFrame(rows)
    assert len(e250) == 5 and len(e600) == 5 and len(base) == 12 and len(kd) == 12 and len(tea) == 4
    df.loc[len(df)] = dict(study="TOTAL (excluding teacher caches and checks, about 10 A100-h)", runs=df.runs.sum(),
                           a100_h=df.a100_h.sum(), source="sum")
    return df


def leakage_checks() -> pd.DataFrame:
    """Results of scripts/verify_landsat_leakage.py (Narval, 1 Oct 2026; JSON at
    $SCRATCH/csf-distill/checks/landsat_leakage.json, transcribed in documentation/distillation.md)."""
    return pd.DataFrame([
        dict(check="image slots (2 images x 2385 pairs)", value=4770),
        dict(check="distinct images by pixel hash", value=333),
        dict(check="tile indices k (locations)", value=12),
        dict(check="val pairs with both images present in train", value="477/477"),
        dict(check="test pairs with both images present in train", value="477/477"),
        dict(check="(k, year) groups with a single distinct image", value="333/333"),
        dict(check="annotated pixels compared across duplicates", value=198829642),
        dict(check="class agreement on those pixels (%)", value=100.0),
        dict(check="lookup oracle SeK on test (no learning)", value=77.87),
        dict(check="lookup oracle Fscd on test", value=95.04),
        dict(check="lookup oracle mIoU on test", value=94.25),
        dict(check="test changed pixels whose two classes are known from train (%)", value=90.8),
        dict(check="SECOND exact train/test duplicates (5936 train, 3388 test images)", value=0),
    ])


def published_second() -> pd.DataFrame:
    """Published SECOND results used for positioning (official 2968/1694 split only).
    Verified against the papers by the Literature track: paper/notes/published_numbers_check.csv.
    Published 'GFLOPs' of ChangeMamba / Mamba-FCS are fvcore-style MAC counts (labelled GFLOPs)."""
    return pd.DataFrame([
        dict(model="PerASCD (ViT-G/16 PerA + ViT-Adapter + CG-Decoder)", params_M=548.17, gmacs=1509.74,
             sek=26.11, fscd=66.41, miou=74.33,
             source="arXiv 2602.13780v3 Table I (SeK/Fscd/mIoU); params and GMACs measured by us (aten counter)",
             verification="paper values checked; reproduced by us to 26.1079 (job 4026049). Paper reports 2853.05 G 'FLOPs' (counter unstated) - not used"),
        dict(model="VMamba-B + CG-Decoder (PerASCD authors)", params_M=113.01, gmacs=352.9, sek=25.31, fscd=65.61,
             miou=74.01, source="released checkpoint (file name and stored metrics); params and aten GMACs measured by us",
             verification="reproduced by us to 25.314 with ImageNet normalisation (job 4265829)"),
        dict(model="Mamba-FCS", params_M=189.54, gmacs=263.15, sek=25.50, fscd=65.78, miou=74.07,
             source="Mamba-FCS arXiv 2508.08232v2 (JSTARS, doi 10.1109/JSTARS.2026.3663066): Table V (accuracy), Table VI (params, 'GFLOPs')",
             verification="checked against the paper (published_numbers_check.csv)"),
        dict(model="MambaSCD-Base (ChangeMamba)", params_M=89.99, gmacs=211.55, sek=22.92, fscd=55.32, miou=73.68,
             source="ChangeMamba arXiv 2404.03425v7 (= TGRS 2024): Table VII (accuracy; Fscd column labelled F1), Table VIII (params, 'GFLOPs')",
             verification="checked against the paper (published_numbers_check.csv)"),
        dict(model="MambaSCD-Tiny (ChangeMamba)", params_M=21.51, gmacs=73.42, sek=22.08, fscd=54.51, miou=73.33,
             source="ChangeMamba arXiv 2404.03425v7: Tables VII and VIII",
             verification="checked against the paper (published_numbers_check.csv)"),
        dict(model="MambaSCD-Tiny, released checkpoint, our metric code", params_M=37.13, gmacs=115.44, sek=23.34,
             fscd=None, miou=None, source="measured by us (README section 2)", verification="measured by us"),
        dict(model="SCanNet (retrained on the official split by PerASCD)", params_M=27.90, gmacs=264.95, sek=23.18,
             fscd=63.53, miou=73.03,
             source="PerASCD arXiv 2602.13780v3 Table I (accuracy); params/'GFLOPs' from ChangeMamba Table VIII",
             verification="checked; SCanNet's own 23.94 uses a 593-pair split of the training set and is NOT comparable"),
    ])


def teacher_precision_and_normalisation() -> pd.DataFrame:
    """Teacher re-evaluation on the full SECOND test set (1 694 pairs). ViT-G: job 4026049 JSON;
    VMamba-B: job 4265829 JSON for precisions (ImageNet normalisation) and job 4263826 for the
    normalisation comparison (values transcribed in documentation/distillation.md, 29 Sep)."""
    rows = []
    for name, f in [("PerASCD ViT-G", LOGS / "step_0_4026049.json"),
                    ("VMamba-B", LOGS / "confirm_vmb/checks/step_0_vmambab_4265829.json")]:
        d = json.load(open(f))
        for prec, r in d["results"].items():
            if isinstance(r, dict) and "ours_eval" in r:
                rows.append(dict(teacher=name, setting=f"precision {prec}", normalisation=(lambda n: n.get("name", "") if isinstance(n, dict) else n)(d.get("normalization", "PerA")),
                                 sek=100 * r["ours_eval"]["sek"], fscd=100 * r["ours_eval"]["fscd"],
                                 miou=100 * r["ours_eval"]["miou"],
                                 latency_b8_ms=r.get("latency", {}).get("median_ms"),
                                 source=f.name))
    for norm, sek in [("PerA (released evaluation code)", 23.732), ("ImageNet", 25.314),
                      ("dataset statistics (commented in released code)", 24.528), ("per-date 0-255", 24.469)]:
        rows.append(dict(teacher="VMamba-B", setting="normalisation, fp32", normalisation=norm, sek=sek,
                         source="job 4263826 (documentation/distillation.md, 29 Sep)"))
    return pd.DataFrame(rows)


def student_design_history() -> pd.DataFrame:
    """Pre-study design results of the student on SECOND (README sections 4-5, Aug-Sep 2026;
    fractions of SeK converted to points). Contribution = effect of keeping the component."""
    return pd.DataFrame([
        dict(component="C2S2 block (checkerboard + MCA-SF + S6 scan), removed", contribution=0.16, ci_lo=-0.04, ci_hi=0.36, verdict="not detectable"),
        dict(component="MCA-SF alone", contribution=0.15, ci_lo=-0.24, ci_hi=0.52, verdict="not detectable"),
        dict(component="change-guided attention (CGA), kept", contribution=-0.01, ci_lo=None, ci_hi=None, verdict="not detectable"),
        dict(component="FFT frequency branch, kept", contribution=-0.46, ci_lo=None, ci_hi=None, verdict="not established"),
        dict(component="wider decoder (+3.5 M, +12 GMACs)", contribution=-0.07, ci_lo=None, ci_hi=None, verdict="not detectable"),
        dict(component="removing the SeK loss term of Mamba-FCS", contribution=1.25, ci_lo=None, ci_hi=None, verdict="established"),
        dict(component="temporal-swap augmentation", contribution=0.96, ci_lo=None, ci_hi=None, verdict="established"),
        dict(component="EMA of weights", contribution=0.74, ci_lo=None, ci_hi=None, verdict="established (0.74-0.87)"),
        dict(component="DySample instead of bilinear upsampling", contribution=0.75, ci_lo=None, ci_hi=None, verdict="established"),
    ]).assign(source="README.md sections 4-5 (pre-study design campaign, ~150 runs, SECOND test, max and final epoch)")


def main():
    compute_cost().to_csv(OUT / "compute_cost.csv", index=False)
    teacher_precision_and_normalisation().to_csv(OUT / "teacher_precision_normalisation.csv", index=False)
    student_design_history().to_csv(OUT / "student_design_history.csv", index=False)
    pd.read_csv(LOGS / "landsat_loc/analyse/pilot_f0_windows.csv").to_csv(OUT / "landsat_pilot_windows.csv", index=False)
    leakage_checks().to_csv(OUT / "landsat_leakage_checks.csv", index=False)
    published_second().to_csv(OUT / "published_second.csv", index=False)
    runs = second_runs(); runs.to_csv(OUT / "second_runs.csv", index=False)
    arms = second_arms(runs); arms.to_csv(OUT / "second_arms.csv", index=False)
    second_pairwise(runs).to_csv(OUT / "second_pairwise.csv", index=False)
    teachers().to_csv(OUT / "teachers_reproduction.csv", index=False)
    a, r, inc = screening()
    a.to_csv(OUT / "screening_arms.csv", index=False); r.to_csv(OUT / "screening_runs.csv", index=False)
    inc.to_csv(OUT / "screening_increments.csv", index=False)
    for f in ["efficiency", "training_cost"]:
        pd.read_csv(LOGS / f"final_tables/{f}.csv").to_csv(OUT / f"{f}.csv", index=False)
    L = landsat()
    for k, v in L.items():
        v.to_csv(OUT / f"{k}.csv", index=False)
    g, r_, p_ = gain_vs_advantage(arms, L["landsat_loc_kd_folds"], L["landsat_loc_kd_runs"])
    g.to_csv(OUT / "gain_vs_teacher_advantage.csv", index=False)
    pd.DataFrame([dict(statistic="pearson_r_gain_vs_teacher_advantage", n=len(g), value=r_, p=p_)]) \
        .to_csv(OUT / "gain_vs_teacher_advantage_corr.csv", index=False)
    print(sorted(p.name for p in OUT.glob("*.csv")))


if __name__ == "__main__":
    main()
