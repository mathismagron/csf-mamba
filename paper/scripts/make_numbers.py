"""Write paper/numbers.tex: one LaTeX macro per number quoted in the text.

Every macro is computed from paper/tables/*.csv (built by build_tables.py), so the prose never
retypes a result. Run from the repository root after build_tables.py:

    python paper/scripts/make_numbers.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "paper" / "tables"
OUT = ROOT / "paper" / "numbers.tex"

M = {}


def put(name, value, fmt="{:.2f}"):
    assert name.isalpha(), name
    assert name not in M, name
    M[name] = fmt.format(value) if not isinstance(value, str) else value


def p_fmt(p):
    """Relation and value for math mode, used as $p\Macro$: '=0.13' or '<10^{-4}'."""
    if p < 1e-4:
        return r"<10^{-4}"
    if p < 0.001:
        return f"={p:.4f}"
    return f"={p:.3f}" if p < 0.01 else f"={p:.2f}"


def signed(x, d=2):
    """Signed number for running text: '+0.09' or '$-$0.04' (true minus sign)."""
    return f"+{x:.{d}f}" if x >= 0 else f"$-${abs(x):.{d}f}"


# ------------------------------------------------------------------ SECOND
a = pd.read_csv(T / "second_arms.csv")
A = {(r.arm, r.epochs): r for r in a.itertuples()}
b120, b200 = A[("no_kd", 120)], A[("no_kd", 200)]
put("SecBaseN", b120.n, "{:d}")
put("SecBaseSek", b120.sek_max_mean); put("SecBaseSekSd", b120.sek_max_sd)
put("SecBaseSekFinal", b120.sek_final_mean); put("SecBaseSekFinalSd", b120.sek_final_sd)
put("SecBaseFscd", b120.fscd_at_max_mean); put("SecBaseMiou", b120.miou_at_max_mean)
put("SecBaseSekFinalTwoHundred", b200.sek_final_mean)
names = {"kd_change_vitg": "Chg", "kd_full_vitg": "Full", "kd_full_vmambab": "Vmb"}
for arm, k in names.items():
    r = A[(arm, 120)]
    put(f"Sec{k}N", r.n, "{:d}")
    put(f"Sec{k}Sek", r.sek_max_mean); put(f"Sec{k}SekSd", r.sek_max_sd)
    put(f"Sec{k}SekFinal", r.sek_final_mean); put(f"Sec{k}SekFinalSd", r.sek_final_sd)
    put(f"Sec{k}Fscd", r.fscd_at_max_mean); put(f"Sec{k}Miou", r.miou_at_max_mean)
    put(f"Sec{k}Gain", r.sek_max_gain); put(f"Sec{k}GainLo", r.sek_max_gain_lo); put(f"Sec{k}GainHi", r.sek_max_gain_hi)
    put(f"Sec{k}GainP", p_fmt(r.sek_max_gain_welch_p))
    put(f"Sec{k}GainFinal", r.sek_final_gain); put(f"Sec{k}GainFinalLo", r.sek_final_gain_lo)
    put(f"Sec{k}GainFinalHi", r.sek_final_gain_hi)
    put(f"Sec{k}VsT", signed(r.sek_max_minus_teacher)); put(f"Sec{k}VsTLo", signed(r.sek_max_minus_teacher_lo))
    put(f"Sec{k}VsTHi", signed(r.sek_max_minus_teacher_hi)); put(f"Sec{k}VsTP", p_fmt(r.sek_max_minus_teacher_p))
    put(f"Sec{k}VsTFinal", signed(r.sek_final_minus_teacher)); put(f"Sec{k}VsTFinalP", p_fmt(r.sek_final_minus_teacher_p))
    put(f"Sec{k}GapClosed", 100 * r.sek_max_gap_closed, "{:.0f}")
e = A[("kd_full_vitg", 200)]
put("SecFullTwoHundredSek", e.sek_max_mean); put("SecFullTwoHundredSekSd", e.sek_max_sd)
put("SecFullTwoHundredSekFinal", e.sek_final_mean)
put("SecFullTwoHundredVsT", signed(e.sek_max_minus_teacher)); put("SecFullTwoHundredVsTLo", signed(e.sek_max_minus_teacher_lo))
put("SecFullTwoHundredVsTHi", signed(e.sek_max_minus_teacher_hi)); put("SecFullTwoHundredVsTP", p_fmt(e.sek_max_minus_teacher_p))
put("SecFullTwoHundredGain", e.sek_max_gain); put("SecFullTwoHundredGainFinal", e.sek_final_gain)

pw = pd.read_csv(T / "second_pairwise.csv").set_index(["comparison", "metric"])
for comp, k in [("semantic_term_increment", "SemInc"), ("vitg_minus_vmambab", "TeacherDiff"),
                ("vmambab_full_minus_vitg_change", "VmbVsChg"), ("kd_full_vitg_e200_minus_e120", "ExtInc")]:
    r = pw.loc[(comp, "sek_max")]
    put(f"{k}", r["mean"]); put(f"{k}Lo", r.lo); put(f"{k}Hi", r.hi); put(f"{k}P", p_fmt(r.p))
    put(f"{k}Pos", f"{int(r.n_positive)}/{int(r.n_pairs)}")
    put(f"{k}Min", r["min"]); put(f"{k}Max", r["max"])

put("TeacherVitgSek", 26.1079, "{:.2f}"); put("TeacherVitgSekExact", 26.1079, "{:.4f}")
put("TeacherVmbSek", 25.314, "{:.2f}"); put("TeacherVmbSekExact", 25.314, "{:.3f}")

eff = pd.read_csv(T / "efficiency.csv").set_index("quantity")
def effv(q, c): return float(eff.loc[q, c])
put("StudentParams", effv("Parameters (M)", "student")); put("TeacherParams", effv("Parameters (M)", "teacher"))
put("RatioParams", effv("Parameters (M)", "ratio"), "{:.1f}")
put("StudentGmacs", effv("GMACs / pair, aten counter", "student")); put("TeacherGmacs", effv("GMACs / pair, aten counter", "teacher"))
put("RatioGmacs", effv("GMACs / pair, aten counter", "ratio"), "{:.1f}")
put("StudentGmacsFvcore", effv("GMACs / pair, fvcore", "student"))
for q, k in [("latency b8 fp32 (ms)", "LatFpThirtyTwo"), ("latency b8 bf16 (ms)", "LatBf"),
             ("latency b8 fp16 (ms)", "LatFpSixteen"), ("latency b1 fp32 (ms)", "LatOneFpThirtyTwo"),
             ("latency b1 bf16 (ms)", "LatOneBf"), ("peak memory b8 fp32 (GB)", "Mem")]:
    put(f"Student{k}", effv(q, "student"), "{:.1f}" if "ms" in q else "{:.2f}")
    put(f"Teacher{k}", effv(q, "teacher"), "{:.1f}" if "ms" in q else "{:.2f}")
    put(f"Ratio{k}", effv(q, "ratio"), "{:.1f}")

# ------------------------------------------------------------------ screening (val)
s = pd.read_csv(T / "screening_arms.csv").set_index("tag")
for tag, k in [("witness", "Wit"), ("ctl-gt-l2", "CtlGt"), ("ctl-gtsmooth-l2", "CtlGts"), ("kdchg-l05", "LHalf"),
               ("kdchg-l1", "LOne"), ("kdchg-l2", "LTwo"), ("kdchg-l4", "LFour"), ("kdchg-l8", "LEight"),
               ("kdchg-l16", "LSixteen"), ("kdsem-l8-t2-all", "SemTwoAll"), ("kdsem-l8-t2-changed", "SemTwoChg"),
               ("kdsem-l8-t4-all", "SemFourAll"), ("kdsem-l8-t4-changed", "SemFourChg"),
               ("kdfeat-s12-l1", "FeatA"), ("kdfeat-s123-l1", "FeatB"), ("kdfeat-s12-l4", "FeatC")]:
    r = s.loc[tag]
    put(f"Scr{k}", r.max_sek_mean); put(f"Scr{k}Delta", signed(r.max_sek_delta))
inc = pd.read_csv(T / "screening_increments.csv")
for tag, vs, k in [("kdsem-l8-t2-all", "kdchg-l8", "SemTwoAll"), ("kdsem-l8-t2-changed", "kdchg-l8", "SemTwoChg"),
                   ("kdsem-l8-t4-all", "kdchg-l8", "SemFourAll"), ("kdsem-l8-t4-changed", "kdchg-l8", "SemFourChg"),
                   ("kdchg-l16", "kdchg-l8", "LSixteen"),
                   ("kdfeat-s12-l1", "kdsem-l8-t2-all", "FeatA"), ("kdfeat-s123-l1", "kdsem-l8-t2-all", "FeatB"),
                   ("kdfeat-s12-l4", "kdsem-l8-t2-all", "FeatC")]:
    r = inc[(inc.tag == tag) & (inc.vs == vs)]
    assert len(r) == 1, (tag, vs, len(r))
    put(f"Inc{k}", signed(float(r.max_sek_incr.iloc[0])))

# ------------------------------------------------------------------ Landsat
lk = pd.read_csv(T / "landsat_leakage_checks.csv").set_index("check").value
put("LkSlots", str(lk["image slots (2 images x 2385 pairs)"])); put("LkDistinct", str(lk["distinct images by pixel hash"]))
put("LkLocations", str(lk["tile indices k (locations)"])); put("LkTest", str(lk["test pairs with both images present in train"]))
put("LkVal", str(lk["val pairs with both images present in train"]))
put("LkPixels", f"{float(lk['annotated pixels compared across duplicates'])/1e6:.1f}")
put("LkOracleSek", float(lk["lookup oracle SeK on test (no learning)"])); put("LkOracleFscd", float(lk["lookup oracle Fscd on test"]))
put("LkOracleMiou", float(lk["lookup oracle mIoU on test"]))
put("LkKnown", float(lk["test changed pixels whose two classes are known from train (%)"]), "{:.1f}")
rs = pd.read_csv(T / "landsat_random_split.csv")
put("RndTwoFifty", rs.sek_test_mean[0]); put("RndSixHundred", rs.sek_test_mean[1]); put("RndSixHundredSd", rs.sek_test_sd[1])
put("PerascdLandsat", rs.sek_test_mean[3])
sl = pd.read_csv(T / "landsat_e600_slopes.csv")
put("RndLastSlope", sl.val_slope_pt_per_10ep.iloc[-1])

kf = pd.read_csv(T / "landsat_loc_kd_folds.csv").set_index("fold")
for f, w in enumerate(["Zero", "One", "Two", "Three"]):
    r = kf.loc[f]
    put(f"F{w}Teacher", r.teacher_test); put(f"F{w}Base", r.base_test); put(f"F{w}BaseSd", r.base_sd)
    put(f"F{w}Kd", r.kd_test); put(f"F{w}KdSd", r.kd_sd); put(f"F{w}Gain", r.gain_kd)
    put(f"F{w}KdMinusT", signed(r.kd_minus_teacher)); put(f"F{w}TMinusB", signed(r.teacher_minus_base))
    put(f"F{w}Locs", r.test_locs.replace(",", ", "))
st = pd.read_csv(T / "landsat_loc_kd_stats.csv").set_index(["analysis", "metric"])
for an, k in [("primary_f123", "Prim"), ("secondary_f0123", "Sec")]:
    r = st.loc[(an, "test_at_best_val")]
    put(f"L{k}Gain", r["mean"]); put(f"L{k}GainLo", r.ci_lo); put(f"L{k}GainHi", r.ci_hi)
    put(f"L{k}GainP", p_fmt(r.p_folds)); put(f"L{k}Pos", f"{int(r.n_pos)}/{int(r.n)}"); put(f"L{k}Wilcox", p_fmt(r.wilcox_p))
    put(f"L{k}Fscd", st.loc[(an, "fscd"), "mean"]); put(f"L{k}Miou", st.loc[(an, "miou"), "mean"])
    put(f"L{k}GainFinal", st.loc[(an, "test_final"), "mean"])
kmt = kf.kd_minus_teacher
put("LPrimKdMinusT", kmt.loc[[1, 2, 3]].mean()); put("LSecKdMinusT", kmt.mean())
from scipy import stats
put("LPrimKdMinusTP", p_fmt(stats.ttest_1samp(kmt.loc[[1, 2, 3]], 0).pvalue))
put("LSecKdMinusTP", p_fmt(stats.ttest_1samp(kmt, 0).pvalue))
put("LTeacherMinusBMean", kf.teacher_minus_base.mean())
put("LBaseMean", kf.base_test.mean()); put("LFoldSd", kf.base_test.std(ddof=1))
put("LKdSlopeMax", kf.kd_val_slope_70_99_max.max())
corr = pd.read_csv(T / "gain_vs_teacher_advantage_corr.csv").iloc[0]
put("CorrR", corr.value); put("CorrP", p_fmt(corr.p)); put("CorrN", int(corr.n), "{:d}")


# equivalence (TOST) of DistilSCD (ViT-G, 120 ep) with the single-run teacher, margin 0.25 SeK
_r = pd.read_csv(T / "second_runs.csv")
_k = _r[(_r.arm == "kd_full_vitg") & (_r.epochs == 120)]
_d = _k.sek_max - 26.1079
put("TostMargin", 0.25)
put("TostP", p_fmt(max(stats.ttest_1samp(_d, -0.25, alternative="greater").pvalue,
                       stats.ttest_1samp(_d, 0.25, alternative="less").pvalue)))
_lo, _hi = stats.t.interval(0.90, len(_d) - 1, _d.mean(), stats.sem(_d))
put("TostLo", signed(_lo)); put("TostHi", signed(_hi))
put("SecFullMaxFinalGap", float((_k.sek_max - _k.sek_final).abs().max()))
put("SecFullBestEpMin", int(_k.epoch_max.min()), "{:d}"); put("SecFullBestEpMax", int(_k.epoch_max.max()), "{:d}")
_b = _r[(_r.arm == "no_kd") & (_r.epochs == 120)]
put("SecBaseBestEpMin", int(_b.epoch_max.min()), "{:d}"); put("SecBaseBestEpMax", int(_b.epoch_max.max()), "{:d}")
# within-SECOND: gain difference between teachers vs teacher-advantage difference
put("TeacherAdvVitg", 26.1079 - A[("no_kd", 120)].sek_max_mean); put("TeacherAdvVmb", 25.314 - A[("no_kd", 120)].sek_max_mean)
# fold teachers: max over epochs (information only) vs DistilSCD
_t = pd.read_csv(T / "landsat_loc_teachers.csv").set_index("fold")
put("LKdMinusTMaxMin", float((kf.kd_test - _t.T_test_max).min()))

cc = pd.read_csv(T / "compute_cost.csv")
put("ComputeTotal", cc.a100_h.iloc[-1], "{:.0f}")
put("ComputeLandsatKd", float(cc[cc.study.str.startswith("Landsat folds: KD")].a100_h.iloc[0]), "{:.1f}")
put("ComputeLandsatTeachers", float(cc[cc.study.str.startswith("Landsat folds: PerASCD")].a100_h.iloc[0]), "{:.1f}")

# ------------------------------------------------------------------ revision controls (wave A)
rs = pd.read_csv(T / "revision_a_stats.csv")
ra = pd.read_csv(T / "revision_a_arms.csv")
rm = pd.read_csv(T / "revision_a_misc.csv").set_index("key").value
rc = pd.read_csv(T / "revision_a_caches.csv").set_index("cache")


def rstat(ds, comp, metric):
    r = rs[(rs.dataset.str.startswith(ds)) & (rs.comparison == comp) & (rs.metric == metric)]
    assert len(r) == 1, (ds, comp, metric, len(r))
    return r.iloc[0]


for key, ds, comp in [("SelfSec", "SECOND", "self_kd_minus_no_kd"), ("KdSelfSec", "SECOND", "kd_vitg_minus_self_kd"),
                      ("SelfTSec", "SECOND", "self_kd_minus_its_teacher"),
                      ("CosSec", "SECOND", "no_kd_cosine_minus_constant"), ("KdCosSec", "SECOND", "kd_vitg_minus_no_kd_cosine"),
                      ("SelfLan", "Landsat", "self_kd_minus_no_kd"), ("KdSelfLan", "Landsat", "kd_vitg_minus_self_kd"),
                      ("CtlLan", "Landsat", "gt_control_minus_no_kd"), ("KdCtlLan", "Landsat", "kd_vitg_minus_gt_control"),
                      ("SelfCtlLan", "Landsat", "self_kd_minus_gt_control"), ("SelfTLan", "Landsat", "self_kd_minus_its_teacher")]:
    m1, m2 = ("sek_max", "sek_final") if ds == "SECOND" else ("test_at_best_val", "test_final")
    for suffix, metric in (("", m1), ("Final", m2)):
        r = rstat(ds, comp, metric)
        put(f"Rev{key}{suffix}", signed(r["mean"]))
        put(f"Rev{key}{suffix}Abs", f"{abs(r['mean']):.2f}")   # unsigned, for "by X" in prose
        put(f"Rev{key}{suffix}Lo", signed(r.lo)); put(f"Rev{key}{suffix}Hi", signed(r.hi))
        put(f"Rev{key}{suffix}P", p_fmt(r.p))
        if not np.isnan(r.n_pairs) and not comp.endswith("its_teacher"):
            put(f"Rev{key}{suffix}Pos", f"{int(r.n_positive)}/{int(r.n_pairs)}")
_sec = ra[ra.dataset == "SECOND"].set_index("arm")
put("RevSelfSecSek", _sec.loc["self_kd"].sek_max_mean); put("RevSelfSecSekSd", _sec.loc["self_kd"].sek_max_sd)
put("RevSelfSecSekFinal", _sec.loc["self_kd"].sek_final_mean)
put("RevCosSek", _sec.loc["no_kd_cosine"].sek_max_mean); put("RevCosSekSd", _sec.loc["no_kd_cosine"].sek_max_sd)
put("RevCosSekFinal", _sec.loc["no_kd_cosine"].sek_final_mean)
put("RevSelfTeacherSec", float(rm["SECOND"]))
_g1 = rstat("SECOND", "self_kd_minus_no_kd", "sek_max")["mean"]; _g2 = rstat("SECOND", "kd_vitg_minus_self_kd", "sek_max")["mean"]
put("RevSelfShare", 100 * _g1 / (_g1 + _g2), "{:.0f}")
_eps = [float(rm[f"eps_fold{f}"]) for f in (1, 2, 3)]
put("RevEpsMin", min(_eps)); put("RevEpsMax", max(_eps))
_l = rc.loc[[f"student_landsat_f{f}_train" for f in (1, 2, 3)]]
put("RevCacheLanDropMin", float((_l.train_sek_full - _l.train_sek_cache).min()), "{:.1f}")
put("RevCacheLanDropMax", float((_l.train_sek_full - _l.train_sek_cache).max()), "{:.1f}")
_s = rc.loc["student_second_train"]
put("RevCacheSecDrop", float(_s.train_sek_full - _s.train_sek_cache))
put("RevComputeHours", float(rm["a100_h_runs_total"]), "{:.0f}")

lines = ["% Generated by paper/scripts/make_numbers.py from paper/tables/*.csv - do not edit by hand."]
lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()]
OUT.write_text("\n".join(lines) + "\n")
print(len(M), "macros ->", OUT)
