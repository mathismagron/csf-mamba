"""Draft figure captions (paper/figures/captions.md), with every number read from paper/tables/*.csv
or from the logs used by make_figures.py. Run after make_figures.py:

    python paper/scripts/make_captions.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "paper" / "tables"
LOGS = ROOT / "logs"
M = "DistilSCD"


def tab(n):
    return pd.read_csv(T / f"{n}.csv")


def f2(x, sign=False):
    return (f"{x:+.2f}" if sign else f"{x:.2f}").replace("-", "\u2212")


def fp(p):
    return "< 0.001" if p < 1e-3 else (f"= {p:.3f}" if p < 0.01 else f"= {p:.2f}")


def main():
    eff = tab("efficiency").set_index("quantity")
    arms = tab("second_arms")
    a = arms[arms.epochs == 120].set_index("arm")
    pub = tab("published_second").set_index("model")
    sa = tab("screening_arms").set_index("tag")
    leak = tab("landsat_leakage_checks").set_index("check")["value"]
    rnd = tab("landsat_random_split")
    kf = tab("landsat_loc_kd_folds")
    st = tab("landsat_loc_kd_stats").set_index(["analysis", "metric"])
    cr = tab("gain_vs_teacher_advantage_corr").iloc[0]
    meta = json.loads((LOGS / "cache/meta.json").read_text())

    gs, gt = eff.loc["GMACs / pair, aten counter", ["student", "teacher"]]
    gr = eff.loc["GMACs / pair, aten counter", "ratio"]
    gfv = eff.loc["GMACs / pair, fvcore", "student"]
    ps, pt, pr = eff.loc["Parameters (M)", ["student", "teacher", "ratio"]]
    nk, kv, kb = a.loc["no_kd"], a.loc["kd_full_vitg"], a.loc["kd_full_vmambab"]
    kc = a.loc["kd_change_vitg"]
    r600 = rnd[rnd.setting.str.contains("600 epochs")].iloc[0]
    pubr = rnd[rnd.setting.str.startswith("PerASCD, published")].iloc[0]
    pilot = pd.read_csv(LOGS / "landsat_loc/runs/landsat_f0_pilot-s1/metrics.csv")
    p1, p2 = st.loc[("primary_f123", "test_at_best_val")], st.loc[("secondary_f0123", "test_at_best_val")]
    n_screen_runs = int(sa.n.sum())
    feat_ratio = [sa.loc[t, "train_h_total"] / sa.loc["kdsem-l8-t2-all", "train_h_total"]
                  for t in ("kdfeat-s12-l1", "kdfeat-s12-l4", "kdfeat-s123-l1")]
    n_tr = sum(1 for l in (ROOT / "splits/SECOND/train.txt").read_text().split() if l)
    n_va = sum(1 for l in (ROOT / "splits/SECOND/val.txt").read_text().split() if l)

    caps = {}
    caps["Fig. 1 — fig1_sek_vs_gmacs (single column)"] = (
        f"Accuracy versus inference cost on the SECOND test set. Knowledge distillation moves the "
        f"{ps:.1f} M-parameter student (grey) from {f2(nk.sek_max_mean)} to {f2(kv.sek_max_mean)} SeK with the "
        f"PerASCD ViT-G teacher ({M}, dark blue) and to {f2(kb.sek_max_mean)} with the VMamba-B teacher "
        f"(light blue), at unchanged cost; the ViT-G teacher ({pt:.0f} M parameters, {gt:.0f} GMACs) reaches "
        f"{f2(kv.teacher_sek)}. Students: mean of n = {int(nk.n)} (no KD) and n = {int(kv.n)} seeds (KD), "
        f"error bars = 95 % CI (smaller than the markers for the KD arms); SeK = maximum test SeK over 120 "
        f"epochs (field convention; final-epoch values in Table II). Teachers: published checkpoints "
        f"re-evaluated under our protocol, single run. GMACs per 512 × 512 pair. Our models and both "
        f"teachers are counted with the same counter (PyTorch aten flop counter), which gives the "
        f"{gr:.0f}× ratio ({gs:.2f} vs {gt:.2f} GMACs; {pr:.1f}× fewer parameters); it ignores the teacher's "
        f"deformable attention and the student's selective scan (fvcore gives {gfv:.2f} GMACs for the student), "
        f"so both counts are lower bounds. Open grey circles: published models on the official split, with "
        f"accuracies from their papers and complexities as published (`GFLOPs' that are fvcore-style MAC "
        f"counts, a different counter from aten, hence not strictly comparable); SCanNet as retrained on the "
        f"official split by the PerASCD authors (complexity from ChangeMamba).")
    caps["Fig. 2 — fig2_method (double column)"] = (
        f"{M}. Offline, once per dataset, the frozen teacher (PerASCD ViT-G) is run on the {len(meta['d4']['files'])} "
        f"views of the dihedral group D4 of every training pair; its logits are stored at the native resolution "
        f"of its decoder ({meta['layout']['shape'][2]} × {meta['layout']['shape'][3]}, {meta['precision']}) with "
        f"2C + 1 channels (C class logits per date, class 0 = unchanged, and one change logit; "
        f"{meta['layout']['shape'][1]} channels on SECOND). During training, the view g produced by the "
        f"student's flip/rotation augmentation is read from the cache (T1/T2 channels swapped with the "
        f"temporal swap) and upsampled bilinearly. The student is a siamese VMamba-mini encoder with "
        f"per-stage concatenation fusion, a change decoder and a shared semantic decoder with a temporal "
        f"embedding, both with DySample upsampling; the change maps guide the semantic decoder. Loss = task "
        f"cross-entropies + soft-target BCE on the change logit (λ_chg = 8) + temperature-scaled KL on the "
        f"semantic distributions renormalised over the real classes (T = 2, λ_sem = 1, all pixels). Only the "
        f"student is used at inference. Inherited FFT-injection and change-guided-attention details are "
        f"omitted (no measurable contribution; Supplement).")
    caps["Fig. 3 — fig3_second_evidence (double column)"] = (
        f"SECOND test results. (a) Per-seed test SeK, maximum over epochs (filled) and final epoch 120 (open), "
        f"for the student without KD (n = {int(nk.n)}), change-only KD from the ViT-G teacher (n = {int(kc.n)}), and "
        f"{M} with the ViT-G (n = {int(kv.n)}) and VMamba-B (n = {int(kb.n)}) teachers; black ticks = means; dashed "
        f"lines = teachers (single run). (b) Mean test SeK per epoch (band = ± 1 s.d. across seeds); the no-KD "
        f"curves are the first 120 epochs of 200-epoch runs with a constant learning rate (their maxima and "
        f"final values match the 120-epoch table exactly); dashed lines = teachers as in (a). (c) {M} minus "
        f"teacher, per seed (SeK max), with the mean and its 95 % CI (one-sample t-test): "
        f"{f2(kv.sek_max_minus_teacher, True)} [{f2(kv.sek_max_minus_teacher_lo)}, {f2(kv.sek_max_minus_teacher_hi)}], "
        f"p {fp(kv.sek_max_minus_teacher_p)} for ViT-G (matches the teacher) and "
        f"{f2(kb.sek_max_minus_teacher, True)} [{f2(kb.sek_max_minus_teacher_lo)}, {f2(kb.sek_max_minus_teacher_hi)}], "
        f"p {fp(kb.sek_max_minus_teacher_p)} for VMamba-B. Teachers are single published checkpoints, so "
        f"their run-to-run variance is not included.")
    caps["Fig. 4 — fig4_screening (double column)"] = (
        f"Pre-registered screening on the SECOND internal validation split ({n_tr} training / {n_va} "
        f"validation pairs; the test set was not used). {len(sa)} arms × 2 seeds = {n_screen_runs} runs of 120 "
        f"epochs; points = seeds, ticks = means; y = maximum validation SeK over epochs. (a) Weight of the "
        f"change-KD term (λ_chg = 0: no KD); the two controls at λ = 2 replace the teacher's soft change target "
        f"by the binary or the smoothed ground truth (same loss, weight and pipeline) and do not improve on "
        f"no KD. (b) Semantic KL on top of change KD (λ_chg = 8): temperature T and pixel mask (all pixels, or "
        f"pixels changed according to the ground truth or the teacher); dashed line = change KD only; the "
        f"retained configuration is T = 2, all pixels. (c) Feature KD added to the retained output KD: "
        f"encoder stages (fraction of the input resolution) and λ_feat; y = paired-by-seed SeK increment over "
        f"output KD, x = training time per run (A100-h; {min(feat_ratio):.1f}–{max(feat_ratio):.1f}× that of "
        f"output KD); peak training memory {sa.loc['kdsem-l8-t2-all', 'peak_gb']:.1f} vs "
        f"{sa.loc['kdfeat-s12-l1', 'peak_gb']:.1f} GB. Two seeds per arm: these are screening results, "
        f"confirmed on the test set with five seeds (Fig. 3).")
    caps["Fig. 5 — fig5_landsat_protocol (double column)"] = (
        f"Landsat-SCD: leakage of the random pair split and the location-disjoint protocol. (a) The "
        f"{leak['tile indices k (locations)']} location indices k of LandsatSCD512 drawn as a 4 × 3 grid (row = "
        f"first digit, column = second digit; this spatial layout is inferred from the indices, not from "
        f"georeferencing), coloured by the fold in which each location is a test location (F0–F3, three "
        f"test locations per fold); \"val Fi\" marks the single validation location of fold i. Fold 0 is "
        f"exploratory (Table VII). (b) Test (solid) and validation (dashed) SeK per epoch of the same student "
        f"without KD on the released random pair split (mean of n = {int(r600.n)} seeds; "
        f"{f2(r600.sek_test_mean)} ± {f2(r600.sek_test_sd)} at epoch 600, still rising) and on the "
        f"location-disjoint fold 0 (pilot run with a 600-epoch budget, 1 seed, stopped at epoch "
        f"{int(pilot.epoch.max())}). Dotted line: a lookup oracle that labels each test pair from the training "
        f"labels of the same two images, without learning ({float(leak['lookup oracle SeK on test (no learning)']):.2f} "
        f"SeK); in the random split, {leak['test pairs with both images present in train']} test pairs have both "
        f"images in the training set. Grey tick: PerASCD as published on the random split "
        f"({pubr.sek_test_mean:.2f}). The random split is the dataset authors' protocol for a single-area time "
        f"series; the location-disjoint folds measure generalisation to unseen locations instead, and the two "
        f"sets of numbers are not comparable.")
    gp = p1["mean"]
    caps["Fig. 6 — fig6_landsat_results (double column)"] = (
        f"Landsat-SCD, location-disjoint folds. (a) Test SeK at the epoch of best fold-validation SeK: "
        f"PerASCD ViT-G teacher retrained by us with the authors' recipe (one run per fold), student without KD "
        f"and {M} (3 seeds each; ticks = means). Fold 0 (shaded) failed the pre-registered teacher gate and was "
        f"run with the gate lifted (exploratory). KD gain on folds 1–3 (primary): +{gp:.2f} SeK (95 % CI "
        f"[{p1.ci_lo:.2f}, {p1.ci_hi:.2f}] across folds; {int(p1.n_pos)}/{int(p1.n)} seed pairs positive); "
        f"all four folds: +{p2['mean']:.2f} ({int(p2.n_pos)}/{int(p2.n)}). (b) KD gain versus the teacher's "
        f"advantage over the no-KD student, for the six teacher/dataset configurations (SECOND: max test SeK, "
        f"5 vs 7 seeds; Landsat folds: 3 seed pairs); error bars = ± 1 s.e. of the gain; open marker = fold 0. "
        f"Points above the dashed y = x line are KD students that outscore their teacher. Pearson "
        f"r = {cr.value:.2f} (n = {int(cr.n)}, p = {cr.p:.2f}).")
    caps["Fig. 7 — fig7_qualitative (double column; produced by plot_qualitative.py after the cluster job)"] = (
        f"Qualitative semantic change maps on SECOND test pairs (top) and Landsat-SCD location-disjoint fold-1 "
        f"test pairs (bottom): images T1 and T2, then the T1 and T2 change maps of the ground truth, the "
        f"student without KD, {M} and the teacher (PerASCD ViT-G; on Landsat, the fold-1 teacher retrained by "
        f"us). White = unchanged; black = changed pixel without a semantic label (ignored by the metric). "
        f"Numbers = per-pair SeK. Pairs were not chosen by eye: test pairs with changes were sorted by their "
        f"changed fraction and taken at evenly spaced quantiles (0.25–0.95). Students: seed 1, checkpoint at "
        f"the max test SeK (SECOND) or the best fold-validation SeK (Landsat).")

    out = ROOT / "paper" / "figures" / "captions.md"
    lines = ["# Draft figure captions", "",
             "Generated by `paper/scripts/make_captions.py` from `paper/tables/*.csv` and the logs used by "
             "`paper/scripts/make_figures.py`; edit the script, not this file. Figures: PNG (300 dpi) + PDF "
             "in `paper/figures/`, IEEE widths (single column 3.5 in, double column 7.16 in).", ""]
    for k, v in caps.items():
        lines += [f"## {k}", "", v, ""]
    out.write_text("\n".join(lines))
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
