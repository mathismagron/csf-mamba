"""Supplementary LaTeX tables, generated from paper/tables/*.csv (never retyped).

    python paper/scripts/make_supp_tables.py   ->  paper/tables_tex/supp_*.tex
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "paper" / "tables"
O = ROOT / "paper" / "tables_tex"
O.mkdir(exist_ok=True)


def f(x, d=2):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def table(name, caption, label, header, rows, colspec, star=False, note=None, size=r"\footnotesize"):
    env = "table*" if star else "table"
    out = [f"\\begin{{{env}}}[!t]", r"\centering", size, f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           f"\\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    out += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}"]
    if note:
        out.append(r"\par\smallskip\parbox{\linewidth}{\scriptsize " + note + "}")
    out.append(f"\\end{{{env}}}")
    (O / f"{name}.tex").write_text("\n".join(out) + "\n")


ARM = {"no_kd": "no KD", "kd_change_vitg": "change KD (ViT-G)", "kd_full_vitg": "DistilSCD (ViT-G)",
       "kd_full_vmambab": "DistilSCD (VMamba-B)"}

# S-I: SECOND per seed
r = pd.read_csv(T / "second_runs.csv").sort_values(["epochs", "arm", "seed"])
rows = [[ARM[x.arm], x.epochs, x.seed, f(x.sek_max), int(x.epoch_max), f(x.sek_final), f(x.fscd_at_max), f(x.miou_at_max)]
        for x in r.itertuples()]
table("supp_second_runs", "SECOND test results per seed.", "tab:s-second-runs",
      ["Arm", "Epochs", "Seed", "SeK max", "Best ep.", "SeK final", "Fscd", "mIoU"], rows, "lrrrrrrr",
      note="Fscd and mIoU at the best epoch. The 200-epoch DistilSCD rows resume the 120-epoch runs "
           "(post-hoc extension); the 200-epoch no-KD rows are the full runs whose first 120 epochs give the 120-epoch rows.")

# S-II: screening per run
s = pd.read_csv(T / "screening_runs.csv").sort_values(["tag", "seed"])
rows = [[x.tag.replace("_", r"\_"), x.seed, f(x.max_sek), int(x.epoch_max), f(x.final_sek), f(x.train_h_total, 1), f(x.peak_gb, 1)]
        for x in s.itertuples()]
table("supp_screening_runs", "Screening runs on the SECOND internal validation split (2\\,671 / 297 pairs).",
      "tab:s-screening-runs", ["Arm tag", "Seed", "SeK max", "Best ep.", "SeK final", "A100-h", "GB"], rows, "lrrrrrr",
      note="Tags: kdchg-l$\\lambda$ = change KD with weight $\\lambda$; ctl-gt / ctl-gtsmooth = ground-truth controls; "
           "kdsem-l8-t$T$-mask = semantic KD; kdfeat-s(stages)-l$\\lambda$ = feature KD; witness = no KD.")

# S-III: Landsat per run
k = pd.read_csv(T / "landsat_loc_kd_runs.csv").sort_values(["fold", "arm", "seed"])
rows = [[x.fold, "DistilSCD" if x.arm == "kd" else "no KD", x.seed, f(x.val_max), int(x.ep_best_val), f(x.test_at_best_val),
         f(x.fscd), f(x.miou), f(x.test_final), f(x.val_slope_70_99, 3)] for x in k.itertuples()]
table("supp_landsat_runs", "Landsat-SCD location-disjoint folds, all runs.", "tab:s-landsat-runs",
      ["Fold", "Arm", "Seed", "Val max", "Best ep.", "Test", "Fscd", "mIoU", "Test final", "Val slope"], rows,
      "rlrrrrrrrr", star=True,
      note="Test, Fscd, mIoU at the best-validation epoch; val slope in SeK points per 10 epochs over epochs 70--99. "
           "Fold 0 is exploratory (teacher gate lifted).")

# S-IV: fold teachers
t = pd.read_csv(T / "landsat_loc_teachers.csv")
rows = [[x.fold, x.test_locs.replace(",", ", "), f(x.T_val), int(x.T_ep), f(x.gate), "yes" if x.gate_pass else "no",
         f(x.T_test_bv), f(x.T_test_final), f(x.T_test_max), f(x.h, 1), f(x.peak, 1)] for x in t.itertuples()]
table("supp_teachers", "Retrained PerASCD ViT-G teachers, one per Landsat-SCD fold (50 epochs, authors' recipe).",
      "tab:s-teachers", ["Fold", "Test tiles", "Val", "Ep.", "Gate", "Pass", "Test", "Test final", "Test max", "A100-h", "GB"],
      rows, "rlrrrlrrrrr", star=True,
      note="Val/Test: SeK at the best-validation epoch; Gate: mean best validation SeK of the three no-KD students of the "
           "fold; Test max is shown for information only (never used for selection).")

# S-V: teacher reproduction
p = pd.read_csv(T / "teacher_precision_normalisation.csv")
rows = [[x.teacher, x.setting, x.normalisation, f(x.sek, 3), f(x.fscd, 3), f(x.miou, 3), f(x.latency_b8_ms, 1)]
        for x in p.itertuples()]
table("supp_teacher_repro", "Re-evaluation of the published SECOND teachers on the full test set (1\\,694 pairs).",
      "tab:s-teacher-repro", ["Teacher", "Setting", "Normalisation", "SeK", "Fscd", "mIoU", "ms (b8)"], rows,
      "lllrrrr", star=True,
      note="Published: ViT-G 26.11 (training log 26.1087); VMamba-B 25.31 (stored in the checkpoint: 0.25314). "
           "Latency at batch 8 on one A100 (the VMamba-B job ran on a different node).")

# S-VI: Landsat budget pilot
w = pd.read_csv(T / "landsat_pilot_windows.csv")
rows = [[x.win, f(x.val_mean), f(x.val_slope, 3), f(x.test_mean), f(x.test_slope, 3)] for x in w.itertuples()]
table("supp_pilot", "Budget pilot on Landsat-SCD fold 0 (student without KD, one seed).", "tab:s-pilot",
      ["Epochs", "Val mean", "Val slope", "Test mean", "Test slope"], rows, "lrrrr",
      note="Slopes in SeK points per 10 epochs. Rule fixed before the run, applied to validation only: first 50-epoch "
           "window with a validation slope below 0.05, i.e. epochs 50--99, hence a 100-epoch budget.")

# S-VII: compute ledger
c = pd.read_csv(T / "compute_cost.csv")
rows = [[x.study.replace("_", r"\_"), x.runs, f(x.a100_h, 1), x.source.replace("_", r"\_")] for x in c.itertuples()]
table("supp_compute", "Compute used in this study.", "tab:s-compute", ["Study", "Runs", "A100-h", "Source"], rows,
      "p{0.62\\linewidth}rrl", star=True)

# S-VIII: student design history
h = pd.read_csv(T / "student_design_history.csv")
rows = [[x.component, f(x.contribution), "--" if pd.isna(x.ci_lo) else f"[{x.ci_lo:.2f}, {x.ci_hi:.2f}]", x.verdict]
        for x in h.itertuples()]
table("supp_student_design", "Student design results obtained before this study (SECOND, SeK points).",
      "tab:s-student-design", ["Component", "Contribution", "95\\,\\% CI", "Verdict"], rows,
      "p{0.5\\linewidth}rrl", star=True,
      note="From the design campaign of the student (about 150 runs, 4--8 seeds per configuration, SECOND test with "
           "best-epoch and final-epoch readings). These results fixed the student recipe that is then frozen in this "
           "paper; they are reported for transparency and are not part of the distillation evidence.")
print(sorted(p.name for p in O.glob("supp_*.tex")))
