"""Publication figures of the DistilSCD paper (IEEE JSTARS), from paper/tables/*.csv and
the per-epoch logs in logs/. Every plotted number is read from those files.

    python paper/scripts/make_figures.py            # all figures
    python paper/scripts/make_figures.py fig1 fig3  # a subset

Output: paper/figures/<name>.png (300 dpi) and .pdf, at IEEE column widths.
Fig. 7 (qualitative maps) is produced by plot_qualitative.py from the .npz written on the
cluster by qualitative_predictions.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import C, COL1, COL2, FOLD, LOGS, METHOD, TABLES, panel_letter, save, setup  # noqa: E402

setup()


def tab(name):
    return pd.read_csv(TABLES / f"{name}.csv")


def ci95(x):
    x = np.asarray(x, float)
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def fmt_p(p):
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def arm(df, name, epochs=120):
    return df[(df.arm == name) & (df.epochs == epochs)]


# ============================================================================ Fig. 1
def fig1():
    eff = tab("efficiency").set_index("quantity")
    pub = tab("published_second").set_index("model")
    runs = tab("second_runs")
    arms = tab("second_arms")
    a120 = arms[arms.epochs == 120].set_index("arm")

    g_s = eff.loc["GMACs / pair, aten counter", "student"]
    g_t = eff.loc["GMACs / pair, aten counter", "teacher"]
    ratio = eff.loc["GMACs / pair, aten counter", "ratio"]
    t_vitg = a120.loc["kd_full_vitg", "teacher_sek"]
    t_vmb = a120.loc["kd_full_vmambab", "teacher_sek"]
    g_vmb = pub.loc["VMamba-B + CG-Decoder (PerASCD authors)", "gmacs"]

    fig, ax = plt.subplots(figsize=(COL1, 2.55), layout="constrained")
    ax.set_xscale("log")

    # published models: GMACs and SeK from their papers (other counters) -> open grey
    pubrows = [("MambaSCD-Tiny (ChangeMamba)", "MambaSCD-T", (0, -8), "center"),
               ("MambaSCD-Base (ChangeMamba)", "MambaSCD-B", (0, -8), "center"),
               ("Mamba-FCS", "Mamba-FCS", (-5, -3), "right"),
               ("SCanNet (retrained on the official split by PerASCD)", "SCanNet", (5, 4), "left")]
    assert set(k for k, *_ in pubrows) <= set(pub.index), "published_second.csv keys changed"
    for key, lab, off, ha in pubrows:
        x, y = pub.loc[key, "gmacs"], pub.loc[key, "sek"]
        ax.plot(x, y, "o", mfc="white", mec=C["published"], mew=1.0, ms=5, zorder=3)
        ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", ha=ha,
                    va="top" if off[1] < 0 else ("bottom" if off[1] > 0 and off[0] == 0 or key.startswith("SCanNet")
                                                 else ("bottom" if off[1] > 0 else "center")),
                    fontsize=7, color="#555555")

    # teachers (aten counter, measured by us; SeK re-evaluated)
    ax.plot(g_t, t_vitg, "s", color=C["teacher"], ms=6, zorder=4)
    ax.annotate("PerASCD ViT-G\nteacher, 548 M", (g_t, t_vitg), xytext=(0, -7),
                textcoords="offset points", ha="center", va="top", fontsize=7, color=C["teacher"])
    ax.plot(g_vmb, t_vmb, "s", color=C["teacher_vmb"], ms=6, zorder=4)
    ax.annotate("VMamba-B\nteacher, 113 M", (g_vmb, t_vmb), xytext=(0, -7), textcoords="offset points",
                ha="center", va="top", fontsize=7, color="#b07800")

    # student arms, mean +/- 95 % CI over seeds
    pts = {}
    for name, col in (("no_kd", C["student"]), ("kd_full_vmambab", C["kd_vmb"]),
                      ("kd_full_vitg", C["kd"])):
        v = arm(runs, name).sek_max
        m = a120.loc[name, "sek_max_mean"]
        assert abs(v.mean() - m) < 1e-9
        pts[name] = m
        ax.errorbar(g_s, m, yerr=ci95(v), fmt="o", color=col, ms=5.5 if name != "kd_full_vitg" else 6.5,
                    elinewidth=0.8, capsize=0, zorder=5, mec="white" if name == "kd_full_vitg" else col,
                    mew=0.6)
    # KD arrows from the no-KD point
    for name, col, rad in (("kd_full_vitg", C["kd"], -0.55), ("kd_full_vmambab", C["kd_vmb"], -0.35)):
        ax.add_patch(FancyArrowPatch((g_s, pts["no_kd"] + 0.12), (g_s, pts[name] - 0.10),
                                     connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                     mutation_scale=7, color=col, lw=1.0, zorder=4))
    ax.annotate("student, no KD (16.5 M)", (g_s, pts["no_kd"]), xytext=(7, 0), textcoords="offset points",
                ha="left", va="center", fontsize=7, color=C["student"])
    ax.annotate(f"{METHOD} (ViT-G)", (g_s, pts["kd_full_vitg"]), xytext=(7, 1),
                textcoords="offset points", ha="left", va="center", fontsize=7, color=C["kd"],
                fontweight="bold")
    ax.annotate(f"{METHOD} (VMamba-B)", (g_s, pts["kd_full_vmambab"]), xytext=(7, -1),
                textcoords="offset points", ha="left", va="center", fontsize=7, color="#2a7fb0")

    gain = a120.loc["kd_full_vitg", "sek_max_gain"]
    ax.annotate(f"+{gain:.1f}\nSeK", (g_s / 1.85, (pts["no_kd"] + pts["kd_full_vitg"]) / 2),
                ha="right", va="center", fontsize=7, color=C["kd"])
    # 51x bracket at the teacher level
    yb = t_vitg + 0.55
    ax.annotate("", xy=(g_s * 1.08, yb), xytext=(g_t / 1.08, yb),
                arrowprops=dict(arrowstyle="<->", color="#333333", lw=0.7, shrinkA=0, shrinkB=0))
    ax.text(np.sqrt(g_s * g_t), yb + 0.07, f"{ratio:.0f}\u00d7 fewer GMACs, same SeK", ha="center",
            va="bottom", fontsize=7, color="#333333")

    ax.set_xlim(11, 3200)
    ax.set_ylim(21.3, 27.4)
    ticks = [20, 50, 100, 200, 500, 1000, 2000]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels(["20", "50", "100", "200", "500", "1k", "2k"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("GMACs per image pair (log scale)")
    ax.set_ylabel("SeK (%), SECOND test")
    ax.text(0.99, 0.01, "higher is better \u2191", transform=ax.transAxes, fontsize=7,
            color="#555555", ha="right", va="bottom")
    return save(fig, "fig1_sek_vs_gmacs")

# ============================================================================ Fig. 3
SECOND_ARMS = [  # (arm, label, colour, curve dirs)
    ("no_kd", "no KD\n(n = 7)", C["student"]),
    ("kd_change_vitg", "change KD\nViT-G", C["kd_chg"]),
    ("kd_full_vitg", f"{METHOD}\nViT-G", C["kd"]),
    ("kd_full_vmambab", f"{METHOD}\nVMamba-B", C["kd_vmb"]),
]


def second_curves():
    """Per-epoch test SeK (%) of every SECOND run, first 120 epochs."""
    dirs = {
        "no_kd": sorted((LOGS / "baseline_augswap_ema").glob("second_mini_chess_crop512-lean-augswap-ema-s*")),
        "kd_change_vitg": sorted((LOGS / "confirm/runs").glob("confirm_kdchg-l8-s*")),
        "kd_full_vitg": sorted((LOGS / "confirm/runs").glob("confirm_kdlogits-s[0-9]*")),
        "kd_full_vmambab": sorted((LOGS / "confirm_vmb/runs").glob("confirm_kdlogits-vmb-s*")),
    }
    runs = tab("second_runs")
    out = {}
    for a, ds in dirs.items():
        mats = []
        for d in ds:
            m = pd.read_csv(d / "metrics.csv")
            assert len(m) >= 120, d
            v = m.sek.values[:120] * 100
            seed = int(d.name.rsplit("-s", 1)[1])
            row = arm(runs, a).set_index("seed").loc[seed]
            # the curve must reproduce the canonical table (max and final)
            assert abs(v.max() - row.sek_max) < 1e-3 and abs(v[-1] - row.sek_final) < 1e-3, d
            mats.append(v)
        out[a] = np.vstack(mats)
    return out


def fig3():
    runs = tab("second_runs")
    arms = tab("second_arms")
    a120 = arms[arms.epochs == 120].set_index("arm")
    t_vitg = a120.loc["kd_full_vitg", "teacher_sek"]
    t_vmb = a120.loc["kd_full_vmambab", "teacher_sek"]
    curves = second_curves()

    fig = plt.figure(figsize=(COL2, 2.45), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.32, 1.18, 0.70])
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))

    # (a) per-seed max and final test SeK
    rng = np.random.default_rng(0)
    for i, (a, lab, col) in enumerate(SECOND_ARMS):
        d = arm(runs, a)
        for j, (key, filled) in enumerate((("sek_max", True), ("sek_final", False))):
            x0 = i + (-0.17 if j == 0 else 0.17)
            v = d[key].values
            xj = x0 + rng.uniform(-0.06, 0.06, len(v))
            axa.plot(xj, v, "o", ms=3.6, mfc=col if filled else "white", mec=col, mew=0.9,
                     zorder=3, alpha=0.95)
            axa.plot([x0 - 0.11, x0 + 0.11], [v.mean()] * 2, "-", color="#222222", lw=1.0, zorder=4)
    for y, col, lab in ((t_vitg, C["teacher"], "ViT-G teacher"), (t_vmb, C["teacher_vmb"], "VMamba-B teacher")):
        axa.axhline(y, color=col, lw=0.9, ls=(0, (4, 2)), zorder=1)
    axa.text(-0.45, t_vitg - 0.05, "ViT-G teacher", color=C["teacher"], fontsize=7, va="top", ha="left")
    axa.text(-0.45, t_vmb - 0.05, "VMamba-B teacher", color="#b07800", fontsize=7, va="top", ha="left")
    axa.set_xticks(range(4))
    axa.set_xticklabels([lab for _, lab, _ in SECOND_ARMS])
    axa.set_xlim(-0.5, 3.5)
    axa.set_ylim(23.0, 26.75)
    axa.set_ylabel("SeK (%), SECOND test")
    # key for filled / open markers
    axa.plot([], [], "o", ms=3.6, color="#444444", label="max over epochs")
    axa.plot([], [], "o", ms=3.6, mfc="white", mec="#444444", label="final epoch (120)")
    axa.legend(loc="lower right", handletextpad=0.3, borderaxespad=0.2)
    axa.set_title("Per-seed test SeK, best and last epoch")
    panel_letter(axa, "a", dx=-0.17)

    # (b) mean test SeK vs epoch
    ep = np.arange(1, 121)
    ends = {}
    for a, lab, col in SECOND_ARMS:
        M = curves[a]
        m, sd = M.mean(0), M.std(0, ddof=1)
        axb.fill_between(ep, m - sd, m + sd, color=col, alpha=0.18, lw=0)
        axb.plot(ep, m, color=col, lw=1.4 if a == "kd_full_vitg" else 1.1, zorder=3)
        ends[a] = m[-1]
    axb.axhline(t_vitg, color=C["teacher"], lw=0.9, ls=(0, (4, 2)), zorder=1)
    axb.axhline(t_vmb, color=C["teacher_vmb"], lw=0.9, ls=(0, (4, 2)), zorder=1)
    names = {"no_kd": "no KD (n = 7)", "kd_change_vitg": "change KD", "kd_full_vitg": METHOD + " (ViT-G)",
             "kd_full_vmambab": METHOD + " (VMamba-B)"}
    ylab = {"no_kd": ends["no_kd"], "kd_change_vitg": ends["kd_change_vitg"] - 0.22,
            "kd_full_vitg": ends["kd_full_vitg"] + 0.12, "kd_full_vmambab": ends["kd_full_vmambab"] + 0.10}
    for a, lab, col in SECOND_ARMS:
        axb.text(122, ylab[a], names[a], color=col if a != "kd_full_vmambab" else "#2a7fb0",
                 fontsize=7, va="center", ha="left",
                 fontweight="bold" if a == "kd_full_vitg" else "normal")
    axb.set_xlim(0, 121)
    axb.set_ylim(20.0, 26.75)
    axb.set_xticks([0, 20, 40, 60, 80, 100, 120])
    axb.set_xlabel("Epoch")
    axb.set_ylabel("Mean SeK (%), SECOND test")
    axb.set_title("Mean test SeK per epoch")
    panel_letter(axb, "b", dx=-0.17)

    # (c) student minus teacher, per seed, with paired 95 % CI
    for i, (a, col, tname) in enumerate((("kd_full_vitg", C["kd"], "ViT-G"),
                                         ("kd_full_vmambab", C["kd_vmb"], "VMamba-B"))):
        r = a120.loc[a]
        dv = arm(runs, a).sek_max.values - r.teacher_sek
        assert abs(dv.mean() - r.sek_max_minus_teacher) < 1e-9
        axc.plot(i + rng.uniform(-0.08, 0.08, len(dv)), dv, "o", ms=3.6, color=col, alpha=0.9, zorder=3)
        axc.errorbar(i + 0.26, r.sek_max_minus_teacher,
                     yerr=[[r.sek_max_minus_teacher - r.sek_max_minus_teacher_lo],
                           [r.sek_max_minus_teacher_hi - r.sek_max_minus_teacher]],
                     fmt="D", ms=4, color="#222222", elinewidth=1.0, capsize=2, zorder=4)
        axc.text(i + 0.1, 0.70, f"p = {fmt_p(r.sek_max_minus_teacher_p)}", ha="center", va="bottom", fontsize=7)
    axc.axhline(0, color="#222222", lw=0.7)
    axc.set_xticks([0, 1])
    axc.set_xticklabels(["ViT-G", "VMamba-B"])
    axc.set_xlabel("Teacher")
    axc.errorbar([], [], yerr=[], fmt="D", ms=4, color="#222222", elinewidth=1.0, capsize=2, label="mean, 95 % CI")
    axc.legend(loc="lower right", handletextpad=0.2, borderaxespad=0.1)
    axc.set_xlim(-0.45, 1.6)
    axc.set_ylim(-0.3, 0.85)
    axc.set_ylabel(f"{METHOD} \u2212 teacher (SeK)")
    axc.set_title("Student minus\nsingle-run teacher")
    panel_letter(axc, "c", dx=-0.42)
    return save(fig, "fig3_second_evidence")

# ============================================================================ Fig. 4
def fig4():
    sr = tab("screening_runs")
    sa = tab("screening_arms").set_index("tag")
    si = tab("screening_increments")
    val = "SeK (%), internal val (max)"

    def seeds(tag, col="max_sek"):
        return sr[sr.tag == tag].sort_values("seed")[col].values

    fig = plt.figure(figsize=(COL2, 2.35), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0])
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))

    # (a) lambda_chg sweep + ground-truth controls at lambda = 2
    lam = [("witness", "0"), ("kdchg-l05", "0.5"), ("kdchg-l1", "1"), ("kdchg-l2", "2"),
           ("kdchg-l4", "4"), ("kdchg-l8", "8"), ("kdchg-l16", "16")]
    xs = np.arange(len(lam))
    means = []
    for x, (tag, _) in zip(xs, lam):
        v = seeds(tag)
        assert abs(v.mean() - sa.loc[tag, "max_sek_mean"]) < 1e-3
        col = C["student"] if tag == "witness" else C["kd_chg"]
        axa.plot([x - 0.07, x + 0.07][:len(v)], v, "o", ms=3.4, color=col, alpha=0.6, mew=0, zorder=3)
        means.append(v.mean())
    axa.plot(xs[1:], means[1:], "-", color=C["kd_chg"], lw=1.2, zorder=2)
    axa.plot(xs[:1], means[:1], "_", color=C["student"], ms=9, mew=1.6, zorder=4)
    axa.plot(xs[1:], means[1:], "_", color=C["kd_chg"], ms=9, mew=1.6, zorder=4)
    for dx, tag, mk, lab in ((0.22, "ctl-gt-l2", "^", "control: binary GT target"),
                             (0.36, "ctl-gtsmooth-l2", "v", "control: smoothed GT target")):
        v = seeds(tag)
        axa.plot([3 + dx] * len(v), v, mk, ms=4, color=C["control"], mfc="white", mew=0.9, zorder=3,
                 label=lab)
    axa.set_xticks(xs)
    axa.set_xticklabels([l for _, l in lam])
    axa.set_xlabel("Change-KD weight $\\lambda_{chg}$ (0 = no KD)")
    axa.set_ylabel(val)
    axa.set_ylim(23.1, 26.6)
    axa.set_xlim(-0.4, len(lam) - 0.6)
    axa.legend(loc="upper left", handletextpad=0.2, borderaxespad=0.1)
    axa.set_title("Change-KD weight and ground-truth controls")
    panel_letter(axa, "a", dx=-0.13)

    # (b) semantic term: temperature x mask, on top of change KD (lambda_chg = 8)
    ref = sa.loc["kdchg-l8", "max_sek_mean"]
    axb.axhline(ref, color=C["kd_chg"], lw=0.9, ls=(0, (4, 2)))
    axb.text(3.45, ref + 0.05, "change KD only ($\\lambda_{chg}$ = 8)", color=C["kd_chg"], fontsize=7,
             ha="right", va="bottom")
    sem = [("kdsem-l8-t2-all", "T = 2\nall"), ("kdsem-l8-t2-changed", "T = 2\nchanged"),
           ("kdsem-l8-t4-all", "T = 4\nall"), ("kdsem-l8-t4-changed", "T = 4\nchanged")]
    for x, (tag, _) in enumerate(sem):
        v = seeds(tag)
        sel = tag == "kdsem-l8-t2-all"
        col = C["kd"] if sel else C["kd_vmb"]
        axb.plot([x - 0.07, x + 0.07], v, "o", ms=3.4, color=col, alpha=0.6, mew=0, zorder=3)
        axb.plot(x, v.mean(), "_", color=col, ms=10, mew=1.8, zorder=4)
    axb.annotate("retained", (0, sa.loc["kdsem-l8-t2-all", "max_sek_mean"]), xytext=(0, 9),
                 textcoords="offset points", ha="center", va="bottom", fontsize=7, color=C["kd"],
                 fontweight="bold")
    axb.set_xticks(range(len(sem)))
    axb.set_xticklabels([l for _, l in sem])
    axb.set_xlabel("Semantic KL: temperature, pixel mask")
    axb.set_ylabel(val)
    axb.set_xlim(-0.5, 3.5)
    axb.set_ylim(25.3, 27.2)
    axb.set_title("Semantic KL settings")
    panel_letter(axb, "b", dx=-0.2)

    # (c) feature KD on top of output KD: increment vs training time
    base = "kdsem-l8-t2-all"
    vb = seeds(base)
    feats = [("kdfeat-s12-l1", "1/8, 1/16; $\\lambda$ = 1"), ("kdfeat-s12-l4", "1/8, 1/16; $\\lambda$ = 4"),
             ("kdfeat-s123-l1", "1/8, 1/16, 1/32; $\\lambda$ = 1")]
    axc.axhline(0, color="#222222", lw=0.7)
    hb = sa.loc[base, "train_h_total"]
    axc.plot(hb, 0, "o", ms=5.5, color=C["kd"], zorder=4)
    axc.annotate(f"output KD\n{sa.loc[base, 'peak_gb']:.1f} GB peak", (hb, 0), xytext=(0, -8),
                 textcoords="offset points", ha="center", va="top", fontsize=7, color=C["kd"])
    offs = {"kdfeat-s12-l1": (-16, -7, "right", "top"), "kdfeat-s12-l4": (-7, 0, "right", "center"),
            "kdfeat-s123-l1": (-6, 5, "right", "bottom")}
    for tag, lab in feats:
        d = seeds(tag) - vb   # paired by seed
        row = si[(si.tag == tag) & (si.vs == base)].iloc[0]
        assert abs(d.mean() - row.max_sek_incr) < 1e-3
        h = sa.loc[tag, "train_h_total"]
        axc.plot([h] * len(d), d, "o", ms=3.4, color=C["feat"], alpha=0.55, mew=0, zorder=3)
        axc.plot(h, d.mean(), "D", ms=4.5, color=C["feat"], zorder=4)
        ox, oy, ha, va = offs[tag]
        axc.annotate(lab, (h, d.mean()), xytext=(ox, oy), textcoords="offset points", ha=ha,
                     va=va, fontsize=7, color="#6b6b1f")
    axc.text(0.98, 0.03, f"feature KD: {sa.loc['kdfeat-s12-l1', 'peak_gb']:.1f} GB peak",
             transform=axc.transAxes, ha="right", va="bottom", fontsize=7, color="#6b6b1f")
    axc.set_xlim(3.0, 12.2)
    axc.set_ylim(-0.75, 0.55)
    axc.set_yticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4])
    axc.set_xlabel("Training time per run (A100-h)")
    axc.set_ylabel("SeK increment over output KD")
    r = [sa.loc[t, "train_h_total"] / hb for t, _ in feats]
    axc.set_title("Feature KD on top of output KD")
    panel_letter(axc, "c", dx=-0.28)
    return save(fig, "fig4_screening")

# ============================================================================ Fig. 5
FOLD_TINT = ["#DDDDDD", "#F5D76E", "#E3A6CB", "#9FD3C7"]   # test-fold tints (Fig. 5a only)
C_RANDOM, C_LOC = "#882255", "#117733"                       # protocol colours (Fig. 5b)


def landsat_folds():
    """Test and val tiles of each location-disjoint fold, read from the released split files."""
    import json
    d = json.loads((LOGS.parent / "splits/LandsatSCD_loc/README.json").read_text())
    folds = d["folds"]
    for f, fd in enumerate(folds):   # the split files must agree with README.json
        for split in ("test", "val"):
            lines = (LOGS.parent / f"splits/LandsatSCD_loc/fold{f}/{split}.txt").read_text().split()
            assert sorted({l.rsplit("_", 1)[1][:2] for l in lines}) == sorted(fd[split])
    return folds, d["locations"]


def mean_curve(paths, col, n=None):
    M = [pd.read_csv(p)[col].values * 100 for p in paths]
    n = n or min(len(m) for m in M)
    return np.vstack([m[:n] for m in M])


def fig5():
    folds, locs = landsat_folds()
    kd_folds = tab("landsat_loc_kd_folds")
    for f, fd in enumerate(folds):
        assert kd_folds.loc[f, "test_locs"].split(",") == fd["test"]
    chk = tab("landsat_leakage_checks").set_index("check")["value"]
    oracle = float(chk["lookup oracle SeK on test (no learning)"])
    rnd = tab("landsat_random_split")
    pub = float(rnd[rnd.setting.str.startswith("PerASCD, published")].sek_test_mean.iloc[0])
    r600 = rnd[rnd.setting.str.contains("600 epochs")].iloc[0]

    fig = plt.figure(figsize=(COL2, 2.45), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[0.78, 1.6])
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # (a) 4 x 3 tile grid (row = first digit, column = second digit; layout inferred)
    test_of = {k: f for f, fd in enumerate(folds) for k in fd["test"]}
    val_of = {fd["val"][0]: f for f, fd in enumerate(folds)}
    for k in locs:
        r, c = int(k[0]), int(k[1])
        f = test_of[k]
        axa.add_patch(Rectangle((c, -r), 0.94, -0.94, facecolor=FOLD_TINT[f], edgecolor="white", lw=0))
        axa.text(c + 0.47, -r - 0.30, k, ha="center", va="center", fontsize=8, fontweight="bold")
        axa.text(c + 0.47, -r - 0.56, f"test F{f}", ha="center", va="center", fontsize=7)
        if k in val_of:
            axa.text(c + 0.47, -r - 0.78, f"val F{val_of[k]}", ha="center", va="center", fontsize=7,
                     color="#333333", style="italic")
    axa.set_xlim(-0.04, 2.98)
    axa.set_ylim(-3.98, 0.02)
    axa.set_aspect("equal")
    axa.axis("off")
    axa.set_title("Location-disjoint folds", loc="right")
    panel_letter(axa, "a", dx=-0.04, dy=1.0)

    # (b) same student, random pair split vs location-disjoint fold 0
    rdirs = sorted((LOGS / "landsat_e600").glob("landsat_base-e600-s*"))
    assert len(rdirs) == int(r600.n)
    rt = mean_curve([d / "metrics_test.csv" for d in rdirs], "sek")
    rv = mean_curve([d / "metrics.csv" for d in rdirs], "sek")
    assert abs(rt[:, -1].mean() - r600.sek_test_mean) < 1e-3
    pdir = LOGS / "landsat_loc/runs/landsat_f0_pilot-s1"
    pt = pd.read_csv(pdir / "metrics_test.csv").sek.values * 100
    pv = pd.read_csv(pdir / "metrics.csv").sek.values * 100
    ert, erv = np.arange(1, rt.shape[1] + 1), np.arange(1, rv.shape[1] + 1)
    axb.plot(ert, rt.mean(0), color=C_RANDOM, lw=1.3)
    axb.plot(erv, rv.mean(0), color=C_RANDOM, lw=0.9, ls=(0, (3, 1.5)))
    axb.plot(np.arange(1, len(pt) + 1), pt, color=C_LOC, lw=1.3)
    axb.plot(np.arange(1, len(pv) + 1), pv, color=C_LOC, lw=0.9, ls=(0, (3, 1.5)))
    axb.axhline(oracle, color=C["oracle"], lw=0.9, ls=":")
    axb.text(8, oracle + 1.0, f"lookup oracle, no learning ({oracle:.2f})", fontsize=7, va="bottom")
    axb.plot([615, 670], [pub, pub], color=C["published"], lw=2.0, solid_capstyle="butt")
    axb.text(642, pub - 1.5, "PerASCD,\npublished", fontsize=7, color="#666666", ha="center", va="top")
    axb.text(320, 51, f"random pair split (n = {len(rdirs)}): still rising at 600 epochs",
             color=C_RANDOM, fontsize=7, ha="center", va="top")
    axb.text(300, pt[299] + 3.0, "location-disjoint fold 0 (pilot, 1 seed): plateau", color=C_LOC,
             fontsize=7, ha="center", va="bottom")
    axb.plot([], [], color="#444444", lw=1.3, label="test")
    axb.plot([], [], color="#444444", lw=0.9, ls=(0, (3, 1.5)), label="validation")
    axb.legend(loc="center right", bbox_to_anchor=(1.0, 0.36), handlelength=2.0)
    axb.set_xlim(0, 685)
    axb.set_xticks([0, 100, 200, 300, 400, 500, 600])
    axb.set_ylim(0, 84)
    axb.set_xlabel("Epoch")
    axb.set_ylabel("SeK (%), student without KD")
    axb.set_title("Random pair split vs. location-disjoint fold")
    panel_letter(axb, "b", dx=-0.09)
    return save(fig, "fig5_landsat_protocol")

# ============================================================================ Fig. 6
def fig6():
    runs = tab("landsat_loc_kd_runs")
    kf = tab("landsat_loc_kd_folds").set_index("fold")
    te = tab("landsat_loc_teachers").set_index("fold")
    gv = tab("gain_vs_teacher_advantage")
    cr = tab("gain_vs_teacher_advantage_corr").iloc[0]
    st = tab("landsat_loc_kd_stats").set_index(["analysis", "metric"])

    fig = plt.figure(figsize=(COL2, 2.5), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # (a) per-fold test SeK at the best-validation epoch
    rng = np.random.default_rng(1)
    axa.axvspan(-0.5, 0.5, color="#f0f0f0", lw=0, zorder=0)
    for f in range(4):
        t = te.loc[f, "T_test_bv"]
        assert abs(kf.loc[f, "teacher_test"] - t) < 1e-9
        axa.plot([f - 0.36, f - 0.16], [t, t], color=C["teacher"], lw=2.0, solid_capstyle="butt", zorder=3)
        for dx, a, col, key in ((0.0, "base", C["student"], "base_test"), (0.26, "kd", C["kd"], "kd_test")):
            v = runs[(runs.arm == a) & (runs.fold == f)].sort_values("seed").test_at_best_val.values
            assert len(v) == 3 and abs(v.mean() - kf.loc[f, key]) < 2e-3
            axa.plot(f + dx + rng.uniform(-0.05, 0.05, 3), v, "o", ms=3.6, color=col, alpha=0.85, mew=0,
                     zorder=3)
            axa.plot([f + dx - 0.09, f + dx + 0.09], [v.mean()] * 2, color="#222222", lw=1.0, zorder=4)
    axa.set_xticks(range(4))
    axa.set_xticklabels(["fold 0\n(exploratory)", "fold 1", "fold 2", "fold 3"])
    axa.set_xlim(-0.5, 3.5)
    axa.set_ylim(17.0, 32.5)
    axa.set_ylabel("SeK (%), test at best-val epoch")
    axa.plot([], [], color=C["teacher"], lw=2.0, label="PerASCD ViT-G teacher (1 run)")
    axa.plot([], [], "o", ms=3.6, color=C["student"], label="student, no KD (3 seeds)")
    axa.plot([], [], "o", ms=3.6, color=C["kd"], label=f"{METHOD} (3 seeds)")
    axa.legend(loc="upper right", handletextpad=0.4, borderaxespad=0.1)
    g = st.loc[("primary_f123", "test_at_best_val")]
    assert g.n_pos == g.n   # every seed pair positive
    axa.set_title("Test SeK per location-disjoint fold")
    panel_letter(axa, "a", dx=-0.11)

    # (b) KD gain vs teacher advantage, 6 configurations
    lim = (-0.6, 3.9)
    axb.plot(lim, lim, color="#888888", lw=0.8, ls=(0, (4, 2)), zorder=1)
    axb.text(3.55, 3.32, "gain = teacher\nadvantage", fontsize=7, color="#666666", ha="right", va="top",
             rotation=0)
    axb.axhline(0, color="#222222", lw=0.6, zorder=1)
    axb.axvline(0, color="#bbbbbb", lw=0.6, zorder=1)
    style = {"SECOND, PerASCD ViT-G": ("o", C["kd"], "SECOND,\nViT-G", (6, 3), "left"),
             "SECOND, VMamba-B": ("o", C["kd_vmb"], "SECOND,\nVMamba-B", (6, -12), "left"),
             "Landsat-SCD fold 0": ("s", C["kd"], "fold 0", (0, 14), "center"),
             "Landsat-SCD fold 1": ("s", C["kd"], "fold 1", (6, 0), "left"),
             "Landsat-SCD fold 2": ("s", C["kd"], "fold 2", (-6, 0), "right"),
             "Landsat-SCD fold 3": ("s", C["kd"], "fold 3", (6, 0), "left")}
    for _, r in gv.iterrows():
        mk, col, lab, off, ha = style[r.config]
        filled = bool(r.gate_pass)
        axb.errorbar(r.teacher_advantage, r.kd_gain, yerr=r.kd_gain_se, fmt=mk, ms=5,
                     mfc=col if filled else "white", mec=col, mew=1.0, color=col, elinewidth=0.8,
                     capsize=0, zorder=3)
        axb.annotate(lab, (r.teacher_advantage, r.kd_gain), xytext=off, textcoords="offset points",
                     ha=ha, va="center", fontsize=7, color="#333333")
    axb.text(0.97, 0.05, f"r = {cr.value:.2f} (n = {int(cr.n)})", transform=axb.transAxes, ha="right",
             va="bottom", fontsize=7)
    axb.plot([], [], "o", ms=5, color="#555555", label="SECOND")
    axb.plot([], [], "s", ms=5, color="#555555", label="Landsat-SCD")
    axb.plot([], [], "s", ms=5, mfc="white", mec="#555555", label="teacher gate failed")
    axb.legend(loc="lower right", bbox_to_anchor=(1.0, 0.11), handletextpad=0.3, borderaxespad=0.1)
    axb.set_xlim(lim)
    axb.set_ylim(-0.2, 3.9)
    axb.set_xlabel("Teacher advantage over no-KD student (SeK)")
    axb.set_ylabel("KD gain (SeK)")
    axb.set_title("KD gain vs. teacher advantage")
    panel_letter(axb, "b", dx=-0.16)
    return save(fig, "fig6_landsat_results")

# ============================================================================ Fig. 2
def read_config(path):
    return dict(tok.split("=", 1) for tok in path.read_text().split() if "=" in tok)


def fig2():
    import json
    meta = json.loads((LOGS / "cache/meta.json").read_text())
    n_pairs, n_ch, h, w = meta["layout"]["shape"]
    n_views = len(meta["d4"]["files"])
    cfg = read_config(LOGS / "confirm/runs/confirm_kdlogits-s1/config.txt")
    eff = tab("efficiency").set_index("quantity")
    p_s, p_t = eff.loc["Parameters (M)", "student"], eff.loc["Parameters (M)", "teacher"]
    g_s = eff.loc["GMACs / pair, aten counter", "student"]
    ncls = (n_ch - 1) // 2

    W, H = COL2, 3.05
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fs = 7
    TEA, STU, GT = C["teacher"], C["kd"], "#555555"

    def box(x, y, w, h, text, ec="#444444", fc="white", lw=0.8, ls="-", bold=False, fsz=fs, tc="#111111"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                    ec=ec, fc=fc, lw=lw, ls=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fsz, color=tc,
                fontweight="bold" if bold else "normal", linespacing=1.15)

    def arrow(p0, p1, col="#444444", lw=0.8, ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color=col, lw=lw,
                                     ls=ls, shrinkA=0, shrinkB=0,
                                     connectionstyle=f"arc3,rad={rad}"))

    # ---- band labels
    ax.text(0.06, 2.93, "Offline, once per dataset", fontsize=8, fontweight="bold", va="top", color=TEA)
    ax.text(0.06, 1.83, "Training", fontsize=8, fontweight="bold", va="top", color=STU)
    ax.plot([0.06, W - 0.06], [1.93, 1.93], color="#cccccc", lw=0.6)

    # ---- offline row
    yo, ho = 2.08, 0.62
    box(0.06, yo, 1.05, ho, "Training pairs\n(T1, T2), 512\u00b2")
    arrow((1.11, yo + ho / 2), (1.33, yo + ho / 2))
    box(1.33, yo, 1.05, ho, f"{n_views} D4 views\n(flip \u00d7 rot90)")
    arrow((2.38, yo + ho / 2), (2.60, yo + ho / 2))
    box(2.60, yo, 1.45, ho, f"PerASCD teacher\nViT-G, {p_t:.0f} M\nfrozen", ec=TEA, fc="#fbe6da", lw=1.0)
    arrow((4.05, yo + ho / 2), (4.27, yo + ho / 2))
    # stacked cache sheets
    for k in range(3)[::-1]:
        ax.add_patch(FancyBboxPatch((4.27 + 0.05 * k, yo + 0.05 * k), 1.95, ho - 0.1,
                                    boxstyle="round,pad=0,rounding_size=0.05", ec=TEA, fc="white", lw=0.8))
    ax.text(4.27 + 0.975, yo + (ho - 0.1) / 2,
            f"Logit cache: {n_views} views, {meta['precision']}, {h}\u00d7{w}\n"
            f"2C+1 = {n_ch} channels:\nsem T1 | sem T2 | change",
            ha="center", va="center", fontsize=fs, linespacing=1.2)

    # ---- training row (student)
    yb = 0.48
    box(0.06, 0.42, 1.05, 0.98, "Augmented pair:\nview g of D4,\ntemporal swap,\nphoto jitter")
    # student frame
    ax.add_patch(FancyBboxPatch((1.27, 0.30), 3.62, 1.43, boxstyle="round,pad=0,rounding_size=0.08",
                                ec=STU, fc="#eaf3fa", lw=1.0, ls=(0, (4, 2))))
    ax.text(1.35, 1.66, f"Student ({METHOD}), {p_s:.1f} M parameters, {g_s:.1f} GMACs",
            fontsize=fs, color=STU, va="top", fontweight="bold")
    box(1.40, 1.00, 1.05, 0.40, "VMamba-mini\nencoder (T1)", ec=STU)
    box(1.40, 0.42, 1.05, 0.40, "VMamba-mini\nencoder (T2)", ec=STU)
    ax.text(1.925, 0.91, "shared weights", fontsize=6.3 if False else fs, ha="center", va="center",
            color=STU, style="italic")
    arrow((1.11, 1.20), (1.40, 1.20))
    arrow((1.11, 0.62), (1.40, 0.62))
    box(2.65, 0.55, 0.68, 0.72, "per-stage\nconcat\nfusion", ec=STU)
    arrow((2.45, 1.20), (2.65, 1.05))
    arrow((2.45, 0.62), (2.65, 0.77))
    box(3.53, 1.00, 1.25, 0.40, "change decoder\n(DySample)", ec=STU)
    box(3.53, 0.40, 1.25, 0.46, "shared semantic decoder\n+ temporal embedding\n(DySample)", ec=STU, fsz=6.6 if False else fs)
    arrow((3.33, 1.05), (3.53, 1.20))
    arrow((3.33, 0.77), (3.53, 0.63))
    arrow((4.15, 1.00), (4.15, 0.86))
    ax.text(4.20, 0.93, "guides", fontsize=fs, va="center", ha="left", color=STU, style="italic")

    # outputs
    ax.plot([4.78, 5.45], [1.20, 1.20], color=STU, lw=0.8)
    ax.plot([4.78, 5.45], [0.63, 0.63], color=STU, lw=0.8)
    ax.text(5.12, 1.30, "change", fontsize=fs, va="bottom", ha="left")
    ax.text(5.12, 0.53, "sem T1,\nsem T2", fontsize=fs, va="top", ha="left")

    # ---- losses
    xl, wl = 5.62, 1.33
    lam_c, t_c = cfg["kd_chg"], cfg["t_chg"]
    lam_s, t_s, mask = cfg["kd_sem"], cfg["t_sem"], cfg["mask"]
    box(xl, 1.33, wl, 0.42, "soft-BCE change KD\n\u03bb$_{chg}$ = " + lam_c, ec=STU, fc="#eaf3fa", lw=1.0)
    box(xl, 0.86, wl, 0.38, "CE task losses\nvs ground truth", ec=GT)
    box(xl, 0.30, wl, 0.48, f"semantic KL, T = {t_s},\n{ncls - 1} real classes,\n\u03bb$_{{sem}}$ = {lam_s}, {mask} pixels",
        ec=STU, fc="#eaf3fa", lw=1.0)
    for y0, y1 in ((1.20, 1.50), (1.20, 1.10), (0.63, 0.98), (0.63, 0.54)):
        arrow((5.45, y0), (xl, y1), col="#777777", lw=0.6)
    # cache -> KD losses (view g of the student's augmentation)
    arrow((5.70, yo), (5.70, 1.75), col=TEA, lw=1.0)
    xr = xl + wl + 0.10
    ax.plot([4.27 + 1.95 + 0.10, xr, xr], [yo + 0.26, yo + 0.26, 0.54], color=TEA, lw=1.0)
    ax.plot([4.27 + 1.95, 4.27 + 2.05], [yo + 0.26, yo + 0.26], color=TEA, lw=1.0)
    arrow((xr, 0.54), (xl + wl, 0.54), col=TEA, lw=1.0)
    ax.text(5.76, 1.86, "view g, bilinear\nupsampling", fontsize=fs, color=TEA, va="center", ha="left")
    ax.text(0.06, 0.08, "Inference: the student alone; teacher, cache and losses are discarded.",
            fontsize=fs, ha="left", va="bottom", color="#333333", style="italic")
    return save(fig, "fig2_method")


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6}

if __name__ == "__main__":
    which = sys.argv[1:] or list(FIGS)
    for w in which:
        FIGS[w]()
