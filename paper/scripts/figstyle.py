"""Shared style for the paper figures (IEEE JSTARS).

Widths: single column 3.5 in, double column 7.16 in. Figures are saved at their exact
physical width (no tight bbox), as PNG (300 dpi) and PDF (Type-42 fonts embedded).
Colours: Okabe-Ito (colour-blind safe); one colour per entity, reused in every figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "paper" / "tables"
LOGS = ROOT / "logs"
FIGDIR = ROOT / "paper" / "figures"

COL1 = 3.5      # in, IEEE single column
COL2 = 7.16     # in, IEEE double column

METHOD = "DistilSCD"

# Okabe-Ito palette bound to entities (threaded across all figures)
C = {
    "student": "#5f5f5f",       # student without KD (grey: the reference point)
    "kd": "#0072B2",            # DistilSCD, ViT-G teacher (focal)
    "kd_vmb": "#56B4E9",        # DistilSCD, VMamba-B teacher (same hue family, lighter)
    "kd_chg": "#009E73",        # change-only KD
    "teacher": "#D55E00",       # PerASCD ViT-G teacher
    "teacher_vmb": "#E69F00",   # VMamba-B teacher
    "published": "#9a9a9a",     # published models (comparators)
    "control": "#CC79A7",       # ground-truth controls
    "oracle": "#000000",
    "feat": "#999933",          # feature KD (olive; distinct from the GT controls)
}
FOLD = ["#999999", "#0072B2", "#E69F00", "#009E73"]   # folds 0..3 (fold 0 exploratory: grey)

SIZES = (8, 7, 7)   # base (titles, axis labels), secondary (legend, annotations), ticks


def setup():
    base, sec, tick = SIZES
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": base, "axes.labelsize": base, "axes.titlesize": base,
        "axes.titlelocation": "left", "legend.fontsize": sec,
        "xtick.labelsize": tick, "ytick.labelsize": tick,
        "axes.linewidth": 0.6, "axes.spines.right": False, "axes.spines.top": False,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "lines.linewidth": 1.2, "patch.linewidth": 0.6,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.3,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "savefig.bbox": "standard", "figure.dpi": 150,
        "mathtext.default": "regular",
    })


def panel_letter(ax, letter, dx=-0.16, dy=1.02):
    """IEEE-style bold panel label '(a)' outside the top-left corner of the axes."""
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontweight="bold",
            fontsize=SIZES[0] + 1, va="bottom", ha="left")


def check_layout(fig):
    """figure-style §9.1: overlapping visible text boxes and text outside the canvas."""
    r = fig.canvas.get_renderer()
    hidden = set()   # tick labels of ticks outside the view interval are never drawn
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                if not (lo - 1e-9 <= tk.get_loc() <= hi + 1e-9):
                    hidden.update({tk.label1, tk.label2})
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible() and t not in hidden]
    texts = [(t, b) for t, b in texts if b.width > 0]
    spines = [(s, s.get_window_extent(r)) for ax in fig.axes
              for s in ax.spines.values() if s.get_visible() and ax.axison]
    ticklabels = {ax: set(ax.get_xticklabels(which="both") + ax.get_yticklabels(which="both"))
                  for ax in fig.axes}
    problems = []
    for i, (a, ba) in enumerate(texts):
        for b, bb in texts[i + 1:]:
            if ba.overlaps(bb):
                problems.append(("text/text", a.get_text(), b.get_text()))
    for t, bt in texts:
        for s, bs in spines:
            if t in ticklabels.get(s.axes, ()):
                continue
            if getattr(t, "_ok_on_spine", False):
                continue
            if bt.overlaps(bs):
                problems.append(("text/spine", t.get_text(), s.axes.get_label() or "axes"))
    W, H = fig.bbox.width, fig.bbox.height
    for t, bt in texts:
        if bt.x0 < -1 or bt.y0 < -1 or bt.x1 > W + 1 or bt.y1 > H + 1:
            problems.append(("outside", t.get_text(), ""))
    return problems


def save(fig, name, verbose=True):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    probs = check_layout(fig)
    fig.savefig(FIGDIR / f"{name}.png", dpi=300)
    fig.savefig(FIGDIR / f"{name}.pdf")
    w, h = fig.get_size_inches()
    if verbose:
        print(f"{name}: {w:.2f} x {h:.2f} in; layout issues: {len(probs)}")
        for p in probs:
            print("   ", p)
    plt.close(fig)
    return probs
