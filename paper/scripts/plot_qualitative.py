"""Fig. 7 (qualitative maps) from the .npz written on the cluster by qualitative_predictions.py.

    scp -r magron13@narval.alliancecan.ca:/scratch/magron13/csf-distill/qualitative logs/
    python paper/scripts/plot_qualitative.py --in logs/qualitative            # 6 SECOND + 4 Landsat
    python paper/scripts/plot_qualitative.py --in logs/qualitative --n-second 4 --n-landsat 3

Rows = image pairs (evenly spaced subset of the candidates, which were themselves chosen at
evenly spaced quantiles of the changed fraction: no visual cherry-picking). Columns = images
T1 / T2, then the T1 and T2 semantic-change maps of the ground truth, the student without KD,
DistilSCD and the teacher. White = unchanged; black = changed pixel without a semantic label
(ignored by the metric). The per-pair SeK of each model is printed in its T1 tile (optional).
Output: paper/figures/fig7_qualitative.{png,pdf}, double-column width.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import COL2, METHOD, save, setup  # noqa: E402

# Colour-blind-aware class palettes (index 0 = unchanged -> white; 255 = unlabelled -> black).
# Class order = distilscd/datasets/second.py and landsat_perascd.py CLASS_NAMES.
PALETTE = {
    "second": (["low vegetation", "non-vegetated ground", "tree", "water", "building", "playground"],
               ["#B8E186", "#BDB8AD", "#1B7837", "#2C7BB6", "#D95F02", "#E7298A"]),
    "landsat_f1": (["farmland", "desert", "building", "water"],
                   ["#66A61E", "#E6AB02", "#D95F02", "#2C7BB6"]),
}
TITLES = {"second": "SECOND test", "landsat_f1": "Landsat-SCD, location-disjoint fold 1 (test)"}
MODELS = [("gt", "ground truth"), ("student", "student, no KD"), ("kd", METHOD), ("teacher", "teacher")]


def colourise(m, colours):
    lut = np.ones((256, 3))
    for i, c in enumerate(colours, start=1):
        lut[i] = [int(c[k:k + 2], 16) / 255 for k in (1, 3, 5)]
    lut[255] = 0.0
    return lut[m]


def load(dirpath, n):
    files = sorted(Path(dirpath).glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz in {dirpath}")
    recs = [dict(np.load(f, allow_pickle=False)) for f in files]
    missing = [str(r["pair"]) for r in recs if "teacher_t1" not in r]
    if missing:
        raise SystemExit(f"teacher predictions missing for {missing[:3]}: run --part teacher")
    recs.sort(key=lambda r: int(r["selection_rank"]))
    pick = np.unique(np.round(np.linspace(0, len(recs) - 1, n)).astype(int))
    return [recs[i] for i in pick]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--n-second", type=int, default=6)
    p.add_argument("--n-landsat", type=int, default=4)
    p.add_argument("--no-sek", action="store_true")
    p.add_argument("--name", default="fig7_qualitative")
    a = p.parse_args()

    setup()
    bands = [(d, load(Path(a.inp) / d, n)) for d, n in (("second", a.n_second), ("landsat_f1", a.n_landsat)) if n > 0]
    ncol = 2 + 2 * len(MODELS)
    nrow = sum(len(r) for _, r in bands)
    tile = COL2 / (ncol + 0.35)
    header, band_gap, legend_h = 0.58, 0.30, 0.30
    H = header + nrow * tile + len(bands) * (band_gap + legend_h)
    fig = plt.figure(figsize=(COL2, H))
    y = H - header
    for b, (dname, recs) in enumerate(bands):
        names, colours = PALETTE[dname]
        ty = (H - 0.03) if b == 0 else (y + 0.06)
        fig.text(0.004, ty / H, TITLES[dname], fontsize=8, fontweight="bold",
                 va="top" if b == 0 else "bottom")
        for r_i, rec in enumerate(recs):
            y -= tile
            cells = [rec["img_t1"], rec["img_t2"]]
            for key, _ in MODELS:
                cells += [colourise(rec[f"{key}_t1"], colours), colourise(rec[f"{key}_t2"], colours)]
            for c, img in enumerate(cells):
                ax = fig.add_axes([(0.35 * tile + c * tile) / COL2, y / H, 0.97 * tile / COL2, 0.97 * tile / H])
                ax.imshow(img, interpolation="nearest")
                ax.set_xticks([]), ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(True), s.set_linewidth(0.3), s.set_color("#888888")
                m = (c - 2) // 2
                if not a.no_sek and c >= 4 and c % 2 == 0:
                    key = MODELS[m][0]
                    ax.text(0.03, 0.03, f"{100 * float(rec[f'{key}_sek']):.1f}", transform=ax.transAxes,
                            fontsize=6, va="bottom", ha="left",
                            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
                if b == 0 and r_i == 0:
                    lab = ["T1", "T2"][c] if c < 2 else f"{MODELS[m][1]}\nT1" if c % 2 == 0 else "\nT2"
                    ax.set_title(lab, fontsize=7, loc="center", pad=2)
        # class legend under the band
        y -= legend_h
        handles = [Patch(fc=c, ec="#888888", lw=0.3, label=n) for n, c in zip(names, colours)]
        handles += [Patch(fc="white", ec="#888888", lw=0.3, label="unchanged")]
        if dname == "second":
            handles += [Patch(fc="black", ec="#888888", lw=0.3, label="unlabelled")]
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.35 * tile / COL2, (y + legend_h / 2) / H),
                   ncol=len(handles), fontsize=7, handlelength=1.0, columnspacing=0.9, handletextpad=0.3)
        y -= band_gap
    save(fig, a.name)


if __name__ == "__main__":
    main()
