"""ε du contrôle `--kd-sem-target gt_ls`, apparié au professeur (révision, vague A).

Le contrôle remplace la distribution sémantique du professeur par la vérité lissée
(1 − ε)·one-hot + ε/(C − 1). Pour que la cible soit aussi « douce » que celle du professeur,
ε est choisi pour que la probabilité de la classe vraie égale la probabilité moyenne que le
professeur lui donne, à la température de la KD (T = 2), sur les pixels changés étiquetés
du train (vue identité du cache) :

    m = E[ softmax(z_t[1..C−1] / T)[y] ]      ε = (1 − m)·(C − 1)/(C − 2)

Seul ce scalaire passe du professeur au contrôle ; aucune information par pixel ou par
classe. Écrit un JSON (eps, m, entropie moyenne, nombre de pixels).

    python scripts/kd_control_eps.py --cache $SCRATCH/csf-distill/cache/perascd_landsat_f1_train \
        --dataset landsat_perascd --data-root $SLURM_TMPDIR/LandsatSCD512 \
        --ids-file splits/LandsatSCD_loc/fold1/train.txt --out $OUT/kd_control_eps.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from distilscd.datasets import DATASETS  # noqa: E402


def matched_eps(m: float, C: int) -> float:
    """ε tel que (1 − ε) + ε/(C − 1) = m, borné à [0, 1]."""
    return float(min(1.0, max(0.0, (1.0 - m) * (C - 1) / (C - 2))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--dataset", default="second", choices=["second", "landsat_perascd"])
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--ids-file", default=None)
    p.add_argument("--t-sem", type=float, default=2.0)
    p.add_argument("--max-pairs", type=int, default=None, help="sous-échantillon régulier (rapide)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cls, C = DATASETS[args.dataset]
    kw = {"ids_file": args.ids_file} if args.ids_file else {}
    ds = cls(args.data_root, split=args.split, transform=None, **kw)
    cache = Path(args.cache)
    ids = [l.strip() for l in (cache / "ids.txt").read_text().splitlines() if l.strip()]
    row = {n: i for i, n in enumerate(ids)}
    d4_0 = np.load(cache / "d4_0.npy", mmap_mode="r")
    if d4_0.shape[1] != 2 * C + 1:
        raise SystemExit(f"cache à {d4_0.shape[1]} canaux, attendu {2 * C + 1}")
    idx = np.arange(len(ds))
    if args.max_pairs and args.max_pairs < len(ds):
        idx = np.linspace(0, len(ds) - 1, args.max_pairs).round().astype(int)
    s_p, s_h, n_pix = 0.0, 0.0, 0
    for i in idx:
        sample = ds[int(i)]
        tz = torch.from_numpy(np.array(d4_0[row[ds.ids[i]]])).float()[None]
        size = sample["sem_t1"].shape[-2:]
        tz = F.interpolate(tz, size=size, mode="bilinear", align_corners=False)[0]
        for lab, sl in ((sample["sem_t1"], slice(1, C)), (sample["sem_t2"], slice(C + 1, 2 * C))):
            valid = (lab >= 1) & (lab <= C - 1)
            if not valid.any():
                continue
            q = F.softmax(tz[sl] / args.t_sem, dim=0)                    # (C−1, H, W)
            y = (lab.clamp(1, C - 1) - 1)
            py = q.gather(0, y[None])[0]
            h = -(q * torch.log(q + 1e-8)).sum(0)
            s_p += float(py[valid].sum())
            s_h += float(h[valid].sum())
            n_pix += int(valid.sum())
    if n_pix == 0:
        raise SystemExit("aucun pixel changé étiqueté")
    m = s_p / n_pix
    res = {"eps": round(matched_eps(m, C), 4), "teacher_mean_p_true": m,
           "teacher_mean_entropy_nats": s_h / n_pix, "uniform_entropy_nats": float(np.log(C - 1)),
           "t_sem": args.t_sem, "num_classes": C, "pixels": n_pix, "pairs": int(len(idx)),
           "cache": str(cache), "ids_file": args.ids_file}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
