"""Cache « élève professeur » : contrôle d'auto-distillation (révision de l'article, vague A).

Question posée par les trois relecteurs internes : le gain de la KD vient-il du savoir du
professeur PerASCD, ou d'une régularisation que n'importe quel professeur — y compris
l'élève lui-même (born-again, Furlanello et al. 2018 ; Yuan et al. 2020) — fournirait ?
On remplace donc le professeur par un élève DistilSCD **entraîné sans KD**, et l'on écrit
un cache au format exact de `scripts/teacher/cache_teacher.py`, lu tel quel par
`distilscd.datasets.teacher_cache.TeacherCacheDataset` : même entraînement, même recette,
mêmes pertes, mêmes λ et T — seule la source des cibles change.

Format (identique au cache PerASCD) : un `.npy` fp16 (N, 2C+1, 128, 128) par élément de D4,
canaux = logits sémantiques T1 (C), T2 (C), logit de changement (1) ; `ids.txt`, `meta.json`.
Pour chaque g de D4 on stocke la sortie de l'élève sur l'entrée transformée, S(g·x).

Différence assumée avec le cache du professeur : l'élève prédit à 512 × 512 ; ses logits
sont moyennés par blocs 4 × 4 vers 128 (la KD les resuréchantillonne en bilinéaire). Le
professeur, lui, produit nativement du 128. La perte d'information est mesurée et écrite
dans `meta.json` (SeK sur le train : sortie pleine résolution vs cache relu).
Logit de changement = bcd[:, 1] − bcd[:, 0] (softmax à 2 classes ⇔ sigmoïde de cette
différence), la convention de `distilscd.losses.distill`.

    python scripts/teacher/cache_student.py --checkpoint $SCRATCH/csf-mamba-runs/<run>/best.pt \
        --dataset second --data-root $SLURM_TMPDIR/SECOND --out-dir $SCRATCH/csf-distill/cache/student_second_train
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from distilscd.datasets import DATASETS  # noqa: E402
from distilscd.distill.d4 import D4, apply_d4  # noqa: E402
from distilscd.evaluation.metrics import SCDEvaluator  # noqa: E402
from distilscd.model import DistilSCDStudent, count_parameters  # noqa: E402

NATIVE = 128
# Point d'efficience figé (identique à scripts/train_kd_*.sbatch).
STUDENT_KW = dict(encoder="vmamba_mini", core="chess", decoder_refine="dw", fft_stages=(0, 1),
                  fusion="concat", cga=True, mcasf=True, upsample="dysample")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="best.pt d'un élève SANS KD (state_dict EMA nu)")
    p.add_argument("--dataset", default="second", choices=["second", "landsat_perascd"])
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--ids-file", default=None, help="landsat_perascd : train.txt du pli")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--backend", default="auto", choices=["auto", "mamba", "ref"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=None, help="nb de lots (test rapide)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def student_blob(model, a, b, precision, device):
    """(B, 2C+1, 512, 512) : logits sém. T1, T2, logit de changement, en fp32."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                        enabled=(precision == "bf16" and device.startswith("cuda"))):
        out = model(a, b)
    d = (out["bcd"][:, 1] - out["bcd"][:, 0]).float()[:, None]
    return torch.cat([out["sem_t1"].float(), out["sem_t2"].float(), d], dim=1)


def as_outputs(blob, C):
    """Blob (2C+1 canaux) -> dict de sorties pour SCDEvaluator (bcd à 2 canaux)."""
    d = blob[:, 2 * C:2 * C + 1]
    return {"bcd": torch.cat([torch.zeros_like(d), d], dim=1),
            "sem_t1": blob[:, :C], "sem_t2": blob[:, C:2 * C]}


@torch.no_grad()
def main():
    args = parse_args()
    device = args.device
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cls, C = DATASETS[args.dataset]
    kw = {"ids_file": args.ids_file} if args.ids_file else {}
    if args.dataset == "landsat_perascd" and not args.ids_file:
        raise SystemExit("--ids-file requis pour landsat_perascd (pli disjoint par emplacement)")
    ds = cls(args.data_root, split=args.split, transform=None, **kw)
    n = len(ds) if args.limit is None else min(len(ds), args.limit * args.batch_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=device.startswith("cuda"))

    model = DistilSCDStudent(num_semantic_classes=C, backend=args.backend, **STUDENT_KW)
    ckpt = Path(args.checkpoint)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "model" in state and "optimizer" in state:
        raise SystemExit(f"{ckpt} est un last.pt (reprise) : donner best.pt (poids EMA nus)")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    chans = 2 * C + 1

    (out / "ids.txt").write_text("\n".join(ds.ids[:n]) + "\n")
    maps = [np.lib.format.open_memmap(out / f"d4_{i}.npy", mode="w+", dtype=np.float16,
                                      shape=(n, chans, NATIVE, NATIVE)) for i in range(len(D4))]
    ev_full, ev_cache = SCDEvaluator(num_classes=C), SCDEvaluator(num_classes=C)
    change_agree = {i: [0.0, 0] for i in range(1, len(D4))}
    t0, done = time.time(), 0
    for bi, batch in enumerate(loader):
        if done >= n:
            break
        m = min(batch["img_t1"].shape[0], n - done)
        a, b = batch["img_t1"][:m].to(device), batch["img_t2"][:m].to(device)
        targets = {k: batch[k][:m].to(device) for k in ("change", "sem_t1", "sem_t2", "unchanged")}
        ident = None
        for gi, (h, k) in enumerate(D4):
            full = student_blob(model, apply_d4(a, h, k), apply_d4(b, h, k), args.precision, device)
            if full.shape[-1] != 4 * NATIVE:
                raise SystemExit(f"sortie {tuple(full.shape)} : tuile 512 attendue")
            low = F.avg_pool2d(full, 4)
            maps[gi][done:done + m] = low.half().cpu().numpy()
            if gi == 0:
                ident = low
                ev_full.add(as_outputs(full, C), targets)
            else:
                ref = apply_d4(ident, h, k)
                agree = ((ref[:, 2 * C] > 0) == (low[:, 2 * C] > 0)).float()
                change_agree[gi][0] += float(agree.sum())
                change_agree[gi][1] += agree.numel()
        rd = torch.from_numpy(np.asarray(maps[0][done:done + m])).to(device).float()
        rd = F.interpolate(rd, (4 * NATIVE, 4 * NATIVE), mode="bilinear", align_corners=False)
        ev_cache.add(as_outputs(rd, C), targets)
        done += m
        if bi % 50 == 0:
            rate = done / max(time.time() - t0, 1e-6)
            print(f"{done}/{n} paires, {rate:.1f} paires/s (×8 vues)", flush=True)
    for mm in maps:
        mm.flush()
    mf, mc = ev_full.compute(), ev_cache.compute()
    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "split": args.split, "n_pairs": n,
        "dataset": args.dataset, "ids_file": args.ids_file, "num_classes": C,
        "teacher": "student_self_distillation", "arch": "DistilSCDStudent " + json.dumps(STUDENT_KW),
        "params": count_parameters(model),
        "precision": args.precision,
        "checkpoint": {"path": str(ckpt), "sha256": sha256(ckpt)},
        "layout": {"shape": [n, chans, NATIVE, NATIVE], "dtype": "float16",
                   "channels": {"sem_t1_logits": [0, C], "sem_t2_logits": [C, 2 * C],
                                "change_logit": [2 * C, 2 * C + 1]},
                   "change_logit_def": "bcd[:,1] - bcd[:,0]",
                   "downsample": "avg_pool2d 4x4 des logits 512 -> 128",
                   "class_0": "jamais cible de la CE de l'élève (ignore_index) : exclu de la KL",
                   "upsample_to_512": "bilinear, align_corners=False (fait par la KD)"},
        "d4": {"files": [f"d4_{i}.npy" for i in range(len(D4))], "elements_hflip_then_rot90k": D4,
               "rule": "sortie de l'élève sur l'entrée transformée, S(g·x), déjà dans le repère transformé"},
        "checks": {
            "train_sek_full_res_identity": float(mf.sek), "train_fscd_full_res_identity": float(mf.fscd),
            "train_sek_cache_reupsampled": float(mc.sek), "train_fscd_cache_reupsampled": float(mc.fscd),
            "train_miou_cache_reupsampled": float(mc.miou),
            "change_agreement_vs_identity": {f"d4_{i}": v[0] / max(v[1], 1) for i, v in change_agree.items()},
        },
        "seconds": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps(meta["checks"], indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
