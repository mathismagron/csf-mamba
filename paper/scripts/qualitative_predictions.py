"""Fig. 7 (qualitative maps): inference of the student without KD, the KD student (DistilSCD)
and the teacher on a few SECOND test pairs and Landsat-SCD fold-1 test pairs.

Runs on the cluster (data and checkpoints live there), in two passes because the student and
the teacher use different virtual environments:

    # pass 1, venv of the student ($SCRATCH/csf-venv-cu12): inputs, ground truth, both students
    python paper/scripts/qualitative_predictions.py --part student --out $OUT \
        --second-root $SLURM_TMPDIR/SECOND --landsat-root $SLURM_TMPDIR/LandsatSCD512
    # pass 2, venv of the teacher ($SCRATCH/perascd-venv): adds the teacher predictions
    python paper/scripts/qualitative_predictions.py --part teacher --out $OUT \
        --perascd-root third_party/PerASCD

Output: $OUT/<dataset>/<pair>.npz, one file per pair, with
    img_t1, img_t2        uint8 (H, W, 3), as read from disk
    gt_change             uint8 (H, W), 1 = changed
    gt_t1, gt_t2          uint8 (H, W), SCD maps: class 1..C-1 on changed pixels, 0 unchanged,
                          255 = changed pixel without semantic label (ignored by the metric)
    <model>_change, <model>_t1, <model>_t2   uint8 predictions, same convention
    <model>_sek           per-pair SeK (our SCDEvaluator, same as the training logs)
for <model> in {student, kd, teacher}; plus `pair`, `dataset`, `change_fraction`,
`selection_rank` and the checkpoint paths. `selection.json` records the selection rule.

Pair selection (fixed rule, no visual cherry-picking): test pairs with a non-zero changed
fraction are sorted by that fraction, and N candidates are taken at evenly spaced quantiles
between 0.25 and 0.95 (SECOND: 24 candidates, Landsat fold 1: 12). The plotting script
(plot_qualitative.py) shows an evenly spaced subset of them (6 and 4 by default).

Prediction rules mirror the evaluation code exactly:
    student: change = argmax(bcd); semantic = argmax(sem_t) * change      (SCDEvaluator)
    teacher: change = sigmoid(logit) > 0.5; semantic = argmax(sem_t) * change
             (scripts/teacher/eval_perascd.py, `legacy` evaluation of the PerASCD authors)
Student checkpoints are `best.pt` (EMA weights): SECOND = max test SeK over epochs (the field
convention used in the paper; seed 1 no-KD max is at epoch 100 of its 200-epoch run, i.e.
inside the first 120 epochs), Landsat = best fold-validation SeK. Seed 1 for both students.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

SCRATCH = os.environ.get("SCRATCH", "/scratch")
CKPT = {
    "second": {
        "student": f"{SCRATCH}/csf-mamba-runs/second_mini_chess_crop512-lean-augswap-ema-s1/best.pt",
        "kd": f"{SCRATCH}/csf-distill/runs/confirm_kdlogits-s1/best.pt",
        "teacher": f"{SCRATCH}/csf-distill/teacher/ckpt/"
                   "PerAChain_40e_mIoU74.33_Sek26.11_Fscd66.41_OA88.70.pth",
    },
    "landsat_f1": {
        "student": f"{SCRATCH}/csf-distill/runs/landsat_f1_base-s1/best.pt",
        "kd": f"{SCRATCH}/csf-distill/runs/landsat_f1_kdlogits-s1/best.pt",
        "teacher": f"{SCRATCH}/csf-distill/teachers/landsat_vitg_f1/best.pt",
    },
}
N_CANDIDATES = {"second": 24, "landsat_f1": 12}
QUANTILES = (0.25, 0.95)
TEACHER_ARCH = "ViT-G/16/1024"      # PerASCD ViT-G, PerA normalisation (set by build_teacher)
# Student recipe (frozen efficiency point; identical to scripts/train_kd_*.sbatch)
STUDENT_KW = dict(encoder="vmamba_mini", core="chess", backend="mamba", decoder_refine="dw",
                  fft_stages=(0, 1), fusion="concat", cga=True, mcasf=True, upsample="dysample")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--part", required=True, choices=["student", "teacher"])
    p.add_argument("--out", required=True)
    p.add_argument("--second-root", default=None)
    p.add_argument("--landsat-root", default=None)
    p.add_argument("--landsat-ids", default=str(REPO / "splits/LandsatSCD_loc/fold1/test.txt"))
    p.add_argument("--perascd-root", default=str(REPO / "third_party/PerASCD"))
    p.add_argument("--msda", default="cuda", choices=["cuda", "pytorch"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--datasets", default="second,landsat_f1")
    return p.parse_args()


def select_pairs(ds, n):
    """Indices of n pairs at evenly spaced quantiles of the changed fraction (non-zero only)."""
    frac = np.array([ds.change_fraction(i) for i in range(len(ds))])
    idx = np.flatnonzero(frac > 0)
    idx = idx[np.argsort(frac[idx], kind="stable")]
    qs = np.linspace(QUANTILES[0], QUANTILES[1], n)
    picks = sorted({int(idx[int(round(q * (len(idx) - 1)))]) for q in qs}, key=lambda i: frac[i])
    return picks, frac


def scd_maps(change, sem_t1, sem_t2):
    """(B,H,W) change in {0,1}, semantic argmax -> uint8 SCD maps (class on changed, else 0)."""
    c = change.astype(np.uint8)
    return c, (sem_t1 * c).astype(np.uint8), (sem_t2 * c).astype(np.uint8)


def per_pair_sek(evaluator_cls, num_classes, bcd, s1, s2, targets):
    ev = evaluator_cls(num_classes=num_classes)
    ev.add({"bcd": bcd, "sem_t1": s1, "sem_t2": s2}, targets)
    return float(ev.compute().sek)


def save_npz(path, **arrays):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- students
def run_students(args):
    import torch
    from distilscd.datasets import DATASETS
    from distilscd.evaluation.metrics import SCDEvaluator
    from distilscd.model import DistilSCDStudent

    out = Path(args.out)
    selection = {"rule": f"non-zero changed fraction, sorted, evenly spaced quantiles {QUANTILES}",
                 "checkpoints": CKPT, "datasets": {}}
    for name in args.datasets.split(","):
        if name == "second":
            cls, ncls = DATASETS["second"]
            ds = cls(args.second_root, split="test")
        else:
            cls, ncls = DATASETS["landsat_perascd"]
            ds = cls(args.landsat_root, split="test", ids_file=args.landsat_ids)
        for k in ("student", "kd"):
            if not Path(CKPT[name][k]).is_file():
                raise SystemExit(f"missing checkpoint: {CKPT[name][k]}")
        picks, frac = select_pairs(ds, N_CANDIDATES[name])
        print(f"[{name}] {len(ds)} test pairs, {len(picks)} candidates", flush=True)
        models = {}
        for k in ("student", "kd"):
            m = DistilSCDStudent(num_semantic_classes=ncls, **STUDENT_KW)
            state = torch.load(CKPT[name][k], map_location="cpu")
            m.load_state_dict(state, strict=True)
            models[k] = m.to(args.device).eval()
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        records = []
        for rank, i in enumerate(picks):
            s = ds[i]
            pair = str(ds.ids[i]).replace("/", "__").removesuffix(".png")
            a = s["img_t1"][None].to(args.device)
            b = s["img_t2"][None].to(args.device)
            gt_c = s["change"].numpy().astype(np.uint8)
            g1, g2 = s["sem_t1"].numpy(), s["sem_t2"].numpy()     # 255 where unlabelled
            rec = {
                "pair": pair, "dataset": name, "change_fraction": float(frac[i]),
                "selection_rank": rank,
                "img_t1": (s["img_t1"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8),
                "img_t2": (s["img_t2"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8),
                "gt_change": gt_c,
                "gt_t1": np.where(gt_c == 1, g1, 0).astype(np.uint8),
                "gt_t2": np.where(gt_c == 1, g2, 0).astype(np.uint8),
            }
            targets = {"change": s["change"][None].to(args.device),
                       "sem_t1": s["sem_t1"][None].to(args.device),
                       "sem_t2": s["sem_t2"][None].to(args.device)}
            with torch.no_grad():
                for k, m in models.items():
                    o = m(a, b)
                    ch = o["bcd"].argmax(1)[0].cpu().numpy()
                    p1 = o["sem_t1"].argmax(1)[0].cpu().numpy()
                    p2 = o["sem_t2"].argmax(1)[0].cpu().numpy()
                    rec[f"{k}_change"], rec[f"{k}_t1"], rec[f"{k}_t2"] = scd_maps(ch, p1, p2)
                    rec[f"{k}_sek"] = per_pair_sek(SCDEvaluator, ncls, o["bcd"], o["sem_t1"],
                                                   o["sem_t2"], targets)
                    rec[f"{k}_ckpt"] = CKPT[name][k]
            save_npz(d / f"{pair}.npz", **rec)
            records.append({"pair": pair, "change_fraction": float(frac[i]),
                            "student_sek": rec["student_sek"], "kd_sek": rec["kd_sek"]})
            print(f"  {pair}: changed {frac[i]:.3f} | SeK no KD {rec['student_sek']:.4f} "
                  f"| KD {rec['kd_sek']:.4f}", flush=True)
        selection["datasets"][name] = {"n_test": len(ds), "pairs": records}
    (out / "selection.json").write_text(json.dumps(selection, indent=1))


# ----------------------------------------------------------------------------- teacher
def run_teacher(args):
    import torch
    sys.path.insert(0, str(REPO / "scripts" / "teacher"))
    import eval_perascd as ev   # noqa: E402  (PerASCD import helpers, normalisation, forward)
    from distilscd.evaluation.metrics import SCDEvaluator

    out = Path(args.out)
    root = Path(args.perascd_root).resolve()
    PerASCD, _, _, compiled = ev.import_legacy(root, args.msda)
    for name in args.datasets.split(","):
        files = sorted((out / name).glob("*.npz"))
        if not files:
            raise SystemExit(f"no student outputs in {out / name}: run --part student first")
        ev.set_num_classes(7 if name == "second" else 5)
        model, info = ev.build_teacher(PerASCD, root, TEACHER_ARCH, CKPT[name]["teacher"])
        model = model.to(args.device).eval()
        print(f"[{name}] teacher {info['arch']} | normalisation {info['normalization']['name']} "
              f"| MSDA {'cuda' if compiled and args.msda == 'cuda' else 'pytorch'}", flush=True)
        for f in files:
            rec = dict(np.load(f, allow_pickle=False))
            a = torch.from_numpy(rec["img_t1"]).permute(2, 0, 1)[None].float().div(255).to(args.device)
            b = torch.from_numpy(rec["img_t2"]).permute(2, 0, 1)[None].float().div(255).to(args.device)
            with torch.no_grad():
                ch, oa, ob = ev.forward(model, a, b, "fp32", args.device)    # PerA normalisation inside
            cm = (torch.sigmoid(ch) > 0.5).squeeze(1).long()
            rec["teacher_change"], rec["teacher_t1"], rec["teacher_t2"] = scd_maps(
                cm[0].cpu().numpy(), oa.argmax(1)[0].cpu().numpy(), ob.argmax(1)[0].cpu().numpy())
            # same SeK computation as the students: 2-channel change logits whose argmax is
            # exactly sigmoid > 0.5 (as in eval_perascd.evaluate)
            bcd = torch.cat([torch.zeros_like(ch), ch], dim=1)
            gt_c = torch.from_numpy(rec["gt_change"].astype(np.int64))[None].to(args.device)
            def sem(gt):   # back to the dataset convention: 255 on unchanged / unlabelled
                t = torch.from_numpy(gt.astype(np.int64))[None].to(args.device)
                return torch.where(t == 0, torch.full_like(t, 255), t)
            targets = {"change": gt_c, "sem_t1": sem(rec["gt_t1"]), "sem_t2": sem(rec["gt_t2"])}
            rec["teacher_sek"] = per_pair_sek(SCDEvaluator, ev.NUM_CLASSES, bcd, oa, ob, targets)
            rec["teacher_ckpt"] = CKPT[name]["teacher"]
            save_npz(f, **rec)
            print(f"  {f.stem}: teacher SeK {float(rec['teacher_sek']):.4f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    a = parse_args()
    run_students(a) if a.part == "student" else run_teacher(a)
