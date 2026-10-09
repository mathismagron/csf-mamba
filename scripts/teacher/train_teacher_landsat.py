"""Entraîne un professeur PerASCD sur un pli Landsat-SCD disjoint par emplacement.

Pourquoi : le checkpoint Landsat des auteurs (non publié) a été entraîné sur le split
aléatoire, qui fuit (12 emplacements, 333 images ; `scripts/verify_landsat_leakage.py`). Il a
donc vu les images de test de tous nos plis. On réentraîne un professeur par pli.

Recette = celle de leur code `legacy` (train.py pour ViT-G, train_Encoders.py pour VMamba-B),
reprise à l'identique sauf mention :
  SGD nesterov, momentum 0,9, wd 1e-5, lr 0,1, warmup linéaire 10 %, puis poly 1,5 vers 0,
  recalculé à chaque micro-itération ; fp16 autocast + GradScaler ; clip 1,5 ;
  drop-path 0,3 ; 50 époques ; loss = 0,5·CE(ignore 0) + BCE pondérée + SSC (τ = 0,01) ;
  augmentations rot90/flip + ColorJitter (0,2 ; 0,2 ; 0,1 ; 0,1), leurs classes CDM*.
  ViT-G : lot 4 × accumulation 2, entrée 448 → sortie 512, poids PerA pré-entraînés.
  VMamba-B : lot 8, poids ImageNet `vssm_base_0229_ckpt_epoch_237.pth`, normalisation
  ImageNet (celle avec laquelle leur checkpoint SECOND se reproduit, cf. distillation.md).
Écarts déclarés : (1) leur sélection se fait sur la Fscd du TEST ; ici best.pt suit la
**SeK de val** du pli, le test n'étant qu'enregistré (`metrics_test.csv`) ; (2) aucun
pseudo-label (commenté chez eux aussi).

    python scripts/teacher/train_teacher_landsat.py --perascd-root third_party/PerASCD \
        --arch ViT-G/16/1024 --pretrained $SCRATCH/csf-distill/pera/<poids PerA>.params \
        --data-root $SLURM_TMPDIR/LandsatSCD512 --fold-dir splits/LandsatSCD_loc/fold0 \
        --out $SCRATCH/csf-distill/teachers/landsat_vitg_f0
    # faisabilité : --max-iters 40 (pic mémoire, s/itération, projection), puis sortie

Reprise automatique sur `<out>/last.pt`.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import eval_perascd as ev  # noqa: E402
from csf_mamba.datasets.landsat_perascd import assert_location_disjoint  # noqa: E402
from csf_mamba.evaluation.metrics import metrics_from_hist  # noqa: E402

NUM_CLASSES = 5
RECIPE = {
    "ViT-G/16/1024": dict(batch=4, accum=2),
    "vmambaB": dict(batch=8, accum=1),
    "ViT-B/16": dict(batch=4, accum=2),     # tests de plomberie uniquement
}
VMAMBA_IMAGENET_PATH = "/data2/sht/checkpoints/vmamba/vssm_base_0229_ckpt_epoch_237.pth"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--perascd-root", required=True)
    p.add_argument("--arch", required=True, choices=list(RECIPE))
    p.add_argument("--pretrained", required=True, help="poids PerA (ViT-G) ou VMamba ImageNet (vmambaB)")
    p.add_argument("--data-root", required=True)
    p.add_argument("--fold-dir", required=True, help="dossier avec train.txt, val.txt, test.txt")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--accum", type=int, default=None)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--power", type=float, default=1.5)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=1.5)
    p.add_argument("--droppath", type=float, default=0.3)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=3701)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--eval-batch", type=int, default=4)
    p.add_argument("--msda", default="auto", choices=["auto", "cuda", "pytorch"])
    p.add_argument("--max-iters", type=int, default=None, help="faisabilité : N micro-itérations puis sortie")
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow-pytorch-msda", action="store_true", help="test CPU de plomberie uniquement")
    p.add_argument("--limit-pairs", type=int, default=None, help="test : n premières paires de chaque liste")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Leur code : modules legacy (avec modules factices si une dépendance d'affichage manque)
# --------------------------------------------------------------------------- #
def import_rs_st(root: Path):
    """datasets.RS_ST importe cv2/skimage/matplotlib au chargement, sans s'en servir dans
    les transformations CDM* utilisées ici. On ne crée un module factice que si la
    dépendance est absente, et on le déclare."""
    stubbed = []
    for name in ("cv2", "skimage", "skimage.io", "skimage.transform", "matplotlib", "matplotlib.pyplot"):
        try:
            importlib.import_module(name)
        except ImportError:
            sys.modules[name] = types.ModuleType(name)
            stubbed.append(name)
    if "skimage" in stubbed:
        sys.modules["skimage"].io = sys.modules["skimage.io"]
        sys.modules["skimage"].transform = sys.modules["skimage.transform"]
    sys.path.insert(0, str(root))
    from datasets import RS_ST as RS
    from utils.loss import CrossEntropyLoss2d, SoftSemanticConsistency, weighted_BCE_logits
    return RS, CrossEntropyLoss2d, SoftSemanticConsistency, weighted_BCE_logits, stubbed


class FoldPairs(Dataset):
    """Paires `<split>/<nom>` d'un pli, transformées par LEURS classes CDM*."""

    def __init__(self, root, ids_file, train, mean, std, RS, limit=None):
        self.root = Path(root)
        self.ids = [l.strip() for l in Path(ids_file).read_text().splitlines() if l.strip()][:limit]
        steps = ([RS.CDMRandomFlipRotate(), RS.CDMColorJitter(brightness=0.2, contrast=0.2,
                                                                saturation=0.1, hue=0.1)] if train else [])
        self.trans = RS.CDMCompose(steps + [RS.CDMToTensor(), RS.CDMNormalize(mean=list(mean), std=list(std))])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        split, name = self.ids[i].split("/")
        d = self.root / split
        a, b, la, lb = (Image.open(d / f / name) for f in ("im1", "im2", "label1", "label2"))
        a, b = a.convert("RGB"), b.convert("RGB")
        return (*self.trans(a, b, la, lb), self.ids[i])


def adjust_lr(optimizer, iter_ratio, init_lr, warmup, min_lr, power):
    """Copie de leur adjust_lr (train.py, branche legacy)."""
    if iter_ratio < warmup:
        lr = init_lr * iter_ratio / warmup
    else:
        lr = min_lr + (init_lr - min_lr) * (((1. - iter_ratio) / (1. - warmup)) ** power)
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def build_model(args, PerASCD):
    """Modèle + rapport de chargement des poids pré-entraînés (leurs chargeurs sont en
    strict=False : on vérifie nous-mêmes que les poids sont bien arrivés)."""
    if args.arch == "vmambaB":
        if "models.SatMAE_temporal" not in sys.modules:
            stub = types.ModuleType("models.SatMAE_temporal")
            stub.get_1d_sincos_pos_embed_from_grid_torch = stub.mae_vit_large_patch16 = None
            sys.modules["models.SatMAE_temporal"] = stub
        from models.Encoders import build_net
        real_load = torch.load
        seen = {}

        def _load(path, *a, **k):
            if str(path) == VMAMBA_IMAGENET_PATH:
                seen["path"] = args.pretrained
                obj = real_load(args.pretrained, *a, **{**k, "weights_only": False})
                seen["state"] = obj["model"] if "model" in obj else obj
                return obj
            return real_load(path, *a, **k)
        torch.load = _load
        try:
            net = build_net("vmambaB", NUM_CLASSES, output_size=512, drop_rate=args.droppath)
        finally:
            torch.load = real_load
        if "state" not in seen:
            raise SystemExit("⛔ build_net n'a pas demandé les poids ImageNet VMamba-B attendus")
        enc_sd = net.encoder.state_dict() if hasattr(net, "encoder") else None
        pre = seen["state"]
        target = enc_sd if enc_sd is not None else net.state_dict()
        hit = [k for k in pre if k in target and target[k].shape == pre[k].shape]
        report = {"pretrained_tensors": len(pre), "loaded": len(hit)}
        same = [k for k in hit if torch.equal(target[k].float().cpu(), pre[k].float().cpu())]
        report["verified_equal"] = len(same)
        if len(same) < 0.9 * len(pre):
            raise SystemExit(f"⛔ poids ImageNet mal chargés : {report}")
        return net, report
    net = PerASCD(in_channels=3, num_classes=NUM_CLASSES, input_size=448, output_size=512,
                  arch=args.arch, droppath=args.droppath, pretrained_pera_path=args.pretrained,
                  is_distilled_pera=False, is_freeze_backbone=False)
    pre = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    pre = {k.replace("teacher.backbone.", ""): v for k, v in pre["model"].items() if k.startswith("teacher.backbone.")}
    sd = net.backbone.state_dict()
    hit = [k for k in pre if k in sd and sd[k].shape == pre[k].shape]
    same = [k for k in hit if torch.equal(sd[k].float(), pre[k].float())]
    report = {"pretrained_tensors": len(pre), "loaded": len(hit), "verified_equal": len(same),
              "not_in_model": sorted(set(pre) - set(sd))[:10]}
    if not pre or len(same) < 0.9 * len(pre):
        raise SystemExit(f"⛔ poids PerA mal chargés : {report}")
    return net, report


@torch.no_grad()
def evaluate(net, loader, device, legacy_eval, get_hist):
    net.eval()
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for a, b, la, lb, _ in loader:
        ch, oa, ob = net(a.to(device).float(), b.to(device).float())
        cm = (torch.sigmoid(ch) > 0.5).squeeze(1).long()
        pa = (oa.argmax(1) * cm).cpu().numpy(); pb = (ob.argmax(1) * cm).cpu().numpy()
        la, lb = la.numpy().astype(np.int64), lb.numpy().astype(np.int64)
        for i in range(len(pa)):
            hist += get_hist(pa[i], la[i], NUM_CLASSES) + get_hist(pb[i], lb[i], NUM_CLASSES)
    fscd, miou, sek = legacy_eval(hist)
    ours = metrics_from_hist(hist)
    assert abs(ours.sek - sek) < 1e-6 and abs(ours.fscd - fscd) < 1e-6, (ours, sek, fscd)
    net.train()
    return {"sek": float(sek), "fscd": float(fscd), "miou": float(miou), "oa": ours.oa}


def main():
    args = parse_args()
    rec = RECIPE[args.arch]
    args.batch_size = args.batch_size or rec["batch"]
    args.accum = args.accum or rec["accum"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = args.device
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    fold = Path(args.fold_dir)
    lists = {s: [l.strip() for l in (fold / f"{s}.txt").read_text().splitlines() if l.strip()]
             for s in ("train", "val", "test")}
    locs = assert_location_disjoint(**lists)

    ev.set_num_classes(NUM_CLASSES)
    norm = ev.set_normalization(args.arch)
    PerASCD, legacy_eval, get_hist, compiled = ev.import_legacy(Path(args.perascd_root).resolve(), args.msda)
    if args.arch != "vmambaB" and not compiled and not args.allow_pytorch_msda:
        raise SystemExit("⛔ MultiScaleDeformableAttention non compilé : entraînement ViT-G impossible "
                         "(le repli PyTorch est trop lent et n'a pas de rétropropagation testée)")
    RS, CE2d, SSC, wBCE, stubbed = import_rs_st(Path(args.perascd_root).resolve())

    net, load_report = build_model(args, PerASCD)
    net = net.to(device)
    n_params = sum(p.numel() for p in net.parameters())
    cfg = {**vars(args), "norm": norm, "locations": {k: sorted(v) for k, v in locs.items()},
           "pairs": {k: len(v) for k, v in lists.items()}, "params_M": round(n_params / 1e6, 2),
           "pretrained_load": load_report, "stubbed_modules": stubbed,
           "msda": "cuda" if compiled else "pytorch",
           "deviations": ["sélection sur SeK de val du pli (eux : Fscd du test)", "pas de pseudo-labels"]}
    (out / "config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({k: cfg[k] for k in ("arch", "params_M", "pairs", "locations", "pretrained_load", "norm")},
                     ensure_ascii=False, default=str), flush=True)

    mk = lambda s, train: FoldPairs(args.data_root, fold / f"{s}.txt", train, norm["mean"], norm["std"], RS,
                                    limit=args.limit_pairs)
    train_loader = DataLoader(mk("train", True), batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    eval_loaders = {s: DataLoader(mk(s, False), batch_size=args.eval_batch, shuffle=False,
                                  num_workers=args.workers, pin_memory=True) for s in ("val", "test")}

    criterion = CE2d(ignore_index=0).to(device)
    criterion_sc = SSC(reduction="mean", tau=args.tau).to(device)
    optimizer = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=args.lr,
                                weight_decay=args.wd, momentum=0.9, nesterov=True)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    start_epoch, best = 0, -1.0
    if (out / "last.pt").exists() and args.max_iters is None:
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck["best_sek"]
        print(f"Reprise depuis {out / 'last.pt'} : époque {start_epoch}, meilleure SeK val {best:.4f}", flush=True)

    all_iters = float(len(train_loader) * args.epochs)
    net.train()
    t_feas = None
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        losses = []
        for i, (a, b, la, lb, _) in enumerate(train_loader):
            running = epoch * len(train_loader) + i + 1
            lr = adjust_lr(optimizer, running / all_iters, args.lr, args.warmup, args.min_lr, args.power)
            a = a.to(device, non_blocking=True).float(); b = b.to(device, non_blocking=True).float()
            bn = (la > 0).unsqueeze(1).to(device, non_blocking=True).float()
            la = la.to(device, non_blocking=True).long(); lb = lb.to(device, non_blocking=True).long()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                ch, oa, ob = net(a, b)
                assert oa.shape[1] == NUM_CLASSES
                loss_seg = criterion(oa, la) + criterion(ob, lb)
                loss_bn = wBCE(ch, bn)
                loss_sc = criterion_sc(oa[:, 1:], ob[:, 1:], bn)
                loss = loss_seg * 0.5 + loss_bn + loss_sc
                if args.accum > 1:
                    loss = loss / args.accum
            scaler.scale(loss).backward()
            if (i + 1) % args.accum == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                if args.clip:
                    nn.utils.clip_grad_norm_(net.parameters(), args.clip)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss) * args.accum)
            if not math.isfinite(losses[-1]):
                raise SystemExit(f"⛔ loss non finie à l'époque {epoch}, itération {i}")
            if args.max_iters is not None:
                sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)
                if i == 4:
                    sync(); t_feas = time.time()
                if i + 1 >= args.max_iters:
                    sync()
                    s_it = (time.time() - t_feas) / (i - 4)
                    per_epoch = s_it * len(train_loader)
                    rep = {"arch": args.arch, "batch": args.batch_size, "accum": args.accum,
                           "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.startswith("cuda") else None,
                           "s_per_micro_iter": s_it, "micro_iters_per_epoch": len(train_loader),
                           "train_h_per_epoch": per_epoch / 3600,
                           "train_h_total_projected": per_epoch * args.epochs / 3600,
                           "loss_first_last": [losses[0], losses[-1]]}
                    (out / "feasibility.json").write_text(json.dumps(rep, indent=1))
                    print(json.dumps(rep, indent=1), flush=True)
                    return
        t_train = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        t1 = time.time()
        res = {s: evaluate(net, l, device, legacy_eval, get_hist) for s, l in eval_loaders.items()}
        t_eval = time.time() - t1
        for s, r in res.items():
            f = out / ("metrics.csv" if s == "val" else "metrics_test.csv")
            if not f.exists():
                f.write_text("epoch,sek,fscd,miou,oa,lr,loss\n")
            with f.open("a") as fh:
                fh.write(f"{epoch},{r['sek']:.5f},{r['fscd']:.5f},{r['miou']:.5f},{r['oa']:.5f},{lr:.6f},{np.mean(losses):.5f}\n")
        tf = out / "timing.csv"
        if not tf.exists():
            tf.write_text("epoch,train_s,eval_s,peak_train_gb\n")
        with tf.open("a") as fh:
            fh.write(f"{epoch},{t_train:.1f},{t_eval:.1f},{peak:.2f}\n")
        print(f"[époque {epoch}] loss {np.mean(losses):.4f} lr {lr:.4f} | val SeK {res['val']['sek']:.4f} "
              f"Fscd {res['val']['fscd']:.4f} | test SeK {res['test']['sek']:.4f} | "
              f"{t_train:.0f} s + {t_eval:.0f} s, {peak:.1f} Go", flush=True)
        if res["val"]["sek"] > best:
            best = res["val"]["sek"]
            torch.save({"epoch": epoch, "model": net.state_dict(), "Sek": res["val"]["sek"],
                        "Fscd": res["val"]["fscd"], "mIoU": res["val"]["miou"], "selected_on": "fold val SeK",
                        "test_at_selection": res["test"]}, out / "best.pt")
            print(f"  -> best.pt (SeK val {best:.4f})", flush=True)
        torch.save({"epoch": epoch, "model": net.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "best_sek": best}, out / "last.pt")


if __name__ == "__main__":
    main()
