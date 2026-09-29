"""Diagnostic : pourquoi le professeur VMamba-B ne reproduit-il pas son score publié ?

Job 4223414 : SeK test 23,73 (fp32) contre 25,31 dans leur journal TensorBoard à
l'époque 42 (celle du checkpoint), alors que le même pipeline reproduit PerASCD
ViT-G à 1e-3 pt près. Données, métrique, normalisation et chargement (strict) sont
donc hors de cause ; reste le calcul de l'encodeur VMamba. Hypothèses testées ici,
chacune sur les MÊMES paires de test :

  default     noyau `selective_scan_cuda` (mamba_ssm) + cross-scan triton ;
  tf32_off    idem, TF32 coupé aussi pour cuDNN (convolutions) ;
  torch_scan  scan sélectif de référence en PyTorch (leur `selective_scan_torch`) ;
  torch_all   scan ET cross-scan/merge en PyTorch : aucune extension compilée.

Pour chaque variante : écart max des logits à `default` sur le premier lot, puis SeK /
Fscd legacy sur les `--pairs` premières paires du test.

Job 4257053 : les quatre variantes de calcul donnent exactement la même SeK — les noyaux
sont hors de cause. `--norms` teste alors la normalisation d'entrée (leur dépôt en a
plusieurs ; la classe utilisée à l'entraînement a pu changer, cf. les lignes commentées
de `train_Encoders.py`) :

  pera      moyenne/écart-type PerA sur [0, 1] (DataPerAAUG, celle de l'étape 0) ;
  pertime   moyenne/écart-type par date sur [0, 255] (MEAN_A/STD_A, MEAN_B/STD_B de
            `datasets/RS_ST.py`, classes `Data` / `Data_test`) ;
  imagenet  moyenne/écart-type ImageNet sur [0, 1] (prétraining VMamba) ;
  raw       images [0, 1] sans normalisation.
  dsstats   moyenne/écart-type commentés dans DataPerAAUG (l. 548-549), sur [0, 1].

Job 4263233 (256 paires) : pera 22.744, pertime 23.144, imagenet 23.858, raw 10.275.

    python scripts/teacher/diag_vmambab.py --perascd-root third_party/PerASCD \\
        --checkpoint .../vmambaB_42e_...pth --data-root $SLURM_TMPDIR/SECOND --pairs 256 --out diag.json
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("eval_perascd", HERE / "eval_perascd.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--perascd-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--pairs", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--variants", default="default,tf32_off,torch_scan,torch_all")
    p.add_argument("--norms", default="", help="p. ex. pera,pertime,imagenet,raw (noyaux par défaut)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    PerASCD, legacy_eval, get_hist, _ = ev.import_legacy(Path(args.perascd_root).resolve(), "pytorch")
    model, info = ev.build_teacher(PerASCD, Path(args.perascd_root).resolve(), "vmambaB", args.checkpoint)
    model = model.cuda().eval()
    vm = sys.modules["models.vmamba"]
    flags0 = {"WITH_SELECTIVESCAN_MAMBA": vm.WITH_SELECTIVESCAN_MAMBA, "WITH_TRITON": vm.WITH_TRITON}
    print("drapeaux de leur vmamba.py :", flags0, "| TF32 matmul", torch.backends.cuda.matmul.allow_tf32,
          "cudnn", torch.backends.cudnn.allow_tf32, flush=True)

    ds = ev.SecondRaw(args.data_root, "test")
    sub = Subset(ds, range(min(args.pairs, len(ds))))
    loader = DataLoader(sub, batch_size=args.batch_size, shuffle=False, num_workers=4)
    first = next(iter(loader))
    a0, b0 = first["img_a"].cuda(), first["img_b"].cuda()

    def setup(name):
        vm.WITH_SELECTIVESCAN_MAMBA = flags0["WITH_SELECTIVESCAN_MAMBA"] and name not in ("torch_scan", "torch_all")
        vm.WITH_TRITON = flags0["WITH_TRITON"] and name != "torch_all"
        torch.backends.cudnn.allow_tf32 = name != "tf32_off"

    report = {"checkpoint": info.get("path"), "flags": flags0, "pairs": len(sub), "variants": {}}
    ref = None
    for name in [v for v in args.variants.split(",") if v]:
        setup(name)
        with torch.no_grad():
            outs = ev.forward(model, a0, b0, "fp32", "cuda")
        if ref is None:
            ref = outs
        gap = [float((o - r).abs().max()) for o, r in zip(outs, ref)]
        t0 = time.time()
        r = ev.evaluate(model, loader, "fp32", "cuda", legacy_eval, get_hist, None)
        report["variants"][name] = {"logit_gap_vs_default": gap, "sek": r["legacy"]["sek"],
                                    "fscd": r["legacy"]["fscd"], "miou": r["legacy"]["miou"],
                                    "seconds": round(time.time() - t0, 1)}
        print(f"{name:11s} écart logits (chg, A, B) {[f'{g:.2e}' for g in gap]} | "
              f"SeK {100 * r['legacy']['sek']:.3f} Fscd {100 * r['legacy']['fscd']:.3f} "
              f"sur {len(sub)} paires | {time.time() - t0:.0f} s", flush=True)
    NORMS = {
        "pera": (torch.tensor(ev.PERA_MEAN), torch.tensor(ev.PERA_STD)) * 2,
        "pertime": (torch.tensor([113.40, 114.08, 116.45]) / 255, torch.tensor([48.30, 46.27, 48.14]) / 255,
                    torch.tensor([111.07, 114.04, 118.18]) / 255, torch.tensor([49.41, 47.01, 47.94]) / 255),
        "imagenet": (torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])) * 2,
        "raw": (torch.zeros(3), torch.ones(3)) * 2,
        # troisième jeu, commenté dans DataPerAAUG (RS_ST.py l. 548-549)
        "dsstats": (torch.tensor([0.4182007312774658, 0.4214799106121063, 0.3991275727748871]),
                    torch.tensor([0.28774282336235046, 0.27541765570640564, 0.2764017581939697])) * 2,
    }
    real_forward = ev.forward
    report["norms"] = {}
    for name in [n for n in args.norms.split(",") if n]:
        setup("default")
        ma, sa, mb, sb = (t.view(1, 3, 1, 1).cuda() for t in NORMS[name])

        def fwd(model_, a, b, prec, device, ma=ma, sa=sa, mb=mb, sb=sb):
            # ev.forward normalise en PerA : on lui passe des images dont la
            # normalisation PerA redonne exactement la normalisation testée.
            pm = torch.tensor(ev.PERA_MEAN, device=a.device).view(1, 3, 1, 1)
            ps = torch.tensor(ev.PERA_STD, device=a.device).view(1, 3, 1, 1)
            a2 = (a - ma) / sa * ps + pm
            b2 = (b - mb) / sb * ps + pm
            return real_forward(model_, a2, b2, prec, device)
        ev.forward = fwd
        t0 = time.time()
        r = ev.evaluate(model, loader, "fp32", "cuda", legacy_eval, get_hist, None)
        ev.forward = real_forward
        report["norms"][name] = {"sek": r["legacy"]["sek"], "fscd": r["legacy"]["fscd"],
                                 "miou": r["legacy"]["miou"]}
        print(f"norm {name:9s} SeK {100 * r['legacy']['sek']:.3f} Fscd {100 * r['legacy']['fscd']:.3f} "
              f"mIoU {100 * r['legacy']['miou']:.3f} sur {len(sub)} paires | {time.time() - t0:.0f} s", flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print("->", args.out)


if __name__ == "__main__":
    main()
