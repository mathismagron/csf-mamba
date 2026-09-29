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
    Path(args.out).write_text(json.dumps(report, indent=1))
    print("->", args.out)


if __name__ == "__main__":
    main()
