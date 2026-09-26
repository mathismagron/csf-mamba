"""Étape P du plan de distillation : l'existant tourne-t-il encore à l'identique ?

Réévalue les `best.pt` des 7 graines du point d'efficience sur SECOND test, avec
**la fonction de validation de l'entraînement** (`scripts.train.validate`), et
compare au maximum de leur `metrics.csv`. Si l'environnement, le code ou les
données avaient dérivé depuis septembre, l'écart le montrerait.

    python -m scripts.check_narval --out $SCRATCH/csf-distill/checks/step_P.json

Critère (documentation/distillation.md, étape P) : |ΔSeK| ≤ 1e-4 en bf16 — la
précision utilisée à la validation pendant l'entraînement — pour les 7 graines.
Le fp32 est rapporté à titre d'information : il n'a jamais servi à sélectionner.

Configuration reconstituée depuis les logs Slurm (`== config`), identique pour
les 7 graines : enc=vmamba_mini dec=dw fusion=concat cga=1 mcasf=1 up=dysample
core=chess fft=[0,1]. Le micro-batch de validation valait 2 : on le garde.
"""

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from csf_mamba.datasets import DATASETS
from csf_mamba.model import CSFMamba, count_parameters
from scripts.train import validate

SEEDS = (1, 2, 3, 4, 5, 6, 42)
TAG = "second_mini_chess_crop512-lean-augswap-ema-s{seed}"
TOL = 1e-4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", default=None, help="défaut : $SCRATCH/csf-mamba-runs")
    p.add_argument("--data-root", default=None, help="défaut : $SCRATCH/SECOND")
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p.add_argument("--precisions", default="bf16,fp32")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--out", required=True)
    return p.parse_args()


def best_row(metrics_csv: Path) -> dict:
    rows = list(csv.DictReader(metrics_csv.open()))
    best = max(rows, key=lambda r: float(r["sek"]))
    return {k: (int(v) if k == "epoch" else float(v)) for k, v in best.items()}


def build_model(device):
    return CSFMamba(
        num_semantic_classes=DATASETS["second"][1], encoder="vmamba_mini",
        core="chess", backend="mamba", decoder_refine="dw", fft_stages=(0, 1),
        fusion="concat", cga=True, mcasf=True, upsample="dysample",
    ).to(device)


@torch.no_grad()
def main():
    args = parse_args()
    import os
    scratch = Path(os.environ["SCRATCH"])
    runs = Path(args.runs_root) if args.runs_root else scratch / "csf-mamba-runs"
    data = Path(args.data_root) if args.data_root else scratch / "SECOND"
    device = "cuda"
    assert torch.cuda.is_available(), "à lancer dans un job GPU"

    ds_cls, num_classes = DATASETS["second"]
    loader = DataLoader(ds_cls(str(data), split="test"), batch_size=args.batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)
    precisions = [p for p in args.precisions.split(",") if p]
    model = build_model(device)
    report = {
        "step": "P", "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": platform.node(), "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "n_test": len(loader.dataset), "params": count_parameters(model),
        "tolerance": TOL, "seeds": {},
    }
    print(f"{len(loader.dataset)} paires de test, {report['gpu']}, torch {torch.__version__}")

    for seed in (int(s) for s in args.seeds.split(",")):
        run = runs / TAG.format(seed=seed)
        ref = best_row(run / "metrics.csv")
        state = torch.load(run / "best.pt", map_location=device)
        model.load_state_dict(state, strict=True)   # lève si un seul tenseur diffère
        entry = {"run": run.name, "csv_best": ref, "eval": {}}
        for prec in precisions:
            t0 = time.time()
            m = validate(model, loader, device, num_classes, use_amp=(prec == "bf16"))
            entry["eval"][prec] = {
                "sek": m.sek, "fscd": m.fscd, "miou": m.miou, "oa": m.oa, "kappa": m.kappa,
                "delta_sek": m.sek - ref["sek"], "seconds": round(time.time() - t0, 1),
            }
            print(f"s{seed:<3} {prec}: SeK {m.sek:.5f} (csv {ref['sek']:.5f}, époque {ref['epoch']}) "
                  f"Δ {m.sek - ref['sek']:+.5f}")
        entry["pass"] = abs(entry["eval"].get("bf16", {"delta_sek": 1})["delta_sek"]) <= TOL
        report["seeds"][str(seed)] = entry

    report["pass"] = all(e["pass"] for e in report["seeds"].values())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print("ÉTAPE P :", "PASS" if report["pass"] else "ÉCHEC", "->", out)


if __name__ == "__main__":
    main()
