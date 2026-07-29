"""Statistiques d'un dataset SCD : ce que contiennent réellement les annotations.

Répond à trois questions, sur un échantillon ALÉATOIRE (et non les N premières
tuiles, qui sont spatialement contiguës et donc non représentatives) :

  1. quelle proportion de pixels change ?
  2. quelles classes apparaissent dans les zones changées, et dans quelles
     proportions ?
  3. quelles TRANSITIONS (classe T1 -> classe T2) se produisent réellement, et
     combien de types distincts ?

La lecture est parallélisée (DataLoader + workers) : à lancer en job, pas sur un
nœud de connexion.

    python -m scripts.dataset_stats --data-root <...> --dataset hi_ucd --split train
"""

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from csf_mamba.datasets import DATASETS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--dataset", default="hi_ucd", choices=sorted(DATASETS))
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--max-samples", type=int, default=1500)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def analyse(dataset, num_classes, max_samples, num_workers, seed):
    random.seed(seed)
    n = min(len(dataset), max_samples)
    indices = random.sample(range(len(dataset)), n)
    loader = DataLoader(Subset(dataset, indices), batch_size=4,
                        num_workers=num_workers, shuffle=False)

    class_counts = torch.zeros(num_classes, dtype=torch.float64)
    transitions = torch.zeros(num_classes, num_classes, dtype=torch.float64)
    tiles_with_change = 0
    changed_px = 0
    valid_px = 0

    for batch in loader:
        change = batch["change"]
        changed = change == 1
        valid = change != 255
        changed_px += changed.sum().item()
        valid_px += valid.sum().item()
        tiles_with_change += changed.flatten(1).any(dim=1).sum().item()

        a, b = batch["sem_t1"][changed], batch["sem_t2"][changed]
        keep = (a < num_classes) & (b < num_classes)
        a, b = a[keep], b[keep]
        if a.numel():
            class_counts += torch.bincount(a, minlength=num_classes).double()
            class_counts += torch.bincount(b, minlength=num_classes).double()
            flat = a.long() * num_classes + b.long()
            transitions += torch.bincount(
                flat, minlength=num_classes ** 2
            ).double().reshape(num_classes, num_classes)

    return dict(n=n, class_counts=class_counts, transitions=transitions,
                tiles_with_change=tiles_with_change,
                change_ratio=100 * changed_px / max(1, valid_px))


def report(name, res, num_classes):
    print(f"\n===== {name} — {res['n']} tuiles tirées au hasard =====")
    print(f"  pixels changés          : {res['change_ratio']:.2f} %")
    print(f"  tuiles avec du changement : {res['tiles_with_change']} / {res['n']} "
          f"({100*res['tiles_with_change']/res['n']:.1f} %)")

    counts = res["class_counts"]
    total = counts.sum()
    if total == 0:
        print("  (aucun pixel changé annoté)")
        return
    pct = 100 * counts / total
    print("\n  classes dans les zones changées :")
    for c in range(1, num_classes):
        marker = "  <-- dominante" if pct[c] == pct[1:].max() else ""
        print(f"    classe {c}: {pct[c]:6.2f} %{marker}")
    print(f"  classes réellement présentes : {(counts[1:] > 0).sum().item()} / {num_classes-1}")

    tr = res["transitions"]
    tr_total = tr.sum()
    off = tr.clone()
    off.fill_diagonal_(0)                    # vraies transitions (classe qui change)
    n_types = int((off > 0).sum().item())
    print(f"\n  types de transition observés (classe T1 != T2) : {n_types}")
    print(f"  part des pixels changés SANS changement de classe : "
          f"{100*tr.diagonal().sum()/max(1.0, tr_total.item()):.1f} %")
    flat = [(off[i, j].item(), i, j) for i in range(num_classes) for j in range(num_classes)]
    flat.sort(reverse=True)
    print("  transitions les plus fréquentes :")
    for v, i, j in flat[:8]:
        if v == 0:
            break
        print(f"    classe {i} -> classe {j} : {100*v/max(1.0, off.sum().item()):5.2f} %")


def main():
    args = parse_args()
    dataset_cls, num_classes = DATASETS[args.dataset]
    for split in args.splits:
        ds = dataset_cls(args.data_root, split=split)
        res = analyse(ds, num_classes, args.max_samples, args.num_workers, args.seed)
        report(f"{args.dataset} / {split} ({len(ds)} tuiles au total)", res, num_classes)


if __name__ == "__main__":
    main()
