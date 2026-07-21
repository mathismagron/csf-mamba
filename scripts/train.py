"""Boucle d'entraînement CSF-Mamba.

Squelette minimal et honnête : il câble modèle + loss + données et fait tourner
une époque. Ce qui reste à brancher avant les vrais runs est marqué TODO — pas
masqué. Lancer via `python -m scripts.train ...` (voir train.sbatch).
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from csf_mamba.datasets.hi_ucd import NUM_SEMANTIC_CLASSES, HiUCDDataset
from csf_mamba.evaluation.metrics import SCDEvaluator, SCDMetrics
from csf_mamba.losses.composite import CSFMambaLoss
from csf_mamba.model import CSFMamba, count_parameters


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--dataset", default="hi_ucd", choices=["hi_ucd"])
    p.add_argument("--encoder", default="conv", choices=["conv", "vmamba_mini", "vmamba_tiny"])
    p.add_argument("--encoder-pretrained", default=None,
                   help="Chemin du checkpoint ImageNet VMamba (cf. download_pretrained.sh)")
    p.add_argument("--core", default="chess", choices=["chess", "l1"])
    p.add_argument("--backend", default="auto", choices=["auto", "mamba", "ref"])
    p.add_argument("--val-split", default="val")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--limit-batches", type=int, default=None,
                   help="Plafonne le nb de batches train/val par époque (run de test).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="runs/dev")
    return p.parse_args()


def build_dataset(args, split):
    if args.dataset == "hi_ucd":
        return HiUCDDataset(args.data_root, split=split)
    raise ValueError(args.dataset)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.output).mkdir(parents=True, exist_ok=True)

    encoder_kwargs = {}
    if args.encoder_pretrained and args.encoder.startswith("vmamba"):
        encoder_kwargs["pretrained_path"] = args.encoder_pretrained

    model = CSFMamba(
        num_semantic_classes=NUM_SEMANTIC_CLASSES,
        encoder=args.encoder, core=args.core, backend=args.backend,
        encoder_kwargs=encoder_kwargs,
    ).to(device)
    print("Paramètres :", count_parameters(model))

    criterion = CSFMambaLoss(num_semantic_classes=NUM_SEMANTIC_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    train_loader = DataLoader(
        build_dataset(args, "train"), batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        build_dataset(args, args.val_split), batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    best_sek = -1.0
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            if args.limit_batches is not None and step >= args.limit_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["img_t1"], batch["img_t2"])
            losses = criterion(outputs, _targets_from_batch(batch))

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            if step % 20 == 0:
                flat = {k: round(v.item(), 4) for k, v in losses.items()}
                print(f"epoch {epoch} step {step} {flat}")

        metrics = validate(model, val_loader, device, limit=args.limit_batches)
        print(f"[val] epoch {epoch} | SeK {metrics.sek:.4f} Fscd {metrics.fscd:.4f} "
              f"mIoU {metrics.miou:.4f} OA {metrics.oa:.4f} kappa {metrics.kappa:.4f}")

        torch.save(model.state_dict(), Path(args.output) / f"epoch_{epoch}.pt")
        if metrics.sek > best_sek:
            best_sek = metrics.sek
            torch.save(model.state_dict(), Path(args.output) / "best.pt")
            print(f"  -> nouveau meilleur SeK {best_sek:.4f}, sauvé dans best.pt")


def _targets_from_batch(batch: dict) -> dict:
    return {
        "change": batch["change"],
        "sem_t1": batch["sem_t1"], "sem_t2": batch["sem_t2"],
        "unchanged": batch["unchanged"],
    }


@torch.no_grad()
def validate(model, loader, device, limit=None) -> SCDMetrics:
    model.eval()
    evaluator = SCDEvaluator(num_classes=NUM_SEMANTIC_CLASSES)
    for step, batch in enumerate(loader):
        if limit is not None and step >= limit:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch["img_t1"], batch["img_t2"])
        evaluator.add(outputs, _targets_from_batch(batch))
    return evaluator.compute()


if __name__ == "__main__":
    main()
