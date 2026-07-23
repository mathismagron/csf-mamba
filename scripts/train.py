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
from csf_mamba.datasets.transforms import train_transform
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
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--limit-batches", type=int, default=None,
                   help="Plafonne le nb de batches train/val par époque (run de test).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--crop-size", type=int, default=256,
                   help="Crop d'entraînement (256 = rapide). 0 = pleine résolution 512.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-6, help="Plancher de la décroissance cosine.")
    p.add_argument("--warmup-iters", type=int, default=1500, help="Montée linéaire du LR.")
    p.add_argument("--sek-warmup-iters", type=int, default=20000,
                   help="Itérations avant d'activer la loss SeK (la sémantique apprend d'abord).")
    p.add_argument("--bcd-change-weight", type=float, default=10.0,
                   help="Poids de la classe 'changement' (rare) dans la loss BCD. Contre le déséquilibre.")
    p.add_argument("--amp", action="store_true", default=True, help="Precision mixte bf16 (défaut).")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="runs/dev")
    p.add_argument("--resume", default="auto",
                   help="'auto' reprend last.pt de --output ; un chemin ; '' pour repartir de zéro.")
    return p.parse_args()


def build_dataset(args, split):
    if args.dataset != "hi_ucd":
        raise ValueError(args.dataset)
    # Crop + augmentation à l'entraînement ; validation en pleine résolution.
    transform = train_transform(args.crop_size) if split == "train" else None
    return HiUCDDataset(args.data_root, split=split, transform=transform)


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

    criterion = CSFMambaLoss(
        num_semantic_classes=NUM_SEMANTIC_CLASSES,
        bcd_change_weight=args.bcd_change_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    train_loader = DataLoader(
        build_dataset(args, "train"), batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        build_dataset(args, args.val_split), batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    steps_per_epoch = args.limit_batches or len(train_loader)
    total_iters = args.epochs * steps_per_epoch
    scheduler = _warmup_cosine(optimizer, args.warmup_iters, total_iters, args.lr, args.min_lr)
    use_amp = args.amp and device == "cuda"

    out_dir = Path(args.output)
    best_sek, start_epoch, global_step = -1.0, 0, 0
    # Reprise : --resume auto -> reprend last.pt du même --output si présent.
    resume_path = out_dir / "last.pt" if args.resume == "auto" else (
        Path(args.resume) if args.resume else None
    )
    if resume_path and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_sek = ckpt["best_sek"]
        global_step = ckpt["global_step"]
        print(f"Reprise depuis {resume_path} : époque {start_epoch}, step {global_step}, "
              f"best SeK {best_sek:.4f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            if args.limit_batches is not None and step >= args.limit_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            apply_sek = global_step >= args.sek_warmup_iters

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(batch["img_t1"], batch["img_t2"])
            # Loss en fp32 (SeK/log sensibles à la précision) : on caste les sorties.
            outputs = {k: (v.float() if torch.is_tensor(v) else v) for k, v in outputs.items()}
            losses = criterion(outputs, _targets_from_batch(batch), apply_sek=apply_sek)

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            if step % 50 == 0:
                flat = {k: round(v.item(), 4) for k, v in losses.items()}
                lr = scheduler.get_last_lr()[0]
                print(f"epoch {epoch} step {step} lr {lr:.2e} sek={'on' if apply_sek else 'off'} {flat}")

        metrics = validate(model, val_loader, device, limit=args.limit_batches, use_amp=use_amp)
        print(f"[val] epoch {epoch} | SeK {metrics.sek:.4f} Fscd {metrics.fscd:.4f} "
              f"mIoU {metrics.miou:.4f} OA {metrics.oa:.4f} kappa {metrics.kappa:.4f}")

        # Log CSV persistant (à côté des checkpoints : survit à un rm du .out).
        csv_path = out_dir / "metrics.csv"
        if not csv_path.exists():
            csv_path.write_text("epoch,sek,fscd,miou,oa,kappa\n")
        with csv_path.open("a") as f:
            f.write(f"{epoch},{metrics.sek:.5f},{metrics.fscd:.5f},"
                    f"{metrics.miou:.5f},{metrics.oa:.5f},{metrics.kappa:.5f}\n")

        # Checkpoint complet (reprise possible) écrasé à chaque époque.
        ckpt = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch,
            "best_sek": best_sek, "global_step": global_step,
        }
        if metrics.sek > best_sek:
            best_sek = metrics.sek
            ckpt["best_sek"] = best_sek
            torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"  -> nouveau meilleur SeK {best_sek:.4f}, sauvé dans best.pt")
        torch.save(ckpt, out_dir / "last.pt")


def _warmup_cosine(optimizer, warmup_iters, total_iters, base_lr, min_lr):
    """Montée linéaire jusqu'à base_lr, puis décroissance cosine jusqu'à min_lr."""
    import math

    def lr_lambda(it):
        if it < warmup_iters:
            return (it + 1) / max(1, warmup_iters)
        progress = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        return (min_lr + (base_lr - min_lr) * cosine) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _targets_from_batch(batch: dict) -> dict:
    return {
        "change": batch["change"],
        "sem_t1": batch["sem_t1"], "sem_t2": batch["sem_t2"],
        "unchanged": batch["unchanged"],
    }


@torch.no_grad()
def validate(model, loader, device, limit=None, use_amp=False) -> SCDMetrics:
    model.eval()
    evaluator = SCDEvaluator(num_classes=NUM_SEMANTIC_CLASSES)
    for step, batch in enumerate(loader):
        if limit is not None and step >= limit:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(batch["img_t1"], batch["img_t2"])
        evaluator.add(outputs, _targets_from_batch(batch))
    return evaluator.compute()


if __name__ == "__main__":
    main()
