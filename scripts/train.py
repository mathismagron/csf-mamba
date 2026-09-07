"""Boucle d'entraînement CSF-Mamba.

Squelette minimal et honnête : il câble modèle + loss + données et fait tourner
une époque. Ce qui reste à brancher avant les vrais runs est marqué TODO — pas
masqué. Lancer via `python -m scripts.train ...` (voir train.sbatch).
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from csf_mamba.datasets import DATASETS
from csf_mamba.datasets.oversample import build_change_index, make_change_sampler
from csf_mamba.datasets.transforms import train_transform
from csf_mamba.ema import ModelEMA
from csf_mamba.evaluation.metrics import SCDEvaluator, SCDMetrics
from csf_mamba.losses.composite import CSFMambaLoss
from csf_mamba.model import CSFMamba, count_parameters


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--dataset", default="hi_ucd", choices=sorted(DATASETS))
    p.add_argument("--encoder", default="conv", choices=["conv", "vmamba_mini", "vmamba_tiny"])
    p.add_argument("--encoder-pretrained", default=None,
                   help="Chemin du checkpoint ImageNet VMamba (cf. download_pretrained.sh)")
    p.add_argument("--core", default="chess", choices=["chess", "l1"])
    p.add_argument("--fusion", default="c2s2", choices=["c2s2", "concat"],
                   help="c2s2 = bloc complet ; concat = ablation TOTALE du C²S² "
                        "(damier + MCA-SF + scan S6) au profit d'un concat 1x1.")
    p.add_argument("--no-cga", dest="cga", action="store_false",
                   help="Retire la Change-Guided Attention du décodeur sémantique.")
    p.add_argument("--no-mcasf", dest="mcasf", action="store_false",
                   help="Retire l'agrégation locale MCA-SF du C²S².")
    p.add_argument("--upsample", default="dysample", choices=["dysample", "bilinear"],
                   help="Rééchantillonnage des décodeurs : appris ou bilinéaire fixe.")
    p.add_argument("--decoder-refine", default="dw", choices=["dw", "full"],
                   help="Raffinement des décodeurs : depthwise (référence) ou 3x3 complet.")
    p.add_argument("--backend", default="auto", choices=["auto", "mamba", "ref"])
    p.add_argument("--val-split", default="val")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--limit-batches", type=int, default=None,
                   help="Plafonne le nb de batches train/val par époque (run de test).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum-steps", type=int, default=1,
                   help="Accumulation de gradient : batch effectif = batch-size x accum-steps. "
                        "Permet de changer la résolution à batch effectif constant.")
    p.add_argument("--crop-size", type=int, default=256,
                   help="Crop d'entraînement (256 = rapide). 0 = pleine résolution 512.")
    p.add_argument("--oversample-change", type=float, default=1.0,
                   help="Poids des tuiles contenant du changement au tirage "
                        "(1 = uniforme). Utile sur Hi-UCD où 9 %% seulement en ont.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-schedule", default="cosine", choices=["cosine", "constant"],
                   help="Après le warmup : décroissance cosine (défaut) ou LR constant.")
    p.add_argument("--min-lr", type=float, default=1e-6, help="Plancher de la décroissance cosine.")
    p.add_argument("--warmup-iters", type=int, default=1500, help="Montée linéaire du LR.")
    p.add_argument("--sek-warmup-iters", type=int, default=20000,
                   help="Itérations avant d'activer la loss SeK (la sémantique apprend d'abord).")
    p.add_argument("--bcd-change-weight", type=float, default=10.0,
                   help="Poids de la classe 'changement' (rare) dans la loss BCD. Contre le déséquilibre.")
    p.add_argument("--lambda-dice", type=float, default=1.0,
                   help="Poids de la loss Dice sur le changement (0 pour désactiver).")
    p.add_argument("--lambda-lovasz", type=float, default=0.0,
                   help="Poids de la loss Lovász sur le changement (optimise l'IoU).")
    p.add_argument("--lambda-deep", type=float, default=0.0,
                   help="Supervision profonde des cartes de changement par stage "
                        "(0 = désactivée, valeur relative au terme BCD principal).")
    p.add_argument("--lambda-sek", type=float, default=0.5,
                   help="Poids du terme SeK différentiable (0 = terme retiré).")
    p.add_argument("--lambda-sc", type=float, default=0.1,
                   help="Poids de la cohérence sémantique L_sc (0 = terme retiré).")
    p.add_argument("--fft-stages", default="0,1",
                   help="Stages portant la branche fréquentielle, séparés par des "
                        "virgules. Chaîne vide = branche FFT retirée.")
    p.add_argument("--lambda-sem-change", type=float, default=0.0,
                   help="Poids de la CE sémantique restreinte aux zones changées "
                        "(0 = désactivé, comme les runs 1-2).")
    p.add_argument("--rot90", action="store_true",
                   help="Ajoute les rotations 90° : groupe diédral complet (8 variantes).")
    p.add_argument("--photometric", type=float, default=0.0,
                   help="Amplitude du jitter luminosité/contraste/saturation, tiré "
                        "INDÉPENDAMMENT par date (0 = désactivé, 0.2 raisonnable).")
    p.add_argument("--temporal-swap", type=float, default=0.0,
                   help="Probabilité d'échanger T1 et T2 (cibles sémantiques avec). "
                        "0 = désactivé. Voir la réserve dans transforms.py : les "
                        "transitions de SECOND sont directionnelles.")
    p.add_argument("--ema-decay", type=float, default=0.0,
                   help="Moyenne mobile exponentielle des poids (0 = désactivée, "
                        "0.9998 raisonnable). La validation et best.pt portent alors "
                        "sur les poids moyennés.")
    p.add_argument("--ema-warmup", type=int, default=2000,
                   help="Itérations de décote de l'EMA au démarrage.")
    p.add_argument("--amp", action="store_true", default=True, help="Precision mixte bf16 (défaut).")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="runs/dev")
    p.add_argument("--resume", default="auto",
                   help="'auto' reprend last.pt de --output ; un chemin ; '' pour repartir de zéro.")
    return p.parse_args()


def build_dataset(args, split):
    dataset_cls, _ = DATASETS[args.dataset]
    # Crop + augmentation à l'entraînement ; validation en pleine résolution.
    transform = train_transform(
        args.crop_size, rot90=args.rot90, photometric=args.photometric,
        temporal_swap=args.temporal_swap,
    ) if split == "train" else None
    return dataset_cls(args.data_root, split=split, transform=transform)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.output).mkdir(parents=True, exist_ok=True)

    _, num_classes = DATASETS[args.dataset]

    encoder_kwargs = {}
    if args.encoder_pretrained and args.encoder.startswith("vmamba"):
        encoder_kwargs["pretrained_path"] = args.encoder_pretrained

    fft_stages = tuple(int(x) for x in args.fft_stages.split(",") if x.strip())
    model = CSFMamba(
        num_semantic_classes=num_classes,
        encoder=args.encoder, core=args.core, backend=args.backend,
        decoder_refine=args.decoder_refine, fft_stages=fft_stages,
        fusion=args.fusion, cga=args.cga, mcasf=args.mcasf,
        upsample=args.upsample,
        encoder_kwargs=encoder_kwargs,
    ).to(device)
    print("Stages FFT :", fft_stages if fft_stages else "aucun (branche retirée)")
    print("Paramètres :", count_parameters(model))

    criterion = CSFMambaLoss(
        num_semantic_classes=num_classes,
        bcd_change_weight=args.bcd_change_weight,
        lambda_dice=args.lambda_dice,
        lambda_deep=args.lambda_deep,
        lambda_sek=args.lambda_sek,
        lambda_sc=args.lambda_sc,
        lambda_sem_change=args.lambda_sem_change,
        lambda_lovasz=args.lambda_lovasz,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    ema = ModelEMA(model, args.ema_decay, args.ema_warmup) if args.ema_decay > 0 else None
    if ema is not None:
        print(f"EMA active : decay {args.ema_decay}, warmup {args.ema_warmup} itérations")

    train_set = build_dataset(args, "train")
    # Sur-échantillonnage des tuiles avec changement (cf. datasets/oversample.py).
    # La VALIDATION reste uniforme : les métriques restent honnêtes.
    train_sampler = None
    if args.oversample_change != 1.0:
        fractions = build_change_index(
            train_set, cache_path=Path(args.output) / "change_index_train.npy"
        )
        train_sampler = make_change_sampler(fractions, args.oversample_change)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        build_dataset(args, args.val_split), batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    micro_per_epoch = args.limit_batches or len(train_loader)
    steps_per_epoch = max(1, micro_per_epoch // args.accum_steps)
    total_iters = args.epochs * steps_per_epoch
    if args.accum_steps > 1:
        print(f"accumulation x{args.accum_steps} : batch effectif "
              f"{args.batch_size * args.accum_steps}, {steps_per_epoch} pas d'optimiseur/époque")
    scheduler = _make_scheduler(optimizer, args.lr_schedule, args.warmup_iters,
                                total_iters, args.lr, args.min_lr)
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
        # Sans ceci, l'EMA repartirait des poids courants à chaque reprise — et
        # tous les runs à 200 époques reprennent (15 h 30 pour 12 h de walltime).
        if ema is not None:
            if "ema" not in ckpt:
                raise SystemExit(
                    f"⛔ --ema-decay demandé mais {resume_path} ne contient pas d'état EMA : "
                    "ce checkpoint vient d'un run SANS EMA. Reprendre dessus produirait "
                    "une moyenne partant du milieu de l'entraînement. Utiliser un "
                    "--output neuf."
                )
            ema.load_state_dict(ckpt["ema"])
            print(f"  état EMA repris : {ema.updates} mises à jour")
        elif "ema" in ckpt:
            raise SystemExit(
                f"⛔ {resume_path} vient d'un run AVEC EMA mais --ema-decay vaut 0 : "
                "la reprise changerait de protocole en cours de run."
            )
        print(f"Reprise depuis {resume_path} : époque {start_epoch}, step {global_step}, "
              f"best SeK {best_sek:.4f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            if args.limit_batches is not None and step >= args.limit_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            apply_sek = global_step >= args.sek_warmup_iters

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(batch["img_t1"], batch["img_t2"])
            # Loss en fp32 (SeK/log sensibles à la précision) : on caste les sorties.
            # `change_maps` est une LISTE de tenseurs : le test `torch.is_tensor`
            # la laissait passer telle quelle, donc en bfloat16, et la CE pondérée
            # de la supervision profonde recevait des logits bf16 avec un poids
            # fp32 -> « expected scalar type BFloat16 but found Float ».
            outputs = {k: _to_fp32(v) for k, v in outputs.items()}
            losses = criterion(outputs, _targets_from_batch(batch), apply_sek=apply_sek)

            # Accumulation : on divise pour que le gradient moyen soit celui du
            # batch effectif, et on ne met à jour que tous les accum_steps.
            (losses["total"] / args.accum_steps).backward()
            if (step + 1) % args.accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                # Après optimizer.step(), et une fois par pas d'optimiseur — pas
                # par micro-batch : sinon la fenêtre de moyennage dépendrait de
                # --accum-steps, et deux runs à batch effectif égal ne seraient
                # plus comparables.
                if ema is not None:
                    ema.update(model)

            if step % (50 * args.accum_steps) == 0:
                flat = {k: round(v.item(), 4) for k, v in losses.items()}
                lr = scheduler.get_last_lr()[0]
                print(f"epoch {epoch} step {step} lr {lr:.2e} sek={'on' if apply_sek else 'off'} {flat}")

        # Avec EMA, tout ce qui est rapporté et sauvé porte sur les poids moyennés :
        # valider le modèle courant et sauver l'EMA (ou l'inverse) mesurerait un
        # modèle et en livrerait un autre.
        eval_model = ema.module if ema is not None else model
        metrics = validate(eval_model, val_loader, device, num_classes,
                           limit=args.limit_batches, use_amp=use_amp)
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
        if ema is not None:
            ckpt["ema"] = ema.state_dict()
        if metrics.sek > best_sek:
            best_sek = metrics.sek
            ckpt["best_sek"] = best_sek
            # best.pt reste un state_dict nu, directement chargeable par
            # scripts.evaluate et scripts.count_gmacs sans qu'ils sachent si
            # l'EMA était active.
            torch.save(eval_model.state_dict(), out_dir / "best.pt")
            print(f"  -> nouveau meilleur SeK {best_sek:.4f}, sauvé dans best.pt")
        torch.save(ckpt, out_dir / "last.pt")


def _to_fp32(v):
    """Caste en fp32 un tenseur, ou chaque tenseur d'une liste/tuple."""
    if torch.is_tensor(v):
        return v.float()
    if isinstance(v, (list, tuple)):
        return type(v)(x.float() if torch.is_tensor(x) else x for x in v)
    return v


def _make_scheduler(optimizer, schedule, warmup_iters, total_iters, base_lr, min_lr):
    """Montée linéaire jusqu'à base_lr, puis `cosine` ou `constant`.

    Le warmup est commun aux deux : démarrer un SSM à plein LR diverge. Seule la
    phase suivante diffère, ce qui isole bien le facteur étudié.

    `constant` sert d'ablation au `cosine` par défaut. Attention à l'interprétation :
    la décroissance du LR est mêlée à la sélection de la meilleure époque, puisque
    `best.pt` retient le meilleur SeK en validation. Un schedule constant produit
    des courbes plus bruitées en fin d'entraînement, donc un maximum sur 100 époques
    mécaniquement plus optimiste. Comparer aussi les valeurs de la DERNIÈRE époque.
    """
    import math

    def lr_lambda(it):
        if it < warmup_iters:
            return (it + 1) / max(1, warmup_iters)
        if schedule == "constant":
            return 1.0
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
def validate(model, loader, device, num_classes, limit=None, use_amp=False) -> SCDMetrics:
    model.eval()
    evaluator = SCDEvaluator(num_classes=num_classes)
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
