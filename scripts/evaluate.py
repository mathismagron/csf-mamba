"""Évaluation d'un checkpoint csf-mamba.

Charge un `best.pt` (ou `last.pt`), calcule le tableau complet des métriques SCD
sur un split, affiche un diagnostic (fraction de changement prédite vs vérité —
pour repérer un collapse « aucun changement »), et sauve quelques visualisations.

    python -m scripts.evaluate \
        --data-root /scratch/<user>/hi-ucd \
        --checkpoint $SCRATCH/csf-mamba-runs/<run>/best.pt \
        --encoder vmamba_mini --output eval_<run>

Se lance dans un JOB GPU (le backbone VMamba a besoin du kernel CUDA).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from csf_mamba.datasets import DATASETS
from csf_mamba.losses.composite import IGNORE_INDEX
from csf_mamba.evaluation.metrics import SCDEvaluator
from csf_mamba.model import CSFMamba

# Palette RGB par classe sémantique (0 = réservé/noir, 1..9 = classes réelles).
PALETTE = np.array([
    [0, 0, 0], [0, 128, 255], [0, 200, 0], [200, 0, 0], [255, 200, 0],
    [128, 128, 128], [180, 0, 180], [255, 128, 0], [150, 100, 50], [0, 100, 0],
], dtype=np.uint8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="hi_ucd", choices=sorted(DATASETS))
    p.add_argument("--encoder", default="vmamba_mini", choices=["conv", "vmamba_mini", "vmamba_tiny"])
    p.add_argument("--core", default="chess", choices=["chess", "l1"])
    p.add_argument("--decoder-refine", default="dw", choices=["dw", "full"])
    # Doivent reproduire l'architecture du checkpoint, sinon le state_dict ne se
    # recharge pas : un modèle `concat` n'a pas les mêmes tenseurs qu'un `c2s2`.
    p.add_argument("--fusion", default="c2s2", choices=["c2s2", "concat"])
    p.add_argument("--no-cga", dest="cga", action="store_false")
    p.add_argument("--no-mcasf", dest="mcasf", action="store_false")
    p.add_argument("--upsample", default="dysample", choices=["dysample", "bilinear"])
    p.add_argument("--backend", default="mamba", choices=["auto", "mamba", "ref"])
    p.add_argument("--split", default="val")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--output", default="eval_out")
    p.add_argument("--viz-samples", type=int, default=8)
    return p.parse_args()


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)


def _to_rgb_image(img_chw: torch.Tensor) -> Image.Image:
    arr = (img_chw.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def _label_to_rgb(label_hw: np.ndarray) -> Image.Image:
    safe = np.where(label_hw == IGNORE_INDEX, 0, label_hw).clip(0, len(PALETTE) - 1)
    return Image.fromarray(PALETTE[safe])


def _binary_to_rgb(mask_hw: np.ndarray) -> Image.Image:
    rgb = np.zeros((*mask_hw.shape, 3), dtype=np.uint8)
    rgb[mask_hw] = (255, 0, 0)  # changement en rouge
    return Image.fromarray(rgb)


def save_panel(sample, out, pred_change, pred_sem2, path):
    """Panneau horizontal : T1 | T2 | changement (GT/pred) | sém T2 (GT/pred)."""
    h = sample["img_t1"].shape[-2]
    tiles = [
        _to_rgb_image(sample["img_t1"]), _to_rgb_image(sample["img_t2"]),
        _binary_to_rgb(sample["change"].cpu().numpy() == 1),
        _binary_to_rgb(pred_change.cpu().numpy() == 1),
        _label_to_rgb(sample["sem_t2"].cpu().numpy()),
        _label_to_rgb(pred_sem2.cpu().numpy()),
    ]
    panel = Image.new("RGB", (sum(t.width for t in tiles) + 5 * len(tiles), h), (255, 255, 255))
    x = 0
    for t in tiles:
        panel.paste(t, (x, 0))
        x += t.width + 5
    panel.save(path)


def _report_confusion(evaluator, out_dir, num_classes):
    """Diagnostic sémantique DANS les zones changées (classes 1..N).

    Répond à la question : le modèle discrimine-t-il les classes, ou prédit-il
    massivement la classe dominante ? Un kappa proche de 0 avec une classe très
    majoritaire signe une distribution dégénérée, pas une architecture en cause.
    """
    hist = evaluator.hist
    np.savetxt(out_dir / "confusion.csv", hist, fmt="%.0f", delimiter=",")

    fg = hist[1:, 1:]                       # bloc sémantique, hors no-change
    total = fg.sum()
    if total == 0:
        print("\n(aucun pixel changé prédit ET annoté : pas de diagnostic sémantique)")
        return

    gt_per_class = fg.sum(axis=0)           # colonnes = vérité
    pred_per_class = fg.sum(axis=1)         # lignes = prédiction
    correct = np.diag(fg)

    print("\n===== Sémantique dans les zones changées =====")
    print("  classe | % vérité | % prédit | rappel")
    for c in range(fg.shape[0]):
        if gt_per_class[c] == 0 and pred_per_class[c] == 0:
            continue
        rappel = correct[c] / gt_per_class[c] if gt_per_class[c] else float("nan")
        print(f"    {c+1:4d} | {100*gt_per_class[c]/total:7.2f}% | "
              f"{100*pred_per_class[c]/total:7.2f}% | {rappel:6.2%}")

    dom_gt = 100 * gt_per_class.max() / total
    dom_pred = 100 * pred_per_class.max() / total
    print(f"  classe la plus fréquente  — vérité {dom_gt:.1f}% | prédite {dom_pred:.1f}%")
    if dom_pred > 80:
        print("  ⚠️ le modèle prédit massivement UNE classe -> kappa ~0 par construction")
    print(f"  matrice complète sauvée dans {out_dir}/confusion.csv")


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_cls, num_classes = DATASETS[args.dataset]
    model = CSFMamba(num_semantic_classes=num_classes,
                     encoder=args.encoder, core=args.core, backend=args.backend,
                     decoder_refine=args.decoder_refine, fusion=args.fusion,
                     cga=args.cga, mcasf=args.mcasf,
                     upsample=args.upsample).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    ds = dataset_cls(args.data_root, split=args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    evaluator = SCDEvaluator(num_classes=num_classes)

    gt_ch, pred_ch, valid_px, saved = 0, 0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch["img_t1"], batch["img_t2"])
        targets = {"change": batch["change"], "sem_t1": batch["sem_t1"],
                   "sem_t2": batch["sem_t2"], "unchanged": batch["unchanged"]}
        evaluator.add(outputs, targets)

        pred_change = outputs["bcd"].argmax(1)
        pred_sem2 = outputs["sem_t2"].argmax(1)
        valid = batch["change"] != IGNORE_INDEX
        gt_ch += ((batch["change"] == 1) & valid).sum().item()
        pred_ch += ((pred_change == 1) & valid).sum().item()
        valid_px += valid.sum().item()

        for i in range(batch["img_t1"].shape[0]):
            if saved >= args.viz_samples:
                break
            sample = {k: batch[k][i] for k in ("img_t1", "img_t2", "change", "sem_t2")}
            save_panel(sample, out_dir, pred_change[i], pred_sem2[i],
                       out_dir / f"sample_{saved:03d}.png")
            saved += 1

    _report_confusion(evaluator, out_dir, num_classes)

    m = evaluator.compute()
    print("\n===== Métriques SCD (split", args.split, ") =====")
    print(f"  SeK   : {m.sek:.4f}")
    print(f"  Fscd  : {m.fscd:.4f}")
    print(f"  mIoU  : {m.miou:.4f}")
    print(f"  OA    : {m.oa:.4f}")
    print(f"  kappa : {m.kappa:.4f}")
    print("\n===== Diagnostic changement =====")
    print(f"  pixels changés VÉRITÉ  : {100 * gt_ch / max(1, valid_px):.2f} %")
    print(f"  pixels changés PRÉDITS : {100 * pred_ch / max(1, valid_px):.2f} %")
    if pred_ch == 0:
        print("  ⚠️ le modèle ne prédit AUCUN changement (collapse) -> renforcer la loss changement.")
    print(f"\n{saved} visualisations sauvées dans {out_dir}/")


if __name__ == "__main__":
    main()
