"""Étape 0-cache : sorties du professeur PerASCD sur SECOND train, pour la KD hors ligne.

Pour chaque paire d'entraînement et chacun des **8 éléments du groupe diédral D4**,
le professeur voit l'image transformée g(x) et l'on stocke sa sortie T(g·x) telle
quelle — pas g·T(x). La KD reste ainsi exactement cohérente avec les rotations et
flips de l'élève (`rot90=1` + flips dans la recette d'efficience), sans supposer
que le professeur est équivariant. L'écart d'équivariance est mesuré au passage.

Stockage (vérifié le 26 septembre, voir documentation/distillation.md §4bis) :
  - résolution **native du décodeur, 128 × 128** : le professeur suréchantillonne
    ensuite en bilinéaire, l'opération est donc exacte (écart 0,0 mesuré) et
    commute avec D4 (écart 1e-6) ;
  - 15 canaux fp16 : logits sémantiques T1 (7), T2 (7), logit de changement (1) ;
  - un `.npy` mémoire-mappé par élément de D4 : (N, 15, 128, 128), soit ≈1,46 Go
    pour les 2 968 paires, ≈11,7 Go au total.

Élément g = (hflip, k) : flip horizontal (axe W) optionnel, PUIS `torch.rot90(·, k,
dims=(H, W))`. Les 8 couples (h ∈ {0,1}, k ∈ {0..3}) couvrent D4 ; un flip vertical
vaut hflip + rot180. `canonical_d4(h, v, k)` ramène n'importe quelle combinaison
des transforms de l'élève (hflip, vflip, puis rot90 k) à son index de cache.

    python scripts/teacher/cache_teacher.py --perascd-root third_party/PerASCD \
        --checkpoint $SCRATCH/csf-distill/teacher/ckpt/PerAChain_40e_...pth \
        --data-root $SCRATCH/SECOND --out-dir $SCRATCH/csf-distill/cache/perascd_second_train
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("eval_perascd", HERE / "eval_perascd.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

# Convention D4 unique, partagée avec l'élève (csf_mamba.distill.d4). Le cache du
# 26 septembre a été écrit avec une copie locale identique de ces fonctions.
from csf_mamba.distill.d4 import D4, apply_d4, canonical_d4  # noqa: E402

CHANNELS = 15                                     # 7 sem T1 + 7 sem T2 + 1 changement
NATIVE = 128


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--perascd-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--arch", default="ViT-G/16/1024", choices=ev.ARCHS)
    p.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--msda", default="auto", choices=["auto", "cuda", "pytorch"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=None, help="nb de lots (test rapide)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def sha256_of(path: Path, published: Path | None) -> str:
    if published and published.is_file():
        for line in published.read_text().splitlines():
            if line.strip().endswith(path.name):
                return line.split()[0]
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def main():
    args = parse_args()
    device = args.device
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    PerASCD, legacy_eval, get_hist, compiled = ev.import_legacy(Path(args.perascd_root).resolve(), args.msda)

    model, info = ev.build_teacher(PerASCD, Path(args.perascd_root).resolve(), args.arch, args.checkpoint)
    meta_ckpt = {"path": args.checkpoint, "arch": args.arch}
    if args.checkpoint.lower() != "none":
        cp = Path(args.checkpoint)
        meta_ckpt.update({"epoch": info.get("epoch"), "renamed_keys": len(info.get("renamed_keys", [])),
                          "sha256": sha256_of(cp, cp.parent.parent / "SHA256SUMS")})
    model = model.to(device).eval()

    ds = ev.SecondRaw(args.data_root, args.split)
    n = len(ds) if args.limit is None else min(len(ds), args.limit * args.batch_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=device.startswith("cuda"))
    (out / "ids.txt").write_text("\n".join(ds.ids[:n]) + "\n")
    maps = [np.lib.format.open_memmap(out / f"d4_{i}.npy", mode="w+", dtype=np.float16,
                                      shape=(n, CHANNELS, NATIVE, NATIVE)) for i in range(len(D4))]

    # Contrôle 1 : la sortie native est bien en 128 et son suréchantillonnage redonne
    # exactement la sortie 512 du modèle.
    first = next(iter(loader))
    a0, b0 = first["img_a"][:1].to(device), first["img_b"][:1].to(device)
    full = ev.forward(model, a0, b0, "fp32", device)
    model.output_size = NATIVE
    nat = ev.forward(model, a0, b0, "fp32", device)
    assert nat[1].shape[-1] == NATIVE, nat[1].shape
    upsample_gap = max(float((F.interpolate(x, (512, 512), mode="bilinear", align_corners=False) - y).abs().max())
                       for x, y in zip(nat, full))

    # Accumulateurs : SeK du professeur sur le train (vue identité, depuis le cache
    # tel qu'il sera relu) et écart d'équivariance T(g·x) vs g·T(x) par élément.
    hist = np.zeros((ev.NUM_CLASSES, ev.NUM_CLASSES))
    equi = {i: {"change_agree": 0.0, "sem_agree_changed": 0.0, "pix": 0, "pix_changed": 0}
            for i in range(1, len(D4))}
    t0, done = time.time(), 0
    for bi, batch in enumerate(loader):
        if done >= n:
            break
        a, b = batch["img_a"].to(device), batch["img_b"].to(device)
        m = min(a.shape[0], n - done)
        a, b = a[:m], b[:m]
        ident = None
        for gi, (h, k) in enumerate(D4):
            ch, sa, sb = ev.forward(model, apply_d4(a, h, k), apply_d4(b, h, k), args.precision, device)
            blob = torch.cat([sa, sb, ch], dim=1)                     # (m, 15, 128, 128)
            maps[gi][done:done + m] = blob.half().cpu().numpy()
            if gi == 0:
                ident = blob
            else:
                ref = apply_d4(ident, h, k)                           # g·T(x)
                c_ref, c_g = ref[:, 14] > 0, blob[:, 14] > 0
                e = equi[gi]
                e["change_agree"] += float((c_ref == c_g).float().sum())
                e["pix"] += c_ref.numel()
                both = c_ref & c_g
                for s in (slice(0, 7), slice(7, 14)):
                    e["sem_agree_changed"] += float(((ref[:, s].argmax(1) == blob[:, s].argmax(1)) & both).sum())
                e["pix_changed"] += 2 * int(both.sum())
        # SeK train à partir de ce qui vient d'être ÉCRIT (fp16), suréchantillonné
        # comme le fera la KD : contrôle de bout en bout du cache.
        rd = torch.from_numpy(np.asarray(maps[0][done:done + m])).to(device).float()
        rd = F.interpolate(rd, (512, 512), mode="bilinear", align_corners=False)
        cm = (rd[:, 14] > 0).long()
        pa, pb = (rd[:, :7].argmax(1) * cm).cpu().numpy(), (rd[:, 7:14].argmax(1) * cm).cpu().numpy()
        la, lb = batch["lbl_a"][:m].numpy(), batch["lbl_b"][:m].numpy()
        for i in range(m):
            hist += get_hist(pa[i], la[i], ev.NUM_CLASSES) + get_hist(pb[i], lb[i], ev.NUM_CLASSES)
        done += m
        if bi % 50 == 0:
            rate = done / max(time.time() - t0, 1e-6)
            print(f"{done}/{n} paires, {rate:.1f} paires/s (×8 vues), reste ≈ {(n - done) / max(rate, 1e-6) / 60:.1f} min",
                  flush=True)
    for mm in maps:
        mm.flush()
    fscd, miou, sek = legacy_eval(hist)
    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "split": args.split, "n_pairs": n,
        "arch": args.arch, "precision": args.precision, "msda": "pytorch" if ev._MSDA.use_pytorch else "cuda",
        "checkpoint": meta_ckpt, "normalization": dict(ev.NORM),
        "layout": {"shape": [n, CHANNELS, NATIVE, NATIVE], "dtype": "float16",
                   "channels": {"sem_t1_logits": [0, 7], "sem_t2_logits": [7, 14], "change_logit": [14, 15]},
                   "class_0": "non-changé : jamais cible de la CE du professeur, à exclure de la KL",
                   "upsample_to_512": "bilinear, align_corners=False (celui du professeur)"},
        "d4": {"files": [f"d4_{i}.npy" for i in range(len(D4))], "elements_hflip_then_rot90k": D4,
               "rule": "sortie du professeur sur l'entrée transformée, T(g·x), déjà dans le repère transformé"},
        "checks": {
            "upsample_native_vs_512_max_abs": upsample_gap,
            "train_sek_identity_view": float(sek), "train_fscd_identity_view": float(fscd),
            "train_miou_identity_view": float(miou),
            "equivariance_vs_identity": {
                f"d4_{i}": {"change_agreement": e["change_agree"] / max(e["pix"], 1),
                            "semantic_agreement_on_changed": e["sem_agree_changed"] / max(e["pix_changed"], 1)}
                for i, e in equi.items()},
        },
        "seconds": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta["checks"], indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
