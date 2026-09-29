"""Étape 0 du plan de distillation : le professeur PerASCD, réévalué sous notre protocole.

Charge le checkpoint SECOND publié (`legacy`, ViT-G/16/1024, 548 M) et l'évalue
sur **notre** copie de SECOND test, en tuile entière 512, avec **trois** calculs de
métrique sur les mêmes prédictions :

  1. `legacy`   : histogramme et `SCDD_eval_from_hist` de leur dépôt, verbatim ;
  2. `ours_hist`: notre `metrics_from_hist` sur le même histogramme (formule seule) ;
  3. `ours_eval`: notre `SCDEvaluator`, chaîne complète de l'élève (masques,
                  convention 0 → ignore), alimenté par les sorties du professeur.

Critères (documentation/distillation.md, étape 0) :
  - SeK legacy fp32 = 26,11 ± 0,05 pt et Fscd = 66,41 ± 0,05 pt ;
  - |legacy − ours_eval| < 1e-4 sur SeK et Fscd ;
  - l'ordre des classes de notre SECOND est celui du professeur (diagnostic de
    permutation : l'identité doit être l'appariement optimal).

Leur validation tournait **sans autocast** (fp32) ; l'opérateur déformable est
toujours calculé en fp32 par leur code. fp16/bf16 sont mesurés pour choisir la
précision du cache et de la KD en ligne.

    python scripts/teacher/eval_perascd.py \
        --perascd-root third_party/PerASCD \
        --checkpoint $SCRATCH/csf-distill/teacher/ckpt/PerAChain_40e_...pth \
        --data-root $SCRATCH/SECOND --out $SCRATCH/csf-distill/checks/step_0.json

Test local sans GPU ni checkpoint (poids aléatoires, ViT-B, opérateur PyTorch) :
    python scripts/teacher/eval_perascd.py --perascd-root ... --checkpoint none \
        --arch ViT-B/16 --device cpu --msda pytorch --precisions fp32 --limit 2 ...
"""

import argparse
import json
import platform
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from csf_mamba.datasets.second import _map_semantic  # noqa: E402
from csf_mamba.evaluation.metrics import SCDEvaluator, metrics_from_hist  # noqa: E402

NUM_CLASSES = 7                      # 0 = non-changé, 1..6 = classes réelles
PERA_MEAN = (0.3585, 0.3741, 0.3155)  # legacy/datasets/RS_ST.py, DataPerAAUG
PERA_STD = (0.1483, 0.1283, 0.1198)
# Valeurs du journal TensorBoard publié, époque 40 (celle du checkpoint).
PUBLISHED = {"sek": 0.261087, "fscd": 0.664138}
# Professeur VMamba-B publié par les mêmes auteurs (vmambaB_42e_mIoU74.01_Sek25.31_Fscd65.61_OA88.37.pth) :
# valeurs du nom de fichier, arrondies à 0,01 pt.
PUBLISHED_BY_ARCH = {"ViT-G/16/1024": PUBLISHED, "ViT-B/16": PUBLISHED,
                     "vmambaB": {"sek": 0.2531, "fscd": 0.6561}}
ARCHS = ["ViT-G/16/1024", "ViT-B/16", "vmambaB"]
TOL_PUBLISHED = 0.0005               # ± 0,05 pt
TOL_CODES = 1e-4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--perascd-root", required=True, help="clone legacy de PerASCD")
    p.add_argument("--checkpoint", required=True, help="chemin du .pth, ou 'none' (test)")
    p.add_argument("--data-root", required=True, help="SECOND au format ChangeMamba")
    p.add_argument("--split", default="test")
    p.add_argument("--arch", default="ViT-G/16/1024", choices=ARCHS)
    p.add_argument("--precisions", default="fp32,fp16,bf16")
    p.add_argument("--msda", default="auto", choices=["auto", "cuda", "pytorch"])
    p.add_argument("--check-msda", action="store_true",
                   help="compare l'opérateur CUDA et sa référence PyTorch sur un lot")
    p.add_argument("--latency", action="store_true", help="lot de 8, médiane de 50 après 10")
    p.add_argument("--flops", action="store_true", help="FLOPs d'une paire (hors op déformable)")
    p.add_argument("--batch-size", type=int, default=4)   # leur val_batch_size
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=None, help="nb de lots (test rapide)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Opérateur déformable : CUDA compilé si présent, sinon référence PyTorch.
# --------------------------------------------------------------------------- #
class _MSDA:
    """Remplace MSDeformAttnFunction ; `use_pytorch` bascule l'implémentation."""
    use_pytorch = False
    cuda_fn = None
    core_pytorch = None

    @classmethod
    def apply(cls, value, shapes, level_start, sampling_locations, attention_weights, im2col):
        if cls.use_pytorch or cls.cuda_fn is None:
            shapes_list = [(int(h), int(w)) for h, w in shapes.tolist()]
            return cls.core_pytorch(value, shapes_list, sampling_locations, attention_weights)
        return cls.cuda_fn.apply(value, shapes, level_start, sampling_locations,
                                 attention_weights, im2col)


class NativeOutput(torch.nn.Module):
    """Réseau `build_net` de leur dépôt (encodeurs non-PerA) construit à la résolution
    native de son décodeur (output_size=128), suréchantillonné ici vers
    `self.output_size` — exactement l'interpolation finale de leur forward (bilinéaire,
    align_corners=False), mais réglable comme `PerASCD.output_size` pour le cache."""

    def __init__(self, net, output_size=512):
        super().__init__()
        self.net, self.output_size = net, output_size

    def forward(self, a, b):
        outs = self.net(a, b)
        if outs[0].shape[-1] == self.output_size:
            return outs
        size = (self.output_size, self.output_size)
        return tuple(torch.nn.functional.interpolate(o, size, mode="bilinear", align_corners=False)
                     for o in outs)


def build_teacher(PerASCD, root: Path, arch: str, checkpoint: str):
    """-> (modèle, infos checkpoint). Sorties (changement, sém. A, sém. B) à 512 ;
    `model.output_size = 128` donne la résolution native (cache)."""
    if arch == "vmambaB":
        # `models/Encoders.py` (branche legacy) importe `models.SatMAE_temporal`, absent
        # du dépôt publié et utilisé seulement par l'encodeur SatMAE : module factice.
        if "models.SatMAE_temporal" not in sys.modules:
            stub = types.ModuleType("models.SatMAE_temporal")

            def _absent(*a, **k):
                raise RuntimeError("SatMAE_temporal absent du dépôt publié")
            stub.get_1d_sincos_pos_embed_from_grid_torch = stub.mae_vit_large_patch16 = _absent
            sys.modules["models.SatMAE_temporal"] = stub
        from models.Encoders import build_net
        # build_net charge des poids ImageNet depuis un chemin de leur machine
        # (/data2/...) avant qu'on charge le checkpoint SCD : on neutralise ce seul appel.
        real_load = torch.load

        def _load(path, *a, **k):
            if str(path).startswith("/data2/"):
                return {"model": {}}
            return real_load(path, *a, **k)
        torch.load = _load
        try:
            model = NativeOutput(build_net("vmambaB", NUM_CLASSES, output_size=128, drop_rate=0.0))
        finally:
            torch.load = real_load
        target = model.net
    else:
        model = PerASCD(in_channels=3, num_classes=NUM_CLASSES, input_size=448, output_size=512,
                        arch=arch, droppath=0.0, pretrained_pera_path=None)
        target = model
    info = {"path": checkpoint, "arch": arch}
    if checkpoint.lower() != "none":
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        state = {k.removeprefix("module."): v for k, v in state.items()}
        state, renamed = rename_cagm_keys(state, target.state_dict())
        target.load_state_dict(state, strict=True)          # lève au moindre écart
        info["renamed_keys"] = renamed
        info.update({k: (float(v) if isinstance(v, (float, np.floating)) else v)
                     for k, v in ckpt.items() if k in ("epoch", "Fscd", "Sek", "mIoU")})
        info["tensors"] = len(state)
        del ckpt, state
    else:
        info["warning"] = "poids aléatoires : test de plomberie uniquement"
    return model, info


def import_legacy(root: Path, msda_mode: str):
    try:
        import MultiScaleDeformableAttention  # noqa: F401
        compiled = True
    except ImportError:
        compiled = False
        if msda_mode == "cuda":
            raise SystemExit("MultiScaleDeformableAttention absent alors que --msda cuda")
        # module factice : leur ms_deform_attn_func l'importe au chargement
        sys.modules["MultiScaleDeformableAttention"] = types.ModuleType("MultiScaleDeformableAttention")
    sys.path.insert(0, str(root))
    from models.PerAChain import PerASCD
    from models.pera_layers.vit_adapter_layers.ops.functions import ms_deform_attn_func as fmod
    from models.pera_layers.vit_adapter_layers.ops.modules import ms_deform_attn as mmod
    from utils.utils import SCDD_eval_from_hist, get_hist

    _MSDA.cuda_fn = fmod.MSDeformAttnFunction if compiled else None
    _MSDA.core_pytorch = fmod.ms_deform_attn_core_pytorch   # fonction nue : pas de liaison
    _MSDA.use_pytorch = (msda_mode == "pytorch") or not compiled
    mmod.MSDeformAttnFunction = _MSDA
    return PerASCD, SCDD_eval_from_hist, get_hist, compiled


# --------------------------------------------------------------------------- #
# Données : notre SECOND (T1/T2/GT_T1/GT_T2/GT_CD), prétraitement du professeur.
# --------------------------------------------------------------------------- #
class SecondRaw(Dataset):
    def __init__(self, root, split):
        self.root = Path(root) / split
        listing = Path(root) / f"{split}.txt"
        if listing.is_file():
            ids = [l.strip() for l in listing.read_text().splitlines() if l.strip()]
            self.ids = [i if i.endswith(".png") else f"{i}.png" for i in ids]
        else:
            self.ids = sorted(p.name for p in (self.root / "T1").glob("*.png"))

    def __len__(self):
        return len(self.ids)

    def _img(self, d, n):   # PIL -> [0,1], comme torchvision to_tensor
        a = np.asarray(Image.open(self.root / d / n).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(a).permute(2, 0, 1)

    def _lbl(self, d, n):
        a = np.asarray(Image.open(self.root / d / n))
        assert a.ndim == 2, f"{d}/{n} n'est pas mono-canal"
        return torch.from_numpy(a.astype(np.int64))

    def __getitem__(self, i):
        n = self.ids[i]
        return {"img_a": self._img("T1", n), "img_b": self._img("T2", n),
                "lbl_a": self._lbl("GT_T1", n), "lbl_b": self._lbl("GT_T2", n),
                "cd": (self._lbl("GT_CD", n) > 0).long()}


def normalize(x):
    m = torch.tensor(PERA_MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(PERA_STD, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


AMP = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def forward(model, a, b, prec, device):
    dt = AMP[prec]
    with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                        dtype=dt or torch.float32, enabled=dt is not None):
        ch, oa, ob = model(normalize(a), normalize(b))
    return ch.float(), oa.float(), ob.float()


def permutation_check(hist):
    """L'ordre des classes de nos labels est-il celui du professeur ?"""
    from scipy.optimize import linear_sum_assignment
    fg = hist[1:, 1:]                      # lignes = prédit, colonnes = vérité
    r, c = linear_sum_assignment(-fg)
    best = [int(x) + 1 for x in c[np.argsort(r)]]
    return {"identity_optimal": best == list(range(1, NUM_CLASSES)),
            "best_assignment_pred_to_gt": best,
            "acc_identity": float(np.trace(fg) / max(fg.sum(), 1)),
            "acc_best": float(fg[r, c].sum() / max(fg.sum(), 1))}


@torch.no_grad()
def evaluate(model, loader, prec, device, legacy_eval, get_hist, limit):
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
    ours = SCDEvaluator(num_classes=NUM_CLASSES)
    checks = {"pixels": 0, "cd_vs_labelA": 0, "labelA_vs_labelB": 0, "pred_change": 0, "gt_change": 0}
    t0 = time.time()
    for k, batch in enumerate(loader):
        if limit is not None and k >= limit:
            break
        a, b = batch["img_a"].to(device), batch["img_b"].to(device)
        ch, oa, ob = forward(model, a, b, prec, device)
        cm = (torch.sigmoid(ch) > 0.5).squeeze(1).long()          # leur seuil
        pa = (oa.argmax(1) * cm).cpu().numpy()                      # leur masquage
        pb = (ob.argmax(1) * cm).cpu().numpy()
        la, lb = batch["lbl_a"].numpy(), batch["lbl_b"].numpy()
        for i in range(pa.shape[0]):
            hist += get_hist(pa[i], la[i], NUM_CLASSES)
            hist += get_hist(pb[i], lb[i], NUM_CLASSES)
        # chaîne complète de l'élève : logits de changement à 2 canaux dont
        # l'argmax vaut exactement « sigmoïde > 0,5 »
        bcd = torch.cat([torch.zeros_like(ch), ch], dim=1)
        ours.add({"bcd": bcd, "sem_t1": oa, "sem_t2": ob},
                 {"change": batch["cd"].to(device),
                  "sem_t1": torch.from_numpy(_map_semantic(la)).to(device),
                  "sem_t2": torch.from_numpy(_map_semantic(lb)).to(device)})
        cd = batch["cd"].numpy()
        checks["pixels"] += cd.size
        checks["cd_vs_labelA"] += int(((la > 0) != (cd > 0)).sum())
        checks["labelA_vs_labelB"] += int(((la > 0) != (lb > 0)).sum())
        checks["pred_change"] += int(cm.sum())
        checks["gt_change"] += int(cd.sum())
    fscd, miou, sek = legacy_eval(hist)
    oh = metrics_from_hist(hist)
    oe = ours.compute()
    res = {
        "legacy": {"sek": float(sek), "fscd": float(fscd), "miou": float(miou)},
        "ours_hist": {"sek": oh.sek, "fscd": oh.fscd, "miou": oh.miou, "oa": oh.oa, "kappa": oh.kappa},
        "ours_eval": {"sek": oe.sek, "fscd": oe.fscd, "miou": oe.miou, "oa": oe.oa, "kappa": oe.kappa},
        "data_checks": checks, "permutation": permutation_check(hist),
        "seconds": round(time.time() - t0, 1),
    }
    res["max_code_gap"] = max(abs(res["legacy"][m] - res["ours_eval"][m]) for m in ("sek", "fscd", "miou"))
    res["hist"] = hist.tolist()
    return res


@torch.no_grad()
def msda_agreement(model, loader, device):
    batch = next(iter(loader))
    a, b = batch["img_a"][:2].to(device), batch["img_b"][:2].to(device)
    _MSDA.use_pytorch = False
    ref = forward(model, a, b, "fp32", device)
    _MSDA.use_pytorch = True
    alt = forward(model, a, b, "fp32", device)
    _MSDA.use_pytorch = False
    out = {}
    for name, x, y in zip(("change", "sem_a", "sem_b"), ref, alt):
        out[name] = {"max_abs_diff": float((x - y).abs().max()),
                     "argmax_agreement": float((x.argmax(1) == y.argmax(1)).float().mean())
                     if x.shape[1] > 1 else float(((x > 0) == (y > 0)).float().mean())}
    return out


@torch.no_grad()
def latency(model, device, prec, batch=8, warmup=10, iters=50):
    a = torch.rand(batch, 3, 512, 512, device=device)
    b = torch.rand(batch, 3, 512, 512, device=device)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        forward(model, a, b, prec, device)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        forward(model, a, b, prec, device)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    med = float(np.median(ts))
    return {"batch": batch, "median_ms": round(1000 * med, 1), "pairs_per_s": round(batch / med, 1),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}


@torch.no_grad()
def flops_one_pair(model, device):
    from torch.utils.flop_counter import FlopCounterMode
    a = torch.rand(1, 3, 512, 512, device=device)
    with FlopCounterMode(display=False) as fc:
        forward(model, a, a.clone(), "fp32", device)
    total = fc.get_total_flops()
    return {"gflops": round(total / 1e9, 1), "gmacs": round(total / 2e9, 1),
            "note": "aten uniquement : l'opérateur déformable (extension C++) n'est pas compté"}


CAGM_KEY = __import__("re").compile(r"^(decoder\.blocks\.\d+\.cagm\.)conv2\.(weight|bias)$")


def rename_cagm_keys(state, target):
    """`cagm.conv2` (checkpoint) -> `cagm.conv_local` (code `legacy` @ a4d808a).

    Le dépôt `legacy` contient DEUX copies de `ChangeAwareGatingModule` :
    `models/Encoders.py` nomme la convolution locale `conv2`, `models/PerAChain.py`
    la nomme `conv_local`. Le checkpoint publié a été produit avec la première
    appellation, le modèle se construit avec la seconde. Les deux classes font
    exactement le même calcul (conv1 3×3 → ReLU → 1×1 vers 2 canaux, sigmoïde,
    × (1 + sigmoïde de la branche globale)) sur des tenseurs de même forme :
    vérifié ligne à ligne le 26 septembre. Le renommage est borné à ce motif, ne
    s'applique que si la clé cible existe et est absente, et vérifie les formes.
    """
    out, renamed = {}, []
    for k, v in state.items():
        m = CAGM_KEY.match(k)
        if m:
            new = f"{m.group(1)}conv_local.{m.group(2)}"
            if new in target and new not in state:
                assert target[new].shape == v.shape, (k, v.shape, target[new].shape)
                out[new] = v
                renamed.append(f"{k} -> {new}")
                continue
        out[k] = v
    if renamed:
        print(f"{len(renamed)} clés renommées (cagm.conv2 -> cagm.conv_local)")
    return out, renamed


def main():
    args = parse_args()
    device = args.device
    PerASCD, legacy_eval, get_hist, compiled = import_legacy(Path(args.perascd_root).resolve(), args.msda)

    model, ckpt_info = build_teacher(PerASCD, Path(args.perascd_root).resolve(), args.arch, args.checkpoint)
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())

    loader = DataLoader(SecondRaw(args.data_root, args.split), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers, pin_memory=device.startswith("cuda"))
    report = {
        "step": 0, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.node(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__, "cuda": torch.version.cuda, "arch": args.arch,
        "params_M": round(n_params / 1e6, 2), "checkpoint": ckpt_info,
        "msda": {"compiled_available": compiled, "used": "pytorch" if _MSDA.use_pytorch else "cuda"},
        "n_pairs": len(loader.dataset), "limit": args.limit, "published": PUBLISHED_BY_ARCH[args.arch], "results": {},
    }
    print(f"{report['params_M']} M paramètres | opérateur déformable : {report['msda']['used']} | "
          f"{len(loader.dataset)} paires")

    if args.check_msda and compiled and device.startswith("cuda") and args.arch != "vmambaB":
        report["msda"]["agreement_fp32"] = msda_agreement(model, loader, device)
        print("accord CUDA / PyTorch :", report["msda"]["agreement_fp32"])

    for prec in [p for p in args.precisions.split(",") if p]:
        r = evaluate(model, loader, prec, device, legacy_eval, get_hist, args.limit)
        report["results"][prec] = r
        print(f"[{prec}] SeK legacy {100*r['legacy']['sek']:.3f} | ours {100*r['ours_eval']['sek']:.3f} | "
              f"Fscd {100*r['legacy']['fscd']:.3f} | écart codes {r['max_code_gap']:.2e} | "
              f"classes dans l'ordre : {r['permutation']['identity_optimal']} | {r['seconds']} s")
        if args.latency and device.startswith("cuda"):
            r["latency"] = latency(model, device, prec)
            print(f"      latence : {r['latency']}")
    if args.flops:
        report["flops"] = flops_one_pair(model, device)
        print("FLOPs :", report["flops"])

    fp32 = report["results"].get("fp32")
    if fp32 is not None and args.limit is None and args.checkpoint.lower() != "none":
        report["criteria"] = {
            "sek_matches_published": abs(fp32["legacy"]["sek"] - PUBLISHED_BY_ARCH[args.arch]["sek"]) <= TOL_PUBLISHED,
            "fscd_matches_published": abs(fp32["legacy"]["fscd"] - PUBLISHED_BY_ARCH[args.arch]["fscd"]) <= TOL_PUBLISHED,
            "codes_agree": fp32["max_code_gap"] < TOL_CODES,
            "class_order_identity": fp32["permutation"]["identity_optimal"],
        }
        report["pass"] = all(report["criteria"].values())
        print("ÉTAPE 0 :", "PASS" if report["pass"] else "ÉCHEC", report["criteria"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
