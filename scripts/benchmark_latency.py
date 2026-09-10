"""Latence et mémoire crête à l'inférence — CSF-Mamba face à MambaSCD.

Pourquoi ce script existe. Toute la thèse d'efficience du projet repose sur des
**paramètres** et des **GMACs**. Aucun des deux ne dit combien de temps le modèle
met réellement. Deux raisons de s'en méfier ici en particulier :

* les noyaux SSM ont un rapport GMACs → temps médiocre (beaucoup de lancements de
  noyaux, peu d'arithmétique par octet lu) ;
* `grid_sample`, au cœur de DySample, est *memory-bound* : il ne compte presque
  rien en MACs et coûte du temps réel.

Si la latence ne suit pas les GMACs, mieux vaut le savoir avant qu'un relecteur
ne le demande. Les deux issues sont publiables — ce qui ne l'est pas, c'est de
revendiquer « 15 % du calcul » sans avoir regardé.

Pièges traités, chacun invalidant la mesure s'il est ignoré :

* **Synchronisation.** CUDA est asynchrone : chronométrer sans
  `torch.cuda.synchronize()` mesure le temps de mise en file, pas le calcul.
* **Warmup.** Les premières itérations paient l'allocation du cache, l'autotuning
  cuDNN et la compilation des noyaux. Elles sont exclues.
* **Médiane, pas moyenne.** Une seule itération perturbée par un voisin de nœud
  décale la moyenne ; la médiane l'ignore.
* **Mémoire remise à zéro** entre configurations, sinon le pic mesuré est celui
  du modèle précédent.
* **Même protocole des deux côtés.** MambaSCD est instancié depuis
  `third_party/` — sa *construction* fonctionne, seul le chargement des poids
  publiés échoue (dépôt refactorisé après publication). Or la latence ne dépend
  pas de la valeur des poids : la comparaison reste valide sans eux.

    python -m scripts.benchmark_latency --fusion concat --encoder vmamba_mini
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

from csf_mamba.datasets import DATASETS
from csf_mamba.model import CSFMamba, count_parameters


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="second", choices=sorted(DATASETS))
    p.add_argument("--encoder", default="vmamba_mini",
                   choices=["conv", "vmamba_mini", "vmamba_tiny"])
    p.add_argument("--core", default="chess", choices=["chess", "l1"])
    p.add_argument("--fusion", default="concat", choices=["c2s2", "concat"])
    p.add_argument("--decoder-refine", default="dw", choices=["dw", "full"])
    p.add_argument("--upsample", default="dysample", choices=["dysample", "bilinear"])
    p.add_argument("--no-cga", dest="cga", action="store_false")
    p.add_argument("--no-mcasf", dest="mcasf", action="store_false")
    p.add_argument("--fft-stages", default="0,1")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch-sizes", default="1,8")
    p.add_argument("--iters", type=int, default=50, help="Itérations chronométrées.")
    p.add_argument("--warmup", type=int, default=10, help="Itérations jetées avant mesure.")
    p.add_argument("--changemamba", action="store_true",
                   help="Mesure AUSSI MambaSCD-Tiny depuis third_party/, même protocole.")
    p.add_argument("--cm-config", default="vssm1/vssm_tiny_224_0229flex.yaml")
    return p.parse_args()


def build_changemamba(cfg_name: str):
    """Instancie MambaSCD sans ses poids — la latence n'en dépend pas."""
    root = Path(__file__).resolve().parents[1] / "third_party" / "ChangeMamba"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from changedetection.configs.config import get_config
    from changedetection.models.ChangeMambaSCD import ChangeMambaSCD
    from changedetection.script.script_utils import get_vssm_kwargs

    cfg_path = root / "changedetection" / "configs" / cfg_name
    if not cfg_path.is_file():
        raise SystemExit(f"config introuvable : {cfg_path}")
    ns = argparse.Namespace(cfg=str(cfg_path), opts=None, batch_size=None, data_path=None,
                            zip=None, cache_mode=None, pretrained=None,
                            encoder_pretrained_path=None, model_checkpoint_path=None,
                            resume=None, resume_training_path=None, accumulation_steps=None,
                            use_checkpoint=None, disable_amp=None, output=None, tag=None,
                            enable_amp=None, optim=None, memory_limit_rate=None,
                            fused_layernorm=None, fused_window_process=None,
                            amp_opt_level=None, throughput=None, traincost=None)
    cfg = get_config(ns)
    return ChangeMambaSCD(output_cd=2, output_clf=7, pretrained=None,
                          **get_vssm_kwargs(cfg))


@torch.inference_mode()
def measure(model, batch, size, iters, warmup, amp):
    """-> (médiane ms, p10 ms, p90 ms, mémoire crête Mo). None si l'OOM survient."""
    device = "cuda"
    x1 = torch.randn(batch, 3, size, size, device=device)
    x2 = torch.randn(batch, 3, size, size, device=device)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else torch.autocast("cuda", enabled=False)

    try:
        for _ in range(warmup):                 # cache d'allocation, autotuning cuDNN
            with ctx:
                model(x1, x2)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()    # sinon on mesure le pic du modèle précédent

        durees = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with ctx:
                model(x1, x2)
            torch.cuda.synchronize()            # CUDA est asynchrone : sans ça on
            durees.append(time.perf_counter() - t0)   # chronomètre la mise en file
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

    pic = torch.cuda.max_memory_allocated() / 2**20
    durees.sort()
    ms = [d * 1000 for d in durees]
    return statistics.median(ms), ms[len(ms) // 10], ms[-1 - len(ms) // 10], pic


def rapport(nom, n_params, batches, size, iters, warmup, model):
    print(f"\n===== {nom} — {n_params / 1e6:.2f} M paramètres, entrée {size}x{size} =====")
    print(f"| précision | batch | médiane (ms) | p10–p90 (ms) | paires/s | mémoire crête |")
    print(f"|---|---|---|---|---|---|")
    for amp, label in ((False, "fp32"), (True, "bf16")):
        for b in batches:
            r = measure(model, b, size, iters, warmup, amp)
            if r is None:
                print(f"| {label} | {b} | — | — | — | **OOM** |")
                continue
            med, p10, p90, pic = r
            print(f"| {label} | {b} | {med:8.2f} | {p10:.2f} – {p90:.2f} | "
                  f"{b / (med / 1000):7.1f} | {pic:7.0f} Mo |")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("Un GPU est requis (les noyaux SSM ne tournent pas sur CPU).")
    print(f"GPU : {torch.cuda.get_device_name(0)}  |  torch {torch.__version__}")
    print(f"protocole : {args.warmup} itérations de chauffe jetées, "
          f"{args.iters} chronométrées, médiane rapportée")

    batches = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    _, num_classes = DATASETS[args.dataset]
    fft_stages = tuple(int(x) for x in args.fft_stages.split(",") if x.strip())

    model = CSFMamba(
        num_semantic_classes=num_classes, encoder=args.encoder, core=args.core,
        backend="auto", decoder_refine=args.decoder_refine, fft_stages=fft_stages,
        fusion=args.fusion, cga=args.cga, mcasf=args.mcasf, upsample=args.upsample,
    ).cuda().eval()

    # Garde-fou repris de count_gmacs : un --fusion accepté mais non propagé a
    # déjà fait mesurer le modèle complet en croyant mesurer la variante allégée.
    attendu = {"c2s2": "C2S2Block", "concat": "ConcatFusion"}[args.fusion]
    obtenu = type(model.c2s2[0]).__name__
    if obtenu != attendu:
        raise SystemExit(f"⛔ --fusion {args.fusion} demandé mais le modèle "
                         f"contient {obtenu}")

    nom = f"CSF-Mamba {args.encoder} / fusion={args.fusion} / dec={args.decoder_refine}"
    rapport(nom, count_parameters(model)["total"], batches, args.size,
            args.iters, args.warmup, model)

    if args.changemamba:
        del model
        torch.cuda.empty_cache()
        cm = build_changemamba(args.cm_config).cuda().eval()
        n = sum(p.numel() for p in cm.parameters())
        rapport(f"MambaSCD ({args.cm_config})", n, batches, args.size,
                args.iters, args.warmup, cm)
        print("\n(Poids non chargés : la latence n'en dépend pas. Le compte de "
              "paramètres, lui, est bien celui de leur architecture.)")


if __name__ == "__main__":
    main()
