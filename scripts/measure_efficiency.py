"""Efficience élève vs professeur, **dans un même job, sur le même GPU** (étape 5).

Pour chacun : paramètres ; GMACs d'une paire 512² par **deux compteurs appliqués
aux deux modèles** ; latence et mémoire crête au protocole du README (lot de 8,
10 itérations de chauffe, médiane de 50, `synchronize` autour de chaque passage)
en fp32, bf16 et fp16 ; lot de 1 en plus pour l'usage embarqué.

Les deux compteurs, et ce que chacun ne voit pas :

* **fvcore** (convention des tableaux d'efficience, celle de `count_gmacs.py`) :
  compte des MACs ; le scan sélectif de l'élève est compté via les handlers de
  ChangeMamba ; l'opérateur déformable (MSDA) du professeur n'a pas de handler et
  est listé dans `unsupported` — non compté.
* **torch.utils.flop_counter** (niveau aten, celui de l'étape 0) : ne voit aucune
  extension CUDA — ni le scan sélectif de l'élève, ni MSDA. Donné pour la
  cohérence avec les 1 509,7 GMACs publiés dans le journal ; l'écart fvcore − aten
  chez l'élève mesure la part du scan sélectif.

Les poids n'influent pas sur ces mesures ; le professeur est quand même chargé
(strict) pour vérifier qu'on mesure bien l'architecture du checkpoint.

    python -m scripts.measure_efficiency --teacher-root third_party/PerASCD \\
        --teacher-ckpt $SCRATCH/csf-distill/teacher/ckpt/PerAChain_40e_...pth --out eff.json
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from csf_mamba.model import CSFMamba, count_parameters

DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}
PERA_MEAN = (0.3585, 0.3741, 0.3155)
PERA_STD = (0.1483, 0.1283, 0.1198)


def build_student():
    # Point d'efficience : recette figée de train_kd_second.sbatch.
    m = CSFMamba(num_semantic_classes=7, encoder="vmamba_mini", core="chess", backend="mamba",
                 decoder_refine="dw", fft_stages=(0, 1), fusion="concat", cga=True, mcasf=True,
                 upsample="dysample")
    assert type(m.c2s2[0]).__name__ == "ConcatFusion"
    return m


class TeacherWrapper(torch.nn.Module):
    """Professeur complet, normalisation PerA incluse (images [0, 1] en entrée)."""

    def __init__(self, root, ckpt):
        super().__init__()
        from csf_mamba.distill.online_teacher import _eval_module
        ev = _eval_module()
        PerASCD, _, _, compiled = ev.import_legacy(Path(root).resolve(), "cuda")
        self.model = PerASCD(in_channels=3, num_classes=ev.NUM_CLASSES, input_size=448,
                             output_size=512, arch="ViT-G/16/1024", droppath=0.0,
                             pretrained_pera_path=None)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = {k.removeprefix("module."): v for k, v in ck["model"].items()}
        state, _ = ev.rename_cagm_keys(state, self.model.state_dict())
        self.model.load_state_dict(state, strict=True)
        del ck, state
        self.msda_compiled = bool(compiled)
        self.register_buffer("mean", torch.tensor(PERA_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(PERA_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, a, b):
        return self.model((a - self.mean) / self.std, (b - self.mean) / self.std)


@torch.no_grad()
def measure(model, batch, dtype, iters, warmup, size=512):
    a = torch.rand(batch, 3, size, size, device="cuda")
    b = torch.rand(batch, 3, size, size, device="cuda")
    ctx = torch.autocast("cuda", dtype=dtype) if dtype else torch.autocast("cuda", enabled=False)
    try:
        for _ in range(warmup):
            with ctx:
                model(a, b)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t = time.perf_counter()
            with ctx:
                model(a, b)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"batch": batch, "oom": True}
    except RuntimeError as exc:                    # p. ex. noyau sans support fp16
        torch.cuda.empty_cache()
        return {"batch": batch, "error": f"{exc}"[:300]}
    ms = sorted(1000 * t for t in ts)
    med = statistics.median(ms)
    return {"batch": batch, "median_ms": round(med, 1), "p10_ms": round(ms[len(ms) // 10], 1),
            "p90_ms": round(ms[-1 - len(ms) // 10], 1), "pairs_per_s": round(batch * 1000 / med, 1),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


@torch.no_grad()
def gmacs_aten(model, size=512):
    from torch.utils.flop_counter import FlopCounterMode
    a = torch.rand(1, 3, size, size, device="cuda")
    with FlopCounterMode(display=False) as fc:
        model(a, a.clone())
    return round(fc.get_total_flops() / 2e9, 2)


def gmacs_fvcore(model, with_ssm_handlers: bool, size=512):
    from fvcore.nn import flop_count
    ops = {}
    if with_ssm_handlers:
        from scripts.count_gmacs import supported_ops
        ops = supported_ops()
    a = torch.randn(1, 3, size, size, device="cuda")
    try:
        g, unsupported = flop_count(model=model, inputs=(a, a.clone()), supported_ops=ops)
    except Exception as exc:                       # traçage JIT impossible : on le dit
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}
    return {"gmacs": round(sum(g.values()), 2),
            "unsupported": {k: int(v) for k, v in sorted(unsupported.items())}}


def profile(name, model, args, dtypes):
    model = model.cuda().eval()
    n = sum(p.numel() for p in model.parameters())
    out = {"name": name, "params_M": round(n / 1e6, 3)}
    print(f"== {name} : {n / 1e6:.2f} M paramètres", flush=True)
    out["gmacs_aten"] = gmacs_aten(model)
    print(f"   GMACs aten : {out['gmacs_aten']}", flush=True)
    out["gmacs_fvcore"] = gmacs_fvcore(model, with_ssm_handlers=name == "student")
    print(f"   GMACs fvcore : {out['gmacs_fvcore']}", flush=True)
    out["latency"] = {}
    for dt in dtypes:
        for bs in args.batches:
            r = measure(model, bs, DTYPES[dt], args.iters, args.warmup)
            out["latency"][f"{dt}_b{bs}"] = r
            print(f"   {dt} lot {bs} : {r}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher-root", required=True)
    ap.add_argument("--teacher-ckpt", required=True)
    ap.add_argument("--batches", default="8,1")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.batches = [int(x) for x in args.batches.split(",")]
    if not torch.cuda.is_available():
        raise SystemExit("GPU requis")
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "cuda": torch.version.cuda, "python": platform.python_version(),
           "protocol": {"size": 512, "batches": args.batches, "warmup": args.warmup,
                        "iters": args.iters, "stat": "median"}}
    st = build_student()
    res["student_breakdown"] = count_parameters(st)
    res["student"] = profile("student", st, args, ["fp32", "bf16", "fp16"])
    del st
    torch.cuda.empty_cache()
    te = TeacherWrapper(args.teacher_root, args.teacher_ckpt)
    res["teacher_msda_compiled"] = te.msda_compiled
    res["teacher"] = profile("teacher", te, args, ["fp32", "bf16", "fp16"])
    s, t = res["student"], res["teacher"]
    ratios = {"params": round(t["params_M"] / s["params_M"], 1),
              "gmacs_aten": round(t["gmacs_aten"] / s["gmacs_aten"], 1)}
    if "gmacs" in s["gmacs_fvcore"] and "gmacs" in t["gmacs_fvcore"]:
        ratios["gmacs_fvcore"] = round(t["gmacs_fvcore"]["gmacs"] / s["gmacs_fvcore"]["gmacs"], 1)
    for k, v in s["latency"].items():
        tv = t["latency"].get(k, {})
        if "median_ms" in v and "median_ms" in tv:
            ratios[f"latency_{k}"] = round(tv["median_ms"] / v["median_ms"], 1)
            ratios[f"mem_{k}"] = round(tv["peak_mem_gb"] / v["peak_mem_gb"], 1)
    res["ratios_teacher_over_student"] = ratios
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(ratios, indent=2))


if __name__ == "__main__":
    main()
