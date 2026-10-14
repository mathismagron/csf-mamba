"""Professeur PerASCD en ligne, réduit à son encodeur (étape 4 : KD de features).

Les logits du professeur viennent du cache (étapes 2–3) ; ses **features** ne sont
pas cachables (≈2,1 To pour 8 vues D4) : on fait tourner son encodeur
(DINOv2 ViT-G/16 + ViT-Adapter) sur les images **exactement** vues par l'élève,
après toute la chaîne d'augmentation (flips, rot90, jitter, échange temporel).
Aucune correspondance de vue n'est donc à gérer ici, contrairement au cache.

Sorties de l'encodeur, par date : 4 cartes à 1/4, 1/8, 1/16, 1/32 de la tuile,
1024 canaux chacune (768 pour ViT-B) — mêmes résolutions que les 4 étages de
l'élève (96/192/384/768 canaux).

Chargement : le code `legacy` de PerASCD et la même procédure que l'étape 0
(`scripts/teacher/eval_perascd.py` : renommage `cagm.conv2`, chargement strict,
opérateur MSDA compilé ou repli PyTorch). Précision : autocast fp16, comme le cache
(choisi à l'étape 0 : SeK 26,1082 contre 26,1079 en fp32).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn

PERA_MEAN = (0.3585, 0.3741, 0.3155)
PERA_STD = (0.1483, 0.1283, 0.1198)
TEACHER_DIM = {"ViT-G/16/1024": 1024, "ViT-B/16": 768}
_EVAL = Path(__file__).resolve().parents[2] / "scripts" / "teacher" / "eval_perascd.py"


def _eval_module():
    spec = importlib.util.spec_from_file_location("eval_perascd", _EVAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class OnlineTeacherEncoder(nn.Module):
    """Encodeur gelé du professeur. `forward(img_t1, img_t2)` -> (feats_t1, feats_t2),
    listes de 4 tenseurs fp16 ; images d'entrée dans [0, 1], comme celles de l'élève."""

    def __init__(self, perascd_root: str, checkpoint: str, arch: str = "ViT-G/16/1024",
                 msda: str = "auto", amp_dtype: torch.dtype | None = torch.float16):
        super().__init__()
        ev = _eval_module()
        PerASCD, _, _, compiled = ev.import_legacy(Path(perascd_root).resolve(), msda)
        full = PerASCD(in_channels=3, num_classes=ev.NUM_CLASSES, input_size=448,
                       output_size=512, arch=arch, droppath=0.0, pretrained_pera_path=None)
        self.info = {"arch": arch, "msda_compiled": bool(compiled), "checkpoint": checkpoint}
        if checkpoint.lower() != "none":
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            state = {k.removeprefix("module."): v for k, v in state.items()}
            state, renamed = ev.rename_cagm_keys(state, full.state_dict())
            full.load_state_dict(state, strict=True)       # lève au moindre écart
            self.info.update(renamed_keys=len(renamed), epoch=ckpt.get("epoch"))
            del ckpt, state
        else:
            self.info["warning"] = "poids aléatoires : test de plomberie uniquement"
        self.backbone = full.backbone                      # décodeur et têtes jetés
        del full
        self.dim = TEACHER_DIM[arch]
        self.amp_dtype = amp_dtype
        self.register_buffer("mean", torch.tensor(PERA_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(PERA_STD).view(1, 3, 1, 1), persistent=False)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        self.info["params_M"] = sum(p.numel() for p in self.backbone.parameters()) / 1e6

    def train(self, mode: bool = True):
        # Toujours en eval : BatchNorm du ViT-Adapter sur ses statistiques, pas de dropout.
        return super().train(False)

    @torch.no_grad()
    def forward(self, img_t1: torch.Tensor, img_t2: torch.Tensor):
        b = img_t1.shape[0]
        x = (torch.cat([img_t1, img_t2], 0).float() - self.mean) / self.std
        # Les deux dates en un seul passage : l'encodeur est siamois et, en eval,
        # indépendant du lot (BatchNorm sur statistiques figées).
        with torch.autocast(device_type=x.device.type, dtype=self.amp_dtype or torch.float32,
                            enabled=self.amp_dtype is not None and x.device.type == "cuda"):
            outs = self.backbone(x)
        return [o[:b].half() for o in outs], [o[b:].half() for o in outs]
