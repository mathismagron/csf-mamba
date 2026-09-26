"""Échantillons d'entraînement + sorties du professeur lues dans le cache hors ligne.

Le cache (`scripts/teacher/cache_teacher.py`) contient, pour chaque paire et chacun
des 8 éléments g de D4, la sortie du professeur sur l'entrée transformée, T(g·x),
à la résolution native de son décodeur (tuile / 4). Ce wrapper :

  1. lit l'échantillon brut (sans transform) ;
  2. applique la chaîne d'augmentation de l'élève en lui demandant d'enregistrer ses
     tirages (clé `transforms.AUG_KEY`) ;
  3. ramène (hflip, vflip, k) à l'index de cache par `canonical_d4` et lit la vue
     correspondante — déjà dans le repère de l'image augmentée ;
  4. échange les canaux T1/T2 du professeur si l'échange temporel a été tiré.

Le jitter photométrique n'a pas d'équivalent dans le cache : le professeur a vu la
vue propre (documenté dans distillation.md). Le crop doit couvrir la tuile entière
(recette d'efficience : crop 512) ; tout autre crop est refusé plutôt qu'approché.

Ajoute `kd_teacher` : tenseur fp16 (15, tuile/4, tuile/4) — logits sémantiques T1
(canaux 0:7), T2 (7:14), logit de changement (14).
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..distill.d4 import D4, canonical_d4
from .transforms import AUG_HFLIP, AUG_KEY, AUG_LEFT, AUG_ROT, AUG_SWAP, AUG_TOP, AUG_VFLIP

CHANNELS = 15
SEM_T1, SEM_T2, CHANGE = slice(0, 7), slice(7, 14), slice(14, 15)


class TeacherCacheDataset(Dataset):
    def __init__(self, base, cache_dir: str, transform=None):
        """`base` : dataset SANS transform (p. ex. SECONDDataset(..., transform=None))."""
        if getattr(base, "transform", None) is not None:
            raise ValueError("le dataset de base doit être construit sans transform")
        self.base, self.transform = base, transform
        self.dir = Path(cache_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        lay = self.meta["layout"]
        if lay["shape"][1] != CHANNELS or lay["dtype"] != "float16":
            raise ValueError(f"cache inattendu : {lay}")
        if [tuple(e) for e in self.meta["d4"]["elements_hflip_then_rot90k"]] != D4:
            raise ValueError("la convention D4 du cache diffère de csf_mamba.distill.d4")
        self.native = lay["shape"][2]
        ids = [l.strip() for l in (self.dir / "ids.txt").read_text().splitlines() if l.strip()]
        self.row = {name: i for i, name in enumerate(ids)}
        absent = [n for n in base.ids if n not in self.row]
        if absent:
            raise KeyError(f"{len(absent)} paires absentes du cache (ex. {absent[:3]})")
        self.files = [self.dir / f for f in self.meta["d4"]["files"]]
        self._maps = None     # ouverts paresseusement, une fois par processus de chargement

    def __len__(self):
        return len(self.base)

    @property
    def ids(self):
        return self.base.ids

    def change_fraction(self, idx: int) -> float:   # pour --oversample-change
        return self.base.change_fraction(idx)

    def _open(self):
        if self._maps is None:
            self._maps = [np.load(f, mmap_mode="r") for f in self.files]
        return self._maps

    def __getitem__(self, idx):
        sample = self.base[idx]
        name = self.base.ids[idx]
        sample[AUG_KEY] = torch.zeros(6, dtype=torch.long)
        if self.transform is not None:
            sample = self.transform(sample)
        aug = sample.pop(AUG_KEY)
        tile = sample["img_t1"].shape[-1]
        if aug[AUG_TOP] or aug[AUG_LEFT] or tile != 4 * self.native:
            raise ValueError(f"crop partiel (top={int(aug[AUG_TOP])}, left={int(aug[AUG_LEFT])}, "
                             f"tuile {tile}) : le cache couvre la tuile entière de {4 * self.native}")
        g = canonical_d4(int(aug[AUG_HFLIP]), int(aug[AUG_VFLIP]), int(aug[AUG_ROT]))
        t = torch.from_numpy(np.array(self._open()[g][self.row[name]]))     # copie (15, n, n)
        if aug[AUG_SWAP]:
            t = torch.cat([t[SEM_T2], t[SEM_T1], t[CHANGE]], dim=0)
        sample["kd_teacher"] = t
        return sample
