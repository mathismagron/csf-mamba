"""Landsat-SCD dans la version prétraitée par les auteurs de PerASCD (`LandsatSCD512`).

Source : HF `SathShen/PerASCD-datasets`, `LandsatSCD512.zip` (1 612 898 211 octets),
dérivée du dump figshare 19946135 (CC BY 4.0) par les scripts publiés dans
`third_party/PerASCD/datasets/LandsatSCD/` (branche legacy) :

1. `rm_aug_samples.py` retire les variantes augmentées hors ligne (`rotate`, `Crop`,
   `ZheDang`) : plus de fuite possible entre splits par une tuile et sa rotation ;
2. `MCDlabel_to_SCDlabel.py` convertit la carte de transitions 0..9 en deux cartes
   sémantiques par date, avec la table publiée
   `MAP_A = [0, 1, 1, 2, 2, 2, 3, 3, 4, 4]`, `MAP_B = [0, 2, 3, 1, 3, 4, 1, 2, 1, 2]` ;
3. `Mask_invalid.py` met à 0 les zones blanches des deux images (hors emprise) ;
4. `split_train_val.py` tire un découpage aléatoire 60/20/20.

Cette version lève les trois blocages de `documentation/landsat-scd.md` (table des
transitions, variantes augmentées, absence de split) et rend nos chiffres
directement comparables à ceux de PerASCD sur ce jeu.

Arborescence (vérifiée le 30 sept. sur l'index de l'archive et une paire ouverte) :

    root/<split>/im1/<id>.png      image T1, RGB 512×512 uint8
    root/<split>/im2/<id>.png      image T2
    root/<split>/label1/<id>.png   sémantique T1, mode L, indices 0..4
    root/<split>/label2/<id>.png   sémantique T2
    splits : train 1431, val 477, test 477 (aucun nom commun)

Pas de carte de changement : PerASCD la dérive par `labels_A > 0` (`train.py`), et
la sémantique vaut 0 hors changement dans les deux dates — c'est notre convention A,
comme SECOND : `0 → ignore`, classes réelles 1..4.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..losses.composite import IGNORE_INDEX

# 5 canaux : index 0 réservé (inchangé), classes réelles 1..4.
NUM_SEMANTIC_CLASSES = 5

# Ordre tiré de leur code, pas deviné : `SCD_ClASSES` de `MCDlabel_to_SCDlabel.py`
# ('0: No change', '1: Farmland', '2: desert', '3: building', '4: water') et
# `ST_CLASSES` commenté pour LandsatSCD dans `datasets/RS_ST.py`.
CLASS_NAMES = ("reserved", "farmland", "desert", "building", "water")

SPLIT_SIZES = {"train": 1431, "val": 477, "test": 477}
ARCHIVE_BYTES = 1_612_898_211


def _map_semantic(index_map: np.ndarray) -> np.ndarray:
    """Classes réelles 1..4 conservées ; 0 (hors changement) -> ignore."""
    mapped = index_map.astype(np.int64)
    mapped[index_map == 0] = IGNORE_INDEX
    return mapped


class LandsatPerASCDDataset(Dataset):
    FOLDERS = ("im1", "im2", "label1", "label2")

    def __init__(self, root: str, split: str = "train", transform=None):
        self.root = Path(root) / split
        self.transform = transform
        self.dirs = {name: self.root / name for name in self.FOLDERS}
        missing = [str(p) for p in self.dirs.values() if not p.is_dir()]
        if missing:
            raise FileNotFoundError(
                "Dossiers LandsatSCD512 introuvables : " + ", ".join(missing)
                + "\nAttendu : <root>/{train,val,test}/{im1,im2,label1,label2} "
                "(archive HF SathShen/PerASCD-datasets, cf. docstring)."
            )
        self.ids = sorted(p.name for p in self.dirs["im1"].glob("*.png"))
        if not self.ids:
            raise RuntimeError(f"aucun échantillon dans {self.dirs['im1']}")
        for name in self.FOLDERS[1:]:
            have = {p.name for p in self.dirs[name].glob("*.png")}
            if have != set(self.ids):
                diff = sorted(set(self.ids) ^ have)
                raise RuntimeError(f"{self.dirs[name]} ne correspond pas à im1/ "
                                   f"({len(diff)} écarts, ex. {diff[:3]})")

    def __len__(self) -> int:
        return len(self.ids)

    def change_fraction(self, idx: int) -> float:
        arr = np.asarray(Image.open(self.dirs["label1"] / self.ids[idx]))
        return float((arr > 0).mean())

    def _load_rgb(self, folder: str, name: str) -> torch.Tensor:
        arr = np.asarray(
            Image.open(self.dirs[folder] / name).convert("RGB"), dtype=np.float32
        ) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _load_index_map(self, folder: str, name: str) -> np.ndarray:
        arr = np.asarray(Image.open(self.dirs[folder] / name))
        if arr.ndim != 2:
            raise ValueError(f"{self.dirs[folder] / name} : carte sémantique attendue "
                             f"mono-canal, reçu {arr.shape}")
        if arr.max() >= NUM_SEMANTIC_CLASSES:
            raise ValueError(f"{self.dirs[folder] / name} : indice {arr.max()} hors 0..4")
        return arr

    def __getitem__(self, idx: int) -> dict:
        name = self.ids[idx]
        lab1 = self._load_index_map("label1", name)
        lab2 = self._load_index_map("label2", name)
        # Changement = sémantique annotée à T1, comme PerASCD (`labels_A > 0`) ;
        # `scripts/check_landsat_perascd.py` vérifie que label1 > 0 ⇔ label2 > 0.
        change = torch.from_numpy((lab1 > 0).astype(np.int64))
        sample = {
            "img_t1": self._load_rgb("im1", name), "img_t2": self._load_rgb("im2", name),
            "sem_t1": torch.from_numpy(_map_semantic(lab1)),
            "sem_t2": torch.from_numpy(_map_semantic(lab2)),
            "change": change, "unchanged": change == 0,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
