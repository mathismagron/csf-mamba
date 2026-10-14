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

import re
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


_NAME = re.compile(r"From(\d{4})To(\d{4})_(\d+)\.png$")


def location_of(ident: str) -> str:
    """Emplacement d'une paire : l'indice `k` de `From<a>To<b>_<k>.png` (vérifié sur les
    pixels : une image distincte par (k, année), aucune partagée entre deux k)."""
    m = _NAME.search(ident)
    if not m:
        raise ValueError(f"nom de paire inattendu : {ident}")
    return m[3]


def assert_location_disjoint(**lists) -> dict:
    """Lève si un emplacement apparaît dans deux listes (fuite). -> {nom: emplacements}."""
    locs = {name: {location_of(i) for i in ids} for name, ids in lists.items()}
    names = list(locs)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            common = locs[names[a]] & locs[names[b]]
            if common:
                raise RuntimeError(f"fuite : emplacements {sorted(common)} à la fois dans "
                                   f"{names[a]} et {names[b]}")
    return locs


def _map_semantic(index_map: np.ndarray) -> np.ndarray:
    """Classes réelles 1..4 conservées ; 0 (hors changement) -> ignore."""
    mapped = index_map.astype(np.int64)
    mapped[index_map == 0] = IGNORE_INDEX
    return mapped


class LandsatPerASCDDataset(Dataset):
    """`ids_file` (facultatif) : liste de paires `<split>/<nom>` relatives à la racine de
    l'archive, p. ex. `splits/LandsatSCD_loc/fold0/test.txt` — découpage disjoint par
    emplacement (`scripts/make_landsat_location_folds.py`). Sans liste, on lit tout le
    dossier `<root>/<split>` (découpage d'origine, qui fuit : cf. docstring du script)."""
    FOLDERS = ("im1", "im2", "label1", "label2")

    def __init__(self, root: str, split: str = "train", transform=None, ids_file: str | None = None):
        self.base = Path(root)
        self.transform = transform
        if ids_file:
            lines = [l.strip() for l in Path(ids_file).read_text().splitlines() if l.strip()]
            bad = [l for l in lines if l.count("/") != 1 or l.split("/")[0] not in ("train", "val", "test")]
            if bad:
                raise ValueError(f"{ids_file} : entrées attendues '<split>/<nom>', ex. fautif {bad[:3]}")
            self.ids = lines
        else:
            d = self.base / split / "im1"
            if not d.is_dir():
                raise FileNotFoundError(
                    f"{d} introuvable.\nAttendu : <root>/{{train,val,test}}/{{im1,im2,label1,label2}} "
                    "(archive HF SathShen/PerASCD-datasets, cf. docstring).")
            self.ids = [f"{split}/{p.name}" for p in sorted(d.glob("*.png"))]
        if not self.ids:
            raise RuntimeError(f"aucun échantillon ({ids_file or self.base / split})")
        if len(set(self.ids)) != len(self.ids):
            raise RuntimeError("identifiants en double")
        missing = [f"{f}/{i}" for i in self.ids for f in self.FOLDERS if not self._path(f, i).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} fichiers absents (ex. {missing[:3]})")

    def _path(self, folder: str, ident: str) -> Path:
        split, name = ident.split("/")
        return self.base / split / folder / name

    def __len__(self) -> int:
        return len(self.ids)

    def change_fraction(self, idx: int) -> float:
        arr = np.asarray(Image.open(self._path("label1", self.ids[idx])))
        return float((arr > 0).mean())

    def _load_rgb(self, folder: str, ident: str) -> torch.Tensor:
        arr = np.asarray(Image.open(self._path(folder, ident)).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _load_index_map(self, folder: str, ident: str) -> np.ndarray:
        path = self._path(folder, ident)
        arr = np.asarray(Image.open(path))
        if arr.ndim != 2:
            raise ValueError(f"{path} : carte sémantique attendue mono-canal, reçu {arr.shape}")
        if arr.max() >= NUM_SEMANTIC_CLASSES:
            raise ValueError(f"{path} : indice {arr.max()} hors 0..4")
        return arr

    def __getitem__(self, idx: int) -> dict:
        ident = self.ids[idx]
        lab1 = self._load_index_map("label1", ident)
        lab2 = self._load_index_map("label2", ident)
        # Changement = sémantique annotée à T1, comme PerASCD (`labels_A > 0`) ;
        # `scripts/check_landsat_perascd.py` vérifie que label1 > 0 ⇔ label2 > 0.
        change = torch.from_numpy((lab1 > 0).astype(np.int64))
        sample = {
            "img_t1": self._load_rgb("im1", ident), "img_t2": self._load_rgb("im2", ident),
            "sem_t1": torch.from_numpy(_map_semantic(lab1)),
            "sem_t2": torch.from_numpy(_map_semantic(lab2)),
            "change": change, "unchanged": change == 0,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
