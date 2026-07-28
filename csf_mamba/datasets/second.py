"""Dataloader SECOND (Semantic Change Detection dataset).

4 662 paires 512×512, résolution 0,5–3 m, 6 classes sémantiques réelles
(non-vegetated ground, tree, low vegetation, water, building, playground).

**Convention native de SECOND = notre convention A** : la sémantique n'est
annotée QUE dans les zones changées, et vaut 0 ailleurs. On mappe donc `0 → 255`
(ignore) et on garde les classes réelles 1..6 — exactement ce que fait Mamba-FCS
(`label_clf[label_clf == 0] = 255`). Aucun décalage d'indices.

Conséquence notable : contrairement à Hi-UCD (sémantique pleine scène), ici la
tête sémantique s'entraîne directement sur la population que SeK mesure. Le
problème de kappa négatif rencontré sur Hi-UCD ne devrait pas s'y poser.

Arborescence attendue (version prétraitée de ChangeMamba, cf.
https://zenodo.org/records/14037769 — cartes sémantiques déjà en mono-canal et
cartes de changement binaires déjà générées) :

    root/<split>/im1/<id>.png       image T1
    root/<split>/im2/<id>.png       image T2
    root/<split>/label1/<id>.png    sémantique T1 (mono-canal, 0..6)
    root/<split>/label2/<id>.png    sémantique T2 (mono-canal, 0..6)
    root/<split>/GT_CD/<id>.png     changement binaire

⚠️ Le dataset ORIGINAL fournit les cartes sémantiques en RGB (pour la
visualisation) et **sans** carte de changement binaire. Utiliser la version
prétraitée ci-dessus, ou convertir au préalable.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..losses.composite import IGNORE_INDEX

# 7 canaux : index 0 réservé (inchangé/non annoté), classes réelles 1..6.
NUM_SEMANTIC_CLASSES = 7

CLASS_NAMES = (
    "reserved", "non-vegetated ground", "tree", "low vegetation",
    "water", "building", "playground",
)


def _map_semantic(index_map: np.ndarray) -> np.ndarray:
    """Classes réelles 1..6 conservées ; 0 (hors changement) -> ignore."""
    mapped = index_map.astype(np.int64)
    mapped[index_map == 0] = IGNORE_INDEX
    return mapped


class SECONDDataset(Dataset):
    def __init__(self, root: str, split: str = "train", transform=None):
        self.root = Path(root) / split
        self.transform = transform
        self.dirs = {
            name: self.root / name
            for name in ("im1", "im2", "label1", "label2", "GT_CD")
        }
        missing = [str(p) for p in self.dirs.values() if not p.is_dir()]
        if missing:
            raise FileNotFoundError(
                "Dossiers SECOND introuvables : " + ", ".join(missing)
                + "\nVérifier l'arborescence (cf. docstring) — utiliser de préférence "
                "la version prétraitée de ChangeMamba (Zenodo 14037769)."
            )

        self.ids = sorted(p.name for p in self.dirs["im1"].glob("*.png"))
        if not self.ids:
            raise RuntimeError(f"aucune image .png dans {self.dirs['im1']}")

    def __len__(self) -> int:
        return len(self.ids)

    def _load_rgb(self, folder: str, name: str) -> torch.Tensor:
        arr = np.asarray(
            Image.open(self.dirs[folder] / name).convert("RGB"), dtype=np.float32
        ) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

    def _load_index_map(self, folder: str, name: str) -> np.ndarray:
        """Carte d'indices mono-canal. Refuse le RGB (dataset non prétraité)."""
        arr = np.asarray(Image.open(self.dirs[folder] / name))
        if arr.ndim != 2:
            raise ValueError(
                f"{self.dirs[folder] / name} a {arr.ndim} dimensions : les cartes "
                "sémantiques doivent être mono-canal. Le dataset SECOND original "
                "les fournit en RGB — utiliser la version prétraitée."
            )
        return arr

    def __getitem__(self, idx: int) -> dict:
        name = self.ids[idx]
        img_t1 = self._load_rgb("im1", name)
        img_t2 = self._load_rgb("im2", name)

        sem_t1 = torch.from_numpy(_map_semantic(self._load_index_map("label1", name)))
        sem_t2 = torch.from_numpy(_map_semantic(self._load_index_map("label2", name)))

        # Carte de changement : binarisée (certains dumps encodent 0/255).
        change_raw = self._load_index_map("GT_CD", name)
        change = torch.from_numpy((change_raw > 0).astype(np.int64))
        unchanged = change == 0  # masque pour L_sc

        sample = {
            "img_t1": img_t1, "img_t2": img_t2,
            "sem_t1": sem_t1, "sem_t2": sem_t2,
            "change": change, "unchanged": unchanged,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
