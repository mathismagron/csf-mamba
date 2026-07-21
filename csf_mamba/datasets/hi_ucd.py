"""Dataloader Hi-UCD (§5.1 du plan).

Chaque masque est un PNG 3 canaux 512×512 :
    canal 1 = sémantique T1 (pleine scène)
    canal 2 = sémantique T2 (pleine scène)
    canal 3 = changement (unchanged / change)

Index bruts :
    sémantique : 0 unlabeled, 1 water, 2 grass, 3 building, 4 green house,
                 5 road, 6 bridge, 7 others, 8 bare land, 9 woodland
    changement : 0 unlabeled, 1 unchanged, 2 change

CONVENTION A (index 0 réservé) — retenue pour être identique à la repro Mamba-FCS
sur SECOND (une seule config de loss pour les deux datasets, cf. §12.1) :
  * Sémantique : PAS de décalage. Les classes réelles restent à 1..9 ; `unlabeled`
    (0) est mappé sur IGNORE_INDEX=255. Les têtes sémantiques sortent donc 10
    canaux (index 0 réservé, jamais une cible → ~0 param en plus), et la SeK-loss
    exclut `non_change_class=0` exactement comme sur SECOND.
  * Changement (cible BCD binaire) : décalé de −1 → 1 unchanged=0, 2 change=1,
    0 unlabeled=255.

Arborescence RÉELLE du dump Hi-UCD-S (vérifiée sur le zip officiel) :
    root/<split>/image/2018/<id>.png     (date T1)
    root/<split>/image/2019/<id>.png     (date T2)
    root/<split>/mask/2018_2019/<id>.png (masque 3 canaux)
Seule la paire 2018→2019 est annotée. Le split `test/` n'a PAS de masque
(labels non fournis) → validation sur le split `val/`. Format et plages de
valeurs des masques vérifiés conformes aux hypothèses ci-dessus (sém 0..9,
changement 0/1/2).
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..losses.composite import IGNORE_INDEX

# 10 canaux : index 0 réservé (unlabeled/ignore), classes réelles 1..9.
NUM_SEMANTIC_CLASSES = 10


def _map_semantic(index_map: np.ndarray) -> np.ndarray:
    """Convention A : classes réelles 1..9 conservées, 0 (unlabeled) -> ignore."""
    mapped = index_map.astype(np.int64)
    mapped[index_map == 0] = IGNORE_INDEX
    return mapped


def _shift_change(index_map: np.ndarray) -> np.ndarray:
    """Cible BCD : décale de −1 ; 0 (unlabeled) devient IGNORE_INDEX."""
    shifted = index_map.astype(np.int64) - 1
    shifted[index_map == 0] = IGNORE_INDEX
    return shifted


class HiUCDDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        year_t1: str = "2018",
        year_t2: str = "2019",
    ):
        self.root = Path(root) / split
        self.transform = transform
        self.year_t1 = year_t1
        self.year_t2 = year_t2
        self.mask_dir = self.root / "mask" / f"{year_t1}_{year_t2}"
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(
                f"{self.mask_dir} introuvable. Vérifier l'arborescence Hi-UCD "
                "(cf. docstring) et le staging sur $SLURM_TMPDIR. Le split `test` "
                "n'a pas de masque — utiliser `val` pour la validation."
            )
        self.ids = sorted(p.stem for p in self.mask_dir.glob("*.png"))
        if not self.ids:
            raise RuntimeError(f"aucun masque .png dans {self.mask_dir}")

    def __len__(self) -> int:
        return len(self.ids)

    def _load_image(self, year: str, sample_id: str) -> torch.Tensor:
        path = self.root / "image" / year / f"{sample_id}.png"
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

    def __getitem__(self, idx: int) -> dict:
        sample_id = self.ids[idx]
        img_t1 = self._load_image(self.year_t1, sample_id)
        img_t2 = self._load_image(self.year_t2, sample_id)

        label = np.asarray(Image.open(self.mask_dir / f"{sample_id}.png"))
        if label.ndim != 3 or label.shape[2] < 3:
            raise ValueError(f"masque {sample_id} attendu 3 canaux, reçu {label.shape}")

        sem_t1 = torch.from_numpy(_map_semantic(label[..., 0]))
        sem_t2 = torch.from_numpy(_map_semantic(label[..., 1]))
        change_raw = label[..., 2]
        # changement : 1 unchanged -> 0, 2 change -> 1 ; 0 unlabeled -> ignore.
        change = _shift_change(change_raw)
        unchanged = torch.from_numpy(change_raw == 1)  # masque pour L_sc

        sample = {
            "img_t1": img_t1, "img_t2": img_t2,
            "sem_t1": sem_t1, "sem_t2": sem_t2,
            "change": torch.from_numpy(change), "unchanged": unchanged,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
