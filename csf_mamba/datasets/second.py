"""Dataloader SECOND (Semantic Change Detection dataset).

4 662 paires 512×512, résolution 0,5–3 m, 6 classes sémantiques réelles
(low vegetation, non-vegetated ground, tree, water, building, playground) — cet
ordre est celui des indices RÉELS du dump, vérifié par `scripts/check_second_classes`
contre les cartes `GT_T*_COLORED`, et non celui de l'énumération de l'article.

**Convention native de SECOND = notre convention A** : la sémantique n'est
annotée QUE dans les zones changées, et vaut 0 ailleurs. On mappe donc `0 → 255`
(ignore) et on garde les classes réelles 1..6 — exactement ce que fait Mamba-FCS
(`label_clf[label_clf == 0] = 255`). Aucun décalage d'indices.

Conséquence notable : contrairement à Hi-UCD (sémantique pleine scène), ici la
tête sémantique s'entraîne directement sur la population que SeK mesure. Le
problème de kappa négatif rencontré sur Hi-UCD ne devrait pas s'y poser.

Arborescence réelle de la version prétraitée ChangeMamba (vérifiée sur le dump
Zenodo 15479555) :

    root/<split>/T1/<id>.png        image T1
    root/<split>/T2/<id>.png        image T2
    root/<split>/GT_T1/<id>.png     sémantique T1 (mono-canal, 0..6)
    root/<split>/GT_T2/<id>.png     sémantique T2 (mono-canal, 0..6)
    root/<split>/GT_CD/<id>.png     changement binaire
    root/<split>.txt                liste officielle des échantillons

Les dossiers `GT_T*_COLORED` sont les cartes RGB de visualisation : ignorés.

⚠️ SECOND n'a que les splits **train** et **test** (pas de `val`) : passer
`--val-split test`. Le dataset ORIGINAL (hors version prétraitée) fournit les
cartes sémantiques uniquement en RGB et sans carte de changement binaire.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..losses.composite import IGNORE_INDEX

# 7 canaux : index 0 réservé (inchangé/non annoté), classes réelles 1..6.
NUM_SEMANTIC_CLASSES = 7

# ⚠️ Ordre VÉRIFIÉ, pas déduit. Une première version suivait l'ordre d'énumération
# de l'article SECOND et se trompait sur les indices 1, 2 et 3. Le contrôle
# (`python -m scripts.check_second_classes`) relève la couleur dominante de chaque
# indice dans `GT_T*_COLORED` : pureté 100 % sur 200 tuiles, aucune ambiguïté.
# Aucune métrique n'en dépend — tout est calculé sur des indices — mais un nom
# faux dans un rapport ou une figure serait une erreur factuelle.
CLASS_NAMES = (
    "reserved", "low vegetation", "non-vegetated ground", "tree",
    "water", "building", "playground",
)


def _map_semantic(index_map: np.ndarray) -> np.ndarray:
    """Classes réelles 1..6 conservées ; 0 (hors changement) -> ignore."""
    mapped = index_map.astype(np.int64)
    mapped[index_map == 0] = IGNORE_INDEX
    return mapped


class SECONDDataset(Dataset):
    def __init__(self, root: str, split: str = "train", transform=None):
        root = Path(root)
        self.root = root / split
        self.transform = transform
        self.dirs = {
            name: self.root / name
            for name in ("T1", "T2", "GT_T1", "GT_T2", "GT_CD")
        }
        missing = [str(p) for p in self.dirs.values() if not p.is_dir()]
        if missing:
            raise FileNotFoundError(
                "Dossiers SECOND introuvables : " + ", ".join(missing)
                + "\nVérifier l'arborescence (cf. docstring). Rappel : SECOND n'a "
                "que les splits 'train' et 'test' (pas de 'val')."
            )

        # Liste officielle du split si présente (root/<split>.txt), sinon glob.
        listing = root / f"{split}.txt"
        if listing.is_file():
            self.ids = [
                line if line.endswith(".png") else f"{line}.png"
                for line in (l.strip() for l in listing.read_text().splitlines())
                if line
            ]
        else:
            self.ids = sorted(p.name for p in self.dirs["T1"].glob("*.png"))
        if not self.ids:
            raise RuntimeError(f"aucun échantillon trouvé pour le split '{split}'")

    def __len__(self) -> int:
        return len(self.ids)

    def change_fraction(self, idx: int) -> float:
        """Fraction de pixels changés, en ne lisant QUE la carte de changement."""
        arr = np.asarray(Image.open(self.dirs["GT_CD"] / self.ids[idx]))
        return float((arr > 0).mean())

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
        img_t1 = self._load_rgb("T1", name)
        img_t2 = self._load_rgb("T2", name)

        sem_t1 = torch.from_numpy(_map_semantic(self._load_index_map("GT_T1", name)))
        sem_t2 = torch.from_numpy(_map_semantic(self._load_index_map("GT_T2", name)))

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
