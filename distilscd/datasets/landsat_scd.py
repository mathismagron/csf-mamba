"""Dataloader Landsat-SCD — troisième jeu de données du projet.

2 425 paires 416×416, résolution 30 m, Landsat de 1990 à 2020 autour de
Tumushuke (Xinjiang, Chine), en bordure du Taklamakan. **4 classes réelles** —
farmland, desert, buildings, water — pour 10 types de transition, plus une
catégorie « sans changement ». Soit 5 catégories, ce qui correspond exactement à
notre **convention A** : index 0 réservé, classes réelles 1..4.

POURQUOI CE JEU
===============
L'énoncé d'efficience du projet ne repose aujourd'hui que sur SECOND, et le
résultat principal — le retrait de la loss SeK — **ne transfère pas** à Hi-UCD.
Un troisième terrain est la seule chose qui manque pour que la conclusion soit
défendable ailleurs que sur un jeu unique. Landsat-SCD est le bon candidat parce
que **Mamba-FCS le rapporte** : la comparaison est directe.

FORMAT, REPRIS DE LEUR DATALOADER ET NON DEVINÉ
===============================================
Source : `third_party/MambaFCS/changedetection/datasets/make_data_loader.py`
(classe `SemanticChangeDetectionDatset_LandSat`) et `configs/train_LANDSAT.yaml`.

    root/A/<id>.png          image T1 (RGB)
    root/B/<id>.png          image T2 (RGB)
    root/labelA/<id>.png     sémantique T1, carte d'indices 0..4
    root/labelB/<id>.png     sémantique T2, carte d'indices 0..4
    root/train_list.txt      listes officielles
    root/val_list.txt
    root/test_list.txt

⚠️ **Il n'y a pas de carte de changement binaire dans le dump.** Mamba-FCS la
dérive des deux cartes sémantiques :

    cd_label[(t1_label > 0) | (t2_label > 0)] = 1

On reprend cette règle à l'identique. Elle dit que la sémantique n'est annotée
que dans les zones changées — même convention que SECOND, et donc `0 → ignore`
pour la sémantique.

⚠️ **Mamba-FCS entraîne sur `train_list.txt` + `val_list.txt`** et évalue sur
`test_list.txt` (1 908 paires d'entraînement pour 477 de test). Pour que notre
comparaison à leur chiffre soit appariée, `split="train"` concatène les deux
listes — c'est le comportement par défaut, et `--landsat-train-only` le
désactive si l'on veut un vrai split de validation.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..losses.composite import IGNORE_INDEX

# 5 canaux : index 0 réservé (inchangé / non annoté), classes réelles 1..4.
NUM_SEMANTIC_CLASSES = 5

# ⚠️ ORDRE NON VÉRIFIÉ. La littérature cite « farmland, desert, buildings,
# water » mais rien ne garantit que ce soit l'ordre des INDICES du dump. Le même
# piège a fait porter de faux noms de classes à SECOND pendant deux semaines, et
# seule une vérification sur les fichiers l'a levé.
#
#     python -m scripts.check_landsat --data-root <racine>
#
# Aucune métrique n'en dépend — tout est calculé sur des indices — mais un nom
# faux dans un rapport ou une figure serait une erreur factuelle.
CLASS_NAMES = ("reserved", "farmland", "desert", "buildings", "water")
CLASS_NAMES_VERIFIED = False


def _map_semantic(index_map: np.ndarray) -> np.ndarray:
    """Classes réelles 1..4 conservées ; 0 (hors changement) -> ignore."""
    mapped = index_map.astype(np.int64)
    mapped[index_map == 0] = IGNORE_INDEX
    return mapped


def apparier_dossiers(root: Path, dossiers=("A", "B", "label")) -> tuple[list, dict]:
    """Noms utilisables dans TOUS les dossiers, et rapport des écarts.

    ⚠️ Le dump figshare n'est pas cohérent d'un dossier à l'autre : `label/`
    contient `…01Zhedang2.png` là où `A/` contient `…01ZheDang2.png`. Sur Linux
    les noms sont sensibles à la casse, donc une jointure naïve par nom de
    fichier échoue — c'est ce qui a fait planter le premier décodage.

    On apparie donc sur le nom **en minuscules**, mais seulement après avoir
    vérifié qu'aucun dossier ne contient deux fichiers ne différant que par la
    casse : dans ce cas l'appariement serait ambigu et il faut s'arrêter.

    -> (noms appariés, rapport) où `rapport[dossier]` donne le nom réel à
       utiliser pour chaque clé.
    """
    par_dossier, collisions = {}, {}
    for d in dossiers:
        m = {}
        for f in (root / d).iterdir():
            if not f.is_file():
                continue
            cle = f.name.lower()
            if cle in m:
                collisions.setdefault(d, []).append((m[cle], f.name))
            m[cle] = f.name
        par_dossier[d] = m

    if collisions:
        detail = "; ".join(f"{d} : {v[:2]}" for d, v in collisions.items())
        raise ValueError(
            "Deux fichiers ne différant que par la casse dans un même dossier — "
            f"l'appariement serait ambigu : {detail}"
        )

    communs = set.intersection(*(set(m) for m in par_dossier.values()))
    # Distinguer « mêmes noms » de « mêmes noms à la casse près » : dire que les
    # dossiers concordent alors qu'ils ne concordent qu'en minuscules serait
    # trompeur, et masquerait précisément le défaut du dump.
    casse_seule = sorted(
        c for c in communs
        if len({par_dossier[d][c] for d in dossiers}) > 1
    )
    rapport = {
        "communs": sorted(communs),
        "par_dossier": par_dossier,
        "exclus": {d: sorted(set(m) - communs) for d, m in par_dossier.items()},
        "casse_seule": casse_seule,
    }
    return sorted(communs), rapport


class LandsatSCDDataset(Dataset):
    def __init__(self, root: str, split: str = "train", transform=None,
                 merge_val_into_train: bool = True):
        root = Path(root)
        self.root = root
        self.transform = transform
        self.dirs = {name: root / name for name in ("A", "B", "labelA", "labelB")}
        missing = [str(p) for p in self.dirs.values() if not p.is_dir()]
        if missing:
            raise FileNotFoundError(
                "Dossiers Landsat-SCD introuvables : " + ", ".join(missing)
                + "\nArborescence attendue (cf. docstring) : A/, B/, labelA/, labelB/ "
                "et les listes train_list.txt / val_list.txt / test_list.txt."
            )

        # Mamba-FCS entraîne sur train + val : on fait pareil pour que la
        # comparaison à leur chiffre soit appariée.
        listes = ["train_list.txt", "val_list.txt"] if (
            split == "train" and merge_val_into_train) else [f"{split}_list.txt"]
        self.ids = []
        for nom in listes:
            chemin = root / nom
            if not chemin.is_file():
                if nom == "val_list.txt" and split == "train":
                    continue  # certains dumps n'ont pas de val : ce n'est pas fatal
                raise FileNotFoundError(f"liste introuvable : {chemin}")
            self.ids += [l.strip() for l in chemin.read_text().splitlines() if l.strip()]
        if not self.ids:
            raise RuntimeError(f"aucun échantillon pour le split '{split}'")

    def __len__(self) -> int:
        return len(self.ids)

    def change_fraction(self, idx: int) -> float:
        """Fraction de pixels changés, sans charger les images.

        Le changement se déduit des deux cartes sémantiques, il faut donc les
        deux — contrairement à SECOND qui fournit une carte dédiée.
        """
        name = self.ids[idx]
        a = np.asarray(Image.open(self.dirs["labelA"] / name))
        b = np.asarray(Image.open(self.dirs["labelB"] / name))
        return float(((a > 0) | (b > 0)).mean())

    def _load_rgb(self, folder: str, name: str) -> torch.Tensor:
        arr = np.asarray(
            Image.open(self.dirs[folder] / name).convert("RGB"), dtype=np.float32
        ) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

    def _load_index_map(self, folder: str, name: str) -> np.ndarray:
        """Carte d'indices mono-canal. Refuse le RGB, comme pour SECOND."""
        arr = np.asarray(Image.open(self.dirs[folder] / name))
        if arr.ndim != 2:
            raise ValueError(
                f"{self.dirs[folder] / name} a {arr.ndim} dimensions : les cartes "
                "sémantiques de Landsat-SCD doivent être mono-canal (indices 0..4). "
                "Un dump en RGB demanderait une table de correspondance couleur -> "
                "indice ; lancer `python -m scripts.check_landsat` pour diagnostiquer."
            )
        return arr

    def __getitem__(self, idx: int) -> dict:
        name = self.ids[idx]
        img_t1 = self._load_rgb("A", name)
        img_t2 = self._load_rgb("B", name)

        brut_t1 = self._load_index_map("labelA", name)
        brut_t2 = self._load_index_map("labelB", name)

        # Règle de Mamba-FCS, reprise verbatim : changé là où l'une des deux
        # cartes porte une classe. Pas de carte binaire dans le dump.
        change = torch.from_numpy(((brut_t1 > 0) | (brut_t2 > 0)).astype(np.int64))

        sem_t1 = torch.from_numpy(_map_semantic(brut_t1))
        sem_t2 = torch.from_numpy(_map_semantic(brut_t2))
        unchanged = change == 0  # masque pour L_sc

        sample = {
            "img_t1": img_t1, "img_t2": img_t2,
            "sem_t1": sem_t1, "sem_t2": sem_t2,
            "change": change, "unchanged": unchanged,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
