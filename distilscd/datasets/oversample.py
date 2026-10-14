"""Sur-échantillonnage des tuiles contenant du changement.

**Le problème.** Sur Hi-UCD, seules ~9,4 % des tuiles d'entraînement contiennent du
changement (contre 99,9 % sur SECOND). Avec un tirage uniforme, 90 % des exemples
n'apportent aucun signal pour localiser ou classer un changement — et l'écart de
performance entre les deux datasets (SeK 0,015 contre 0,19) suit ce rapport de
densité.

**Le principe.** On donne aux tuiles avec changement un poids `oversample` fois
supérieur, et on tire avec remise. Avec un facteur 3 sur Hi-UCD, la densité passe de
9,4 % à ~24 %.

**Le compromis à surveiller.** Un facteur élevé fait revoir souvent les mêmes ~1 130
tuiles (risque de sur-apprentissage) et raréfie les exemples négatifs (risque de
faux positifs). D'où un défaut modéré et deux garde-fous mesurables : la courbe de
SeK en validation (qui redescendrait) et le diagnostic « % de changement prédit »
(qui augmenterait). La validation reste **strictement uniforme** : les métriques
restent honnêtes.
"""

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class _ChangeFractionOnly(Dataset):
    """Vue légère d'un dataset : ne lit que le masque de changement."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.change_fraction(idx)


def build_change_index(dataset, cache_path=None, num_workers=8, verbose=True):
    """Fraction de changement par tuile, en ne lisant que les masques.

    Met en cache sur disque : un run repris ne recalcule pas.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.is_file():
            fractions = np.load(cache_path)
            if len(fractions) == len(dataset):
                if verbose:
                    print(f"index de changement lu depuis {cache_path}")
                return fractions
            if verbose:
                print(f"cache {cache_path} obsolète ({len(fractions)} != {len(dataset)}), "
                      "recalcul")

    start = time.time()
    loader = DataLoader(_ChangeFractionOnly(dataset), batch_size=64,
                        num_workers=num_workers, shuffle=False)
    fractions = torch.cat([b.float() for b in loader]).numpy()

    if verbose:
        with_change = (fractions > 0).sum()
        print(f"index de changement construit en {time.time() - start:.0f} s : "
              f"{with_change}/{len(fractions)} tuiles avec changement "
              f"({100 * with_change / len(fractions):.1f} %)")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, fractions)
    return fractions


def make_change_sampler(fractions, oversample, verbose=True):
    """WeightedRandomSampler donnant un poids `oversample` aux tuiles avec changement.

    Taille d'époque inchangée (len(fractions) tirages, avec remise).
    """
    has_change = fractions > 0
    weights = np.where(has_change, float(oversample), 1.0)

    if verbose:
        n = len(fractions)
        density_before = 100 * has_change.sum() / n
        density_after = 100 * weights[has_change].sum() / weights.sum()
        repeats = (weights[has_change].sum() / weights.sum() * n
                   / max(1, has_change.sum()))
        print(f"sur-échantillonnage x{oversample} : densité de signal "
              f"{density_before:.1f} % -> {density_after:.1f} % "
              f"(chaque tuile avec changement vue ~{repeats:.1f} fois/époque)")

    return WeightedRandomSampler(
        torch.from_numpy(weights).double(), num_samples=len(fractions), replacement=True
    )
