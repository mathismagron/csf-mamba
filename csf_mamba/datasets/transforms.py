"""Transforms pour l'entraînement SCD.

Appliqués sur le dict d'échantillon complet (images + toutes les cibles) pour
garantir un recadrage/retournement COHÉRENT entre les modalités.
"""

import random

import torch

# Champs spatiaux d'un échantillon Hi-UCD et leur nombre de dims.
_CHW_FIELDS = ("img_t1", "img_t2")            # (C, H, W)
_HW_FIELDS = ("sem_t1", "sem_t2", "change", "unchanged")  # (H, W)


class RandomCrop:
    """Crop aléatoire commun à tous les champs. Sert aussi d'augmentation."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, sample: dict) -> dict:
        _, h, w = sample["img_t1"].shape
        s = self.size
        if h < s or w < s:
            raise ValueError(f"crop {s} > image {h}x{w}")
        top = random.randint(0, h - s)
        left = random.randint(0, w - s)
        for k in _CHW_FIELDS:
            sample[k] = sample[k][:, top:top + s, left:left + s].contiguous()
        for k in _HW_FIELDS:
            sample[k] = sample[k][top:top + s, left:left + s].contiguous()
        return sample


class RandomFlip:
    """Retournement horizontal/vertical aléatoire, commun à tous les champs."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def _flip(self, sample: dict, dim_chw: int, dim_hw: int):
        for k in _CHW_FIELDS:
            sample[k] = torch.flip(sample[k], dims=[dim_chw])
        for k in _HW_FIELDS:
            sample[k] = torch.flip(sample[k], dims=[dim_hw])

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            self._flip(sample, dim_chw=2, dim_hw=1)  # horizontal
        if random.random() < self.p:
            self._flip(sample, dim_chw=1, dim_hw=0)  # vertical
        return sample


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


def train_transform(crop_size: int):
    """Augmentation d'entraînement : crop aléatoire + flips. None si crop_size<=0."""
    if crop_size and crop_size > 0:
        return Compose([RandomCrop(crop_size), RandomFlip()])
    return None
