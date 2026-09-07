"""Transforms pour l'entraînement SCD.

Appliqués sur le dict d'échantillon complet (images + toutes les cibles) pour
garantir un recadrage/retournement COHÉRENT entre les modalités.

⚠️ **À la recette retenue, le crop ne fait rien.** `--crop-size 512` vaut la
taille de la tuile SECOND : `RandomCrop` tire alors toujours l'unique fenêtre
possible. Il ne restait donc que les deux flips, soit **4 variantes** pour
2 968 paires d'entraînement — le régime d'augmentation le plus pauvre du projet,
et il est resté invisible jusqu'en septembre parce que le crop *semblait* en être
une. D'où les transforms ci-dessous, toutes désactivées par défaut pour que les
runs déjà mesurés restent reproductibles à l'identique.
"""

import random

import torch

# Champs spatiaux d'un échantillon et leur nombre de dims.
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


class RandomRot90:
    """Rotation d'un multiple de 90°, commune à tous les champs.

    Avec les deux flips, on obtient le groupe diédral complet : 8 variantes au
    lieu de 4. Licite ici — une image aérienne n'a pas d'orientation privilégiée,
    contrairement à une photo au sol où le ciel est en haut.
    """

    def __call__(self, sample: dict) -> dict:
        k = random.randint(0, 3)
        if k == 0:
            return sample
        for f in _CHW_FIELDS:
            sample[f] = torch.rot90(sample[f], k, dims=[1, 2]).contiguous()
        for f in _HW_FIELDS:
            sample[f] = torch.rot90(sample[f], k, dims=[0, 1]).contiguous()
        return sample


class PhotometricJitter:
    """Jitter luminosité / contraste / saturation, **indépendant par date**.

    L'indépendance est le point. SECOND présente de fortes différences
    radiométriques entre T1 et T2 (dates, saisons, capteurs) qui corrèlent avec
    le changement sans le causer : un modèle peut apprendre à signaler « la
    radiométrie globale a bougé » plutôt que « le sol a changé ». Perturber
    chaque date séparément casse ce raccourci.

    N'agit que sur les images ; les cibles ne sont pas touchées.
    """

    def __init__(self, strength: float = 0.2):
        self.strength = strength

    def _jitter_one(self, img: torch.Tensor) -> torch.Tensor:
        s = self.strength
        img = img * random.uniform(1 - s, 1 + s)                       # luminosité
        mean = img.mean()
        img = (img - mean) * random.uniform(1 - s, 1 + s) + mean       # contraste
        gray = img.mean(dim=0, keepdim=True)
        img = (img - gray) * random.uniform(1 - s, 1 + s) + gray       # saturation
        return img.clamp_(0.0, 1.0)

    def __call__(self, sample: dict) -> dict:
        for f in _CHW_FIELDS:
            sample[f] = self._jitter_one(sample[f])
        return sample


class RandomTemporalSwap:
    """Échange T1 <-> T2, cibles sémantiques échangées avec.

    ⚠️ **Ce n'est pas une symétrie exacte de la tâche.** Le masque de changement,
    lui, est bien invariant à l'ordre. Mais les *transitions* sémantiques de
    SECOND sont directionnelles : « végétation -> bâtiment » y est bien plus
    fréquent que l'inverse, parce que les paires vont toujours du passé vers le
    futur. Échanger fabrique donc des échantillons tirés d'une distribution qui
    n'apparaît jamais au test.

    Deux effets opposés, et c'est une question empirique de savoir lequel domine :
    on double les données et on force un raisonnement bi-temporel réel, mais on
    dilue un a priori directionnel que le jeu de test, lui, respecte. D'où
    l'ablation séparée du reste de l'augmentation (cf. lot A du plan).
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            sample["img_t1"], sample["img_t2"] = sample["img_t2"], sample["img_t1"]
            sample["sem_t1"], sample["sem_t2"] = sample["sem_t2"], sample["sem_t1"]
            # `change` et `unchanged` sont invariants à l'ordre : rien à faire.
        return sample


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


def train_transform(crop_size: int, rot90: bool = False, photometric: float = 0.0,
                    temporal_swap: float = 0.0):
    """Augmentation d'entraînement. Les défauts reproduisent les runs d'août.

    Aux valeurs par défaut, la chaîne est exactement celle de la campagne d'août
    — crop + flips si `crop_size > 0`, rien du tout sinon. Les runs déjà mesurés
    restent donc comparables aux nouveaux sans réserve.
    """
    extra = rot90 or photometric > 0 or temporal_swap > 0
    if not (crop_size and crop_size > 0) and not extra:
        return None
    steps = []
    if crop_size and crop_size > 0:
        steps.append(RandomCrop(crop_size))
    steps.append(RandomFlip())
    if rot90:
        steps.append(RandomRot90())
    if photometric > 0:
        steps.append(PhotometricJitter(photometric))
    if temporal_swap > 0:
        steps.append(RandomTemporalSwap(temporal_swap))
    return Compose(steps)
