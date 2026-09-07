"""Non-régression des augmentations ajoutées en septembre et de l'EMA.

Trois défauts sont visés, tous silencieux — ils ne lèvent aucune erreur, ils
dégradent seulement le résultat, et le projet en a déjà connu trois du même
genre (« paramètre accepté mais non propagé ») :

1. une transform qui désynchronise images et cibles (le modèle apprendrait sur
   des étiquettes décalées) ;
2. `train_transform` aux valeurs par défaut qui ne reproduirait plus la chaîne
   d'août, rendant les 7 graines déjà mesurées incomparables aux nouvelles ;
3. un état EMA perdu à la reprise — invisible, alors que tous les runs à
   200 époques reprennent au moins une fois.

    python -m pytest tests/test_augment_ema.py -v
"""

import random

import torch
from torch import nn

from csf_mamba.datasets.transforms import (
    Compose, PhotometricJitter, RandomCrop, RandomFlip, RandomRot90,
    RandomTemporalSwap, train_transform,
)
from csf_mamba.ema import ModelEMA

H = W = 16


def _sample():
    """Échantillon où chaque pixel est identifiable, pour suivre les permutations.

    Les cibles portent le même motif que le canal 0 de l'image : toute transform
    correcte les déplace exactement de la même façon, ce qui rend une
    désynchronisation détectable par simple égalité.
    """
    ramp = torch.arange(H * W, dtype=torch.float32).view(H, W)
    return {
        "img_t1": torch.stack([ramp, ramp + 1000, ramp + 2000]),
        "img_t2": torch.stack([ramp + 3000, ramp + 4000, ramp + 5000]),
        "sem_t1": ramp.long(), "sem_t2": ramp.long() + 10000,
        "change": ramp.long(), "unchanged": ramp.long(),
    }


def _assert_synchronized(s, msg):
    """Le canal 0 de img_t1 et les cartes HW doivent avoir subi la MÊME géométrie."""
    ref = s["img_t1"][0]
    for field in ("sem_t1", "change", "unchanged"):
        assert torch.equal(s[field].float(), ref), f"{msg} : {field} désynchronisé"


def test_geometric_transforms_keep_fields_aligned():
    random.seed(0)
    for name, t in [("flip", RandomFlip(p=1.0)), ("rot90", RandomRot90()),
                    ("crop", RandomCrop(8))]:
        for _ in range(20):
            _assert_synchronized(t(_sample()), name)


def test_rot90_covers_the_four_rotations():
    """Sinon la rotation serait câblée mais toujours nulle — un no-op invisible."""
    random.seed(0)
    seen = {tuple(RandomRot90()(_sample())["change"].flatten()[:4].tolist())
            for _ in range(60)}
    assert len(seen) == 4, f"{len(seen)} rotations distinctes au lieu de 4"


def test_photometric_touches_images_only_and_each_date_independently():
    random.seed(0)
    s = _sample()
    s["img_t1"] = torch.rand(3, H, W)
    s["img_t2"] = s["img_t1"].clone()
    before = {k: s[k].clone() for k in ("sem_t1", "sem_t2", "change", "unchanged")}
    out = PhotometricJitter(0.3)(s)

    for k, v in before.items():
        assert torch.equal(out[k], v), f"{k} modifié par un jitter photométrique"
    # Deux dates initialement identiques doivent diverger : c'est tout l'intérêt
    # du tirage indépendant (casser le raccourci « la radiométrie a bougé »).
    assert not torch.allclose(out["img_t1"], out["img_t2"])
    for f in ("img_t1", "img_t2"):
        assert out[f].min() >= 0.0 and out[f].max() <= 1.0, "sortie hors de [0, 1]"


def test_temporal_swap_exchanges_dates_and_leaves_change_invariant():
    s = _sample()
    original = {k: v.clone() for k, v in s.items()}
    out = RandomTemporalSwap(p=1.0)(s)

    assert torch.equal(out["img_t1"], original["img_t2"])
    assert torch.equal(out["img_t2"], original["img_t1"])
    assert torch.equal(out["sem_t1"], original["sem_t2"])
    assert torch.equal(out["sem_t2"], original["sem_t1"])
    # Le masque de changement, lui, est bien invariant à l'ordre des dates.
    assert torch.equal(out["change"], original["change"])
    assert torch.equal(out["unchanged"], original["unchanged"])


def test_defaults_reproduce_the_august_pipeline():
    """Les 7 graines déjà mesurées doivent rester comparables aux runs à venir."""
    t = train_transform(512)
    assert isinstance(t, Compose)
    assert [type(x).__name__ for x in t.transforms] == ["RandomCrop", "RandomFlip"]
    assert train_transform(0) is None, "crop<=0 sans augmentation doit rester un no-op"
    assert train_transform(0, rot90=True) is not None


def test_train_transform_wires_every_flag():
    """Un flag accepté mais absent de la chaîne est le défaut n°3 du projet."""
    t = train_transform(512, rot90=True, photometric=0.2, temporal_swap=0.5)
    assert [type(x).__name__ for x in t.transforms] == [
        "RandomCrop", "RandomFlip", "RandomRot90", "PhotometricJitter",
        "RandomTemporalSwap",
    ]


def test_ema_tracks_then_lags_behind_the_model():
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    ema = ModelEMA(model, decay=0.99, warmup=0)
    start = ema.module.weight.detach().clone()

    with torch.no_grad():
        model.weight.add_(10.0)
    ema.update(model)

    moved = (ema.module.weight - start).abs().mean().item()
    gap = (ema.module.weight - model.weight).abs().mean().item()
    assert 0 < moved < 10.0, "l'EMA doit bouger, mais pas suivre le modèle d'un bloc"
    assert gap > 1.0, "l'EMA colle au modèle : le lissage ne sert à rien"


def test_ema_warmup_prevents_sticking_to_initialization():
    """Sans décote, une decay de 0.9998 laisserait l'EMA à l'init pendant 200 pas."""
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    with_warmup = ModelEMA(model, decay=0.9998, warmup=2000)
    without = ModelEMA(model, decay=0.9998, warmup=0)
    with torch.no_grad():
        model.weight.add_(10.0)
    with_warmup.update(model)
    without.update(model)
    assert (with_warmup.module.weight - model.weight).abs().mean() < \
           (without.module.weight - model.weight).abs().mean()


def test_ema_survives_a_checkpoint_round_trip():
    """Le défaut le plus coûteux : tous les runs à 200 époques reprennent."""
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    ema = ModelEMA(model, decay=0.99, warmup=0)
    for _ in range(5):
        with torch.no_grad():
            model.weight.add_(0.1)
        ema.update(model)

    state = ema.state_dict()
    restored = ModelEMA(nn.Linear(4, 4), decay=0.5, warmup=999)
    restored.load_state_dict(state)

    assert restored.updates == ema.updates == 5
    assert restored.decay == 0.99 and restored.warmup == 0, \
        "decay/warmup non repris : la fenêtre de moyennage changerait à la reprise"
    assert torch.equal(restored.module.weight, ema.module.weight)


def test_ema_copies_integer_buffers_instead_of_averaging_them():
    """Moyenner un compteur entier lèverait une erreur ou le figerait à zéro."""
    model = nn.BatchNorm2d(3)
    ema = ModelEMA(model, decay=0.9, warmup=0)
    model(torch.randn(4, 3, 8, 8))          # incrémente num_batches_tracked
    ema.update(model)
    assert ema.module.num_batches_tracked.item() == model.num_batches_tracked.item()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("tous les tests passent")
