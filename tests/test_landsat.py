"""Non-régression du dataloader Landsat-SCD.

Le format a été repris du code de Mamba-FCS
(`SemanticChangeDetectionDatset_LandSat`), pas deviné. Ces tests vérifient que
notre implémentation en respecte les trois règles qui comptent :

1. **La carte de changement est DÉRIVÉE** des deux cartes sémantiques — le dump
   n'en contient pas — selon `(labelA > 0) | (labelB > 0)`.
2. **La sémantique vaut 0 hors changement**, donc `0 -> ignore_index`, comme sur
   SECOND. Se tromper ici ferait apprendre au modèle une classe « rien » qu'il
   prédirait partout.
3. **L'entraînement concatène train + val**, comme Mamba-FCS, sans quoi notre
   chiffre ne serait pas comparable au leur.

    python -m pytest tests/test_landsat.py -v
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from csf_mamba.datasets import DATASETS
from csf_mamba.datasets.landsat_scd import (
    CLASS_NAMES, CLASS_NAMES_VERIFIED, NUM_SEMANTIC_CLASSES, LandsatSCDDataset,
)
from csf_mamba.losses.composite import IGNORE_INDEX

TAILLE = 32


def _dump(root: Path, n_train=3, n_val=2, n_test=2):
    for d in ("A", "B", "labelA", "labelB"):
        (root / d).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    i = 0
    for nom, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        ids = []
        for _ in range(n):
            f = f"{i:04d}.png"; ids.append(f); i += 1
            for d in ("A", "B"):
                Image.fromarray(rng.integers(0, 256, (TAILLE, TAILLE, 3), np.uint8)).save(root / d / f)
            chg = rng.random((TAILLE, TAILLE)) < 0.4
            for d in ("labelA", "labelB"):
                Image.fromarray((rng.integers(1, 5, (TAILLE, TAILLE)) * chg).astype(np.uint8)).save(root / d / f)
        (root / f"{nom}_list.txt").write_text("\n".join(ids))
    return root


def test_le_dataset_est_enregistre_avec_cinq_canaux():
    cls, n = DATASETS["landsat_scd"]
    assert cls is LandsatSCDDataset
    assert n == NUM_SEMANTIC_CLASSES == 5, "4 classes réelles + l'index 0 réservé"
    assert len(CLASS_NAMES) == 5


def test_train_concatene_train_et_val_comme_mamba_fcs():
    with tempfile.TemporaryDirectory() as d:
        r = _dump(Path(d))
        assert len(LandsatSCDDataset(r, "train")) == 5, "train + val attendus"
        assert len(LandsatSCDDataset(r, "train", merge_val_into_train=False)) == 3
        assert len(LandsatSCDDataset(r, "test")) == 2


def test_le_changement_est_derive_des_deux_semantiques():
    """La règle de Mamba-FCS : changé là où l'une des deux cartes porte une classe."""
    with tempfile.TemporaryDirectory() as d:
        r = Path(d)
        for x in ("A", "B", "labelA", "labelB"):
            (r / x).mkdir(parents=True)
        a = np.zeros((TAILLE, TAILLE), np.uint8); a[:8, :] = 2   # classe sur T1 seulement
        b = np.zeros((TAILLE, TAILLE), np.uint8); b[8:16, :] = 3  # sur T2 seulement
        for x in ("A", "B"):
            Image.fromarray(np.zeros((TAILLE, TAILLE, 3), np.uint8)).save(r / x / "0.png")
        Image.fromarray(a).save(r / "labelA" / "0.png")
        Image.fromarray(b).save(r / "labelB" / "0.png")
        (r / "test_list.txt").write_text("0.png")

        s = LandsatSCDDataset(r, "test")[0]
        attendu = torch.from_numpy(((a > 0) | (b > 0)).astype(np.int64))
        assert torch.equal(s["change"], attendu), "le OU de Mamba-FCS n'est pas respecté"
        assert s["change"][:16].all() and not s["change"][16:].any()
        assert torch.equal(s["unchanged"], s["change"] == 0)


def test_la_semantique_hors_changement_devient_ignore():
    """Sans cela, le modèle apprendrait une classe « rien » et la prédirait partout."""
    with tempfile.TemporaryDirectory() as d:
        s = LandsatSCDDataset(_dump(Path(d)), "test")[0]
        for k in ("sem_t1", "sem_t2"):
            hors = s[k][s["change"] == 0]
            assert (hors == IGNORE_INDEX).all(), f"{k} : pixels inchangés non ignorés"
            dedans = s[k][s["change"] == 1]
            assert dedans.min() >= 1 and dedans.max() <= 4, "classes hors de 1..4"


def test_formes_et_plage_des_images():
    with tempfile.TemporaryDirectory() as d:
        s = LandsatSCDDataset(_dump(Path(d)), "test")[0]
        assert s["img_t1"].shape == (3, TAILLE, TAILLE)
        assert s["sem_t1"].shape == s["change"].shape == (TAILLE, TAILLE)
        assert 0.0 <= s["img_t1"].min() and s["img_t1"].max() <= 1.0, "images hors [0, 1]"


def test_un_label_rgb_est_refuse_explicitement():
    """Un dump coloré doit produire un message clair, pas un plantage obscur."""
    with tempfile.TemporaryDirectory() as d:
        r = Path(d)
        for x in ("A", "B", "labelA", "labelB"):
            (r / x).mkdir(parents=True)
        for x in ("A", "B"):
            Image.fromarray(np.zeros((TAILLE, TAILLE, 3), np.uint8)).save(r / x / "0.png")
        for x in ("labelA", "labelB"):
            Image.fromarray(np.zeros((TAILLE, TAILLE, 3), np.uint8)).save(r / x / "0.png")
        (r / "test_list.txt").write_text("0.png")
        try:
            LandsatSCDDataset(r, "test")[0]
            assert False, "un label RGB devrait être refusé"
        except ValueError as e:
            assert "mono-canal" in str(e) and "check_landsat" in str(e)


def test_les_noms_de_classes_sont_marques_non_verifies():
    """Le même piège a fait porter de faux noms à SECOND pendant deux semaines."""
    assert CLASS_NAMES_VERIFIED is False, (
        "si les noms ont été vérifiés sur le dump, passer le drapeau à True "
        "et le dire dans le journal"
    )


def test_fraction_de_changement_sans_charger_les_images():
    with tempfile.TemporaryDirectory() as d:
        ds = LandsatSCDDataset(_dump(Path(d)), "test")
        f = ds.change_fraction(0)
        assert 0.0 <= f <= 1.0


if __name__ == "__main__":
    for nom, fn in sorted(globals().items()):
        if nom.startswith("test_"):
            fn(); print(f"  ok  {nom}")
    print("tous les tests passent")
