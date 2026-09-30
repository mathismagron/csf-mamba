"""LandsatSCD512 (version PerASCD) : dataloader et vérification, sur un jeu synthétique."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from csf_mamba.datasets import DATASETS
from csf_mamba.datasets.landsat_perascd import LandsatPerASCDDataset
from csf_mamba.losses.composite import IGNORE_INDEX

REPO = Path(__file__).resolve().parents[1]


def _write_pair(d: Path, name: str, rng, l1=None, l2=None):
    for f in ("im1", "im2", "label1", "label2"):
        (d / f).mkdir(parents=True, exist_ok=True)
    for f in ("im1", "im2"):
        Image.fromarray(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)).save(d / f / name)
    if l1 is None:
        l1 = np.zeros((512, 512), np.uint8)
        l2 = np.zeros((512, 512), np.uint8)
        l1[:100, :100], l2[:100, :100] = 1, 2      # farmland -> desert
        l1[200:250, :], l2[200:250, :] = 4, 1      # water -> farmland
    Image.fromarray(l1, mode="L").save(d / "label1" / name)
    Image.fromarray(l2, mode="L").save(d / "label2" / name)


@pytest.fixture
def root(tmp_path):
    rng = np.random.default_rng(0)
    for split, n in (("train", 3), ("val", 2), ("test", 2)):
        for i in range(n):
            _write_pair(tmp_path / split, f"From1990To1993_{split}{i:02d}.png", rng)
    return tmp_path


def test_registry():
    cls, n = DATASETS["landsat_perascd"]
    assert cls is LandsatPerASCDDataset and n == 5


def test_sample(root):
    ds = LandsatPerASCDDataset(str(root), "train")
    assert len(ds) == 3
    s = ds[0]
    assert s["img_t1"].shape == (3, 512, 512) and float(s["img_t1"].max()) <= 1.0
    ch = s["change"].numpy()
    assert ch.sum() == 100 * 100 + 50 * 512
    assert set(np.unique(s["sem_t1"].numpy())) == {1, 4, IGNORE_INDEX}
    assert set(np.unique(s["sem_t2"].numpy())) == {1, 2, IGNORE_INDEX}
    assert (s["sem_t1"].numpy()[ch == 0] == IGNORE_INDEX).all()
    assert s["unchanged"].numpy().sum() == 512 * 512 - ch.sum()
    assert abs(ds.change_fraction(0) - ch.mean()) < 1e-9


def test_mismatched_folders(root):
    (root / "train" / "label2" / "From1990To1993_train00.png").unlink()
    with pytest.raises(RuntimeError):
        LandsatPerASCDDataset(str(root), "train")


def test_bad_index(root):
    rng = np.random.default_rng(1)
    l1 = np.full((512, 512), 7, np.uint8)
    _write_pair(root / "train", "bad.png", rng, l1, l1)
    ds = LandsatPerASCDDataset(str(root), "train")
    with pytest.raises(ValueError):
        ds[ds.ids.index("bad.png")]


def _check(root):
    return subprocess.run([sys.executable, "-m", "scripts.check_landsat_perascd", "--data-root", str(root)],
                          cwd=REPO, capture_output=True, text=True)


def test_check_counts_fail_on_synthetic(root):
    # comptes synthétiques (3/2/2) ≠ 1431/477/477 : le contrôle doit refuser
    r = _check(root)
    assert r.returncode == 1 and "1431 attendues" in r.stdout


def test_check_detects_disagreement_and_self_transition(root, monkeypatch, capsys):
    import scripts.check_landsat_perascd as chk
    monkeypatch.setattr(chk, "SPLIT_SIZES", {"train": 4, "val": 2, "test": 2})
    rng = np.random.default_rng(2)
    l1 = np.zeros((512, 512), np.uint8); l2 = l1.copy()
    l1[:10, :10] = 3; l2[:10, :10] = 3          # building -> building
    l1[20:30, :10] = 2                          # T1 annotée, T2 non
    _write_pair(root / "train", "odd.png", rng, l1, l2)
    monkeypatch.setattr(sys, "argv", ["x", "--data-root", str(root)])
    with pytest.raises(SystemExit) as e:
        chk.main()
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "divergent" in out and "elle-même" in out and "attendues" not in out
