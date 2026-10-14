"""Tests des contrôles de la révision (vague A), CPU.

1. `sem_target="gt_ls"` : cible = distribution de probabilité, one-hot lissé à l'intérieur
   des zones étiquetées, uniforme ailleurs ; la perte NE DÉPEND PAS des logits sémantiques du
   professeur (aucune fuite de son savoir) ; `teacher` reste le défaut, à l'identique.
2. `matched_eps` : bornes et formule.
3. `cache_student.py` de bout en bout (élève conv. aléatoire, SECOND synthétique) : le cache
   relu par `TeacherCacheDataset` vaut, vue par vue, la sortie de l'élève sur l'entrée
   transformée moyennée 4 × 4 ; meta.json au format attendu.

    PYTHONPATH=. python -m pytest -q tests/test_revision_controls.py
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from distilscd.datasets.second import SECONDDataset
from distilscd.datasets.teacher_cache import TeacherCacheDataset
from distilscd.distill.d4 import D4, apply_d4
from distilscd.losses.distill import DistillLoss

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_distill import NATIVE, TILE, _make_second  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _outputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"bcd": torch.randn(2, 2, TILE, TILE, generator=g),
            "sem_t1": torch.randn(2, 7, TILE, TILE, generator=g),
            "sem_t2": torch.randn(2, 7, TILE, TILE, generator=g)}


def _labels():
    change = torch.zeros(2, TILE, TILE, dtype=torch.long)
    change[:, 8:24, 4:20] = 1
    s1 = torch.full((2, TILE, TILE), 255, dtype=torch.long)
    s2 = s1.clone()
    s1[change == 1], s2[change == 1] = 3, 5
    return change, s1, s2


def test_sem_gt_ls_target_and_independence():
    change, s1, s2 = _labels()
    out = _outputs()
    ta, tb = torch.randn(2, 15, NATIVE, NATIVE), torch.randn(2, 15, NATIVE, NATIVE)
    tb[:, 14] = ta[:, 14]                     # même logit de changement, sémantique différente
    loss = DistillLoss(0.0, 1.0, 1.0, 2.0, "all", sem_target="gt_ls", sem_eps=0.2)
    la = loss(out, ta, change, sem_gt=(s1, s2))["kd_sem"]
    lb = loss(out, tb, change, sem_gt=(s1, s2))["kd_sem"]
    assert torch.equal(la, lb) and la > 0
    # la KD proprement dite dépend, elle, du professeur, et reste le défaut
    kd = DistillLoss(0.0, 1.0, 1.0, 2.0, "all")
    assert not torch.equal(kd(out, ta, change)["kd_sem"], kd(out, tb, change)["kd_sem"])
    assert torch.equal(kd(out, ta, change)["kd_sem"],
                       DistillLoss(0.0, 1.0, 1.0, 2.0, "all", sem_target="teacher")(out, ta, change,
                                                                                    sem_gt=(s1, s2))["kd_sem"])
    q = loss._sem_gt_target(s1, 7, (NATIVE, NATIVE), (TILE, TILE))
    assert q.shape == (2, 6, TILE, TILE)
    assert torch.allclose(q.sum(1), torch.ones(2, TILE, TILE), atol=1e-5)
    inner = q[:, :, 14:18, 10:14]             # loin des bords : one-hot lissé exact
    assert torch.allclose(inner[:, 2], torch.full_like(inner[:, 2], 0.8 + 0.2 / 6), atol=1e-5)
    assert torch.allclose(q[:, :, 0:2, 26:], torch.full_like(q[:, :, 0:2, 26:], 1 / 6), atol=1e-5)
    try:
        loss(out, ta, change)
    except ValueError:
        pass
    else:
        raise AssertionError("gt_ls sans sem_gt doit échouer")


def test_matched_eps():
    m = _load(REPO / "scripts/kd_control_eps.py", "kd_control_eps")
    assert m.matched_eps(1.0, 7) == 0.0
    assert abs(m.matched_eps(1 / 6, 7) - 1.0) < 1e-9
    eps = m.matched_eps(0.7, 5)
    assert abs((1 - eps) + eps / 4 - 0.7) < 1e-9


def test_cache_student_end_to_end():
    cs = _load(REPO / "scripts/teacher/cache_student.py", "cache_student")
    from distilscd.model import DistilSCDStudent
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _make_second(tmp / "SECOND", np.random.default_rng(0))
        cs.NATIVE = NATIVE
        cs.STUDENT_KW = dict(encoder="conv")
        torch.manual_seed(0)
        model = DistilSCDStudent(num_semantic_classes=7, backend="ref", **cs.STUDENT_KW).eval()
        torch.save(model.state_dict(), tmp / "best.pt")
        argv = sys.argv
        sys.argv = ["cache_student", "--checkpoint", str(tmp / "best.pt"), "--dataset", "second",
                    "--data-root", str(tmp / "SECOND"), "--out-dir", str(tmp / "cache"),
                    "--backend", "ref", "--device", "cpu", "--batch-size", "4", "--workers", "0"]
        try:
            cs.main()
        finally:
            sys.argv = argv
        meta = json.loads((tmp / "cache" / "meta.json").read_text())
        assert meta["layout"]["shape"] == [6, 15, NATIVE, NATIVE]
        assert meta["teacher"] == "student_self_distillation"
        base = SECONDDataset(str(tmp / "SECOND"), split="train", transform=None)
        cached = TeacherCacheDataset(base, str(tmp / "cache"))     # format accepté tel quel
        assert len(cached) == 6
        maps = [np.load(tmp / "cache" / f"d4_{i}.npy") for i in range(8)]
        s = base[2]
        a, b = s["img_t1"][None], s["img_t2"][None]
        with torch.no_grad():
            for gi, (h, k) in enumerate(D4):
                o = model(apply_d4(a, h, k), apply_d4(b, h, k))
                blob = torch.cat([o["sem_t1"], o["sem_t2"], (o["bcd"][:, 1] - o["bcd"][:, 0])[:, None]], 1)
                exp = F.avg_pool2d(blob, 4)[0].numpy()
                row = cached.row[base.ids[2]]
                assert np.allclose(maps[gi][row].astype(np.float32), exp, atol=2e-2), gi
        assert torch.allclose(cached[2]["kd_teacher"].float(), torch.from_numpy(maps[0][cached.row[base.ids[2]]]).float())
