"""Tests de la distillation hors ligne (CPU, sans GPU ni vrai cache).

1. La chaîne d'augmentation sans clé `_aug` est inchangée (mêmes sorties, même
   consommation du générateur aléatoire) : les runs sans KD restent identiques.
2. `SECONDDataset(ids_file=..., image_split=...)` sélectionne bien le sous-ensemble.
3. **Alignement du cache, de bout en bout** : un faux professeur parfaitement
   informé (ses logits valent la vérité terrain, vue par vue) est écrit dans un cache
   au format réel ; après la vraie chaîne d'augmentation de l'élève (flips, rot90,
   jitter, échange temporel), la vue lue doit coïncider avec les cibles augmentées.
   Témoin : lire toujours la vue 0 doit échouer — le test est sensible.
4. `DistillLoss` est nulle et de gradient nul quand l'élève reproduit le professeur.

    PYTHONPATH=. python -m pytest -q tests/test_distill.py
"""

import copy
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from csf_mamba.datasets.second import SECONDDataset
from csf_mamba.datasets.teacher_cache import TeacherCacheDataset
from csf_mamba.datasets.transforms import AUG_KEY, train_transform
from csf_mamba.distill.d4 import D4, apply_d4
from csf_mamba.losses.distill import DistillLoss

TILE, NATIVE, N = 32, 8, 6


def _make_second(root: Path, rng):
    """SECOND synthétique, cartes constantes par blocs 4×4 (alignées sur la grille native)."""
    for d in ("T1", "T2", "GT_T1", "GT_T2", "GT_CD"):
        (root / "train" / d).mkdir(parents=True, exist_ok=True)
    natives = {}
    for i in range(N):
        name = f"{i:05d}.png"
        cd = rng.integers(0, 2, (NATIVE, NATIVE)).astype(np.uint8)
        s1 = np.where(cd > 0, rng.integers(1, 7, (NATIVE, NATIVE)), 0).astype(np.uint8)
        s2 = np.where(cd > 0, rng.integers(1, 7, (NATIVE, NATIVE)), 0).astype(np.uint8)
        up = lambda a: np.kron(a, np.ones((4, 4), np.uint8))
        for d in ("T1", "T2"):
            Image.fromarray(rng.integers(0, 255, (TILE, TILE, 3), dtype=np.uint8)).save(root / "train" / d / name)
        Image.fromarray(up(s1)).save(root / "train" / "GT_T1" / name)
        Image.fromarray(up(s2)).save(root / "train" / "GT_T2" / name)
        Image.fromarray(up(cd) * 255).save(root / "train" / "GT_CD" / name)
        natives[name] = (cd, s1, s2)
    (root / "train.txt").write_text("\n".join(f"{i:05d}" for i in range(N)))
    return natives


def _make_cache(cache: Path, natives: dict):
    """Faux professeur « oracle » : T(g·x) = logits de la vérité vue dans le repère g."""
    cache.mkdir(parents=True, exist_ok=True)
    names = sorted(natives)
    for gi, (h, k) in enumerate(D4):
        arr = np.zeros((len(names), 15, NATIVE, NATIVE), np.float16)
        for r, name in enumerate(names):
            cd, s1, s2 = (apply_d4(torch.from_numpy(a.astype(np.int64)), h, k).numpy() for a in natives[name])
            for c in range(7):
                arr[r, c] = np.where(s1 == c, 10, -10)
                arr[r, 7 + c] = np.where(s2 == c, 10, -10)
            arr[r, 14] = np.where(cd > 0, 10, -10)
        np.save(cache / f"d4_{gi}.npy", arr)
    (cache / "ids.txt").write_text("\n".join(names) + "\n")
    (cache / "meta.json").write_text(json.dumps({
        "layout": {"shape": [len(names), 15, NATIVE, NATIVE], "dtype": "float16"},
        "d4": {"files": [f"d4_{i}.npy" for i in range(8)], "elements_hflip_then_rot90k": D4}}))


def _sample():
    return {"img_t1": torch.rand(3, TILE, TILE), "img_t2": torch.rand(3, TILE, TILE),
            "sem_t1": torch.randint(0, 7, (TILE, TILE)), "sem_t2": torch.randint(0, 7, (TILE, TILE)),
            "change": torch.randint(0, 2, (TILE, TILE)), "unchanged": torch.zeros(TILE, TILE, dtype=torch.bool)}


def test_transforms_unchanged_without_aug_key():
    tf = train_transform(TILE, rot90=True, photometric=0.2, temporal_swap=0.5)
    for seed in range(20):
        base = _sample()
        a, b = copy.deepcopy(base), copy.deepcopy(base)
        b[AUG_KEY] = torch.zeros(6, dtype=torch.long)
        random.seed(seed); ra = tf(a); next_a = random.random()
        random.seed(seed); rb = tf(b); next_b = random.random()
        assert next_a == next_b
        for k in ra:
            assert torch.equal(ra[k], rb[k]), k


def test_second_ids_subset():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make_second(root, np.random.default_rng(0))
        ids = root / "val.txt"
        ids.write_text("00001\n00004\n")
        ds = SECONDDataset(str(root), split="val", ids_file=str(ids), image_split="train")
        assert ds.ids == ["00001.png", "00004.png"] and len(ds) == 2
        bad = root / "bad.txt"
        bad.write_text("99999\n")
        try:
            SECONDDataset(str(root), split="val", ids_file=str(bad), image_split="train")
            raise AssertionError("un identifiant absent aurait dû lever")
        except FileNotFoundError:
            pass


def _check(ds, draws, force_view0=False):
    bad = 0
    for t in range(draws):
        random.seed(t)
        if force_view0:
            import csf_mamba.datasets.teacher_cache as tc
            orig = tc.canonical_d4
            tc.canonical_d4 = lambda h, v, k: 0
        try:
            s = ds[t % len(ds)]
        finally:
            if force_view0:
                tc.canonical_d4 = orig
        kd = s["kd_teacher"].float()
        chg = s["change"][2::4, 2::4]
        ok = torch.equal((kd[14] > 0).long(), chg)
        for sl, key in ((slice(0, 7), "sem_t1"), (slice(7, 14), "sem_t2")):
            lab = s[key][2::4, 2::4]
            valid = lab != 255
            ok &= torch.equal(kd[sl].argmax(0)[valid], lab[valid])
        bad += not ok
    return bad


def test_cache_alignment_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        natives = _make_second(root, np.random.default_rng(1))
        _make_cache(root / "cache", natives)
        tf = train_transform(TILE, rot90=True, photometric=0.2, temporal_swap=0.5)
        base = SECONDDataset(str(root), split="train", transform=None)
        ds = TeacherCacheDataset(base, str(root / "cache"), transform=tf)
        assert _check(ds, 200) == 0
        assert _check(ds, 200, force_view0=True) > 50       # témoin : le test voit l'erreur


def test_distill_loss_zero_at_match():
    torch.manual_seed(0)
    teacher = torch.randn(2, 15, NATIVE, NATIVE)
    up = torch.nn.functional.interpolate(teacher, size=(TILE, TILE), mode="bilinear", align_corners=False)
    bcd = torch.stack([torch.zeros_like(up[:, 14]), up[:, 14]], 1).requires_grad_(True)
    s1 = torch.cat([torch.zeros_like(up[:, :1]), up[:, 1:7]], 1).requires_grad_(True)
    s2 = torch.cat([torch.zeros_like(up[:, :1]), up[:, 8:14]], 1).requires_grad_(True)
    out = {"bcd": bcd, "sem_t1": s1, "sem_t2": s2}
    change = torch.randint(0, 2, (2, TILE, TILE))
    for mask in ("changed", "all"):
        loss = DistillLoss(1.0, 1.0, 1.0, 2.0, mask)
        terms = loss(out, teacher, change)
        assert terms["kd_change"].abs() < 1e-5 and terms["kd_sem"].abs() < 1e-5, terms
        sum(terms.values()).backward()
        assert max(float(g.abs().max()) for g in (bcd.grad, s1.grad, s2.grad)) < 1e-5
        bcd.grad = s1.grad = s2.grad = None
    far = {"bcd": -bcd.detach(), "sem_t1": -s1.detach(), "sem_t2": -s2.detach()}
    t = DistillLoss(1.0, 1.0, 1.0, 2.0, "all")(far, teacher, change)
    assert t["kd_change"] > 0.1 and t["kd_sem"] > 0.1


def test_change_target_controls():
    """Contrôles `gt` / `gt_smooth` : la cible ne dépend plus du professeur, et
    `teacher` reste le comportement par défaut, à l'identique."""
    torch.manual_seed(1)
    change = torch.zeros(2, TILE, TILE, dtype=torch.long)
    change[:, 8:24, 4:20] = 1                       # un carré : bords nets
    teacher_a = torch.randn(2, 15, NATIVE, NATIVE)
    teacher_b = torch.randn(2, 15, NATIVE, NATIVE)
    d = torch.randn(2, TILE, TILE)
    out = {"bcd": torch.stack([torch.zeros_like(d), d], 1)}
    ref = DistillLoss(1.0)(out, teacher_a, change)["kd_change"]
    assert torch.equal(ref, DistillLoss(1.0, change_target="teacher")(out, teacher_a, change)["kd_change"])
    for tgt in ("gt", "gt_smooth"):
        la = DistillLoss(1.0, change_target=tgt)(out, teacher_a, change)["kd_change"]
        lb = DistillLoss(1.0, change_target=tgt)(out, teacher_b, change)["kd_change"]
        assert torch.equal(la, lb), tgt             # indépendant du professeur
    # gt : BCE brute sur la vérité (entropie nulle)
    exp = torch.nn.functional.binary_cross_entropy_with_logits(d, change.float())
    got = DistillLoss(2.0, change_target="gt")(out, teacher_a, change)["kd_change"]
    assert torch.allclose(got, 2.0 * exp, atol=1e-6)
    # gt_smooth : dans [0, 1], égale à la vérité loin des bords, fractionnaire sur les bords
    loss = DistillLoss(1.0, change_target="gt_smooth")
    p = loss._change_target(None, teacher_a, change, (TILE, TILE))
    assert p.min() >= 0 and p.max() <= 1
    assert torch.all(p[:, 14:18, 10:14] == 1) and torch.all(p[:, 0:2, 26:] == 0)
    frac = ((p > 0.01) & (p < 0.99)).float().mean()
    assert 0 < frac < 0.5, float(frac)


if __name__ == "__main__":
    for f in (test_transforms_unchanged_without_aug_key, test_second_ids_subset,
              test_cache_alignment_end_to_end, test_distill_loss_zero_at_match,
              test_change_target_controls):
        f(); print("OK", f.__name__)
