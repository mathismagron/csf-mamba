"""Tests de la KD de features (étape 4), CPU.

1. `FeatureDistillLoss` : nulle quand le professeur vaut la sortie de l'adaptateur,
   ≈ 1 pour des features indépendantes, invariante à une échelle / un biais du
   professeur (LayerNorm sur les canaux), gradient vers l'élève ET l'adaptateur.
2. `CSFMamba(return_encoder_feats)` : désactivé, sorties inchangées ; activé, les
   features exposées sont celles de l'encodeur.
3. (si PERASCD_ROOT est défini) `OnlineTeacherEncoder` ViT-B aléatoire : 4 échelles
   aux tailles de l'élève, fp16, sans gradient, deux dates cohérentes avec un
   passage séparé.

    PYTHONPATH=. python tests/test_distill_feat.py
"""
import os

import torch

from csf_mamba.losses.distill import FeatureDistillLoss
from csf_mamba.model import CSFMamba

CH = (96, 192, 384, 768)


def _student(b=2, size=64):
    return [torch.randn(b, c, size // (4 * 2 ** i), size // (4 * 2 ** i), requires_grad=True)
            for i, c in enumerate(CH)]


def test_feature_loss_properties():
    torch.manual_seed(0)
    loss = FeatureDistillLoss(CH, (1, 2, 3), teacher_dim=32)
    s1, s2 = _student(), _student()
    with torch.no_grad():
        t1 = [loss.adapters[str(i)](s1[i]) if str(i) in loss.adapters else None for i in range(4)]
        t2 = [loss.adapters[str(i)](s2[i]) if str(i) in loss.adapters else None for i in range(4)]
    zero = loss(s1, s2, t1, t2)["kd_feat"]
    assert zero.abs() < 1e-4, zero
    # invariance échelle / biais du professeur
    t1b = [None if t is None else 5 * t + 3 for t in t1]
    t2b = [None if t is None else 0.2 * t - 1 for t in t2]
    assert loss(s1, s2, t1b, t2b)["kd_feat"].abs() < 1e-4
    # professeur indépendant : ≈ 1 ; gradients vers l'élève et les adaptateurs
    r1 = [None if t is None else torch.randn_like(t) for t in t1]
    r2 = [None if t is None else torch.randn_like(t) for t in t2]
    far = loss(s1, s2, r1, r2)["kd_feat"]
    assert 0.8 < far < 1.2, far
    far.backward()
    assert s1[1].grad is not None and s1[1].grad.abs().sum() > 0
    assert s1[0].grad is None                       # étage non distillé
    assert all(p.grad is not None for p in loss.parameters())


def test_model_encoder_feats_flag():
    torch.manual_seed(0)
    m = CSFMamba(num_semantic_classes=7, encoder="conv", backend="ref").eval()
    a, b = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        o0 = m(a, b)
        m.return_encoder_feats = True
        o1 = m(a, b)
        enc = m.encoder(a)
    assert set(o1) - set(o0) == {"enc_t1", "enc_t2"}
    for k in o0:
        x, y = o0[k], o1[k]
        if torch.is_tensor(x):
            assert torch.equal(x, y), k
    assert all(torch.equal(u, v) for u, v in zip(o1["enc_t1"], enc))


def test_online_teacher_vitb():
    root = os.environ.get("PERASCD_ROOT")
    if not root:
        print("  (PERASCD_ROOT absent : test du professeur en ligne sauté)")
        return
    from csf_mamba.distill.online_teacher import OnlineTeacherEncoder
    t = OnlineTeacherEncoder(root, "none", arch="ViT-B/16", msda="pytorch")
    a, b = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)
    f1, f2 = t(a, b)
    assert [tuple(x.shape) for x in f1] == [(2, 768, 16, 16), (2, 768, 8, 8), (2, 768, 4, 4), (2, 768, 2, 2)]
    assert all(x.dtype == torch.float16 and not x.requires_grad for x in f1 + f2)
    g1, _ = t(a, a)                                 # t1 inchangée : mêmes features
    assert all(torch.allclose(x.float(), y.float(), atol=1e-2) for x, y in zip(f1, g1))
    t.train()
    assert not t.training


if __name__ == "__main__":
    for f in (test_feature_loss_properties, test_model_encoder_feats_flag, test_online_teacher_vitb):
        f(); print("OK", f.__name__)
