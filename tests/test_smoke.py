"""Smoke test CPU de bout en bout.

Ne vérifie PAS la qualité du modèle — seulement que la plomberie tient :
formes de sortie conformes à la spec §7, forward/backward sans crash, budget
paramètres. Tourne en quelques secondes sur CPU avec backend='ref', sans GPU,
sans dataset, sans kernel CUDA.

    python -m pytest tests/test_smoke.py -v
    # ou directement :  python tests/test_smoke.py
"""

import torch

from csf_mamba.datasets.hi_ucd import NUM_SEMANTIC_CLASSES
from csf_mamba.losses.composite import CSFMambaLoss
from csf_mamba.model import CSFMamba, count_parameters
from csf_mamba.modules.c2s2 import C2S2Block
from csf_mamba.modules.chessboard import deinterleave, interleave

B, H, W = 2, 64, 64  # petit pour rester rapide sur CPU


def _dummy_batch(num_classes=NUM_SEMANTIC_CLASSES):
    return {
        "img_t1": torch.randn(B, 3, H, W),
        "img_t2": torch.randn(B, 3, H, W),
        "change": torch.randint(0, 2, (B, H, W)),
        # convention A : classes réelles 1..num_classes-1 (index 0 réservé, jamais cible)
        "sem_t1": torch.randint(1, num_classes, (B, H, W)),
        "sem_t2": torch.randint(1, num_classes, (B, H, W)),
        "unchanged": torch.randint(0, 2, (B, H, W)).bool(),
    }


def test_interleave_roundtrip():
    """L'entrelacement damier doit être exactement inversible."""
    xa = torch.randn(B, 8, 16, 16)
    xb = torch.randn(B, 8, 16, 16)
    ra, rb = deinterleave(interleave(xa, xb), 16, 16)
    assert torch.allclose(xa, ra, atol=1e-6), "X^a non reconstruit"
    assert torch.allclose(xb, rb, atol=1e-6), "X^b non reconstruit"


def test_c2s2_preserves_shape():
    """Le C²S²-Block consomme (F1,F2) et rend un Z de même forme."""
    for core in ("chess", "l1"):
        block = C2S2Block(channels=32, core=core, backend="ref")
        f1, f2 = torch.randn(B, 32, 16, 16), torch.randn(B, 32, 16, 16)
        z = block(f1, f2)
        assert z.shape == f1.shape, f"core={core}: {z.shape} != {f1.shape}"


def test_model_output_shapes():
    """Formes conformes à la spec §7 pour l'encodeur conv (CPU)."""
    model = CSFMamba(num_semantic_classes=NUM_SEMANTIC_CLASSES, encoder="conv", backend="ref")
    batch = _dummy_batch()
    out = model(batch["img_t1"], batch["img_t2"])

    assert out["bcd"].shape == (B, 2, H, W), out["bcd"].shape
    assert out["sem_t1"].shape == (B, NUM_SEMANTIC_CLASSES, H, W), out["sem_t1"].shape
    assert out["sem_t2"].shape == (B, NUM_SEMANTIC_CLASSES, H, W), out["sem_t2"].shape
    assert len(out["change_maps"]) == 4, "une carte de changement par stage attendue"


def test_forward_backward_and_budget():
    """Le backward passe (loss différentiable) et on affiche le budget params."""
    model = CSFMamba(num_semantic_classes=NUM_SEMANTIC_CLASSES, encoder="conv", backend="ref")
    criterion = CSFMambaLoss(num_semantic_classes=NUM_SEMANTIC_CLASSES)
    batch = _dummy_batch()
    out = model(batch["img_t1"], batch["img_t2"])

    targets = {
        "change": batch["change"],
        "sem_t1": batch["sem_t1"], "sem_t2": batch["sem_t2"],
        "unchanged": batch["unchanged"],
    }
    losses = criterion(out, targets)
    losses["total"].backward()

    grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    assert any(grads), "aucun gradient calculé"

    params = count_parameters(model)
    print("\nBudget paramètres (encodeur conv jouet) :")
    for name, n in params.items():
        print(f"  {name:20s} {n/1e6:7.2f} M")


def test_vmamba_constructs_within_budget():
    """VMamba-mini se construit et le modèle complet tient la cible Piste A.

    Forward NON testé : le backbone VMamba exige le kernel CUDA selective_scan
    (forward_type v3noz), indisponible sur CPU. Ici on vérifie seulement le
    câblage et le budget paramètres. Skip propre si ChangeMamba n'est pas cloné.
    """
    try:
        model = CSFMamba(num_semantic_classes=NUM_SEMANTIC_CLASSES, encoder="vmamba_mini")
    except ImportError as exc:
        print(f"⊘ VMamba non disponible (attendu si third_party absent) : {exc}")
        return
    total = count_parameters(model)["total"] / 1e6
    print(f"\nModèle complet (VMamba-mini) : {total:.2f} M params")
    assert total < 22.0, f"budget Piste A dépassé : {total:.2f} M"


if __name__ == "__main__":
    test_interleave_roundtrip()
    print("✓ interleave roundtrip")
    test_c2s2_preserves_shape()
    print("✓ C²S² preserve les formes (chess + l1)")
    test_model_output_shapes()
    print("✓ formes de sortie conformes §7")
    test_forward_backward_and_budget()
    print("✓ forward/backward OK")
    test_vmamba_constructs_within_budget()
    print("✓ VMamba-mini construit dans le budget")
