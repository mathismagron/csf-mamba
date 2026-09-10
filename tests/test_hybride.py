"""Non-régression de la piste hybride Mamba/Transformer.

Trois garanties, dans l'ordre d'importance :

1. **Le modèle de référence est inchangé.** `attn_stages=()` ne construit rien et
   ne modifie aucune sortie. Sans cette garantie, une piste expérimentale
   contaminerait 150 entraînements déjà mesurés.
2. **L'hybride part de la ligne de base.** Le LayerScale initialisé à 1e-5 et la
   convolution de position initialisée à zéro rendent le bloc quasi identitaire :
   à l'initialisation, l'hybride reproduit le modèle de référence à ~1e-4 près.
   Il ne peut donc pas partir plus bas, et un résultat négatif voudra dire que
   l'idée ne prend pas — pas que l'initialisation l'a tuée.
3. **Le bloc fait bien ce qu'il prétend** : il mélange les deux dates, et il n'est
   pas invariant par permutation spatiale (sans quoi il ne pourrait rien
   localiser).

    python -m pytest tests/test_hybride.py -v
"""

import torch

from csf_mamba.datasets.hi_ucd import NUM_SEMANTIC_CLASSES
from csf_mamba.experimental import BiTemporalAttention
from csf_mamba.model import CSFMamba, count_parameters

B, H, W = 2, 32, 32


def _model(**kw):
    torch.manual_seed(0)
    return CSFMamba(num_semantic_classes=NUM_SEMANTIC_CLASSES, encoder="conv",
                    backend="ref", fusion="concat", **kw).eval()


def test_le_modele_de_reference_est_inchange():
    """attn_stages vide : aucun paramètre ajouté, aucun sous-module construit."""
    ref = _model()
    assert len(ref.attn) == 0 and ref.attn_stages == ()
    assert "attn" in count_parameters(ref) and count_parameters(ref)["attn"] == 0


def test_l_hybride_part_de_la_ligne_de_base():
    """À l'initialisation, l'attention ne doit rien changer aux sorties.

    Mesuré sur la MÊME instance, attention activée puis désactivée : c'est la
    seule façon d'isoler la contribution du bloc. Comparer deux modèles construits
    séparément mesurerait aussi la différence d'initialisation des autres modules.
    """
    torch.manual_seed(1)
    m = _model(attn_stages=(3,))
    x1, x2 = torch.randn(B, 3, H, W), torch.randn(B, 3, H, W)
    with torch.no_grad():
        avec = m(x1, x2)
        stages, m.attn_stages = m.attn_stages, ()
        sans = m(x1, x2)
        m.attn_stages = stages
    for k in ("bcd", "sem_t1", "sem_t2"):
        ecart = (avec[k] - sans[k]).abs().max().item()
        print(f"    {k:<8} écart max {ecart:.2e}")
        assert ecart < 1e-3, f"{k} : écart {ecart:.2e} à l'initialisation"


def test_les_autres_modules_sont_initialises_a_l_identique():
    """L'attention est construite en dernier : à graine égale, tout le reste
    reçoit exactement les mêmes poids qu'en référence. La comparaison hybride /
    référence est donc appariée jusque dans l'initialisation."""
    ref, hyb = _model(), _model(attn_stages=(3,))
    r, h = ref.state_dict(), hyb.state_dict()
    communs = [k for k in r if k in h]
    assert len(communs) == len(r), "des poids de la référence ont disparu"
    differents = [k for k in communs if not torch.equal(r[k], h[k])]
    assert not differents, f"{len(differents)} poids diffèrent, ex. {differents[:3]}"
    print(f"    {len(communs)} tenseurs partagés, tous identiques")


def test_le_bloc_melange_bien_les_deux_dates():
    """Changer f2 doit modifier la sortie de f1 : sinon aucune comparaison n'a lieu."""
    torch.manual_seed(2)
    blk = BiTemporalAttention(64, depth=1, num_heads=4, layer_scale=1.0).eval()
    f1, f2 = torch.randn(B, 64, 8, 8), torch.randn(B, 64, 8, 8)
    with torch.no_grad():
        o1_a, _ = blk(f1, f2)
        o1_b, _ = blk(f1, torch.randn(B, 64, 8, 8))
    assert not torch.allclose(o1_a, o1_b, atol=1e-6), \
        "la sortie de T1 ne dépend pas de T2 — le bloc ne compare rien"


def test_le_bloc_n_est_pas_invariant_par_permutation_spatiale():
    """Sans information de position, l'attention ne pourrait rien localiser."""
    torch.manual_seed(3)
    blk = BiTemporalAttention(64, depth=1, num_heads=4, layer_scale=1.0).eval()
    # La convolution de position est initialisée à zéro : on lui donne des poids
    # pour tester ce qu'elle apporte une fois entraînée.
    with torch.no_grad():
        blk.pos.weight.normal_(0, 0.1)
    f1, f2 = torch.randn(B, 64, 8, 8), torch.randn(B, 64, 8, 8)
    perm = torch.randperm(8)
    with torch.no_grad():
        o_direct, _ = blk(f1, f2)
        o_permute, _ = blk(f1[:, :, perm], f2[:, :, perm])
    assert not torch.allclose(o_direct[:, :, perm], o_permute, atol=1e-5), \
        "le bloc est invariant par permutation : il ne peut pas localiser"


def test_formes_et_retropropagation():
    torch.manual_seed(4)
    m = _model(attn_stages=(2, 3), attn_depth=1)
    x1, x2 = torch.randn(B, 3, H, W), torch.randn(B, 3, H, W)
    out = m(x1, x2)
    assert out["bcd"].shape == (B, 2, H, W)
    assert out["sem_t1"].shape == (B, NUM_SEMANTIC_CLASSES, H, W)
    out["bcd"].sum().backward()
    g = m.attn["3"].blocks[0].gamma1.grad
    assert g is not None and g.abs().sum() > 0, "le LayerScale ne reçoit pas de gradient"


def test_resolution_variable():
    """La position est conditionnelle : le bloc doit accepter toute résolution."""
    blk = BiTemporalAttention(32, depth=1, num_heads=4).eval()
    for hw in (4, 8, 16):
        o1, o2 = blk(torch.randn(1, 32, hw, hw), torch.randn(1, 32, hw, hw))
        assert o1.shape == o2.shape == (1, 32, hw, hw)


if __name__ == "__main__":
    for nom, fn in sorted(globals().items()):
        if nom.startswith("test_"):
            fn(); print(f"  ok  {nom}")
    print("tous les tests passent")
