"""CSF-Mamba : assemblage de bout en bout (spec §7 du plan).

    encodeur siamois  ->  {X^T1_i}, {X^T2_i}
    C²S²-Block/stage  ->  Z_i  (fusion bi-temporelle)
    FFT stages 1–2    ->  injection spatio-fréquentielle (haute résolution)
    décodeur BCD      ->  Y_BCD, {CM_i}
    décodeur SCD      ->  Y^T1, Y^T2  (guidés par {CM_i})

Note SCD/SeK : le réseau produit les logits sémantiques par date. La carte SCD
« from-to » consommée par la SeK-loss se construit en masquant la sémantique par
la carte de changement (comme Mamba-FCS). Cette construction n'est pas encore
câblée ici — elle doit être reprise verbatim du dépôt Mamba-FCS pour garantir la
comparabilité (§12.1). Tant qu'elle manque, la loss saute proprement les termes
SeK/mIoU (voir CSFMambaLoss).
"""

import torch
from torch import nn

from .backbone.encoder import STAGE_CHANNELS, build_encoder
from .decoders.binary import BinaryDecoder
from .decoders.semantic import SharedSemanticDecoder
from .modules.c2s2 import C2S2Block, ConcatFusion
from .modules.fusion import FFTBranch


class CSFMamba(nn.Module):
    def __init__(
        self,
        num_semantic_classes: int = 10,  # convention A : 0 réservé, 1..9 réelles
        num_change: int = 2,
        encoder: str = "conv",
        core: str = "chess",
        backend: str = "auto",
        fft_stages: tuple[int, ...] = (0, 1),
        channels: tuple[int, ...] = STAGE_CHANNELS,
        decoder_refine: str = "dw",
        upsample: str = "dysample",
        fusion: str = "c2s2",
        cga: bool = True,
        mcasf: bool = True,
        encoder_kwargs: dict | None = None,
        # --- piste expérimentale, hors cadre initial (cf. documentation/hybride.md).
        # Vide = rien n'est construit et le modèle est identique au modèle de
        # référence, à l'octet près. Un test le vérifie.
        attn_stages: tuple[int, ...] = (),
        attn_depth: int = 2,
        attn_heads: int = 8,
        attn_mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.channels = channels
        self.fft_stages = set(fft_stages)

        self.encoder = build_encoder(encoder, **(encoder_kwargs or {}))
        # `fusion="concat"` retire le C²S² entier — damier, MCA-SF et scan S6 —
        # au profit d'une concaténation + 1x1. C'est l'ablation de la contribution
        # architecturale elle-même.
        block = {"c2s2": C2S2Block, "concat": ConcatFusion}[fusion]
        self.c2s2 = nn.ModuleList(
            block(c, core=core, backend=backend, mcasf=mcasf) for c in channels
        )
        self.fft = nn.ModuleDict(
            {str(i): FFTBranch(channels[i]) for i in self.fft_stages}
        )
        self.binary_decoder = BinaryDecoder(
            channels, num_change, refine=decoder_refine, upsample=upsample
        )
        self.semantic_decoder = SharedSemanticDecoder(
            channels, num_semantic_classes, num_change,
            refine=decoder_refine, upsample=upsample, cga=cga,
        )

        # Hybride Mamba/Transformer : attention bi-temporelle jointe aux stages
        # demandés, appliquée AVANT la fusion pour enrichir les traits de chaque
        # date du contexte global et de ceux de l'autre date.
        #
        # ⚠️ Construite EN DERNIER, et ce n'est pas un détail de style. Chaque
        # module consomme le générateur aléatoire à sa création : la placer plus
        # haut décalerait l'initialisation de tous les modules suivants, et un run
        # hybride ne différerait plus d'un run de référence à graine égale par la
        # seule attention, mais aussi par des décodeurs initialisés autrement. En
        # dernier, l'encodeur, la fusion, la FFT et les deux décodeurs reçoivent
        # exactement les mêmes poids qu'en référence : la comparaison est appariée
        # jusque dans l'initialisation.
        self.attn_stages = tuple(sorted(set(attn_stages)))
        self.attn = nn.ModuleDict()
        if self.attn_stages:
            from .experimental import BiTemporalAttention
            for i in self.attn_stages:
                if not 0 <= i < len(channels):
                    raise ValueError(f"stage d'attention {i} hors de [0, {len(channels)})")
                self.attn[str(i)] = BiTemporalAttention(
                    channels[i], depth=attn_depth, num_heads=attn_heads,
                    mlp_ratio=attn_mlp_ratio,
                )

    def forward(self, img_t1: torch.Tensor, img_t2: torch.Tensor) -> dict:
        feats_t1 = self.encoder(img_t1)
        feats_t2 = self.encoder(img_t2)

        # Attention bi-temporelle avant la fusion (liste vide par défaut).
        if self.attn_stages:
            feats_t1, feats_t2 = list(feats_t1), list(feats_t2)
            for i in self.attn_stages:
                feats_t1[i], feats_t2[i] = self.attn[str(i)](feats_t1[i], feats_t2[i])

        fused = []
        for i, (block, f1, f2) in enumerate(zip(self.c2s2, feats_t1, feats_t2)):
            z = block(f1, f2)
            if i in self.fft_stages:
                z = self.fft[str(i)](z, f1, f2)
            fused.append(z)

        bcd = self.binary_decoder(fused)
        scd = self.semantic_decoder(bcd["features"], bcd["change_maps"])

        return {
            "bcd": bcd["y_bcd"],
            "change_maps": bcd["change_maps"],
            "sem_t1": scd["sem_t1"],
            "sem_t2": scd["sem_t2"],
            "feat_t1": scd["feat_t1"],
            "feat_t2": scd["feat_t2"],
        }


def count_parameters(model: nn.Module) -> dict:
    """Décompte par sous-module, pour suivre la cible ~15M (§11-5)."""
    breakdown = {}
    for name, child in model.named_children():
        breakdown[name] = sum(p.numel() for p in child.parameters())
    breakdown["total"] = sum(p.numel() for p in model.parameters())
    return breakdown
