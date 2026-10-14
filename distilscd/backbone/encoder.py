"""Encodeur siamois : une image -> 4 features multi-échelles {X_i}.

Contrat commun (spec §7), pour une entrée (B, 3, H, W) :
    stage 1 -> (B,  96, H/4,  W/4)
    stage 2 -> (B, 192, H/8,  W/8)
    stage 3 -> (B, 384, H/16, W/16)
    stage 4 -> (B, 768, H/32, W/32)

Deux implémentations derrière ce contrat :
  * ConvEncoder  — jouet conv pur, léger, tourne sur CPU. Sert aux tests de
    formes et au debug sans GPU ni poids pré-entraînés.
  * VMambaTinyEncoder — le vrai backbone du plan (VMamba-Tiny ImageNet). Importé
    paresseusement depuis third_party/ChangeMamba ; c'est le budget paramètres.

Le reste du modèle ne dépend que du contrat, jamais de l'implémentation.
"""

from torch import nn

STAGE_CHANNELS = (96, 192, 384, 768)


class ConvEncoder(nn.Module):
    """Encodeur conv de substitution. Ne PAS utiliser pour les vrais runs."""

    def __init__(self, in_channels: int = 3, channels: tuple[int, ...] = STAGE_CHANNELS):
        super().__init__()
        self.channels = channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=7, stride=4, padding=3),
            nn.GroupNorm(1, channels[0]),
            nn.GELU(),
        )
        self.stages = nn.ModuleList()
        for prev, cur in zip(channels[:-1], channels[1:]):
            self.stages.append(
                nn.Sequential(
                    nn.Conv2d(prev, cur, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(1, cur),
                    nn.GELU(),
                    nn.Conv2d(cur, cur, kernel_size=3, padding=1, groups=cur),
                    nn.GELU(),
                )
            )

    def forward(self, x):
        feats = [self.stem(x)]
        for stage in self.stages:
            feats.append(stage(feats[-1]))
        return feats  # [stage1, stage2, stage3, stage4]


import sys
from pathlib import Path

# Racine du dépôt ChangeMamba cloné (cf. scripts/setup_third_party.sh). Ses
# imports internes sont en `from changedetection...`, donc c'est ce dossier — et
# non third_party/ — qui doit être sur le sys.path.
_CHANGEMAMBA_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "ChangeMamba"

# Kwargs VSSM alignés EXACTEMENT sur le checkpoint ImageNet téléchargeable
# vssm_tiny_0230_ckpt_epoch_262.pth (Zenodo, cf. scripts/download_pretrained.sh) :
# depths=[2,2,5,2], MLP présent. Vérifié en inspectant les clés du checkpoint —
# avec depths=[2,2,4,2] (yaml 0229flex) un bloc du stage 3 ne se chargeait pas.
#
# Budgets mesurés (backbone seul, dims=96, depths=[2,2,5,2]) :
#   tiny (mlp_ratio=4.0)  -> modèle complet ~35 M (hors cible 15M)
#   mini (mlp_ratio=-1.0) -> modèle complet ~20 M (= Piste A, §11-5)
# Le « VMamba-Tiny ~14M » du plan correspond en fait à la config MINI (MLP off).
_VMAMBA_BASE_KWARGS = dict(
    patch_size=4,
    in_chans=3,
    depths=[2, 2, 5, 2],
    dims=96,                    # VSSM étend en [96, 192, 384, 768]
    ssm_d_state=1,
    ssm_ratio=2.0,
    ssm_dt_rank="auto",
    ssm_conv=3,
    ssm_conv_bias=False,
    forward_type="v3noz",       # ⚠️ SelectiveScanCore : kernel CUDA requis au forward
    downsample_version="v3",
    patchembed_version="v2",
    norm_layer="ln2d",          # -> channel_first=True, sortie déjà (B,C,H,W)
    drop_path_rate=0.2,
)
VMAMBA_MLP_RATIO = {"tiny": 4.0, "mini": -1.0}


class VMambaTinyEncoder(nn.Module):
    """Adaptateur autour du backbone VMamba (dépôt ChangeMamba).

    Import différé : rien ne casse tant qu'on ne l'instancie pas, donc le reste
    du paquet reste importable sur une machine sans le dépôt cloné. Le forward de
    `Backbone_VSSM` renvoie déjà la liste des 4 features (B,C,H,W) — notre contrat.

    ⚠️ Le forward exige le kernel CUDA `selective_scan` (via forward_type v3noz) :
    ne tourne PAS sur CPU. Pour les tests CPU, utiliser ConvEncoder. La
    construction, elle, marche partout (utile pour compter les paramètres).

    variant : "mini" (13M, MLP désactivé, tient la cible) ou "tiny" (28M, MLP on).
    """

    def __init__(self, pretrained_path: str | None = None, variant: str = "mini"):
        super().__init__()
        if variant not in VMAMBA_MLP_RATIO:
            raise ValueError(f"variant inconnu : {variant!r} (attendu 'mini' ou 'tiny')")
        self.channels = STAGE_CHANNELS
        self.variant = variant

        if str(_CHANGEMAMBA_ROOT) not in sys.path:
            sys.path.insert(0, str(_CHANGEMAMBA_ROOT))
        try:
            from changedetection.models.Mamba_backbone import Backbone_VSSM
        except ImportError as exc:
            raise ImportError(
                f"VMamba introuvable ({exc}). Cloner ChangeMamba via "
                "scripts/setup_third_party.sh, installer einops/timm/fvcore/triton, "
                "ou utiliser encoder='conv' pour les tests CPU."
            ) from exc

        self.vssm = Backbone_VSSM(
            out_indices=(0, 1, 2, 3),
            pretrained=pretrained_path,
            mlp_ratio=VMAMBA_MLP_RATIO[variant],
            **_VMAMBA_BASE_KWARGS,
        )

    def forward(self, x):
        return list(self.vssm(x))


def build_encoder(name: str = "conv", **kwargs) -> nn.Module:
    if name == "conv":
        return ConvEncoder(**kwargs)
    if name in {"vmamba_mini", "vmamba"}:
        return VMambaTinyEncoder(variant="mini", **kwargs)
    if name == "vmamba_tiny":
        return VMambaTinyEncoder(variant="tiny", **kwargs)
    raise ValueError(f"encodeur inconnu : {name!r}")
