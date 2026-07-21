#!/bin/bash
# Télécharge le backbone VMamba-Tiny pré-entraîné ImageNet (Zenodo).
#
# C'est le MÊME checkpoint pour les variantes mini et tiny : le chargement fait du
# shape-matching (mini ignore simplement les poids MLP qu'il n'a pas). Vérifié :
#   tiny -> 218 poids chargés, 0 mismatch (seuls outnorm/classifier hors backbone)
#   mini -> 152 poids chargés, 0 mismatch (poids MLP ignorés)
#
# Sur Alliance Canada, lancer sur un nœud de connexion (accès réseau), puis le
# checkpoint sera stagé avec le dataset.
set -euo pipefail

DEST="${1:-pretrained_weight}"
mkdir -p "$DEST"
CKPT="$DEST/vssm_tiny_0230_ckpt_epoch_262.pth"
URL="https://zenodo.org/api/records/15479555/files/vssm_tiny_0230_ckpt_epoch_262.pth/content"

if [[ -f "$CKPT" ]]; then
    echo "== déjà présent : $CKPT =="
else
    echo "== téléchargement VMamba-Tiny ImageNet (~123 Mo) =="
    curl -L -o "$CKPT" "$URL"
fi
ls -la "$CKPT"
echo "Utiliser : --encoder-pretrained $CKPT"
