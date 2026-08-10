#!/bin/bash
# Télécharge le backbone VMamba-Tiny pré-entraîné ImageNet (Zenodo).
#
# C'est le MÊME checkpoint pour les variantes mini et tiny : le chargement fait du
# shape-matching (mini ignore simplement les poids MLP qu'il n'a pas). Vérifié :
#   tiny -> 218 poids chargés, 0 mismatch (seuls outnorm/classifier hors backbone)
#   mini -> 152 poids chargés, 0 mismatch (poids MLP ignorés)
#
# Le même dépôt Zenodo héberge aussi les checkpoints SCD entraînés par
# ChangeMamba, nécessaires à `scripts/evaluate_changemamba.py` :
#
#   FILES="MambaSCD_Tiny_SECOND_SeK_0.2208.pth" scripts/download_pretrained.sh
#
# Pour voir tout ce que contient le dépôt :
#   curl -s https://zenodo.org/api/records/15479555 \
#     | python3 -c "import sys,json;[print(f['key']) for f in json.load(sys.stdin)['files']]"
#
# Sur Alliance Canada, lancer sur un nœud de connexion (accès réseau), puis le
# checkpoint sera stagé avec le dataset.
set -euo pipefail

RECORD=15479555
DEST="${1:-pretrained_weight}"
FILES="${FILES:-vssm_tiny_0230_ckpt_epoch_262.pth}"

mkdir -p "$DEST"
for NAME in $FILES; do
    CKPT="$DEST/$NAME"
    if [[ -f "$CKPT" ]]; then
        echo "== déjà présent : $CKPT =="
    else
        echo "== téléchargement $NAME =="
        # -f : échoue sur une réponse HTTP d'erreur, au lieu d'écrire une page
        # HTML dans un fichier .pth qui ne planterait qu'au chargement.
        curl -fL -o "$CKPT" \
            "https://zenodo.org/api/records/$RECORD/files/$NAME/content"
    fi
    ls -la "$CKPT"
done
echo
echo "Backbone   : --encoder-pretrained $DEST/vssm_tiny_0230_ckpt_epoch_262.pth"
echo "Checkpoints SCD : utilisés par scripts/evaluate_changemamba.sbatch (CKPT=...)"
