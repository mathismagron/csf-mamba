#!/bin/bash
# Télécharge le dataset SECOND prétraité (ChangeMamba) — SUR UN NŒUD DE CONNEXION.
#
# Version prétraitée : cartes sémantiques déjà en mono-canal et cartes de
# changement binaires déjà générées. Le dataset ORIGINAL fournit les cartes
# sémantiques en RGB et sans carte de changement — inutilisable tel quel.
# Utiliser cette version garantit en plus le même prétraitement que ChangeMamba,
# donc des chiffres comparables.
#
# Usage :  bash scripts/download_second.sh [destination]
#          (défaut : $SCRATCH/SECOND)
set -euo pipefail

DEST="${1:-$SCRATCH/SECOND}"
URL="https://zenodo.org/api/records/15479555/files/SECOND.zip/content"
ARCHIVE="$(dirname "$DEST")/SECOND.zip"

mkdir -p "$(dirname "$DEST")"

if [[ -f "$ARCHIVE" ]]; then
    echo "== archive déjà présente : $ARCHIVE =="
else
    echo "== téléchargement de SECOND.zip (3,84 Go) =="
    curl -L --fail -C - -o "$ARCHIVE" "$URL"   # -C - : reprend si interrompu
fi

echo "== décompression vers $(dirname "$DEST") =="
unzip -q -o "$ARCHIVE" -d "$(dirname "$DEST")"

echo "== arborescence obtenue =="
find "$(dirname "$DEST")" -maxdepth 4 -type d \( -name 'im1' -o -name 'label1' -o -name 'GT_CD' \) | head
echo
echo "Vérifier ensuite que le dataloader lit correctement (cf. RUN.md)."
