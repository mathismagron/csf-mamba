#!/bin/bash
# LandsatSCD512 (version PerASCD de Landsat-SCD) vers $SCRATCH, sur un nœud de CONNEXION
# (les nœuds de calcul n'ont pas internet).
#
#   bash scripts/download_landsat_perascd.sh
#
# Garde l'archive ($SCRATCH/LandsatSCD512.zip) : les jobs la décompressent dans
# $SLURM_TMPDIR (unzip vérifie le CRC de chaque fichier), comme SECOND.zip.
# Extrait aussi une copie dans $SCRATCH/LandsatSCD512 pour la vérification.
set -euo pipefail
URL="https://huggingface.co/datasets/SathShen/PerASCD-datasets/resolve/main/LandsatSCD512.zip"
ZIP="$SCRATCH/LandsatSCD512.zip"
DEST="$SCRATCH/LandsatSCD512"
EXPECTED=1612898211

wget -c -O "$ZIP" "$URL"
SIZE=$(stat -c %s "$ZIP")
[ "$SIZE" -eq "$EXPECTED" ] || { echo "⛔ taille $SIZE, $EXPECTED attendue : relancer (wget -c reprend)"; exit 1; }
sha256sum "$ZIP" | tee "$SCRATCH/LandsatSCD512.zip.sha256"

# L'archive n'a pas de dossier racine : train/, val/, test/ au premier niveau.
TOP=$(unzip -Z1 "$ZIP" | cut -d/ -f1 | sort -u | tr '\n' ' ')
[ "$TOP" = "test train val " ] || { echo "⛔ premier niveau inattendu : $TOP"; exit 1; }
mkdir -p "$DEST"
unzip -q -n "$ZIP" -d "$DEST"
for s in train val test; do
    echo "   $s : $(ls "$DEST/$s/im1" | wc -l) paires"
done
echo "== suite : python -m scripts.check_landsat_perascd --data-root $DEST"
