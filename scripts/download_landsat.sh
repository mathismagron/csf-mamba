#!/bin/bash
# Télécharge et installe Landsat-SCD depuis figshare.
#
# ⚠️ À LANCER SUR UN NŒUD DE CONNEXION : les nœuds de calcul d'Alliance Canada
# n'ont pas d'accès réseau. C'est aussi l'usage recommandé — les nœuds de
# connexion sont faits pour les transferts de données.
#
#   scripts/download_landsat.sh                 # vers $SCRATCH/Landsat-SCD
#   scripts/download_landsat.sh /autre/chemin
#
# Source : figshare 19946135, « Landsat-SCD_dataset.zip », 4 400 617 661 octets,
# licence CC BY 4.0. 2 425 paires 416x416, 4 classes réelles, résolution 30 m.
#
# Le téléchargement reprend là où il s'est arrêté (`curl -C -`) : une session de
# nœud de connexion coupée ne coûte pas les 4 Gio déjà transférés.
set -euo pipefail

URL="https://ndownloader.figshare.com/files/35495660"
TAILLE_ATTENDUE=4400617661
DEST="${1:-${SCRATCH:?SCRATCH non défini}/Landsat-SCD}"
ZIP="$(dirname "$DEST")/Landsat-SCD_dataset.zip"

echo "== destination : $DEST"
echo "== archive     : $ZIP"

if [ -d "$DEST/A" ]; then
    echo "✓ $DEST/A existe déjà — rien à faire."
    echo "  Vérifier le dump : python -m scripts.check_landsat --data-root $DEST"
    exit 0
fi

# --- 1. Place disponible. L'archive et son contenu coexistent un moment.
besoin_gio=9
dispo_gio=$(df -BG --output=avail "$(dirname "$DEST")" | tail -1 | tr -dc '0-9')
echo "== espace : ${dispo_gio} Gio disponibles, ~${besoin_gio} nécessaires"
if [ "${dispo_gio:-0}" -lt "$besoin_gio" ]; then
    echo "⛔ espace insuffisant. Sur Alliance Canada : diskusage_report"
    exit 1
fi

# --- 2. Téléchargement reprenable.
echo "== téléchargement (4,1 Gio, reprenable)"
curl -L -C - --retry 5 --retry-delay 10 -o "$ZIP" "$URL"

taille=$(stat -c %s "$ZIP")
if [ "$taille" -ne "$TAILLE_ATTENDUE" ]; then
    echo "⛔ taille inattendue : $taille au lieu de $TAILLE_ATTENDUE octets."
    echo "   Téléchargement incomplet — relancer ce script, il reprendra."
    exit 1
fi
echo "✓ taille conforme : $taille octets"

# --- 3. Regarder DANS l'archive avant d'extraire.
# Beaucoup de dumps enferment tout dans un dossier racine. L'extraire à l'aveugle
# donnerait $DEST/Landsat-SCD_dataset/A/ au lieu de $DEST/A/, et le dataloader
# échouerait sur une arborescence introuvable.
# ⚠️ Ne JAMAIS mettre `head` dans un pipeline ici. Sous `set -o pipefail`, il
# ferme le tuyau, `unzip` reçoit SIGPIPE, le script sort en 141 — SANS message,
# puisque `set -e` interrompt avant tout affichage. C'est précisément ce qui a
# fait échouer la première exécution. On capture d'abord, on filtre ensuite, et
# awk lit son flux jusqu'au bout.
echo "== structure de l'archive (10 premières entrées)"
listing=$(unzip -l "$ZIP")
printf '%s\n' "$listing" | sed -n '4,13p'
racine=$(printf '%s\n' "$listing" \
    | awk 'NR>3 && $4 != "" && !vu {split($4, p, "/"); print p[1]; vu=1}')
echo "== premier segment de chemin : « $racine »"

# --- 4. Extraction, puis normalisation de l'arborescence.
TMP="$(dirname "$DEST")/.landsat-extract.$$"
mkdir -p "$TMP"
echo "== extraction"
unzip -q "$ZIP" -d "$TMP"

if [ -d "$TMP/A" ]; then
    src="$TMP"
else
    # Un seul dossier racine : on descend dedans.
    src=$(find "$TMP" -maxdepth 2 -type d -name A -print -quit || true)
    if [ -z "$src" ]; then
        echo "⛔ aucun dossier A/ trouvé dans l'archive. Contenu extrait :"
        find "$TMP" -maxdepth 2 | sed -n '1,20p'    # sed lit tout : pas de SIGPIPE
        echo "   Ne PAS lancer d'entraînement : le format diffère de celui attendu."
        exit 1
    fi
    src=$(dirname "$src")
fi
echo "== racine réelle des données : $src"

mkdir -p "$(dirname "$DEST")"
mv "$src" "$DEST"
rm -rf "$TMP"

echo
echo "== contenu installé"
ls -1 "$DEST" | sed -n '1,20p'

echo
echo "✓ Installé dans $DEST"
echo "  ÉTAPE SUIVANTE, obligatoire avant tout entraînement :"
echo "    python -m scripts.check_landsat --data-root $DEST"
echo
echo "  L'archive peut être supprimée ensuite : rm $ZIP"
