#!/bin/bash
# Clone les dépôts de référence dans third_party/ (git-ignoré).
#
# Ces dépôts ne font PAS partie du code propre de csf-mamba. Ils servent à :
#   - fournir VMamba-Tiny (backbone) via ChangeMamba,
#   - reproduire les baselines Mamba-FCS / ChessMamba à protocole identique,
#   - lever verbatim la SeK-loss de Mamba-FCS (§12.1) et le module L1 de CSSM.
# Vérifier les licences avant toute publication.
set -euo pipefail

DEST="third_party"
mkdir -p "$DEST"

clone() {  # url  dossier
    if [[ -d "$DEST/$2/.git" ]]; then
        echo "== $2 déjà présent, skip =="
    else
        git clone --depth 1 "$1" "$DEST/$2"
    fi
}

clone https://github.com/ChenHongruixuan/ChangeMamba.git ChangeMamba
clone https://github.com/Buddhi19/MambaFCS.git           MambaFCS
clone https://github.com/DingLei14/ChessMamba.git        ChessMamba
clone https://github.com/Elman295/CSSM.git               CSSM

echo "third_party prêt. Rappel : ces dépôts sont des références, pas notre code."
