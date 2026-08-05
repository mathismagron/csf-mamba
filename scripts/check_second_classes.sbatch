#!/bin/bash
#SBATCH --job-name=csf-classes
#SBATCH --account=def-hervete
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=csf-classes-%j.out
#
# Vérifie l'association indice -> classe sémantique de SECOND.
#
# Pas de --gres=gpu : le script ne fait que lire des PNG. Un job CPU seul
# démarre bien plus vite qu'un job GPU, et les nœuds de calcul n'ont pas la
# limitation d'entrées/sorties imposée sur le nœud de connexion.
#
# Le premier job (194247) a fini en TIMEOUT à 20 min sans écrire une seule ligne,
# malgré python -u : il n'a jamais atteint son premier print. La lecture de PNG
# n'était donc pas en cause. Deux mesures : le script n'importe plus torch (il
# lit CLASS_NAMES par analyse syntaxique) et le venv est retiré au profit de
# scipy-stack. Traces horodatées ajoutées pour situer tout blocage résiduel.
#
#   sbatch scripts/check_second_classes.sbatch
#   DATE=T1 LIMIT=500 sbatch --export=ALL,DATE,LIMIT scripts/check_second_classes.sbatch
set -euo pipefail

SPLIT="${SPLIT:-train}"
DATE="${DATE:-T2}"
LIMIT="${LIMIT:-200}"

# Pas de venv : le script n'utilise que numpy et Pillow, tous deux fournis par
# scipy-stack. Le premier job a passé 20 minutes sans produire une seule ligne,
# bloqué avant son premier print — le venv sur Lustre était le seul suspect
# lourd, il est donc retiré du chemin.
module load python/3.11 scipy-stack
cd "$HOME/csf-mamba"

python -u -m scripts.check_second_classes \
    --data-root "$SCRATCH/SECOND" \
    --split "$SPLIT" \
    --date "$DATE" \
    --limit "$LIMIT"
