#!/bin/bash
#SBATCH --job-name=csf-classes
#SBATCH --account=def-hervete
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --output=csf-classes-%j.out
#
# Vérifie l'association indice -> classe sémantique de SECOND.
#
# Pas de --gres=gpu : le script ne fait que lire des PNG. Un job CPU seul
# démarre bien plus vite qu'un job GPU, et les nœuds de calcul n'ont pas la
# limitation d'entrées/sorties imposée sur le nœud de connexion.
#
#   sbatch scripts/check_second_classes.sbatch
#   DATE=T1 LIMIT=500 sbatch --export=ALL,DATE,LIMIT scripts/check_second_classes.sbatch
set -euo pipefail

SPLIT="${SPLIT:-train}"
DATE="${DATE:-T2}"
LIMIT="${LIMIT:-200}"

module load python/3.11
source "$SCRATCH/csf-venv-cu12/bin/activate"
cd "$HOME/csf-mamba"

python -u -m scripts.check_second_classes \
    --data-root "$SCRATCH/SECOND" \
    --split "$SPLIT" \
    --date "$DATE" \
    --limit "$LIMIT"
