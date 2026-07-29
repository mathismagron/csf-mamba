#!/bin/bash
#SBATCH --job-name=csf-stats
#SBATCH --account=def-hervete
#SBATCH --qos=cc-debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:30:0
#SBATCH --output=csf-stats-%j.out
#
# Statistiques d'un dataset (CPU seulement, pas de GPU) : taux de changement,
# distribution des classes dans les zones changées, matrice de transition.
# Lecture parallélisée : à lancer en job, trop lent sur un nœud de connexion.
#
#   sbatch scripts/dataset_stats.sbatch                                  # Hi-UCD
#   DATASET=second DATA=$SCRATCH/SECOND SPLITS="train test" \
#       sbatch --export=ALL,DATASET,DATA,SPLITS scripts/dataset_stats.sbatch
set -euo pipefail

DATASET="${DATASET:-hi_ucd}"
DATA="${DATA:-/scratch/magron13/hi-ucd}"
SPLITS="${SPLITS:-train val}"
N="${N:-1500}"

module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"

cd "$HOME/csf-mamba"
python -m scripts.dataset_stats \
    --data-root "$DATA" \
    --dataset "$DATASET" \
    --splits $SPLITS \
    --max-samples "$N" \
    --num-workers 8
