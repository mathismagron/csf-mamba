#!/bin/bash
#SBATCH --job-name=csf-gmacs
#SBATCH --account=def-hervete
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=csf-gmacs-%j.out
#
# Compte GMACs + paramètres. Nécessite un GPU (kernel selective_scan).
# Compare les deux variantes de backbone pour le tableau d'efficience.
#
#   sbatch scripts/count_gmacs.sbatch
#   SIZE=256 sbatch --export=ALL,SIZE scripts/count_gmacs.sbatch
set -euo pipefail

SIZE="${SIZE:-512}"

module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"
cd "$HOME/csf-mamba"

for ENC in vmamba_mini vmamba_tiny; do
    echo "########## $ENC (taille $SIZE) — début $(date +%H:%M:%S)"
    python -u -m scripts.count_gmacs --encoder "$ENC" --size "$SIZE" --dataset second
    echo "########## $ENC terminé $(date +%H:%M:%S)"
done
