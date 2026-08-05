#!/bin/bash
#SBATCH --job-name=csf-gmacs
#SBATCH --account=def-hervete
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=csf-gmacs-%j.out
#
# Compte GMACs + paramètres. Nécessite un GPU (kernel selective_scan).
# Compare les deux variantes de backbone pour le tableau d'efficience.
#
#   sbatch scripts/count_gmacs.sbatch
#   SIZE=256 sbatch --export=ALL,SIZE scripts/count_gmacs.sbatch
#
# ⚠️ Le traçage fvcore d'un modèle SSM est LENT : compter ~30-45 min par variante.
# Le job à 1 h avec les deux backbones a fini en TIMEOUT (job 173649). D'où les 3 h,
# et ENCODERS pour ne mesurer que ce qui manque plutôt que tout reprendre :
#   ENCODERS=vmamba_mini DECODER_REFINE=full \
#     sbatch --export=ALL,ENCODERS,DECODER_REFINE scripts/count_gmacs.sbatch
set -euo pipefail

SIZE="${SIZE:-512}"
ENCODERS="${ENCODERS:-vmamba_mini vmamba_tiny}"
DECODER_REFINE="${DECODER_REFINE:-dw}"

module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"
cd "$HOME/csf-mamba"

for ENC in $ENCODERS; do
    echo "########## $ENC / décodeur $DECODER_REFINE (taille $SIZE) — début $(date +%H:%M:%S)"
    python -u -m scripts.count_gmacs --encoder "$ENC" --size "$SIZE" --dataset second \
        --decoder-refine "$DECODER_REFINE"
    echo "########## $ENC terminé $(date +%H:%M:%S)"
done
