#!/bin/bash
#SBATCH --job-name=csf-gmacs
#SBATCH --account=def-hervete
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=logs/csf-gmacs-%j.out
#
# Le dossier logs/ doit exister avant la soumission (mkdir -p logs) : Slurm
# n'ouvre pas le fichier de sortie si son dossier est absent, et le job échoue.
#
# Compte GMACs + paramètres. Nécessite un GPU (kernel selective_scan).
# Compare les deux variantes de backbone pour le tableau d'efficience.
#
#   sbatch scripts/count_gmacs.sbatch
#   SIZE=256 sbatch --export=ALL,SIZE scripts/count_gmacs.sbatch
#
# ⚠️ Deux TIMEOUTs successifs (173649 à 1 h, 190707 à 5 h). Le second n'a produit
# AUCUNE ligne Python après la bannière du shell : le blocage est donc en amont
# du traçage, pas dedans. Le script et ce job tracent maintenant chaque étape
# avec son horodatage — modules, venv, imports, CUDA, construction, traçage —
# pour situer le blocage au lieu de le deviner.
#
# ENCODERS permet de ne mesurer qu'une variante plutôt que les deux :
#   ENCODERS=vmamba_mini DECODER_REFINE=full \
#     sbatch --export=ALL,ENCODERS,DECODER_REFINE scripts/count_gmacs.sbatch
set -euo pipefail

SIZE="${SIZE:-512}"
ENCODERS="${ENCODERS:-vmamba_mini vmamba_tiny}"
DECODER_REFINE="${DECODER_REFINE:-dw}"
FUSION="${FUSION:-c2s2}"           # c2s2 | concat (retire tout le bloc)
CGA="${CGA:-1}"                    # 0 = sans Change-Guided Attention
MCASF="${MCASF:-1}"                # 0 = sans MCA-SF
UPSAMPLE="${UPSAMPLE:-dysample}"   # dysample | bilinear

echo "== démarrage $(date +%H:%M:%S) sur $(hostname)"
module load python/3.11 cuda/12.2
echo "== modules chargés $(date +%H:%M:%S)"
source "$SCRATCH/csf-venv-cu12/bin/activate"
echo "== venv activé $(date +%H:%M:%S)"
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd "$HOME/csf-mamba"

for ENC in $ENCODERS; do
    echo "########## $ENC / fusion=$FUSION / décodeur $DECODER_REFINE / up=$UPSAMPLE / cga=$CGA mcasf=$MCASF (taille $SIZE) — début $(date +%H:%M:%S)"
    python -u -m scripts.count_gmacs --encoder "$ENC" --size "$SIZE" --dataset second \
        --decoder-refine "$DECODER_REFINE" \
        --fusion "$FUSION" \
        --upsample "$UPSAMPLE" \
        ${CGA:+$([ "$CGA" = 0 ] && echo --no-cga)} \
        ${MCASF:+$([ "$MCASF" = 0 ] && echo --no-mcasf)}
    echo "########## $ENC terminé $(date +%H:%M:%S)"
done
