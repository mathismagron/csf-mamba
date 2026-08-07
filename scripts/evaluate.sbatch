#!/bin/bash
#SBATCH --job-name=csf-eval
#SBATCH --account=def-hervete
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=logs/csf-eval-%j.out
#
# Le dossier logs/ doit exister avant la soumission (mkdir -p logs) : Slurm
# n'ouvre pas le fichier de sortie si son dossier est absent, et le job échoue.
#
# Évalue un checkpoint : tableau complet des métriques + diagnostic + visualisations.
#
# Usage (variables d'environnement, avec valeurs par défaut = run SECOND) :
#   sbatch scripts/evaluate.sbatch
#   RUN=hiucd_mini_chess_v3_semch DATASET=hi_ucd DATA=$SCRATCH/hi-ucd SPLIT=val \
#       sbatch --export=ALL,RUN,DATASET,DATA,SPLIT scripts/evaluate.sbatch
#
# ENCODER et DECODER_REFINE doivent reproduire la config d'entraînement :
#   RUN=second_mini_chess_crop512-decfull DECODER_REFINE=full \
#       sbatch --export=ALL,RUN,DECODER_REFINE scripts/evaluate.sbatch
# Astuce : pour un démarrage plus rapide, ajouter la QOS sur la ligne de
# commande — sbatch --qos=cc-debug ... — plutôt que de la coder en dur
# (elle est limitée à un job simultané et parfois indisponible).
set -euo pipefail

RUN="${RUN:-second_mini_chess}"
DATASET="${DATASET:-second}"
DATA="${DATA:-$SCRATCH/SECOND}"
SPLIT="${SPLIT:-test}"
CKPT_NAME="${CKPT_NAME:-best.pt}"
# Doivent correspondre à l'entraînement du checkpoint, sinon le state_dict ne
# se recharge pas (le décodeur élargi n'a pas la même forme que la référence).
ENCODER="${ENCODER:-vmamba_mini}"
DECODER_REFINE="${DECODER_REFINE:-dw}"

module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"

cd "$HOME/csf-mamba"
python -m scripts.evaluate \
    --data-root "$DATA" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --checkpoint "$SCRATCH/csf-mamba-runs/$RUN/$CKPT_NAME" \
    --encoder "$ENCODER" \
    --decoder-refine "$DECODER_REFINE" \
    --backend mamba \
    --output "$SCRATCH/csf-mamba-runs/$RUN/eval_$SPLIT"
