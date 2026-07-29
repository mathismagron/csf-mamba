#!/bin/bash
#SBATCH --job-name=csf-second
#SBATCH --account=def-hervete
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=csf-second-%j.out
#
# Entraînement sur SECOND — le benchmark de référence du domaine.
# 2 968 paires train / 1 694 test (split officiel). ~371 steps/époque, ~3-4 h.
#
# Réglages SPÉCIFIQUES à SECOND (ne pas copier ceux de Hi-UCD) :
#   --val-split test        SECOND n'a pas de split 'val'
#   --sek-warmup-iters 0    comme Mamba-FCS (SEK_START_ITER=0 sur SECOND)
#   --lambda-sem-change 0   redondant : ce_sem y est DÉJÀ restreint au changement
#
# DÉFAUT = sans compensation de déséquilibre (WEIGHT=1, DICE=0). SECOND a ~20 %
# de pixels changés : il est quasi équilibré et n'en a pas besoin. Le premier run
# (WEIGHT=5 DICE=1) sur-détectait le changement d'un facteur 1,5 (30,9 % prédits
# contre 20,1 % réels), ce qui dégradait précision, Fscd, mIoU et SeK.
#
# Ablations (un seul facteur à la fois, dossier de sortie distinct) :
#   sbatch scripts/train_second.sbatch                                   # défaut
#   WEIGHT=5 DICE=1 TAG=w5d1 sbatch --export=ALL,WEIGHT,DICE,TAG scripts/train_second.sbatch
set -euo pipefail

WEIGHT="${WEIGHT:-1}"        # poids de la classe changement dans la CE
DICE="${DICE:-0}"            # poids de la loss Dice
EPOCHS="${EPOCHS:-100}"
TAG="${TAG:-nobal}"          # suffixe du dossier de sortie

module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "== staging SECOND vers \$SLURM_TMPDIR =="
cp -r "$SCRATCH/SECOND" "$SLURM_TMPDIR/"
DATA_ROOT="$SLURM_TMPDIR/SECOND"

CKPT="$SCRATCH/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth"
OUT="$SCRATCH/csf-mamba-runs/second_mini_chess_$TAG"
echo "== config : weight=$WEIGHT dice=$DICE epochs=$EPOCHS -> $OUT =="

python -m scripts.train \
    --data-root "$DATA_ROOT" \
    --dataset second \
    --val-split test \
    --encoder vmamba_mini \
    --encoder-pretrained "$CKPT" \
    --core chess \
    --backend mamba \
    --crop-size 256 \
    --batch-size 8 \
    --bcd-change-weight "$WEIGHT" \
    --lambda-dice "$DICE" \
    --lambda-sem-change 0 \
    --sek-warmup-iters 0 \
    --epochs "$EPOCHS" \
    --seed 42 \
    --resume auto \
    --output "$OUT"
