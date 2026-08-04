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
#
# Résolution : l'entraînement se fait sur des crops 256 mais la validation sur des
# tuiles 512 entières — pour un modèle SSM, cela quadruple la longueur de séquence
# entre les deux régimes. Pour tester l'effet de ce décalage, entraîner en 512 :
#   CROP=512 BATCH=2 ACCUM=4 TAG=crop512 \
#     sbatch --time=24:00:00 --export=ALL,CROP,BATCH,ACCUM,TAG scripts/train_second.sbatch
# BATCH=2 x ACCUM=4 conserve le batch effectif de 8 : SEULE la résolution change
# (512² en batch 8 sature l'A100). Compter ~4x le temps : ~14 h au lieu de ~3-4 h.
set -euo pipefail

WEIGHT="${WEIGHT:-1}"        # poids de la classe changement dans la CE
DICE="${DICE:-0}"            # poids de la loss Dice
EPOCHS="${EPOCHS:-100}"
LOVASZ="${LOVASZ:-0}"        # poids de la loss Lovász (optimise l'IoU)
CROP="${CROP:-256}"          # taille du crop d'entraînement (512 = tuile entière)
BATCH="${BATCH:-8}"          # micro-batch
ACCUM="${ACCUM:-1}"          # accumulation : batch effectif = BATCH x ACCUM
DECODER_REFINE="${DECODER_REFINE:-dw}"  # dw = référence, full = décodeur élargi
ENCODER="${ENCODER:-vmamba_mini}"  # vmamba_mini (13,8M) ou vmamba_tiny (28M)
TAG="${TAG:-nobal}"          # suffixe du dossier de sortie

module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "== staging SECOND vers \$SLURM_TMPDIR =="
cp -r "$SCRATCH/SECOND" "$SLURM_TMPDIR/"
DATA_ROOT="$SLURM_TMPDIR/SECOND"

CKPT="$SCRATCH/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth"
OUT="$SCRATCH/csf-mamba-runs/second_mini_chess_$TAG"
echo "== config : weight=$WEIGHT dice=$DICE lovasz=$LOVASZ crop=$CROP batch=$BATCH x accum=$ACCUM enc=$ENCODER dec=$DECODER_REFINE epochs=$EPOCHS -> $OUT =="

python -m scripts.train \
    --data-root "$DATA_ROOT" \
    --dataset second \
    --val-split test \
    --encoder "$ENCODER" \
    --decoder-refine "$DECODER_REFINE" \
    --encoder-pretrained "$CKPT" \
    --core chess \
    --backend mamba \
    --crop-size "$CROP" \
    --batch-size "$BATCH" \
    --accum-steps "$ACCUM" \
    --bcd-change-weight "$WEIGHT" \
    --lambda-dice "$DICE" \
    --lambda-lovasz "$LOVASZ" \
    --lambda-sem-change 0 \
    --sek-warmup-iters 0 \
    --epochs "$EPOCHS" \
    --seed 42 \
    --resume auto \
    --output "$OUT"
