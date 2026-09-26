#!/bin/bash
#SBATCH --job-name=perascd-cache
#SBATCH --account=def-hervete
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:30:00
#SBATCH --output=logs/perascd-cache-%j.out
#
# Étape 0-cache : sorties du professeur sur les 2 968 paires de SECOND train, pour
# les 8 éléments de D4, en fp16 (SeK test fp16 = fp32 à 0,0003 pt près, étape 0),
# à la résolution native 128² du décodeur. ≈11,7 Go dans
# $SCRATCH/csf-distill/cache/perascd_second_train/.
#
#   cd ~/csf-mamba && sbatch scripts/teacher/cache_teacher.sbatch
#
# Durée attendue : 2 968 × 8 vues à ~16 paires/s (lot de 8, mesuré à l'étape 0)
# ≈ 25 min de calcul, plus lecture et écriture.
set -euo pipefail

REPO="$HOME/csf-mamba"
PERASCD="$REPO/third_party/PerASCD"
CKPT="$SCRATCH/csf-distill/teacher/ckpt/PerAChain_40e_mIoU74.33_Sek26.11_Fscd66.41_OA88.70.pth"
OUT="$SCRATCH/csf-distill/cache/perascd_second_train"

module load python/3.11 cuda/12.2
source "$SCRATCH/perascd-venv/bin/activate"
cd "$REPO"
echo "== csf-mamba $(git rev-parse --short HEAD) | PerASCD $(git -C "$PERASCD" rev-parse --short HEAD)"
python -c "import torch, MultiScaleDeformableAttention" \
    || { echo "opérateur déformable absent : lancer d'abord eval_perascd.sbatch"; exit 1; }

python scripts/teacher/cache_teacher.py \
    --perascd-root "$PERASCD" \
    --checkpoint "$CKPT" \
    --data-root "$SCRATCH/SECOND" \
    --split train --precision fp16 --batch-size 8 \
    --out-dir "$OUT"
du -sh "$OUT"
