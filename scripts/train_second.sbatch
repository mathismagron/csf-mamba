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
# 2 968 paires train / 1 694 test (split officiel, lu depuis train.txt/test.txt).
# ~371 steps/époque, soit ~4x moins que Hi-UCD : compter ~3-4 h pour 100 époques.
#
# Réglages SPÉCIFIQUES à SECOND (ne pas copier ceux de Hi-UCD) :
#   --val-split test        SECOND n'a pas de split 'val'
#   --sek-warmup-iters 0    comme Mamba-FCS (SEK_START_ITER=0 sur SECOND) : la
#                           sémantique y est annotée seulement sur le changement,
#                           donc le kappa devient positif vite.
#   --lambda-sem-change 0   redondant ici : ce_sem est DÉJÀ restreint aux zones
#                           changées par la nature des labels. C'était un
#                           correctif propre à Hi-UCD (sémantique pleine scène).
#
# Soumission :  cd $HOME/csf-mamba && sbatch scripts/train_second.sbatch
set -euo pipefail

module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Staging sur le stockage local du nœud (dataset petit : copie directe).
echo "== staging SECOND vers \$SLURM_TMPDIR =="
cp -r "$SCRATCH/SECOND" "$SLURM_TMPDIR/"
DATA_ROOT="$SLURM_TMPDIR/SECOND"

CKPT="$SCRATCH/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth"
OUT="$SCRATCH/csf-mamba-runs/second_mini_chess"

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
    --bcd-change-weight 5 \
    --lambda-dice 1.0 \
    --lambda-sem-change 0 \
    --sek-warmup-iters 0 \
    --epochs 100 \
    --seed 42 \
    --resume auto \
    --output "$OUT"
