#!/bin/bash
#SBATCH --job-name=csf-test
#SBATCH --account=def-hervete
#SBATCH --qos=cc-debug
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0:20:0
#SBATCH --output=csf-test-%j.out
#
# Run de test rapide (QOS debug) : vérifie que le kernel + VMamba + tout le
# pipeline tournent sur GPU. 10 batches, 1 époque, batch 2 (le 512x512 pleine
# résolution est gourmand — batch 8 sature l'A100 40G). Pas un vrai entraînement.
# Soumission :  cd $HOME/csf-mamba && sbatch scripts/test.sbatch
set -euo pipefail

module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$HOME/csf-mamba"
python -m scripts.train \
    --data-root /scratch/magron13/hi-ucd \
    --dataset hi_ucd \
    --encoder vmamba_mini \
    --encoder-pretrained "$SCRATCH/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth" \
    --backend mamba \
    --epochs 1 --limit-batches 10 --batch-size 2 \
    --output "$SCRATCH/csf-test"
