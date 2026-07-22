#!/bin/bash
#SBATCH --job-name=csf-mamba
#SBATCH --account=def-hervete         # allocation Narval (Prof. Eric Hervet)
#SBATCH --gres=gpu:a100:1             # Narval : A100 (sm_80)
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=csf-mamba-%j.out
#
# Vrai entraînement (crops 256, batch 8, 50 époques). Reprise auto : si le job
# atteint la limite de temps, resoumettre le MÊME script reprend depuis last.pt.
#
# Prérequis (faits UNE fois sur un nœud de connexion) :
#   bash scripts/setup_env.sh
#   bash scripts/download_pretrained.sh $SCRATCH/pretrained_weight
#   # créer l'archive du dataset (une fois) :
#   cd /scratch/magron13/hi-ucd && tar -cf $SCRATCH/hi-ucd.tar train val test
#
# Soumission :  cd $HOME/csf-mamba && sbatch scripts/train.sbatch
set -euo pipefail

module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Dataset stagé sur le stockage LOCAL du nœud (rapide vs 40k fichiers sur Lustre).
echo "== staging dataset vers \$SLURM_TMPDIR =="
mkdir -p "$SLURM_TMPDIR/hi-ucd"
tar -xf "$SCRATCH/hi-ucd.tar" -C "$SLURM_TMPDIR/hi-ucd"
DATA_ROOT="$SLURM_TMPDIR/hi-ucd"

CKPT="$SCRATCH/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth"
# Sortie FIXE (pas $SLURM_JOB_ID) pour que la reprise retrouve last.pt.
OUT="$SCRATCH/csf-mamba-runs/hiucd_mini_chess"

python -m scripts.train \
    --data-root "$DATA_ROOT" \
    --dataset hi_ucd \
    --encoder vmamba_mini \
    --encoder-pretrained "$CKPT" \
    --core chess \
    --backend mamba \
    --crop-size 256 \
    --batch-size 8 \
    --epochs 50 \
    --seed 42 \
    --resume auto \
    --output "$OUT"
