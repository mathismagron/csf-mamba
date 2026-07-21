#!/bin/bash
#SBATCH --job-name=csf-mamba
#SBATCH --account=def-hervete         # allocation Narval (Prof. Eric Hervet)
#SBATCH --gres=gpu:a100:1             # Narval : A100 (sm_80)
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#
# Prérequis (faits UNE fois sur un nœud de connexion) :
#   bash scripts/setup_env.sh            # venv cu12 + kernel dans $SCRATCH
#   bash scripts/download_pretrained.sh $SCRATCH/pretrained_weight
#   # + dataset Hi-UCD archivé dans $SCRATCH/hi-ucd.tar
#
# Soumission :  cd $HOME/csf-mamba && sbatch scripts/train.sbatch
set -euo pipefail

# cuda/12.2 : DOIT matcher le torch cu12 du venv (cf. setup_env.sh).
module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"

# Dataset : la copie maître est dans $SCRATCH, mais on la décompresse sur le
# stockage LOCAL du nœud ($SLURM_TMPDIR) — bien plus rapide que lire 40k petits
# fichiers depuis Lustre pendant tout l'entraînement.
echo "== staging dataset vers \$SLURM_TMPDIR =="
# L'archive contient train/ val/ test/ à la racine (créée avec :
#   cd <dossier Hi-UCD dézippé> && tar -cf $SCRATCH/hi-ucd.tar train val test)
mkdir -p "$SLURM_TMPDIR/hi-ucd"
tar -xf "$SCRATCH/hi-ucd.tar" -C "$SLURM_TMPDIR/hi-ucd"
DATA_ROOT="$SLURM_TMPDIR/hi-ucd"

CKPT="$SCRATCH/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth"

python -m scripts.train \
    --data-root "$DATA_ROOT" \
    --dataset hi_ucd \
    --encoder vmamba_mini \
    --encoder-pretrained "$CKPT" \
    --core chess \
    --backend auto \
    --seed 42 \
    --output "$SCRATCH/csf-mamba-runs/$SLURM_JOB_ID"
