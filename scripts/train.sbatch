#!/bin/bash
#SBATCH --job-name=csf-mamba
#SBATCH --account=aip-CHANGEME        # <- allocation AIP tamIA : aip-<nom_du_prof>
#SBATCH --gres=gpu:h100:1             # tamIA : H100 80G. NB : jobs = nœuds entiers,
                                      #   l'alloc peut donner tout le nœud. Vérifier
                                      #   le nb de GPU/nœud sur la page tamIA.
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#
# Prérequis (faits UNE fois sur un nœud de connexion) :
#   bash scripts/setup_env.sh            # venv + kernel dans $SCRATCH
#   bash scripts/download_pretrained.sh $SCRATCH/pretrained_weight
#   # + dataset Hi-UCD archivé dans $SCRATCH/hi-ucd.tar
#
# Soumission :  cd $HOME/csf-mamba && sbatch scripts/train.sbatch
set -euo pipefail

module load python/3.11 cuda cudnn arrow
source "$SCRATCH/csf-venv/bin/activate"

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
