#!/bin/bash
#SBATCH --job-name=csf-cmeval
#SBATCH --account=def-hervete
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=logs/csf-cmeval-%j.out
#
# Le dossier logs/ doit exister avant la soumission (mkdir -p logs) : Slurm
# n'ouvre pas le fichier de sortie si son dossier est absent, et le job échoue.
#
# Évalue un checkpoint ChangeMamba publié avec NOTRE code de métriques : même
# split, mêmes images, même évaluateur des deux côtés. C'est la comparaison la
# plus rigoureuse possible avec le SOTA, et elle donne au passage le compte de
# paramètres exact de leur modèle.
#
#   sbatch scripts/evaluate_changemamba.sbatch
#   CKPT=/chemin/vers/autre.pth CONFIG=vssm1/vssm_base_224.yaml \
#     sbatch --export=ALL,CKPT,CONFIG scripts/evaluate_changemamba.sbatch
#
# 3 h de walltime : ~1 h d'évaluation, plus l'ouverture du venv sur Lustre qui
# coûte à elle seule près d'une heure (mesuré : 43 min pour le seul import torch).
set -euo pipefail

CKPT="${CKPT:-$SCRATCH/pretrained_weight/MambaSCD_Tiny_SECOND_SeK_0.2208.pth}"
CONFIG="${CONFIG:-vssm1/vssm_tiny_224_0229flex.yaml}"
SPLIT="${SPLIT:-test}"

if [ ! -f "$CKPT" ]; then
    echo "checkpoint introuvable : $CKPT"
    echo
    echo "Le récupérer depuis le dépôt Zenodo 15479555 — le même que le backbone"
    echo "VMamba et le SECOND prétraité. Sur un nœud de CONNEXION (accès réseau) :"
    echo
    echo "  FILES=\"$(basename "$CKPT")\" \\"
    echo "    scripts/download_pretrained.sh \$SCRATCH/pretrained_weight"
    echo
    echo "puis relancer ce job. Variante Base disponible :"
    echo "  MambaSCD_Base_SECOND_SeK_0.2292.pth"
    exit 1
fi

module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"
cd "$HOME/csf-mamba"

python -u -m scripts.evaluate_changemamba \
    --checkpoint "$CKPT" \
    --config "$CONFIG" \
    --data-root "$SCRATCH/SECOND" \
    --split "$SPLIT"
