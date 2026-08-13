#!/bin/bash
#SBATCH --job-name=csf-audit
#SBATCH --account=def-hervete
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=logs/csf-audit-%j.out
#
# Le dossier logs/ doit exister avant la soumission (mkdir -p logs).
#
# Audit du comptage de GMACs. Trace quatre modèles (les trois de ChangeMamba
# plus le nôtre), d'où 48 Go et 3 h : le traçage fvcore d'un SSM est lent et
# MambaSCD-Base est gros.
#
#   sbatch scripts/audit_gmacs.sbatch
#   FUSION=concat sbatch --export=ALL,FUSION scripts/audit_gmacs.sbatch
set -euo pipefail

FUSION="${FUSION:-c2s2}"
ENCODER="${ENCODER:-vmamba_mini}"

module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"
cd "$HOME/csf-mamba"

python -u -m scripts.audit_gmacs --fusion "$FUSION" --encoder "$ENCODER"
