#!/bin/bash
#SBATCH --job-name=csf-preflight
#SBATCH --account=def-hervete
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/csf-preflight-%j.out
#
# Le dossier logs/ doit exister avant la soumission (mkdir -p logs).
#
# Contrôle préalable de l'audit GMACs : dépôt tiers, configs, et comptes de
# paramètres des trois modèles ChangeMamba comparés aux valeurs publiées.
# Pas de --gres=gpu : la construction d'un modèle ne demande pas le kernel CUDA,
# seul le forward le fait. Un job CPU démarre presque tout de suite.
set -euo pipefail
module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"
cd "$HOME/csf-mamba"
python -u -m scripts.audit_gmacs --params-only
