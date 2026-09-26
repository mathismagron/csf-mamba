#!/bin/bash
#SBATCH --job-name=csf-check-P
#SBATCH --account=def-hervete
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/csf-check-P-%j.out
#
# Étape P du plan de distillation (documentation/distillation.md).
#   1. tests CPU du dépôt ;
#   2. réévaluation des 7 best.pt du point d'efficience sur SECOND test,
#      en bf16 (comme à l'entraînement) et en fp32.
# Critère : |ΔSeK| ≤ 1e-4 en bf16 pour les 7 graines.
#
#   cd ~/csf-mamba && mkdir -p logs && sbatch scripts/check_narval.sbatch
#
# ~7 × 2 évaluations de 1 694 paires : 30 à 60 min sur A100. Le temps demandé
# (≤ 3 h) ouvre la partition la plus courte, donc l'attente la plus brève.
set -euo pipefail

module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"
cd "$HOME/csf-mamba"
OUT="$SCRATCH/csf-distill/checks"
mkdir -p "$OUT"

echo "== commit : $(git rev-parse --short HEAD)  ($(git status --porcelain | wc -l) fichiers modifiés)"
pip freeze > "$OUT/freeze_csf-venv-cu12_${SLURM_JOB_ID}.txt"

echo "== 1. tests CPU"
# Un test en échec ne bloque pas la réévaluation : les deux résultats comptent.
TESTS=OK
if python -c "import pytest" 2>/dev/null; then
    PYTHONPATH=. python -m pytest -q tests || TESTS=ECHEC
else
    for t in tests/test_*.py; do echo "-- $t"; PYTHONPATH=. python "$t" || TESTS=ECHEC; done
fi
echo "== tests : $TESTS"

echo "== 2. réévaluation des 7 graines"
python -m scripts.check_narval --out "$OUT/step_P_${SLURM_JOB_ID}.json"
