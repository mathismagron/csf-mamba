#!/bin/bash
#SBATCH --job-name=csf-eff
#SBATCH --account=def-hervete
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=logs/csf-eff-%j.out
#
# Étape 5 : efficience élève vs professeur, même job, même GPU, protocole du README
# (lot de 8, chauffe 10, médiane de 50) — scripts/measure_efficiency.py.
# Prérequis : build MSDA partagé (scripts/teacher/build_msda_student.sbatch).
#
#   cd ~/csf-mamba && sbatch scripts/measure_efficiency.sbatch
set -euo pipefail
REPO="$HOME/csf-mamba"
PERASCD="$REPO/third_party/PerASCD"
TCKPT="$SCRATCH/csf-distill/teacher/ckpt/PerAChain_40e_mIoU74.33_Sek26.11_Fscd66.41_OA88.70.pth"
OUT="$SCRATCH/csf-distill/checks/efficiency_${SLURM_JOB_ID}.json"

module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"
export PYTHONPATH="$SCRATCH/csf-distill/msda_site${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
echo "== commit $(git rev-parse --short HEAD) ($(git status --porcelain | wc -l) fichiers modifiés) | job $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
python -c "import torch, MultiScaleDeformableAttention as m; print('MSDA :', m.__file__)"
python -u -m scripts.measure_efficiency --teacher-root "$PERASCD" --teacher-ckpt "$TCKPT" --out "$OUT"
echo "== résultats : $OUT"
