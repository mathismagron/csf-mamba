#!/bin/bash
#SBATCH --job-name=vmambab-diag
#SBATCH --account=def-hervete
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=logs/vmambab-diag-%j.out
#
# Diagnostic de l'écart de reproduction du professeur VMamba-B (23,73 contre 25,31,
# job 4223414) : noyaux compilés contre implémentations PyTorch de référence, TF32.
#   cd ~/csf-mamba && sbatch scripts/teacher/diag_vmambab.sbatch
set -euo pipefail
REPO="$HOME/csf-mamba"
PERASCD="$REPO/third_party/PerASCD"
CKPT="$SCRATCH/csf-distill/teacher_vmambab/ckpt/vmambaB_42e_mIoU74.01_Sek25.31_Fscd65.61_OA88.37.pth"
module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"
cd "$REPO"
echo "== commit $(git rev-parse --short HEAD) | job $SLURM_JOB_ID"
pip list 2>/dev/null | grep -i -E "^(mamba|causal|triton|torch |timm)" || true
python -c "import torch, selective_scan_cuda as s; print('selective_scan_cuda :', s.__file__)"
unzip -q "$SCRATCH/SECOND.zip" -x '*_COLORED/*' -d "$SLURM_TMPDIR"
python -u scripts/teacher/diag_vmambab.py --perascd-root "$PERASCD" --checkpoint "$CKPT" \
    --data-root "$SLURM_TMPDIR/SECOND" --pairs 256 \
    --out "$SCRATCH/csf-distill/checks/diag_vmambab_${SLURM_JOB_ID}.json"
