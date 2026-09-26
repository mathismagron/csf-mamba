#!/bin/bash
#SBATCH --job-name=perascd-step0
#SBATCH --account=def-hervete
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=logs/perascd-step0-%j.out
#
# Étape 0 du plan de distillation : le professeur PerASCD réévalué sous notre
# protocole (SECOND test, tuile 512, notre code de métrique ET le leur).
# Prérequis, une fois, sur un nœud de connexion : bash scripts/teacher/setup_teacher.sh
#
#   cd ~/csf-mamba && mkdir -p logs && sbatch scripts/teacher/eval_perascd.sbatch
#
# Au premier passage, compile l'opérateur MultiScaleDeformableAttention dans le
# venv du professeur (≈5 min, son setup.py exige un GPU visible) ; ensuite il
# est réutilisé.
set -euo pipefail

REPO="$HOME/csf-mamba"
PERASCD="$REPO/third_party/PerASCD"
TEACHER="$SCRATCH/csf-distill/teacher"
CKPT="$TEACHER/ckpt/PerAChain_40e_mIoU74.33_Sek26.11_Fscd66.41_OA88.70.pth"
OUT="$SCRATCH/csf-distill/checks"
mkdir -p "$OUT"

module load python/3.11 cuda/12.2
source "$SCRATCH/perascd-venv/bin/activate"
cd "$REPO"
echo "== csf-mamba $(git rev-parse --short HEAD) | PerASCD $(git -C "$PERASCD" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== opérateur déformable"
if ! python -c "import torch, MultiScaleDeformableAttention" 2>/dev/null; then
    BUILD="$SLURM_TMPDIR/msda"
    cp -r "$PERASCD/models/pera_layers/vit_adapter_layers/ops" "$BUILD"
    rm -rf "$BUILD/build" "$BUILD/dist" "$BUILD"/*.egg-info   # artefacts de leur machine
    ( cd "$BUILD" && TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=8 \
        pip install --no-index --no-build-isolation . )
fi
python -c "import torch, MultiScaleDeformableAttention as m; print('   compilé :', m.__file__)"

echo "== évaluation"
python scripts/teacher/eval_perascd.py \
    --perascd-root "$PERASCD" \
    --checkpoint "$CKPT" \
    --data-root "$SCRATCH/SECOND" \
    --precisions fp32,fp16,bf16 \
    --check-msda --latency --flops \
    --out "$OUT/step_0_${SLURM_JOB_ID}.json"
