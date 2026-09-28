#!/bin/bash
#SBATCH --job-name=msda-build
#SBATCH --account=def-hervete
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=0:45:00
#SBATCH --output=logs/msda-build-%j.out
#
# Étape 4 : compile UNE fois l'opérateur MultiScaleDeformableAttention du
# professeur pour le venv de l'ÉLÈVE (csf-venv-cu12, torch 2.5.1, CUDA 12.2) et le
# range dans $SCRATCH/csf-distill/msda_site. train_kd_second.sbatch l'utilise s'il
# s'importe, sinon recompile dans le job (≈12 min, job 4176343).
#
#   cd ~/csf-mamba && sbatch scripts/teacher/build_msda_student.sbatch
set -euo pipefail

REPO="$HOME/csf-mamba"
PERASCD="$REPO/third_party/PerASCD"
DEST="$SCRATCH/csf-distill/msda_site"

module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"
cd "$REPO"

BUILD="$SLURM_TMPDIR/msda_src"; STAGE="$SLURM_TMPDIR/msda_site"
cp -r "$PERASCD/models/pera_layers/vit_adapter_layers/ops" "$BUILD"
rm -rf "$BUILD/build" "$BUILD/dist" "$BUILD"/*.egg-info
t0=$(date +%s)
( cd "$BUILD" && TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=12 \
    pip install -q --no-index --no-build-isolation --no-deps --target "$STAGE" . )
PYTHONPATH="$STAGE" python -c "import torch, MultiScaleDeformableAttention as m; print('compilé :', m.__file__)"
python - <<EOF > "$STAGE/BUILD_INFO.txt"
import torch, platform
print(f"torch {torch.__version__} cuda {torch.version.cuda} python {platform.python_version()}")
EOF
echo "PerASCD $(git -C "$PERASCD" rev-parse HEAD) job $SLURM_JOB_ID $(date '+%F %T')" >> "$STAGE/BUILD_INFO.txt"
cat "$STAGE/BUILD_INFO.txt"
echo "compilation : $(( $(date +%s) - t0 )) s"

# Remplacement atomique : un job qui démarre pendant la copie voit l'ancien dossier
# ou le nouveau, jamais un dossier à moitié écrit.
rm -rf "$DEST.tmp"; cp -r "$STAGE" "$DEST.tmp"
rm -rf "$DEST.old"; [ -d "$DEST" ] && mv "$DEST" "$DEST.old"
mv "$DEST.tmp" "$DEST"; rm -rf "$DEST.old"
ls -la "$DEST"
