#!/bin/bash
# Étape 0 — installation du professeur PerASCD. À lancer sur un nœud de CONNEXION
# (internet requis ; les nœuds de calcul n'y ont pas accès) :
#
#   cd ~/csf-mamba && bash scripts/teacher/setup_teacher.sh
#
# Ce que fait le script, idempotent (le relancer ne refait que ce qui manque) :
#   1. clone PerASCD, branche `legacy`, figée au commit vérifié le 25 septembre ;
#   2. télécharge le checkpoint SECOND publié (4,1 Go) et en note le sha256 ;
#   3. crée un venv SÉPARÉ de celui de l'élève (le professeur n'y installe rien).
# L'opérateur CUDA MultiScaleDeformableAttention est compilé plus tard, dans le
# job GPU (son setup.py refuse de compiler sans GPU visible).
set -euo pipefail

REPO="$HOME/csf-mamba"
PERASCD="$REPO/third_party/PerASCD"          # third_party/ est git-ignoré
COMMIT="a4d808a6cfb5df7efeee186730ac26b4504c9ed6"
TEACHER="$SCRATCH/csf-distill/teacher"
VENV="$SCRATCH/perascd-venv"
ZIP_NAME="PerASCD_260128115444_vitg01min0Clip15LsscTau001.zip"
URL="https://huggingface.co/SathShen/PerASCD-Checkpoint/resolve/main/$ZIP_NAME"
PTH_NAME="PerAChain_40e_mIoU74.33_Sek26.11_Fscd66.41_OA88.70.pth"

echo "== 1. code PerASCD (legacy @ ${COMMIT:0:7})"
if [ ! -d "$PERASCD/.git" ]; then
    git clone --branch legacy https://github.com/SathShen/PerASCD.git "$PERASCD"
fi
git -C "$PERASCD" fetch -q origin legacy
git -C "$PERASCD" -c advice.detachedHead=false checkout -q "$COMMIT"
echo "   $(git -C "$PERASCD" log -1 --format='%h %ad %s' --date=short)"

echo "== 2. checkpoint SECOND publié"
mkdir -p "$TEACHER"
if [ ! -f "$TEACHER/ckpt/$PTH_NAME" ]; then
    if [ ! -f "$TEACHER/$ZIP_NAME" ]; then
        wget -q --show-progress -c -O "$TEACHER/$ZIP_NAME.part" "$URL"
        mv "$TEACHER/$ZIP_NAME.part" "$TEACHER/$ZIP_NAME"
    fi
    unzip -n -q "$TEACHER/$ZIP_NAME" -d "$TEACHER/ckpt"
fi
ls -la "$TEACHER/ckpt"
if [ ! -f "$TEACHER/SHA256SUMS" ]; then
    (cd "$TEACHER" && sha256sum "$ZIP_NAME" "ckpt/$PTH_NAME" > SHA256SUMS)
fi
cat "$TEACHER/SHA256SUMS"
# Le zip n'est plus utile une fois extrait et haché ; on le garde tant que le
# sha256 n'a pas été reporté dans le journal, puis : rm "$TEACHER/$ZIP_NAME".

echo "== 3. venv du professeur ($VENV)"
module load python/3.11 cuda/12.2
if [ ! -d "$VENV" ]; then
    virtualenv --no-download "$VENV"
fi
source "$VENV/bin/activate"
pip install -q --no-index --upgrade pip
# Même torch que l'élève (2.5.1, CUDA 12) : comparaisons de latence à pile égale.
pip install -q --no-index torch==2.5.1 torchvision==0.20.1 timm numpy scipy pillow setuptools wheel ninja
python - <<'EOF'
import torch, torchvision, timm, numpy, scipy
print(f"   torch {torch.__version__} (CUDA {torch.version.cuda}) | torchvision {torchvision.__version__} "
      f"| timm {timm.__version__} | numpy {numpy.__version__} | scipy {scipy.__version__}")
EOF
pip freeze > "$TEACHER/freeze_perascd-venv.txt"
echo "OK. Étape suivante : sbatch scripts/teacher/eval_perascd.sbatch"
