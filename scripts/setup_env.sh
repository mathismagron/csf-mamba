#!/bin/bash
# Setup de l'environnement sur Narval (Alliance) — À LANCER SUR UN NŒUD DE CONNEXION.
#
# Pourquoi le nœud de connexion : c'est là qu'il y a internet (télécharger torch
# PyPI + compiler le kernel). Le venv est persistant dans $SCRATCH (on ne le
# reconstruit pas à chaque job — on ne veut pas recompiler le kernel CUDA).
#
#   Layout retenu :
#     repo   -> $HOME/csf-mamba            (code, sauvegardé)
#     venv   -> $SCRATCH/csf-venv-cu12     (grand, non sauvegardé)
#     data   -> $SCRATCH/hi-ucd(.tar)      (grand)
#     poids  -> $SCRATCH/pretrained_weight/
#
# IMPORTANT — pourquoi CUDA 12 et pas le wheelhouse :
#   Le wheelhouse Alliance ne fournit QUE torch 2.13 (compilé CUDA 13.2). Or CUDA
#   13 a retiré des symboles CUB (cub::LaneId, CTA_SYNC, ShuffleDown...) utilisés
#   par le kernel selective_scan de ChangeMamba -> il ne compile pas. On installe
#   donc torch CUDA 12 depuis PyPI (le kernel a été écrit pour CUDA 12) et on
#   charge cuda/12.2 dont le CUB contient encore ces symboles. Combo validé.
#
# Usage :  cd $HOME/csf-mamba && bash scripts/setup_env.sh
set -euo pipefail

REPO="${REPO:-$HOME/csf-mamba}"
VENV="${VENV:-$SCRATCH/csf-venv-cu12}"

module load python/3.11 cuda/12.2

echo "== création du venv persistant : $VENV =="
virtualenv --no-download "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip

# torch CUDA 12 depuis PyPI (PAS le wheelhouse, qui n'a que du CUDA 13).
# 2.4.1/0.19.1 = paire appariée, compatible ChangeMamba.
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# Reste des dépendances (torch tire triton tout seul en version compatible).
pip install numpy pillow scipy einops timm fvcore

# Kernel CUDA selective_scan du backbone VMamba (VMamba ne tourne PAS sans lui).
# En CUDA 12, compute_70 est encore supporté et les symboles CUB existent : le
# setup.py de ChangeMamba compile tel quel, aucun patch nécessaire.
echo "== compilation du kernel selective_scan (peut prendre plusieurs minutes) =="
pip install --no-build-isolation "$REPO/third_party/ChangeMamba/kernels/selective_scan"

# Kernel fusionné pour NOTRE C²S² (optionnel). S'il échoue, backend='ref' (plus
# lent mais fonctionnel sur GPU aussi).
pip install mamba_ssm causal_conv1d \
    || echo "== mamba_ssm indisponible : utiliser --backend ref =="

python - <<'PY'
import torch
print("torch", torch.__version__, "| version CUDA de torch:", torch.version.cuda)
PY
echo "venv prêt : $VENV"
echo "NB : tester l'import du kernel dans un JOB GPU (pas sur le nœud de connexion)."
