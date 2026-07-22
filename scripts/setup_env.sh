#!/bin/bash
# Setup de l'environnement sur Narval (Alliance) — À LANCER SUR UN NŒUD DE CONNEXION.
#
# Tout vient du wheelhouse Alliance (--no-index) : builds `+computecanada`
# cohérents entre eux et avec les modules CUDA du cluster. Le venv est persistant
# dans $SCRATCH (on ne recompile pas le kernel à chaque job).
#
#   Layout :
#     repo   -> $HOME/csf-mamba            (code, sauvegardé)
#     venv   -> $SCRATCH/csf-venv-cu12     (grand, non sauvegardé)
#     data   -> $SCRATCH/hi-ucd/           (grand)
#     poids  -> $SCRATCH/pretrained_weight/
#
# CHOIX DE VERSIONS (durement gagnés) :
#   - torch 2.5.1 (CUDA 12) : imposé par mamba_ssm (torch~=2.5). CUDA 12 (pas 13)
#     car le kernel selective_scan utilise des symboles CUB retirés en CUDA 13.
#   - torch + torchvision doivent être le MÊME build : on les installe ENSEMBLE en
#     premier (sinon une install ultérieure tire un torchvision dépareillé ->
#     `torchvision::nms does not exist`).
#   - mamba_ssm : kernel fusionné pour NOTRE C²S². Sans lui, backend='ref' (scan
#     Python) = trop lent pour du 512² (le step ne finit jamais). Son __init__ est
#     cassé (transformers récent) mais on l'importe par sous-module dans le code.
#   - selective_scan : kernel du backbone VMamba, compilé contre le torch installé.
#
# Usage :  cd $HOME/csf-mamba && bash scripts/setup_env.sh
set -euo pipefail

REPO="${REPO:-$HOME/csf-mamba}"
VENV="${VENV:-$SCRATCH/csf-venv-cu12}"

module load python/3.11 cuda/12.2

echo "== création du venv persistant : $VENV =="
virtualenv --no-download "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip

# torch + torchvision APPARIÉS, en premier, tout du wheelhouse.
pip install --no-index torch==2.5.1 torchvision==0.20.1

# Reste des dépendances de base (triton importé par VMamba au chargement).
pip install --no-index numpy pillow scipy einops timm fvcore triton

# Kernel fusionné mamba_ssm pour notre C²S² (torch déjà en 2.5.1 -> pas d'upgrade).
pip install --no-index mamba_ssm causal_conv1d

# Kernel CUDA selective_scan du backbone VMamba, compilé EN DERNIER (contre le
# torch définitif). En CUDA 12 les symboles CUB existent : compile tel quel.
echo "== compilation du kernel selective_scan (peut prendre plusieurs minutes) =="
pip install --no-build-isolation "$REPO/third_party/ChangeMamba/kernels/selective_scan"

# Vérif non bloquante (les kernels CUDA ne s'importent qu'avec un GPU : un échec
# ici sur le nœud de connexion est NORMAL, le vrai test se fait dans un job GPU).
python - <<'PY' || true
import torch, torchvision, torchvision.ops  # noqa: F401
print("torch", torch.__version__, "| torchvision", torchvision.__version__)
PY
echo "venv prêt : $VENV — tester le forward GPU dans un job."
