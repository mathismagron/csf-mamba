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
# POURQUOI torch 2.4.1 (et pas le défaut 2.13) :
#   `pip install --no-index torch` prend la version vedette 2.13 = CUDA 13, qui a
#   retiré des symboles CUB utilisés par le kernel selective_scan -> ne compile
#   pas. On épingle torch 2.4.1 (CUDA 12), pour lequel le kernel compile. On
#   charge cuda/12.2 (le CUB de CUDA 12 contient encore ces symboles).
#
#   torch ET torchvision doivent être le MÊME build (+computecanada). C'est
#   pourquoi on les installe ENSEMBLE en premier, en --no-index : sinon une install
#   ultérieure (timm) peut tirer un torchvision PyPI dépareillé -> erreur
#   `torchvision::nms does not exist`.
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
pip install --no-index torch==2.4.1 torchvision==0.19.1

# Reste des dépendances (wheelhouse). triton est importé par VMamba au chargement
# (même si notre forward_type ne l'utilise pas) -> obligatoire.
pip install --no-index numpy pillow scipy einops timm fvcore triton

# Kernel CUDA selective_scan du backbone VMamba (VMamba ne tourne PAS sans lui).
# En CUDA 12, les symboles CUB existent et compute_70 est supporté : compile tel quel.
echo "== compilation du kernel selective_scan (peut prendre plusieurs minutes) =="
pip install --no-build-isolation "$REPO/third_party/ChangeMamba/kernels/selective_scan"

# Kernel fusionné pour NOTRE C²S² (optionnel). S'il échoue, backend='ref' (plus lent).
pip install --no-index mamba_ssm causal_conv1d \
    || echo "== mamba_ssm indisponible : utiliser --backend ref =="

python - <<'PY'
import torch, torchvision, torchvision.ops  # noqa: F401
print("torch", torch.__version__, "| torchvision", torchvision.__version__)
PY
echo "venv prêt : $VENV"
echo "NB : tester le forward GPU dans un JOB (le kernel ne s'importe qu'avec un GPU)."
