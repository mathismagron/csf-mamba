#!/bin/bash
# Setup de l'environnement sur tamIA — À LANCER SUR UN NŒUD DE CONNEXION.
#
# Pourquoi le nœud de connexion : les nœuds de CALCUL de tamIA n'ont PAS internet.
# Tout ce qui télécharge ou compile se fait donc ici, une fois. Le venv est
# persistant dans $SCRATCH (on ne le reconstruit pas à chaque job — indispensable
# car on compile un kernel CUDA qu'on ne veut pas recompiler à chaque fois).
#
#   Layout retenu :
#     repo   -> $HOME/csf-mamba        (code, sauvegardé)
#     venv   -> $SCRATCH/csf-venv      (grand, non sauvegardé)
#     data   -> $SCRATCH/hi-ucd(.tar)  (grand)
#     poids  -> $SCRATCH/pretrained_weight/
#
# Usage :  cd $HOME/csf-mamba && bash scripts/setup_env.sh
set -euo pipefail

REPO="${REPO:-$HOME/csf-mamba}"
VENV="${VENV:-$SCRATCH/csf-venv}"

# Adapter les versions aux modules réellement dispo sur tamIA (module avail).
module load python/3.11 cuda cudnn arrow

echo "== création du venv persistant : $VENV =="
virtualenv --no-download "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip

# torch/torchvision : IMPÉRATIVEMENT le build Alliance (wheelhouse) pour que la
# compilation CUDA du kernel corresponde au driver du cluster.
pip install --no-index torch torchvision numpy pillow scipy

# Dépendances du backbone VMamba. Depuis un nœud de connexion, internet marche ;
# on tente d'abord le wheelhouse, sinon PyPI.
for pkg in einops timm fvcore triton; do
    pip install --no-index "$pkg" || pip install "$pkg"
done

# Kernel CUDA selective_scan du backbone VMamba (VMamba ne tourne PAS sans lui).
# --no-build-isolation : compile avec le torch déjà installé.
echo "== compilation du kernel selective_scan (peut prendre plusieurs minutes) =="
pip install --no-build-isolation "$REPO/third_party/ChangeMamba/kernels/selective_scan"

# Kernel fusionné pour NOTRE C²S² (optionnel). S'il échoue, backend='ref' (plus
# lent mais fonctionnel sur GPU aussi).
pip install --no-index mamba_ssm causal_conv1d \
    || pip install mamba_ssm causal_conv1d \
    || echo "== mamba_ssm indisponible : utiliser --backend ref =="

python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA dispo à l'import:", torch.cuda.is_available())
try:
    import selective_scan_cuda_core  # noqa: F401
    print("selective_scan compilé : OK")
except Exception as e:
    print("selective_scan ABSENT :", e)
PY
echo "venv prêt : $VENV"
