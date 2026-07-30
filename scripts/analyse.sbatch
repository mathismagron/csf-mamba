#!/bin/bash
#SBATCH --job-name=csf-analyse
#SBATCH --account=def-hervete
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:15:0
#SBATCH --output=csf-analyse-%j.out
#
# Analyses légères (CPU) sur les sorties d'évaluation déjà produites. À lancer en
# job : sur un nœud de connexion, le simple démarrage de Python depuis le venv
# (sur Lustre) prend déjà plusieurs minutes.
#
# Par défaut : décomposition localisation/sémantique de TOUTES les matrices de
# confusion trouvées.
#
#   sbatch scripts/analyse.sbatch
# Astuce : pour un démarrage plus rapide, ajouter la QOS sur la ligne de
# commande — sbatch --qos=cc-debug ... — plutôt que de la coder en dur
# (elle est limitée à un job simultané et parfois indisponible).
set -euo pipefail

module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"
cd "$HOME/csf-mamba"

shopt -s nullglob
FILES=("$SCRATCH"/csf-mamba-runs/*/eval_*/confusion.csv)
if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "Aucune matrice de confusion trouvée — lancer d'abord scripts/evaluate.sbatch"
    exit 0
fi
python -m scripts.confusion_report "${FILES[@]}"
