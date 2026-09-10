#!/bin/bash
#SBATCH --job-name=csf-latence
#SBATCH --account=def-hervete
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/csf-latence-%j.out
#
# Le dossier logs/ doit exister avant la soumission (mkdir -p logs).
#
# Lot C1 : latence et mémoire crête à l'inférence. Toute la thèse d'efficience du
# projet repose sur des paramètres et des GMACs ; ni l'un ni l'autre ne dit
# combien de temps le modèle met vraiment. Les noyaux SSM ont un mauvais rapport
# GMACs -> temps, et grid_sample (DySample) est memory-bound.
#
# ⚠️ UN SEUL job à la fois sur le nœud, et rien d'autre en parallèle : une mesure
# de latence partage le GPU avec ce qui tourne à côté. Ne pas lancer pendant que
# des entraînements occupent l'allocation si on peut l'éviter.
#
#   sbatch scripts/benchmark_latency.sbatch                       # le point d'efficience
#   ENCODER=vmamba_tiny sbatch --export=ALL,ENCODER scripts/benchmark_latency.sbatch
#
# 2 h : la mesure prend quelques minutes, l'ouverture du venv sur Lustre domine.
set -euo pipefail

ENCODER="${ENCODER:-vmamba_mini}"
FUSION="${FUSION:-concat}"
DECODER_REFINE="${DECODER_REFINE:-dw}"
SIZE="${SIZE:-512}"
BATCHES="${BATCHES:-1,8}"
ITERS="${ITERS:-50}"
CHANGEMAMBA="${CHANGEMAMBA:-1}"   # 0 pour ne mesurer que le nôtre
# Surcharges de LEUR config. 'MODEL.VSSM.MLP_RATIO 0.0' retrouve la variante à
# 21,51 M de leur table publiée ; sans cela on chronomètre celle du dépôt actuel,
# à 37,13 M, ce qui gonfle notre avantage.
CM_OPTS="${CM_OPTS:-}"

echo "== démarrage $(date +%H:%M:%S) sur $(hostname)"
module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
cd "$HOME/csf-mamba"

echo "== config : enc=$ENCODER fusion=$FUSION dec=$DECODER_REFINE size=$SIZE batches=$BATCHES iters=$ITERS cm=$CHANGEMAMBA cm_opts=[${CM_OPTS:-aucune}] =="

python -u -m scripts.benchmark_latency \
    --encoder "$ENCODER" \
    --fusion "$FUSION" \
    --decoder-refine "$DECODER_REFINE" \
    --size "$SIZE" \
    --batch-sizes "$BATCHES" \
    --iters "$ITERS" \
    ${CHANGEMAMBA:+$([ "$CHANGEMAMBA" = 1 ] && echo --changemamba)} \
    ${CM_OPTS:+--cm-opts "$CM_OPTS"}

echo "== terminé $(date +%H:%M:%S)"
