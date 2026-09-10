#!/bin/bash
#SBATCH --job-name=csf-hybride
#SBATCH --account=def-hervete
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/csf-hybride-%j.out
#
# ⚗️ PISTE EXPÉRIMENTALE — HORS DU CADRE INITIAL DU STAGE.
#
# Lanceur SÉPARÉ, volontairement. Les runs sont tagués `hyb-*`, ils ne peuvent
# donc pas se glisser dans un groupe de la campagne de référence : les tags de
# celle-ci commencent tous par `crop512-`, et aggregate_seeds regroupe par nom.
#
# Ce qu'on teste : une attention bi-temporelle JOINTE aux stages profonds, en
# plus du modèle de référence. Conception et justification chiffrée :
# documentation/hybride.md
#
# La recette d'entraînement est figée ici (voir BASE ci-dessous) pour que la seule
# différence avec le run de référence correspondant soit l'attention. Un
# entraînement hybride et un entraînement de référence lancés à la même graine
# partagent d'ailleurs exactement la même initialisation pour tous les autres
# modules (l'attention est construite en dernier) : la comparaison est appariée
# jusque dans les poids de départ.
#
#   sbatch scripts/train_hybrid.sbatch                       # stage 4, profondeur 2
#   ATTN_STAGES=2,3 ATTN_DEPTH=1 SEED=2 TAG=hyb-s23-d1-s2 \
#     sbatch --export=ALL,ATTN_STAGES,ATTN_DEPTH,SEED,TAG scripts/train_hybrid.sbatch
#
# Coût de l'attention, calculé analytiquement (entrée 512²) :
#   stage 1 : 208,57 GMACs/bloc  — inabordable
#   stage 2 :  28,19             — inabordable
#   stage 3 :   5,64 GMACs, 1,18 M
#   stage 4 :   2,82 GMACs, 4,72 M   <- défaut
# Défaut (stage 4, profondeur 2) : 25,92 M et 37,06 GMACs, contre 16,48 et 31,42
# pour le point d'efficience — et toujours sous les 37,13 M / 115,44 GMACs de
# MambaSCD.
set -euo pipefail

# BASE choisit sur quelle recette l'attention se greffe. Ce n'est pas un détail :
#   eff  — point d'efficience (16,48 M, SeK 0,2387). Isole proprement l'effet de
#          l'attention, mais battre Mamba-FCS (0,2550) lui demanderait +0,0163,
#          soit davantage que le plus gros effet jamais établi dans ce projet
#          (+0,0125 pour le retrait de la loss SeK). Question scientifique.
#   perf — point de performance (32,58 M, SeK 0,2484), encodeur tiny et
#          supervision profonde. Il ne reste que +0,0066 à trouver pour dépasser
#          Mamba-FCS. Tentative de record.
BASE="${BASE:-eff}"
case "$BASE" in
  eff)  ENCODER=vmamba_mini; LAMBDA_DEEP=0   ;;
  perf) ENCODER=vmamba_tiny; LAMBDA_DEEP=1.0 ;;
  *) echo "⛔ BASE doit valoir 'eff' ou 'perf', pas '$BASE'"; exit 1 ;;
esac

ATTN_STAGES="${ATTN_STAGES:-3}"       # stages attentionnés (0..3), vide = référence
ATTN_DEPTH="${ATTN_DEPTH:-2}"         # blocs Transformer par stage
ATTN_HEADS="${ATTN_HEADS:-8}"
ATTN_MLP="${ATTN_MLP:-2.0}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-200}"
TAG="${TAG:-hyb-${BASE}-s${ATTN_STAGES//,/}-d${ATTN_DEPTH}-s${SEED}}"

case "$TAG" in
  hyb-*) ;;
  *) echo "⛔ le tag doit commencer par 'hyb-' pour rester séparé de la campagne"; exit 1 ;;
esac

module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "== staging SECOND vers \$SLURM_TMPDIR =="
cp -r "$SCRATCH/SECOND" "$SLURM_TMPDIR/"
DATA_ROOT="$SLURM_TMPDIR/SECOND"

CKPT="$SCRATCH/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth"
OUT="$SCRATCH/csf-mamba-runs/second_mini_chess_$TAG"
echo "== config HYBRIDE : base=$BASE enc=$ENCODER deep=$LAMBDA_DEEP attn_stages=[$ATTN_STAGES] depth=$ATTN_DEPTH heads=$ATTN_HEADS mlp=$ATTN_MLP seed=$SEED epochs=$EPOCHS -> $OUT =="

# Recette du point d'efficience, figée : seule l'attention distingue ce run.
python -u -m scripts.train \
    --data-root "$DATA_ROOT" --dataset second --val-split test \
    --encoder "$ENCODER" --encoder-pretrained "$CKPT" \
    --core chess --fusion concat --upsample dysample --decoder-refine dw \
    --backend mamba \
    --crop-size 512 --batch-size 2 --accum-steps 4 \
    --bcd-change-weight 1 --lambda-dice 0 --lambda-lovasz 0 \
    --lambda-sem-change 0 --sek-warmup-iters 0 \
    --lambda-deep "$LAMBDA_DEEP" --lambda-sek 0 --lambda-sc 0.1 --fft-stages 0,1 \
    --lr-schedule constant --epochs "$EPOCHS" --seed "$SEED" \
    --rot90 --photometric 0.2 --temporal-swap 0.5 --ema-decay 0.9998 \
    --attn-stages "$ATTN_STAGES" \
    --attn-depth "$ATTN_DEPTH" \
    --attn-heads "$ATTN_HEADS" \
    --attn-mlp-ratio "$ATTN_MLP" \
    --output "$OUT"
