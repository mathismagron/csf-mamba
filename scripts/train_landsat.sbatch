#!/bin/bash
#SBATCH --job-name=csf-landsat
#SBATCH --account=def-hervete
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/csf-landsat-%j.out
#
# Troisième jeu de données. Le but n'est pas d'optimiser Landsat-SCD mais de
# savoir si l'énoncé d'efficience du projet, démontré sur SECOND, TRANSFÈRE.
# C'est la réserve n°1 du README, et la seule qui bloque une publication.
#
# ⚠️ LANCER D'ABORD LA VÉRIFICATION DU DUMP :
#     python -m scripts.check_landsat --data-root $SCRATCH/Landsat-SCD
#   Le dataloader a été écrit d'après le code de Mamba-FCS, pas d'après les
#   fichiers. Personne n'a encore regardé le dump.
#
#   BASE=eff  sbatch --export=ALL,BASE,SEED scripts/train_landsat.sbatch
#   BASE=lean sbatch --export=ALL,BASE,SEED scripts/train_landsat.sbatch
set -euo pipefail

BASE="${BASE:-eff}"
case "$BASE" in
  lean) ENCODER=vmamba_mini; LAMBDA_DEEP=0;   ROT90=0; PHOTO=0;   TSWAP=0;   EMA=0 ;;
  eff)  ENCODER=vmamba_mini; LAMBDA_DEEP=0;   ROT90=1; PHOTO=0.2; TSWAP=0.5; EMA=0.9998 ;;
  perf) ENCODER=vmamba_tiny; LAMBDA_DEEP=1.0; ROT90=1; PHOTO=0.2; TSWAP=0.5; EMA=0.9998 ;;
  *) echo "⛔ BASE doit valoir lean, eff ou perf — pas '$BASE'"; exit 1 ;;
esac

# ⚠️ 305 époques et non 200, et ce n'est pas un détail. Le résultat « LR constant
# sur 200 époques » a été établi sur SECOND, qui compte 2 968 paires
# d'entraînement : 371 pas d'optimiseur par époque, 74 200 au total. Landsat-SCD
# en compte ~1 948, soit 243 pas par époque. Transposer 200 ÉPOQUES donnerait
# 48 600 pas — 35 % de budget en moins, sans l'avoir décidé. On transpose donc le
# nombre de PAS, ce qui fait 305 époques.
EPOCHS="${EPOCHS:-305}"

# Tuiles de 416 : le crop couvre la tuile entière, comme 512 sur SECOND.
CROP="${CROP:-416}"
# 416² fait 66 % des pixels de 512² : on peut doubler le micro-batch et diviser
# l'accumulation, à batch effectif égal (8).
BATCH="${BATCH:-4}"
ACCUM="${ACCUM:-2}"

# ⚠️⚠️ LE RÉGLAGE LE PLUS RISQUÉ DE CE FICHIER.
# WEIGHT=1 et DICE=0 (pas de compensation de déséquilibre) est le réglage de
# SECOND, où la retirer a rapporté +0,032. Sur Hi-UCD, où seuls 1,4 % des pixels
# changent, elle est INDISPENSABLE — sans elle le modèle s'effondre à SeK 0,000.
# Le journal en a tiré une règle explicite : « un réglage anti-déséquilibre ne se
# recopie pas d'un jeu de données à l'autre ».
# Landsat-SCD annonce ~19 % de changement, proche des 20,1 % de SECOND, d'où ce
# défaut — mais `check_landsat` MESURE le taux réel et avertit s'il est faible.
# Si c'est le cas : WEIGHT=20 DICE=1.
WEIGHT="${WEIGHT:-1}"
DICE="${DICE:-0}"

SEED="${SEED:-1}"
TAG="${TAG:-landsat-${BASE}-s${SEED}}"

module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "== staging Landsat-SCD vers \$SLURM_TMPDIR =="
cp -r "$SCRATCH/Landsat-SCD" "$SLURM_TMPDIR/"
DATA_ROOT="$SLURM_TMPDIR/Landsat-SCD"

CKPT="$SCRATCH/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth"
OUT="$SCRATCH/csf-mamba-runs/landsat_$TAG"
CONF="base=$BASE enc=$ENCODER deep=$LAMBDA_DEEP crop=$CROP batch=$BATCH x accum=$ACCUM weight=$WEIGHT dice=$DICE rot90=$ROT90 photo=$PHOTO tswap=$TSWAP ema=$EMA seed=$SEED epochs=$EPOCHS"
echo "== config LANDSAT : $CONF -> $OUT =="

# Empreinte de configuration : même garde-fou que les autres lanceurs.
mkdir -p "$OUT"
STAMP="$OUT/config.txt"
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" != "$CONF" ]; then
    echo "⛔ COLLISION : $OUT a déjà servi à une AUTRE configuration."
    echo "   gravée  : $(cat "$STAMP")"
    echo "   demandée: $CONF"
    exit 1
fi
printf '%s\n' "$CONF" > "$STAMP"

python -u -m scripts.train \
    --data-root "$DATA_ROOT" --dataset landsat_scd --val-split test \
    --encoder "$ENCODER" --encoder-pretrained "$CKPT" \
    --core chess --fusion concat --upsample dysample --decoder-refine dw \
    --backend mamba \
    --crop-size "$CROP" --batch-size "$BATCH" --accum-steps "$ACCUM" \
    --bcd-change-weight "$WEIGHT" --lambda-dice "$DICE" --lambda-lovasz 0 \
    --lambda-sem-change 0 --sek-warmup-iters 0 \
    --lambda-deep "$LAMBDA_DEEP" --lambda-sek 0 --lambda-sc 0.1 --fft-stages 0,1 \
    --lr-schedule constant --epochs "$EPOCHS" --seed "$SEED" \
    ${ROT90:+$([ "$ROT90" = 1 ] && echo --rot90)} \
    --photometric "$PHOTO" --temporal-swap "$TSWAP" --ema-decay "$EMA" \
    --output "$OUT"
