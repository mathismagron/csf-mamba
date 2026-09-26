#!/bin/bash
#SBATCH --job-name=csf-kd
#SBATCH --account=def-hervete
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/csf-kd-%j.out
#
# Étapes 1 à 3 du plan de distillation (documentation/distillation.md) : le point
# d'efficience, à l'identique de ses 7 graines de référence, avec ou sans KD.
#
# MODE=screen  (défaut) entraînement sur splits/SECOND/train.txt (2 671), sélection
#              sur splits/SECOND/val.txt (297) — le test n'est jamais regardé.
# MODE=confirm entraînement sur les 2 968 paires, validation sur le test (convention
#              du projet), pour les 5 graines de l'étape 5.
#
#   # témoin sans KD (étape 1, criblage)
#   SEED=1 TAG=witness sbatch --export=ALL,SEED,TAG scripts/train_kd_second.sbatch
#   # KD du changement (étape 2)
#   SEED=1 LKD_CHG=1 TAG=kdchg-l1 sbatch --export=ALL,SEED,LKD_CHG,TAG scripts/train_kd_second.sbatch
#   # + KD sémantique (étape 3)
#   SEED=1 LKD_CHG=1 LKD_SEM=1 KD_T_SEM=2 KD_MASK=all TAG=kdsem-t2-all \
#       sbatch --export=ALL,SEED,LKD_CHG,LKD_SEM,KD_T_SEM,KD_MASK,TAG scripts/train_kd_second.sbatch
#
# Durée : 120 époques ≈ 8–10 h (à confirmer par sacct sur les runs de septembre).
# Reprise automatique sur last.pt : si le walltime tombe, resoumettre la même ligne.
set -euo pipefail

MODE="${MODE:-screen}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-120}"
LKD_CHG="${LKD_CHG:-0}"        # λ KD changement
KD_T_CHG="${KD_T_CHG:-1}"      # température KD changement
LKD_SEM="${LKD_SEM:-0}"        # λ KD sémantique
KD_T_SEM="${KD_T_SEM:-2}"      # température KD sémantique
KD_MASK="${KD_MASK:-changed}"  # changed | all
TAG="${TAG:?donner un TAG explicite (witness, kdchg-l1, ...)}"

REPO="$HOME/csf-mamba"
CACHE_SRC="$SCRATCH/csf-distill/cache/perascd_second_train"
OUT="$SCRATCH/csf-distill/runs/${MODE}_${TAG}-s${SEED}"
mkdir -p "$OUT"

# Recette d'efficience, figée : ligne `== config` des 7 graines de référence.
RECIPE="weight=1 dice=0 lovasz=0 crop=512 batch=2 x accum=4 enc=vmamba_mini dec=dw fusion=concat cga=1 mcasf=1 up=dysample core=chess deep=0 sek=0 sc=0.1 fft=[0,1] lr=constant rot90=1 photo=0.2 tswap=0.5 ema=0.9998"
CONF="$RECIPE seed=$SEED epochs=$EPOCHS mode=$MODE kd_chg=$LKD_CHG t_chg=$KD_T_CHG kd_sem=$LKD_SEM t_sem=$KD_T_SEM mask=$KD_MASK"
echo "== config : $CONF -> $OUT =="
STAMP="$OUT/config.txt"
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" != "$CONF" ]; then
    echo "⛔ COLLISION : $OUT a déjà servi à une AUTRE configuration."
    echo "   gravée  : $(cat "$STAMP")"; echo "   demandée: $CONF"; exit 1
fi
printf '%s\n' "$CONF" > "$STAMP"

module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
echo "== commit $(git rev-parse --short HEAD) ($(git status --porcelain | wc -l) fichiers modifiés) | job $SLURM_JOB_ID"
echo "$(date '+%F %T') job=$SLURM_JOB_ID commit=$(git rev-parse --short HEAD)" >> "$OUT/slurm_jobs.txt"

# Staging par UNE archive : `cp -r` de ~33 000 petits fichiers depuis /scratch a
# bloqué un job entier le 26 septembre (job 4030416, 1 h sans finir). Archive à
# créer une fois, sur un nœud de connexion :
#   tar -cf $SCRATCH/csf-distill/SECOND_stage.tar -C $SCRATCH --exclude='*_COLORED' SECOND
# Sans archive, lecture directe dans /scratch (même résultat, plus lent).
STAGE_TAR="$SCRATCH/csf-distill/SECOND_stage.tar"
t0=$(date +%s)
if [ -f "$STAGE_TAR" ]; then
    echo "== staging SECOND : extraction de $STAGE_TAR"
    tar -xf "$STAGE_TAR" -C "$SLURM_TMPDIR"
    DATA_ROOT="$SLURM_TMPDIR/SECOND"
else
    echo "⚠️  pas d'archive $STAGE_TAR : lecture directe dans \$SCRATCH/SECOND"
    DATA_ROOT="$SCRATCH/SECOND"
fi
echo "   $(( $(date +%s) - t0 )) s ; $(ls "$DATA_ROOT/train/T1" | wc -l) paires train, $(ls "$DATA_ROOT/test/T1" | wc -l) test"

KD_ARGS=()
if [ "$LKD_CHG" != "0" ] || [ "$LKD_SEM" != "0" ]; then
    echo "== staging du cache du professeur (≈12 Go, 10 gros fichiers)"
    t0=$(date +%s)
    cp -r "$CACHE_SRC" "$SLURM_TMPDIR/kdcache"
    echo "   $(( $(date +%s) - t0 )) s"
    KD_ARGS=(--kd-cache "$SLURM_TMPDIR/kdcache"
             --lambda-kd-change "$LKD_CHG" --kd-t-change "$KD_T_CHG"
             --lambda-kd-sem "$LKD_SEM" --kd-t-sem "$KD_T_SEM" --kd-sem-mask "$KD_MASK")
fi

case "$MODE" in
    screen)  SPLIT_ARGS=(--train-ids splits/SECOND/train.txt --val-ids splits/SECOND/val.txt) ;;
    confirm) SPLIT_ARGS=(--val-split test) ;;
    *) echo "MODE inconnu : $MODE"; exit 1 ;;
esac

python -m scripts.train \
    --data-root "$DATA_ROOT" --dataset second "${SPLIT_ARGS[@]}" \
    --encoder vmamba_mini --encoder-pretrained "$SCRATCH/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth" \
    --decoder-refine dw --core chess --fusion concat --upsample dysample --backend mamba \
    --crop-size 512 --batch-size 2 --accum-steps 4 \
    --bcd-change-weight 1 --lambda-dice 0 --lambda-lovasz 0 --lambda-sem-change 0 \
    --rot90 --photometric 0.2 --temporal-swap 0.5 --ema-decay 0.9998 \
    --sek-warmup-iters 0 --lambda-deep 0 --lambda-sek 0 --lambda-sc 0.1 --fft-stages 0,1 \
    --lr-schedule constant --epochs "$EPOCHS" --seed "$SEED" \
    "${KD_ARGS[@]}" \
    --resume auto --output "$OUT"
