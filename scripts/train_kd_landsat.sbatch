#!/bin/bash
#SBATCH --job-name=csf-kd-lsat
#SBATCH --account=def-hervete
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/csf-kd-lsat-%j.out
#
# Distillation, deuxième jeu : Landsat-SCD dans la version prétraitée de PerASCD
# (LandsatSCD512, csf_mamba/datasets/landsat_perascd.py). Même élève, même recette
# que scripts/train_kd_second.sbatch — seuls changent les données et le protocole de
# sélection, parce que ce jeu a un vrai split de validation :
#
#   entraînement sur train (1431), best.pt choisi sur val (477), test (477) évalué à
#   chaque époque sans rôle dans la sélection -> metrics.csv (val) + metrics_test.csv.
#   Rapport : test à l'époque du meilleur val, et test à la dernière époque.
#
# Budget en PAS, pas en époques : SECOND 2968 paires / batch effectif 8 = 371 pas x 120
# époques = 44 520 pas ; ici 1431 paires -> 178 pas/époque -> 250 époques (44 500 pas).
#
# Prérequis (nœud de connexion) : bash scripts/download_landsat_perascd.sh, puis
#   python -m scripts.check_landsat_perascd --data-root $SCRATCH/LandsatSCD512 \
#       --out $SCRATCH/csf-distill/checks/landsat_perascd_check.json
#
#   # baseline sans KD, 5 graines
#   for s in 1 2 3 4 5; do SEED=$s TAG=base sbatch --export=ALL,SEED,TAG scripts/train_kd_landsat.sbatch; done
#   # KD (quand un cache professeur Landsat existe) :
#   SEED=1 LKD_CHG=8 LKD_SEM=1 KD_T_SEM=2 KD_MASK=all KD_TEACHER=perascd TAG=kdlogits \
#       sbatch --export=ALL,SEED,LKD_CHG,LKD_SEM,KD_T_SEM,KD_MASK,KD_TEACHER,TAG scripts/train_kd_landsat.sbatch
#
# ⛔ Le split d'origine (sans FOLD) FUIT : 12 emplacements, 333 images, toutes les images de
# test présentes dans l'entraînement (scripts/verify_landsat_leakage.py). Il ne sert plus que
# de référence. Les runs de l'étude utilisent FOLD=0..3 (splits/LandsatSCD_loc, disjoints par
# emplacement) ; sortie landsat_f<F>_<TAG>-s<SEED>, cache <teacher>_landsat_f<F>_train.
#   FOLD=0 SEED=1 EPOCHS=600 TAG=pilot sbatch --time=14:00:00 --export=ALL,FOLD,SEED,EPOCHS,TAG scripts/train_kd_landsat.sbatch
# Reprise automatique sur last.pt : si le walltime tombe, resoumettre la même ligne.
set -euo pipefail

SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-250}"
LKD_CHG="${LKD_CHG:-0}"
KD_T_CHG="${KD_T_CHG:-1}"
LKD_SEM="${LKD_SEM:-0}"
KD_T_SEM="${KD_T_SEM:-2}"
KD_MASK="${KD_MASK:-changed}"
KD_TEACHER="${KD_TEACHER:-perascd}"   # cache : $SCRATCH/csf-distill/cache/<teacher>_landsat[_f<F>]_train
FOLD="${FOLD:-}"                       # 0..3 : pli disjoint par emplacement ; vide = split d'origine (fuit)
TAG="${TAG:?donner un TAG explicite (base, kdlogits, ...)}"

REPO="$HOME/csf-mamba"
if [ -n "$FOLD" ]; then
    case "$FOLD" in 0|1|2|3) ;; *) echo "⛔ FOLD doit valoir 0..3"; exit 1 ;; esac
    FD="$REPO/splits/LandsatSCD_loc/fold$FOLD"
    [ -f "$FD/train.txt" ] || { echo "⛔ $FD absent"; exit 1; }
    OUT="$SCRATCH/csf-distill/runs/landsat_f${FOLD}_${TAG}-s${SEED}"
    IDS_ARGS=(--train-ids "$FD/train.txt" --val-ids "$FD/val.txt" --test-ids "$FD/test.txt")
    CACHE_NAME="${KD_TEACHER}_landsat_f${FOLD}_train"
else
    OUT="$SCRATCH/csf-distill/runs/landsat_${TAG}-s${SEED}"
    IDS_ARGS=()
    CACHE_NAME="${KD_TEACHER}_landsat_train"
fi
mkdir -p "$OUT"

RECIPE="weight=1 dice=0 lovasz=0 crop=512 batch=2 x accum=4 enc=vmamba_mini dec=dw fusion=concat cga=1 mcasf=1 up=dysample core=chess deep=0 sek=0 sc=0.1 fft=[0,1] lr=constant rot90=1 photo=0.2 tswap=0.5 ema=0.9998"
CONF="dataset=landsat_perascd $RECIPE seed=$SEED epochs=$EPOCHS select=val report=test"
if [ -n "$FOLD" ]; then CONF="$CONF fold=$FOLD split=location"; fi
KD_ARGS=()
if [ "$LKD_CHG" != "0" ] || [ "$LKD_SEM" != "0" ]; then
    CACHE_SRC="$SCRATCH/csf-distill/cache/$CACHE_NAME"
    [ -f "$CACHE_SRC/meta.json" ] || { echo "⛔ cache professeur absent : $CACHE_SRC"; exit 1; }
    CONF="$CONF kd_chg=$LKD_CHG t_chg=$KD_T_CHG kd_sem=$LKD_SEM t_sem=$KD_T_SEM mask=$KD_MASK teacher=$KD_TEACHER"
fi
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

# Staging depuis l'archive (un gros fichier, CRC vérifié par unzip), comme SECOND.zip.
ZIP="$SCRATCH/LandsatSCD512.zip"
[ -f "$ZIP" ] || { echo "⛔ $ZIP absent : bash scripts/download_landsat_perascd.sh (nœud de connexion)"; exit 1; }
t0=$(date +%s)
DATA_ROOT="$SLURM_TMPDIR/LandsatSCD512"
unzip -q "$ZIP" -d "$DATA_ROOT"
N=$(for s in train val test; do ls "$DATA_ROOT/$s/im1" | wc -l; done | tr '\n' ' ')
echo "   $(( $(date +%s) - t0 )) s ; train/val/test : $N"
[ "$N" = "1431 477 477 " ] || { echo "⛔ LandsatSCD512 incomplet (attendu 1431 477 477)"; exit 1; }

if [ -n "${CACHE_SRC:-}" ]; then
    echo "== staging du cache du professeur"
    t0=$(date +%s)
    cp -r "$CACHE_SRC" "$SLURM_TMPDIR/kdcache"
    echo "   $(( $(date +%s) - t0 )) s"
    KD_ARGS=(--kd-cache "$SLURM_TMPDIR/kdcache"
             --lambda-kd-change "$LKD_CHG" --kd-t-change "$KD_T_CHG"
             --lambda-kd-sem "$LKD_SEM" --kd-t-sem "$KD_T_SEM" --kd-sem-mask "$KD_MASK")
fi

python -m scripts.train \
    --data-root "$DATA_ROOT" --dataset landsat_perascd --val-split val --test-split test "${IDS_ARGS[@]}" \
    --encoder vmamba_mini --encoder-pretrained "$SCRATCH/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth" \
    --decoder-refine dw --core chess --fusion concat --upsample dysample --backend mamba \
    --crop-size 512 --batch-size 2 --accum-steps 4 \
    --bcd-change-weight 1 --lambda-dice 0 --lambda-lovasz 0 --lambda-sem-change 0 \
    --rot90 --photometric 0.2 --temporal-swap 0.5 --ema-decay 0.9998 \
    --sek-warmup-iters 0 --lambda-deep 0 --lambda-sek 0 --lambda-sc 0.1 --fft-stages 0,1 \
    --lr-schedule constant --epochs "$EPOCHS" --seed "$SEED" \
    "${KD_ARGS[@]}" \
    --resume auto --output "$OUT"
