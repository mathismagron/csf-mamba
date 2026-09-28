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
#   # contrôle : même terme, cible = vérité lissée (gt_smooth) ou brute (gt)
#   SEED=1 LKD_CHG=2 KD_TGT=gt_smooth TAG=ctl-gtsmooth-l2 sbatch --export=ALL,SEED,LKD_CHG,KD_TGT,TAG scripts/train_kd_second.sbatch
#   # + KD sémantique (étape 3)
#   SEED=1 LKD_CHG=1 LKD_SEM=1 KD_T_SEM=2 KD_MASK=all TAG=kdsem-t2-all \
#       sbatch --export=ALL,SEED,LKD_CHG,LKD_SEM,KD_T_SEM,KD_MASK,TAG scripts/train_kd_second.sbatch
#   # + KD de features, professeur en ligne (étape 4) : ≈11 h en criblage (330 s/époque,
#   # job 4176343) -> walltime allongé ; compiler d'abord MSDA une fois :
#   #   sbatch scripts/teacher/build_msda_student.sbatch
#   SEED=1 LKD_CHG=8 LKD_SEM=1 KD_T_SEM=2 KD_MASK=all LKD_FEAT=1 KD_FEAT_STAGES=1,2 TAG=kdfeat-s12 \
#       sbatch --time=16:00:00 --export=ALL,SEED,LKD_CHG,LKD_SEM,KD_T_SEM,KD_MASK,LKD_FEAT,KD_FEAT_STAGES,TAG \
#       scripts/train_kd_second.sbatch
#
# Durée mesurée (vague 1, 27 sept.) : 120 époques en MODE=screen = 4 h 44 – 5 h 07
# (≈135–143 s d'entraînement + 8 s de val par époque ; cache KD sans surcoût mesurable).
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
KD_TGT="${KD_TGT:-teacher}"    # cible du terme de changement : teacher | gt | gt_smooth (contrôles)
LKD_FEAT="${LKD_FEAT:-0}"      # λ KD de features (professeur en ligne)
KD_FEAT_STAGES="${KD_FEAT_STAGES:-1,2}"  # étages 0..3 = 1/4, 1/8, 1/16, 1/32
TAG="${TAG:?donner un TAG explicite (witness, kdchg-l1, ...)}"

REPO="$HOME/csf-mamba"
CACHE_SRC="$SCRATCH/csf-distill/cache/perascd_second_train"
OUT="$SCRATCH/csf-distill/runs/${MODE}_${TAG}-s${SEED}"
mkdir -p "$OUT"

# Recette d'efficience, figée : ligne `== config` des 7 graines de référence.
RECIPE="weight=1 dice=0 lovasz=0 crop=512 batch=2 x accum=4 enc=vmamba_mini dec=dw fusion=concat cga=1 mcasf=1 up=dysample core=chess deep=0 sek=0 sc=0.1 fft=[0,1] lr=constant rot90=1 photo=0.2 tswap=0.5 ema=0.9998"
CONF="$RECIPE seed=$SEED epochs=$EPOCHS mode=$MODE kd_chg=$LKD_CHG t_chg=$KD_T_CHG kd_sem=$LKD_SEM t_sem=$KD_T_SEM mask=$KD_MASK"
# Ajouté seulement hors défaut : les config.txt de la vague 1 restent valides.
if [ "$KD_TGT" != "teacher" ]; then
    [ "$LKD_CHG" != "0" ] || { echo "⛔ KD_TGT=$KD_TGT sans LKD_CHG"; exit 1; }
    CONF="$CONF tgt=$KD_TGT"
fi
if [ "$LKD_FEAT" != "0" ]; then
    CONF="$CONF kd_feat=$LKD_FEAT feat_stages=$KD_FEAT_STAGES"
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

# Staging depuis SECOND.zip, l'archive d'origine (Zenodo 15479555, 3,84 Go) dont
# $SCRATCH/SECOND a été extrait. Deux incidents du 26 septembre motivent ce choix :
# `cp -r` des ~33 000 petits fichiers a bloqué un job 1 h (4030416), et une archive
# tar recréée à partir d'eux a été retrouvée tronquée à 16 Mo (4039676). Un seul gros
# fichier se lit bien même quand /scratch est lent, et `unzip` vérifie le CRC de
# chaque fichier : une corruption fait échouer le job au lieu de passer en silence.
SECOND_ZIP="$SCRATCH/SECOND.zip"
t0=$(date +%s)
if [ -f "$SECOND_ZIP" ]; then
    echo "== staging SECOND : unzip de $SECOND_ZIP ($(stat -c '%s octets, modifié %y' "$SECOND_ZIP"))"
    unzip -q "$SECOND_ZIP" -x '*_COLORED/*' -d "$SLURM_TMPDIR"
    DATA_ROOT="$SLURM_TMPDIR/SECOND"
else
    echo "⚠️  pas de $SECOND_ZIP : lecture directe dans \$SCRATCH/SECOND"
    DATA_ROOT="$SCRATCH/SECOND"
fi
N_TRAIN=$(ls "$DATA_ROOT/train/T1" 2>/dev/null | wc -l)
N_TEST=$(ls "$DATA_ROOT/test/T1" 2>/dev/null | wc -l)
echo "   $(( $(date +%s) - t0 )) s ; $N_TRAIN paires train, $N_TEST test"
# Garde-fou : une extraction incomplète ne doit jamais atteindre l'entraînement.
if [ "$N_TRAIN" -ne 2968 ] || [ "$N_TEST" -ne 1694 ]; then
    echo "⛔ SECOND incomplet dans $DATA_ROOT (attendu 2968 / 1694). Contenu :"
    ls -la "$SLURM_TMPDIR" "$DATA_ROOT" 2>&1 | head -20
    exit 1
fi

KD_ARGS=()
if [ "$LKD_CHG" != "0" ] || [ "$LKD_SEM" != "0" ]; then
    echo "== staging du cache du professeur (≈12 Go, 10 gros fichiers)"
    t0=$(date +%s)
    cp -r "$CACHE_SRC" "$SLURM_TMPDIR/kdcache"
    echo "   $(( $(date +%s) - t0 )) s"
    KD_ARGS=(--kd-cache "$SLURM_TMPDIR/kdcache"
             --lambda-kd-change "$LKD_CHG" --kd-t-change "$KD_T_CHG"
             --lambda-kd-sem "$LKD_SEM" --kd-t-sem "$KD_T_SEM" --kd-sem-mask "$KD_MASK"
             --kd-change-target "$KD_TGT")
fi

FEAT_ARGS=()
if [ "$LKD_FEAT" != "0" ]; then
    PERASCD="$REPO/third_party/PerASCD"
    TCKPT="$SCRATCH/csf-distill/teacher/ckpt/PerAChain_40e_mIoU74.33_Sek26.11_Fscd66.41_OA88.70.pth"
    [ -d "$PERASCD/models" ] && [ -f "$TCKPT" ] || { echo "⛔ professeur absent : $PERASCD / $TCKPT"; exit 1; }
    # Opérateur MSDA : build partagé de scripts/teacher/build_msda_student.sbatch
    # s'il s'importe, sinon compilé dans le job, dans un dossier local (≈12 min) —
    # le venv partagé de l'élève n'est jamais modifié.
    echo "== opérateur MultiScaleDeformableAttention (venv élève)"
    t0=$(date +%s)
    SITE="$SCRATCH/csf-distill/msda_site"
    if PYTHONPATH="$SITE" python -c "import torch, MultiScaleDeformableAttention" 2>/dev/null; then
        echo "   build partagé : $(head -1 "$SITE/BUILD_INFO.txt" 2>/dev/null)"
    else
        echo "   pas de build partagé utilisable : compilation dans le job"
        BUILD="$SLURM_TMPDIR/msda_src"; SITE="$SLURM_TMPDIR/msda_site"
        cp -r "$PERASCD/models/pera_layers/vit_adapter_layers/ops" "$BUILD"
        rm -rf "$BUILD/build" "$BUILD/dist" "$BUILD"/*.egg-info
        ( cd "$BUILD" && TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=12 \
            pip install -q --no-index --no-build-isolation --no-deps --target "$SITE" . )
    fi
    export PYTHONPATH="$SITE${PYTHONPATH:+:$PYTHONPATH}"
    python -c "import torch, MultiScaleDeformableAttention as m; print('   compilé :', m.__file__)"
    echo "   $(( $(date +%s) - t0 )) s ; PerASCD $(git -C "$PERASCD" rev-parse --short HEAD)"
    FEAT_ARGS=(--lambda-kd-feat "$LKD_FEAT" --kd-feat-stages "$KD_FEAT_STAGES"
               --teacher-root "$PERASCD" --teacher-ckpt "$TCKPT" --teacher-msda cuda)
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
    "${KD_ARGS[@]}" "${FEAT_ARGS[@]}" \
    --resume auto --output "$OUT"
