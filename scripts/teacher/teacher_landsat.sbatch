#!/bin/bash
#SBATCH --job-name=lsat-teacher
#SBATCH --account=def-hervete
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/lsat-teacher-%j.out
#
# Professeur Landsat-SCD par pli disjoint par emplacement (splits/LandsatSCD_loc/fold<F>).
#   ARCH=vitg (PerASCD ViT-G/16 PerA, 548 M ; défaut) | vmambab (VMamba-B + CG-Decoder, repli)
#   MODE=feas  : 40 micro-itérations -> pic mémoire, s/itération, durée projetée (feasibility.json)
#   MODE=train : 50 époques, best.pt sur la SeK de val du pli, test enregistré ; reprise sur last.pt
#   MODE=cache : logits du best.pt sur le train du pli (8 vues D4, calcul fp32, stockage fp16)
#                -> $SCRATCH/csf-distill/cache/<perascd|vmambab>_landsat_f<F>_train
#                Garde-fou : GATE_SEK (SeK val du pilote élève, en fraction) obligatoire ;
#                pas de cache si le professeur ne la dépasse pas.
#
#   FOLD=0 MODE=feas PERA_WEIGHTS=$SCRATCH/csf-distill/pera/<fichier>.params \
#       sbatch --export=ALL,FOLD,MODE,PERA_WEIGHTS scripts/teacher/teacher_landsat.sbatch
#   FOLD=0 MODE=train PERA_WEIGHTS=... sbatch --time=20:00:00 --export=ALL,FOLD,MODE,PERA_WEIGHTS scripts/teacher/teacher_landsat.sbatch
#   FOLD=0 MODE=cache GATE_SEK=0.xxxx sbatch --export=ALL,FOLD,MODE,GATE_SEK scripts/teacher/teacher_landsat.sbatch
set -euo pipefail

FOLD="${FOLD:?FOLD=0..3}"
MODE="${MODE:?MODE=feas|train|cache}"
ARCH="${ARCH:-vitg}"
case "$FOLD" in 0|1|2|3) ;; *) echo "⛔ FOLD doit valoir 0..3"; exit 1 ;; esac
case "$ARCH" in
    vitg)    ARCH_PY="ViT-G/16/1024"; PRE="${PERA_WEIGHTS:-}"; TNAME=perascd ;;
    vmambab) ARCH_PY="vmambaB"; PRE="${VMB_WEIGHTS:-$SCRATCH/pretrained_weight/vssm_base_0229_ckpt_epoch_237.pth}"; TNAME=vmambab ;;
    *) echo "⛔ ARCH doit valoir vitg ou vmambab"; exit 1 ;;
esac

REPO="$HOME/csf-mamba"
PERASCD="$REPO/third_party/PerASCD"
FD="$REPO/splits/LandsatSCD_loc/fold$FOLD"
OUT="$SCRATCH/csf-distill/teachers/landsat_${ARCH}_f${FOLD}"
CACHE="$SCRATCH/csf-distill/cache/${TNAME}_landsat_f${FOLD}_train"
[ -f "$FD/train.txt" ] || { echo "⛔ $FD absent"; exit 1; }
[ -d "$PERASCD/models" ] || { echo "⛔ PerASCD legacy absent : $PERASCD"; exit 1; }
mkdir -p "$OUT"

module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
echo "== $MODE | $ARCH_PY | pli $FOLD | commit $(git rev-parse --short HEAD) ($(git status --porcelain | wc -l) modifiés) | PerASCD $(git -C "$PERASCD" rev-parse --short HEAD) | job $SLURM_JOB_ID"
echo "$(date '+%F %T') mode=$MODE job=$SLURM_JOB_ID commit=$(git rev-parse --short HEAD)" >> "$OUT/slurm_jobs.txt"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Opérateur déformable de ViT-Adapter : build partagé (scripts/teacher/build_msda_student.sbatch).
if [ "$ARCH" = vitg ]; then
    SITE="$SCRATCH/csf-distill/msda_site"
    export PYTHONPATH="$SITE${PYTHONPATH:+:$PYTHONPATH}"
    python -c "import torch, MultiScaleDeformableAttention as m; print('   MSDA :', m.__file__)" \
        || { echo "⛔ MSDA non importable depuis $SITE : sbatch scripts/teacher/build_msda_student.sbatch"; exit 1; }
fi

ZIP="$SCRATCH/LandsatSCD512.zip"
[ -f "$ZIP" ] || { echo "⛔ $ZIP absent"; exit 1; }
DATA="$SLURM_TMPDIR/LandsatSCD512"
t0=$(date +%s); unzip -q "$ZIP" -d "$DATA"
N=$(for s in train val test; do ls "$DATA/$s/im1" | wc -l; done | tr '\n' ' ')
echo "== LandsatSCD512 : $(( $(date +%s) - t0 )) s ; $N"
[ "$N" = "1431 477 477 " ] || { echo "⛔ archive incomplète"; exit 1; }

case "$MODE" in
  feas|train)
    [ -n "$PRE" ] && [ -f "$PRE" ] || { echo "⛔ poids pré-entraînés absents : '$PRE' (PERA_WEIGHTS / VMB_WEIGHTS)"; exit 1; }
    EXTRA=(); [ "$MODE" = feas ] && EXTRA=(--max-iters 40)
    python -u scripts/teacher/train_teacher_landsat.py --perascd-root "$PERASCD" --arch "$ARCH_PY" \
        --pretrained "$PRE" --data-root "$DATA" --fold-dir "$FD" --out "$OUT" --workers 10 "${EXTRA[@]}"
    ;;
  cache)
    GATE="${GATE_SEK:?GATE_SEK (SeK val du pilote élève, fraction) obligatoire}"
    [ -f "$OUT/best.pt" ] || { echo "⛔ $OUT/best.pt absent"; exit 1; }
    [ -e "$CACHE/meta.json" ] && { echo "⛔ $CACHE existe déjà : le déplacer avant de régénérer"; exit 1; }
    python - "$OUT/best.pt" "$GATE" <<'EOF'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sek, gate = float(ck["Sek"]), float(sys.argv[2])
print(f"   professeur : SeK val {sek:.4f} (époque {ck['epoch']}) ; test à la sélection {ck['test_at_selection']}")
print(f"   garde-fou : SeK val professeur {sek:.4f} > élève pilote {gate:.4f} ? {sek > gate}")
if not sek > gate:
    raise SystemExit("⛔ professeur pas meilleur que l'élève sur la val du pli : pas de cache")
EOF
    python scripts/teacher/cache_teacher.py --perascd-root "$PERASCD" --checkpoint "$OUT/best.pt" \
        --arch "$ARCH_PY" --dataset landsat_perascd --ids-file "$FD/train.txt" --data-root "$DATA" \
        --split train --precision fp32 --batch-size 8 --out-dir "$CACHE" --msda "$([ "$ARCH" = vitg ] && echo cuda || echo auto)"
    (cd "$CACHE" && sha256sum d4_*.npy ids.txt meta.json > SHA256SUMS)
    du -sh "$CACHE"
    ;;
  *) echo "⛔ MODE inconnu : $MODE"; exit 1 ;;
esac
