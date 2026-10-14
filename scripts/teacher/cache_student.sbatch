#!/bin/bash
#SBATCH --job-name=csf-cache-student
#SBATCH --account=def-hervete
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/csf-cache-student-%j.out
#
# Cache du contrôle d'auto-distillation (révision, vague A) : l'élève DistilSCD SANS KD,
# graine 1, sert de professeur. Même format que le cache PerASCD (scripts/teacher/cache_student.py).
#
#   sbatch scripts/teacher/cache_student.sbatch                                   # SECOND
#   for f in 1 2 3; do FOLD=$f sbatch --export=ALL,FOLD scripts/teacher/cache_student.sbatch; done
#
# Professeurs (best.pt = poids EMA) :
#   SECOND  : second_mini_chess_crop512-lean-augswap-ema-s1 (meilleur SeK test, époque 100,
#             même convention de sélection que le checkpoint publié de PerASCD)
#   Landsat : landsat_f<F>_base-s1 (meilleur SeK de validation du pli)
# Les élèves distillés utilisent donc les graines 2, 3 (et 4 sur SECOND) : jamais la graine
# de leur professeur.
set -euo pipefail

FOLD="${FOLD:-}"
REPO="$HOME/csf-mamba"
module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"
cd "$REPO"
echo "== commit $(git rev-parse --short HEAD) | job $SLURM_JOB_ID"

if [ -z "$FOLD" ]; then
    CKPT="$SCRATCH/csf-mamba-runs/second_mini_chess_crop512-lean-augswap-ema-s1/best.pt"
    OUT="$SCRATCH/csf-distill/cache/student_second_train"
    unzip -q "$SCRATCH/SECOND.zip" -x '*_COLORED/*' -d "$SLURM_TMPDIR"
    N=$(ls "$SLURM_TMPDIR/SECOND/train/T1" | wc -l)
    [ "$N" -eq 2968 ] || { echo "⛔ SECOND incomplet ($N)"; exit 1; }
    DATA_ARGS=(--dataset second --data-root "$SLURM_TMPDIR/SECOND" --split train)
else
    case "$FOLD" in 1|2|3|0) ;; *) echo "⛔ FOLD 0..3"; exit 1 ;; esac
    CKPT="$SCRATCH/csf-distill/runs/landsat_f${FOLD}_base-s1/best.pt"
    OUT="$SCRATCH/csf-distill/cache/student_landsat_f${FOLD}_train"
    unzip -q "$SCRATCH/LandsatSCD512.zip" -d "$SLURM_TMPDIR/LandsatSCD512"
    DATA_ARGS=(--dataset landsat_perascd --data-root "$SLURM_TMPDIR/LandsatSCD512" --split train
               --ids-file "$REPO/splits/LandsatSCD_loc/fold$FOLD/train.txt")
fi
[ -f "$CKPT" ] || { echo "⛔ checkpoint absent : $CKPT"; exit 1; }
if [ -f "$OUT/meta.json" ]; then echo "⛔ $OUT existe déjà (meta.json) : le supprimer d'abord"; exit 1; fi
echo "== professeur élève : $CKPT -> $OUT"

# Écriture locale puis copie : un job interrompu ne laisse jamais de cache partiel.
TMP="$SLURM_TMPDIR/cache_out"
python scripts/teacher/cache_student.py --checkpoint "$CKPT" "${DATA_ARGS[@]}" \
    --out-dir "$TMP" --backend mamba --precision fp32 --batch-size 8 --workers 6
mkdir -p "$(dirname "$OUT")"
cp -r "$TMP" "$OUT.partial" && mv "$OUT.partial" "$OUT"
du -sh "$OUT"
