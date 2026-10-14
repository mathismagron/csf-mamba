#!/bin/bash
#SBATCH --job-name=distilscd-qual
#SBATCH --account=def-hervete
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/distilscd-qual-%j.out
# Fig. 7 of the paper (qualitative maps): inference only, no training.
# Student without KD, DistilSCD (KD student) and the PerASCD ViT-G teacher on 24 SECOND test
# candidates and 12 Landsat-SCD fold-1 test candidates (selection rule in
# paper/scripts/qualitative_predictions.py). Expected run time: well under 1 h (unzip of the two
# archives + ~36 forward passes per model; the ViT-G teacher is loaded twice).
#
#   sbatch paper/scripts/qualitative.sbatch
#   # then, locally:
#   scp -r magron13@narval.alliancecan.ca:/scratch/magron13/csf-distill/qualitative logs/
#   python paper/scripts/plot_qualitative.py --in logs/qualitative
set -euo pipefail

REPO="${REPO:-$HOME/csf-mamba}"
OUT="${OUT:-$SCRATCH/csf-distill/qualitative}"
mkdir -p "$OUT"
cd "$REPO"
echo "== commit $(git rev-parse --short HEAD) | job $SLURM_JOB_ID | out $OUT"
echo "$(date '+%F %T') job=$SLURM_JOB_ID commit=$(git rev-parse --short HEAD)" >> "$OUT/slurm_jobs.txt"

module load python/3.11 cuda/12.2

# ---- data: staged from the original archives (same procedure as the training jobs)
t0=$(date +%s)
unzip -q "$SCRATCH/SECOND.zip" -x '*_COLORED/*' -d "$SLURM_TMPDIR"
SECOND_ROOT="$SLURM_TMPDIR/SECOND"
N_TEST=$(ls "$SECOND_ROOT/test/T1" | wc -l)
[ "$N_TEST" -eq 1694 ] || { echo "SECOND test incomplete: $N_TEST pairs (expected 1694)"; exit 1; }
LANDSAT_ROOT="$SLURM_TMPDIR/LandsatSCD512"
unzip -q "$SCRATCH/LandsatSCD512.zip" -d "$LANDSAT_ROOT"
echo "   staging $(( $(date +%s) - t0 )) s"

# ---- pass 1: students (venv of the student: VMamba / mamba_ssm kernels)
source "$SCRATCH/csf-venv-cu12/bin/activate"
python -u paper/scripts/qualitative_predictions.py --part student --out "$OUT" \
    --second-root "$SECOND_ROOT" --landsat-root "$LANDSAT_ROOT" \
    --landsat-ids splits/LandsatSCD_loc/fold1/test.txt
deactivate

# ---- pass 2: teacher (venv of the teacher: PerASCD legacy code + compiled MSDA operator)
source "$SCRATCH/perascd-venv/bin/activate"
python -u paper/scripts/qualitative_predictions.py --part teacher --out "$OUT" \
    --perascd-root "$REPO/third_party/PerASCD" --msda cuda
deactivate

ls -la "$OUT"/second "$OUT"/landsat_f1 | head -50
echo "== done $(date '+%F %T')"
