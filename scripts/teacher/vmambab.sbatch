#!/bin/bash
#SBATCH --job-name=vmambab-teacher
#SBATCH --account=def-hervete
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=logs/vmambab-teacher-%j.out
#
# Écart de capacité : second professeur, VMamba-B + CG-Decoder publié par les auteurs
# de PerASCD (vmambaB_42e_mIoU74.01_Sek25.31_Fscd65.61_OA88.37.pth, ≈113 M).
#   1. évaluation sur SECOND test sous notre protocole (fp32/fp16/bf16, latence, FLOPs) ;
#   2. cache de ses logits sur les 2 968 paires d'entraînement, 8 vues D4, même format
#      que le cache PerASCD (calcul fp32, stockage fp16).
# Tourne dans le venv de l'ÉLÈVE : le VMamba du professeur a besoin des noyaux
# `selective_scan_cuda` (mamba_ssm) et de triton, absents du venv PerASCD.
#
# Prérequis, sur un nœud de CONNEXION (internet) :
#   D=$SCRATCH/csf-distill/teacher_vmambab && mkdir -p $D && cd $D
#   wget -c https://huggingface.co/SathShen/PerASCD-Checkpoint/resolve/main/vmambaB_260427223524_Lr1e1min0W01C15D03LsscTau001.zip
#   unzip -n vmambaB_260427223524_Lr1e1min0W01C15D03LsscTau001.zip -d ckpt && sha256sum *.zip ckpt/*.pth > SHA256SUMS
#
#   cd ~/csf-mamba && sbatch scripts/teacher/vmambab.sbatch
set -euo pipefail

REPO="$HOME/csf-mamba"
PERASCD="$REPO/third_party/PerASCD"
T="$SCRATCH/csf-distill/teacher_vmambab"
CKPT="$T/ckpt/vmambaB_42e_mIoU74.01_Sek25.31_Fscd65.61_OA88.37.pth"
CHECKS="$SCRATCH/csf-distill/checks"
CACHE="$SCRATCH/csf-distill/cache/vmambab_second_train"
mkdir -p "$CHECKS"
[ -f "$CKPT" ] || { echo "⛔ checkpoint absent : $CKPT (voir les prérequis en tête du script)"; exit 1; }
[ -e "$CACHE/meta.json" ] && { echo "⛔ $CACHE existe déjà : le déplacer avant de régénérer"; exit 1; }

module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"
cd "$REPO"
echo "== commit $(git rev-parse --short HEAD) ($(git status --porcelain | wc -l) fichiers modifiés) | PerASCD $(git -C "$PERASCD" rev-parse --short HEAD) | job $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'EOF'
import importlib
for m in ("torch", "timm", "scipy", "triton", "selective_scan_cuda"):
    try:
        mod = importlib.import_module(m); print(f"   {m:20s} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        print(f"   {m:20s} ABSENT ({e})")
        if m != "selective_scan_cuda":
            raise SystemExit(f"⛔ {m} manquant dans csf-venv-cu12")
EOF
# sans selective_scan_cuda, leur vmamba.py bascule sur un scan PyTorch pur : juste, mais lent.

# SECOND depuis l'archive d'origine (CRC vérifié), comme les runs d'entraînement.
t0=$(date +%s)
unzip -q "$SCRATCH/SECOND.zip" -x '*_COLORED/*' -d "$SLURM_TMPDIR"
DATA="$SLURM_TMPDIR/SECOND"
echo "== SECOND : $(( $(date +%s) - t0 )) s ; $(ls $DATA/train/T1 | wc -l) train, $(ls $DATA/test/T1 | wc -l) test"

echo "== 1. évaluation sur test"
python scripts/teacher/eval_perascd.py --perascd-root "$PERASCD" --checkpoint "$CKPT" --arch vmambaB \
    --data-root "$DATA" --precisions fp32,fp16,bf16 --latency --flops \
    --out "$CHECKS/step_0_vmambab_${SLURM_JOB_ID}.json"

# Garde-fou : pas de cache (donc pas de runs d'entraînement dépendants) si le
# professeur ne reproduit pas son score publié sous notre protocole.
python - "$CHECKS/step_0_vmambab_${SLURM_JOB_ID}.json" <<'EOF2'
import json, sys
r = json.load(open(sys.argv[1]))
print("   critères :", r.get("criteria"))
if not r.get("pass"):
    raise SystemExit("⛔ reproduction du professeur VMamba-B en échec : cache non généré")
EOF2

echo "== 2. cache des logits (train, 8 vues D4)"
python scripts/teacher/cache_teacher.py --perascd-root "$PERASCD" --checkpoint "$CKPT" --arch vmambaB \
    --data-root "$DATA" --split train --precision fp32 --batch-size 8 --out-dir "$CACHE"
(cd "$CACHE" && sha256sum d4_*.npy ids.txt meta.json > SHA256SUMS)
du -sh "$CACHE"
