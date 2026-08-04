#!/bin/bash
#SBATCH --job-name=csf-mamba
#SBATCH --account=def-hervete         # allocation Narval (Prof. Eric Hervet)
#SBATCH --gres=gpu:a100:1             # Narval : A100 (sm_80)
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=csf-mamba-%j.out
#
# Entraînement sur Hi-UCD (crops 256, batch 8, AMP bf16, LR cosine+warmup,
# warmup SeK). ~14 h pour 100 époques. Reprise auto : si le job atteint la limite
# de temps, resoumettre le MÊME script repart depuis last.pt.
#
# Prérequis (une fois, sur un nœud de connexion) :
#   bash scripts/setup_env.sh
#   bash scripts/download_pretrained.sh $SCRATCH/pretrained_weight
#   cd /scratch/<user>/hi-ucd && tar -cf $SCRATCH/hi-ucd.tar train val
#
# DÉFAUT = WEIGHT 5. Hi-UCD n'a que ~1,4 % de pixels changés (train), donc une
# compensation reste nécessaire (sans elle, le run 1 avait collapsé vers « aucun
# changement », SeK = 0). Mais WEIGHT=20 sur-détectait d'un facteur 3,6 (7,35 %
# prédits contre 2,05 % réels), ce qui dégradait la précision. 5 est un
# compromis à valider.
#
# Ablations (un seul facteur à la fois, dossier de sortie distinct) :
#   sbatch scripts/train.sbatch                                          # défaut
#   WEIGHT=20 TAG=w20 sbatch --export=ALL,WEIGHT,TAG scripts/train.sbatch
#   SEMCH=0 TAG=nosemch sbatch --export=ALL,SEMCH,TAG scripts/train.sbatch
#   OVERSAMPLE=3 TAG=ov3 sbatch --export=ALL,OVERSAMPLE,TAG scripts/train.sbatch
#   ENCODER=vmamba_tiny TAG=tiny sbatch --export=ALL,ENCODER,TAG scripts/train.sbatch
#
# Résolution : l'entraînement se fait sur des crops 256 mais la validation sur des
# tuiles 512 — pour un modèle SSM cela quadruple la longueur de séquence entre les
# deux régimes. Supprimer ce décalage a rapporté +0,026 de SeK sur SECOND :
#   CROP=512 BATCH=2 ACCUM=4 TAG=crop512 \
#     sbatch --time=24:00:00 --export=ALL,CROP,BATCH,ACCUM,TAG scripts/train.sbatch
# BATCH=2 x ACCUM=4 conserve le batch effectif de 8 : SEULE la résolution change
# (512² en batch 8 sature l'A100). Compter ~4x le temps.
#
# OVERSAMPLE : seules ~9,4 % des tuiles d'entraînement Hi-UCD contiennent du
# changement. Un facteur 3 porte la densité de signal à ~24 %, un facteur 10 à
# ~52 %. Contrepartie : les ~1 130 tuiles utiles sont revues 2,5 à 5 fois par
# époque (risque de sur-apprentissage) et les négatifs se raréfient (risque de
# faux positifs). Surveiller la courbe de SeK et le %% de changement prédit.
set -euo pipefail

WEIGHT="${WEIGHT:-5}"        # poids de la classe changement dans la CE
DICE="${DICE:-1.0}"          # poids de la loss Dice
SEMCH="${SEMCH:-1.0}"        # CE sémantique restreinte aux zones changées
LOVASZ="${LOVASZ:-0}"        # poids de la loss Lovász (optimise l'IoU)
OVERSAMPLE="${OVERSAMPLE:-1}" # poids des tuiles AVEC changement au tirage
EPOCHS="${EPOCHS:-100}"
CROP="${CROP:-256}"          # taille du crop d'entraînement (512 = tuile entière)
BATCH="${BATCH:-8}"          # micro-batch
ACCUM="${ACCUM:-1}"          # accumulation : batch effectif = BATCH x ACCUM
DECODER_REFINE="${DECODER_REFINE:-dw}"  # dw = référence, full = décodeur élargi
ENCODER="${ENCODER:-vmamba_mini}"  # vmamba_mini (13,8M) ou vmamba_tiny (28M)
TAG="${TAG:-w5}"             # suffixe du dossier de sortie

module load python/3.11 cuda/12.2
source "$SCRATCH/csf-venv-cu12/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Dataset stagé sur le stockage LOCAL du nœud (rapide vs 40k fichiers sur Lustre).
echo "== staging dataset vers \$SLURM_TMPDIR =="
mkdir -p "$SLURM_TMPDIR/hi-ucd"
tar -xf "$SCRATCH/hi-ucd.tar" -C "$SLURM_TMPDIR/hi-ucd"
DATA_ROOT="$SLURM_TMPDIR/hi-ucd"

CKPT="$SCRATCH/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth"
OUT="$SCRATCH/csf-mamba-runs/hiucd_mini_chess_$TAG"
echo "== config : weight=$WEIGHT dice=$DICE lovasz=$LOVASZ semch=$SEMCH oversample=$OVERSAMPLE enc=$ENCODER dec=$DECODER_REFINE crop=$CROP batch=$BATCH x accum=$ACCUM epochs=$EPOCHS -> $OUT =="

python -m scripts.train \
    --data-root "$DATA_ROOT" \
    --dataset hi_ucd \
    --encoder "$ENCODER" \
    --decoder-refine "$DECODER_REFINE" \
    --encoder-pretrained "$CKPT" \
    --core chess \
    --backend mamba \
    --crop-size "$CROP" \
    --batch-size "$BATCH" \
    --accum-steps "$ACCUM" \
    --bcd-change-weight "$WEIGHT" \
    --lambda-dice "$DICE" \
    --lambda-lovasz "$LOVASZ" \
    --oversample-change "$OVERSAMPLE" \
    --lambda-sem-change "$SEMCH" \
    --epochs "$EPOCHS" \
    --seed 42 \
    --resume auto \
    --output "$OUT"
