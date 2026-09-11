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
# --- Réglages propres aux blocs Transformer (lot du 11 septembre).
# Le premier lot les entraînait avec le réglage du reste du modèle ; c'est la
# réserve principale contre son résultat négatif. Voir documentation/hybride.md.
ATTN_LS="${ATTN_LS:-1e-5}"        # init du LayerScale (1e-1 = CaiT peu profond)
ATTN_DROP="${ATTN_DROP:-0.0}"     # dropout dans les blocs
ATTN_LRS="${ATTN_LRS:-1.0}"       # multiplicateur de LR pour l'attention seule
ATTN_NOWD="${ATTN_NOWD:-0}"       # 1 = pas de weight decay sur gammas/normes/biais
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-200}"
# Suffixe de tag pour les réglages non par défaut : sans lui, un second lot
# viserait le même dossier que le premier. Le garde-fou d'empreinte le refuserait
# — correctement — mais mieux vaut que le tag se distingue de lui-même.
#
# ⚠️ Écrit en `if` et non en `[ ... ] && ...` : sous `set -e`, un test faux en fin
# de ligne fait avorter le script entier.
OPT_SUFFIX=""
if [ "$ATTN_LS" != "1e-5" ]; then OPT_SUFFIX="${OPT_SUFFIX}-ls${ATTN_LS}"; fi
if [ "$ATTN_DROP" != "0.0" ]; then OPT_SUFFIX="${OPT_SUFFIX}-dr${ATTN_DROP}"; fi
if [ "$ATTN_LRS" != "1.0" ]; then OPT_SUFFIX="${OPT_SUFFIX}-lrs${ATTN_LRS}"; fi
if [ "$ATTN_NOWD" = "1" ]; then OPT_SUFFIX="${OPT_SUFFIX}-nowd"; fi

TAG="${TAG:-hyb-${BASE}-s${ATTN_STAGES//,/}-d${ATTN_DEPTH}${OPT_SUFFIX}-s${SEED}}"

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
CONF="base=$BASE enc=$ENCODER deep=$LAMBDA_DEEP attn_stages=[$ATTN_STAGES] depth=$ATTN_DEPTH heads=$ATTN_HEADS mlp=$ATTN_MLP ls=$ATTN_LS drop=$ATTN_DROP lrs=$ATTN_LRS nowd=$ATTN_NOWD seed=$SEED epochs=$EPOCHS"
echo "== config HYBRIDE : $CONF -> $OUT =="

# ⚠️ Empreinte de configuration — garde-fou contre la collision de dossiers.
# Si une variable ne se propage pas (préfixe oublié dans --export, faute de
# frappe), le tag retombe sur sa valeur par défaut et DEUX configurations
# différentes visent le même dossier. `--resume auto` y reprendrait alors les
# checkpoints de l'autre run, sans le moindre message. On grave la configuration
# au premier passage et on refuse de démarrer si elle a changé — une reprise
# légitime, elle, retrouve la même empreinte et passe.
mkdir -p "$OUT"
STAMP="$OUT/config.txt"
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" != "$CONF" ]; then
    echo "⛔ COLLISION : $OUT a déjà servi à une AUTRE configuration."
    echo "   gravée  : $(cat "$STAMP")"
    echo "   demandée: $CONF"
    echo "   Reprendre ici mélangerait deux entraînements. Changer de TAG."
    exit 1
fi
printf '%s\n' "$CONF" > "$STAMP"


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
    --attn-layer-scale "$ATTN_LS" \
    --attn-dropout "$ATTN_DROP" \
    --attn-lr-scale "$ATTN_LRS" \
    ${ATTN_NOWD:+$([ "$ATTN_NOWD" = 1 ] && echo --attn-no-wd)} \
    --output "$OUT"
