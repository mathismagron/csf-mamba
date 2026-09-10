#!/bin/bash
# Resoumet un job d'entraînement interrompu, à configuration IDENTIQUE.
#
# Pourquoi un script plutôt qu'un copier-coller : la ligne `== config` du log
# porte 24 variables. En retaper une de travers produit un run qui rejoint son
# groupe sans rien signaler, et la moyenne du groupe ne veut alors plus rien
# dire — un run à `ema=0` glissé parmi trois à `ema=0.9998` ne lève aucune
# erreur, il fausse juste le résultat. Ce script relit la bannière et réémet
# exactement ce qu'elle contient.
#
#   scripts/resubmit.sh 2667720            # affiche la commande, ne lance rien
#   scripts/resubmit.sh 2667720 12:00:00   # walltime de la reprise
#   GO=1 scripts/resubmit.sh 2667720 12:00:00   # soumet pour de bon
#
# La reprise s'appuie sur `--resume auto` : le run repart de `last.pt` du même
# dossier de sortie, avec l'optimiseur, le scheduler, le compteur d'époques, le
# meilleur SeK et — depuis le 7 septembre — l'état de l'EMA.
set -euo pipefail

JOB="${1:?usage: scripts/resubmit.sh <jobid> [walltime]}"
TIME="${2:-12:00:00}"
LOG="logs/csf-second-${JOB}.out"

[ -f "$LOG" ] || { echo "⛔ log introuvable : $LOG"; exit 1; }

BANNER=$(grep -m1 '== config' "$LOG" || true)
[ -n "$BANNER" ] || { echo "⛔ aucune ligne '== config' dans $LOG"; exit 1; }

val() {
    # Ancré sur l'espace précédent : sans cela « enc= » attraperait aussi
    # « dec= », et « sek= » n'importe quel suffixe.
    printf '%s' "$BANNER" | grep -oE "(^|[[:space:]])$1=[^[:space:]]+" \
        | head -1 | sed -E "s/^[[:space:]]*$1=//"
}

OUT=$(printf '%s' "$BANNER" | sed -E 's/.*-> (.*) ==$/\1/')
TAG=$(basename "$OUT" | sed -E 's/^second_mini_chess_//')

export WEIGHT=$(val weight)   DICE=$(val dice)       LOVASZ=$(val lovasz)
export CROP=$(val crop)       BATCH=$(val batch)     ACCUM=$(val accum)
export ENCODER=$(val enc)     DECODER_REFINE=$(val dec)
export FUSION=$(val fusion)   CGA=$(val cga)         MCASF=$(val mcasf)
export UPSAMPLE=$(val up)     CORE=$(val core)       SEED=$(val seed)
export LAMBDA_DEEP=$(val deep) LAMBDA_SEK=$(val sek) LAMBDA_SC=$(val sc)
export LR_SCHEDULE=$(val lr)  EPOCHS=$(val epochs)
export ROT90=$(val rot90)     PHOTO=$(val photo)
export TSWAP=$(val tswap)     EMA=$(val ema)
export FFT_STAGES=$(printf '%s' "$BANNER" | grep -oE 'fft=\[[^]]*\]' | sed -E 's/fft=\[(.*)\]/\1/')
export TAG

# Aucune variable ne doit être vide : une valeur manquante retomberait
# silencieusement sur le défaut du sbatch, qui n'est pas forcément celui du run.
for v in WEIGHT DICE LOVASZ CROP BATCH ACCUM ENCODER DECODER_REFINE FUSION CGA \
         MCASF UPSAMPLE CORE SEED LAMBDA_DEEP LAMBDA_SEK LAMBDA_SC LR_SCHEDULE \
         EPOCHS ROT90 PHOTO TSWAP EMA TAG; do
    [ -n "${!v}" ] || { echo "⛔ $v n'a pas pu être relu dans la bannière"; exit 1; }
done

echo "bannière relue :"
echo "  $BANNER"
echo "variables réémises (à comparer ligne à ligne avec la bannière) :"
printf '  %-16s %s\n' \
    WEIGHT "$WEIGHT" DICE "$DICE" LOVASZ "$LOVASZ" \
    CROP "$CROP" BATCH "$BATCH" ACCUM "$ACCUM" \
    ENCODER "$ENCODER" DECODER_REFINE "$DECODER_REFINE" FUSION "$FUSION" \
    CGA "$CGA" MCASF "$MCASF" UPSAMPLE "$UPSAMPLE" CORE "$CORE" \
    LAMBDA_DEEP "$LAMBDA_DEEP" LAMBDA_SEK "$LAMBDA_SEK" LAMBDA_SC "$LAMBDA_SC" \
    FFT_STAGES "$FFT_STAGES" LR_SCHEDULE "$LR_SCHEDULE" \
    ROT90 "$ROT90" PHOTO "$PHOTO" TSWAP "$TSWAP" EMA "$EMA" \
    EPOCHS "$EPOCHS" SEED "$SEED" TAG "$TAG"
echo

DONE=0
[ -f "$OUT/metrics.csv" ] && DONE=$(( $(wc -l < "$OUT/metrics.csv") - 1 ))
echo "job $JOB  ->  $TAG"
echo "  époques déjà écrites : $DONE / $EPOCHS   (reprise à l'époque $DONE)"
[ -f "$OUT/last.pt" ] || echo "  ⚠️ pas de last.pt : le run repartirait de zéro"
if [ "$EMA" != "0" ]; then
    echo "  EMA active ($EMA) : l'état doit être dans last.pt, sinon le script d'entraînement refuse"
fi
echo "  walltime demandé : $TIME"

EXPORT=ALL,WEIGHT,DICE,LOVASZ,EPOCHS,CROP,BATCH,ACCUM,FUSION,CGA,MCASF,UPSAMPLE,CORE,LAMBDA_DEEP,LAMBDA_SEK,LAMBDA_SC,FFT_STAGES,LR_SCHEDULE,ENCODER,DECODER_REFINE,ROT90,PHOTO,TSWAP,EMA,SEED,TAG

if [ "${GO:-0}" = "1" ]; then
    sbatch --time="$TIME" --export="$EXPORT" scripts/train_second.sbatch
else
    echo
    echo "  (essai à blanc — relancer avec GO=1 pour soumettre)"
    echo "  sbatch --time=$TIME --export=$EXPORT scripts/train_second.sbatch"
fi
