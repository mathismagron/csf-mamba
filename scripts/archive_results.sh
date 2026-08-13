#!/bin/bash
# Archive tout ce qui doit survivre à la perte d'accès au cluster.
#
# Priorité au petit et à l'irremplaçable. Un `metrics.csv` pèse quelques kilo-
# octets et contient TOUS les résultats d'un run, époque par époque : c'est lui
# qui porte la valeur scientifique, pas le checkpoint. Les poids sont énormes et
# ne servent qu'à refaire des prédictions ; ils sont archivés à part.
#
#   scripts/archive_results.sh              # archive légère (métriques + logs + figures)
#   WITH_CKPT="crop512-best crop512-lean" scripts/archive_results.sh
#
# Puis, depuis le PORTABLE :
#   scp 'magron13@narval.alliancecan.ca:~/csf-archive-*.tar.gz' ~/
set -euo pipefail

RUNS="$SCRATCH/csf-mamba-runs"
DEST="$SLURM_TMPDIR/csf-archive"
[ -d "${SLURM_TMPDIR:-}" ] || DEST="$HOME/csf-archive-tmp"
rm -rf "$DEST"; mkdir -p "$DEST"

echo "== métriques de tous les runs (l'essentiel) =="
n=0
for d in "$RUNS"/*/; do
    nom=$(basename "$d")
    if [ -f "$d/metrics.csv" ]; then
        mkdir -p "$DEST/metrics/$nom"
        cp "$d/metrics.csv" "$DEST/metrics/$nom/"
        n=$((n+1))
    fi
done
echo "   $n fichiers metrics.csv"

echo "== sorties d'évaluation : matrices de confusion et panneaux =="
for d in "$RUNS"/*/eval_*/; do
    [ -d "$d" ] || continue
    rel=${d#$RUNS/}
    mkdir -p "$DEST/eval/$rel"
    cp -n "$d"/*.csv "$d"/*.png "$DEST/eval/$rel/" 2>/dev/null || true
done

echo "== logs Slurm =="
mkdir -p "$DEST/logs"
cp "$HOME/csf-mamba/logs/"*.out "$DEST/logs/" 2>/dev/null || true

echo "== tableaux d'agrégation, figés au moment de l'archivage =="
mkdir -p "$DEST/tableaux"
cd "$HOME/csf-mamba"
python -m scripts.aggregate_seeds --ref second_mini_chess_crop512 --min-epochs 95 \
    "$RUNS"/second_mini_chess_* > "$DEST/tableaux/global.txt" 2>&1 || true
python -m scripts.aggregate_seeds --ref second_mini_chess_crop512-nosek --min-epochs 95 \
    "$RUNS"/second_mini_chess_crop512-nosek* > "$DEST/tableaux/ablations.txt" 2>&1 || true

echo "== manifeste =="
{
    echo "Archive CSF-Mamba — $(date -Iseconds)"
    echo "Dépôt git : $(git -C "$HOME/csf-mamba" rev-parse HEAD)"
    echo
    echo "Runs archivés :"
    ls -1 "$DEST/metrics"
} > "$DEST/MANIFESTE.txt"

TAR="$HOME/csf-archive-$(date +%Y%m%d).tar.gz"
tar -czf "$TAR" -C "$(dirname "$DEST")" "$(basename "$DEST")"
echo; echo "== archive légère : $TAR ($(du -h "$TAR" | cut -f1)) =="

if [ -n "${WITH_CKPT:-}" ]; then
    TARC="$HOME/csf-checkpoints-$(date +%Y%m%d).tar"
    rm -f "$TARC"
    for motif in $WITH_CKPT; do
        for d in "$RUNS"/*"$motif"*/; do
            [ -f "$d/best.pt" ] || continue
            tar -rf "$TARC" -C "$RUNS" "$(basename "$d")/best.pt"
            echo "   + $(basename "$d")/best.pt"
        done
    done
    echo "== checkpoints : $TARC ($(du -h "$TARC" | cut -f1)) =="
fi
rm -rf "$DEST"
