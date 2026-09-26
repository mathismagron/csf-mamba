#!/bin/bash
# Audit de Narval, EN LECTURE SEULE : rien n'est modifié, déplacé ni soumis.
# À lancer sur un nœud de CONNEXION :   bash ~/narval_audit.sh
# Écrit tout dans ~/narval_audit_<date>.txt (et l'affiche en même temps).
#
# Répond à trois questions du plan de distillation :
#   - les 7 graines du point d'efficience (augswap-ema) sont-elles encore là ?
#   - quelle marge GPU reste-t-il sur def-hervete ?
#   - comment le stockage est-il organisé (et où serait un élève « 3 étages ») ?
#
# Les .pt ne sont jamais lus (stat seulement) : leur date d'accès, qui compte
# pour la purge du scratch, n'est donc pas rafraîchie par cet audit.
set -u
OUT="$HOME/narval_audit_$(date +%Y%m%d_%H%M).txt"
exec > >(tee "$OUT") 2>&1
T() { echo; echo "===== $* ====="; }
RUNS="$SCRATCH/csf-mamba-runs"
PROJ="$HOME/projects/def-hervete"

T "0. Identité"
date; hostname
echo "USER=$USER  HOME=$HOME  SCRATCH=$SCRATCH"
ls -ld "$HOME"/projects/* "$HOME"/nearline/* 2>/dev/null

T "1. Quotas"
diskusage_report 2>/dev/null || echo "(diskusage_report indisponible)"
lfs quota -h -u "$USER" /scratch 2>/dev/null

T "2. Purge du scratch (fichiers annoncés à la suppression)"
if [ -e "/scratch/to_delete/$USER" ]; then
    echo "LISTE DE PURGE PRÉSENTE :"; ls -la "/scratch/to_delete/$USER"
    head -40 "/scratch/to_delete/$USER" 2>/dev/null
else
    echo "aucune liste /scratch/to_delete/$USER"
fi

T "3. Arborescence (niveau 1)"
for d in "$HOME" "$SCRATCH" "$PROJ" "$PROJ/$USER"; do
    [ -d "$d" ] || continue
    echo "--- $d"; ls -la --time-style=long-iso "$d" 2>/dev/null | head -60
done

T "3b. Volumes (niveau 1, 5 min max par entrée)"
for d in "$HOME" "$SCRATCH" "$PROJ/$USER"; do
    [ -d "$d" ] || continue
    echo "--- $d"
    for x in "$d"/* "$d"/.[!.]*; do
        [ -e "$x" ] || continue
        timeout 300 du -sh "$x" 2>/dev/null || echo "  >300s  $x (non mesuré)"
    done | sort -h | tail -25
done

T "4. Dépôts git (profondeur 3) : branches, stash, worktrees, encodeur"
find "$HOME" "$SCRATCH" "$PROJ/$USER" -maxdepth 3 -name .git 2>/dev/null | while read -r g; do
    r=$(dirname "$g"); echo "--- $r"
    git -C "$r" remote get-url origin 2>/dev/null
    git -C "$r" branch -a --no-color 2>/dev/null | head -10
    git -C "$r" log -3 --format='%h %ad %s' --date=short 2>/dev/null
    echo "modifs non commitées : $(git -C "$r" status --porcelain 2>/dev/null | wc -l)"
    git -C "$r" status --porcelain 2>/dev/null | head -15
    git -C "$r" stash list 2>/dev/null | head -5
    git -C "$r" worktree list 2>/dev/null
    grep -rn "out_indices" "$r"/csf_mamba 2>/dev/null | head -3
done

T "5. Tous les runs : époques écrites, nombre de .pt, taille"
if [ -d "$RUNS" ]; then
    echo "nombre de dossiers : $(ls -1 "$RUNS" | wc -l)"
    for d in "$RUNS"/*/; do
        n=$(basename "$d")
        e="-"; [ -f "$d/metrics.csv" ] && e=$(( $(wc -l < "$d/metrics.csv") - 1 ))
        c=$(ls "$d"/*.pt 2>/dev/null | wc -l)
        printf "%-60s ep=%-4s pt=%-2s %s\n" "$n" "$e" "$c" "$(du -sh "$d" 2>/dev/null | cut -f1)"
    done
else
    echo "ABSENT : $RUNS"
fi

T "6. Point d'efficience (*augswap-ema*) : détail"
for d in "$RUNS"/*augswap-ema*/; do
    [ -d "$d" ] || { echo "aucun dossier *augswap-ema*"; break; }
    echo "--- $d"
    cat "$d/config.txt" 2>/dev/null
    echo "fichiers (taille, date de modif, date d'accès) :"
    find "$d" -maxdepth 1 -type f -printf '  %10s  mod %TY-%Tm-%Td  acc %AY-%Am-%Ad  %f\n' 2>/dev/null
    if [ -f "$d/metrics.csv" ]; then
        head -1 "$d/metrics.csv"; tail -2 "$d/metrics.csv"
    fi
done

T "7. Données, poids, archives"
ls -la "$SCRATCH/SECOND" 2>/dev/null | head
for s in train test; do
    [ -d "$SCRATCH/SECOND/$s" ] && echo "$s : $(ls "$SCRATCH/SECOND/$s"/* -d 2>/dev/null | xargs -n1 basename | tr '\n' ' ') | T1 = $(ls "$SCRATCH/SECOND/$s/T1" 2>/dev/null | wc -l) fichiers"
done
ls -la "$SCRATCH/pretrained_weight" 2>/dev/null
ls -la "$HOME"/csf-archive-*.tar.gz 2>/dev/null

T "7b. Environnement de l'élève"
ls -la "$SCRATCH/csf-venv-cu12/bin/python" 2>/dev/null || echo "venv absent"
if [ -f "$SCRATCH/csf-venv-cu12/bin/activate" ]; then
    ( module load python/3.11 cuda/12.2 >/dev/null 2>&1
      source "$SCRATCH/csf-venv-cu12/bin/activate"
      pip list 2>/dev/null | grep -iE '^(torch|torchvision|mamba[-_]ssm|causal[-_]conv1d|selective[-_]scan[a-z_-]*|timm|triton|transformers|numpy)[[:space:]]' )
fi

T "8. Allocation et priorité (fairshare)"
sacctmgr -nP show assoc user="$USER" format=account,partition,qos 2>/dev/null
echo "--- sshare GPU"
sshare -l -A def-hervete_gpu -u "$USER" 2>/dev/null || sshare -l -u "$USER"
echo "--- sshare CPU"
sshare -l -A def-hervete_cpu -u "$USER" 2>/dev/null | tail -3

T "8b. GPU-heures consommées depuis le 1er juillet 2026, par mois"
sacct -X -u "$USER" -S 2026-07-01 -n -P --format=JobID,Account,Start,Elapsed,AllocTRES,State 2>/dev/null |
awk -F'|' '
  $2 ~ /hervete/ {
    t=$4; d=0; if (t ~ /-/) { split(t,a,"-"); d=a[1]; t=a[2] }
    n=split(t,b,":"); s=(n==3)? b[1]*3600+b[2]*60+b[3] : b[1]*60+b[2]; s+=d*86400
    g=0; if (match($5,/gres\/gpu[^=,]*=[0-9]+/)) { x=substr($5,RSTART,RLENGTH); sub(/.*=/,"",x); g=x }
    if (g>0) { m=substr($3,1,7); h[m]+=g*s/3600; nj[m]++; tot+=g*s/3600 }
  }
  END { for (m in h) printf "  %s  %4d jobs GPU  %7.1f GPU-h\n", m, nj[m], h[m]; printf "  TOTAL %7.1f GPU-h\n", tot }' | sort

T "9. Limites des partitions GPU et file actuelle"
sinfo -o "%P %l %G" 2>/dev/null | grep -i gpu | sort -u | head -15
squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %.12l %R" 2>/dev/null

echo; echo "Rapport écrit dans : $OUT"
