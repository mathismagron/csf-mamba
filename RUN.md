# Lancer csf-mamba sur tamIA

Cluster IA dédié (H100 80G). **Particularités qui dictent la marche à suivre :**
- Compte = allocation **AIP** : `--account=aip-<nom_du_prof>` (ton prof doit t'ajouter).
- Nœuds de calcul **sans internet** → tout téléchargement/compilation sur un
  **nœud de connexion**.
- Jobs = **nœuds entiers**.

**Layout retenu :**
| Quoi | Où | Pourquoi |
|---|---|---|
| Code (repo git) | `$HOME/csf-mamba` | 50 Go, sauvegardé |
| venv + kernel compilé | `$SCRATCH/csf-venv` | grand, persistant (compilé une fois) |
| Dataset | `$SCRATCH/hi-ucd.tar` | 20 To |
| Poids ImageNet | `$SCRATCH/pretrained_weight/` | — |

---

## Une fois pour toutes (sur un nœud de connexion)

```bash
# 1. Récupérer le code dans $HOME
cd $HOME
git clone <url-de-ton-repo> csf-mamba
cd csf-mamba

# 2. Cloner les dépôts de référence (VMamba) — a besoin d'internet
bash scripts/setup_third_party.sh

# 3. Créer le venv + compiler le kernel selective_scan (dans $SCRATCH)
bash scripts/setup_env.sh
#    -> vérifie à la fin : "selective_scan compilé : OK"

# 4. Télécharger les poids ImageNet du backbone (dans $SCRATCH)
bash scripts/download_pretrained.sh $SCRATCH/pretrained_weight

# 5. Dataset Hi-UCD dans $SCRATCH. Le zip officiel se dézippe en train/ val/ test/
#    (paire annotée 2018→2019 ; test/ sans masque). Créer l'archive ainsi :
#      cd <dossier Hi-UCD dézippé>          # là où sont train/ val/ test/
#      tar -cf $SCRATCH/hi-ucd.tar train val test
#    Transfert depuis ton PC (le zip fait ~33 Go) :
#      rsync -avP Hi-UCD.zip <user>@tamia:/scratch/<user>/   # puis dézipper là-bas
ls $SCRATCH/hi-ucd.tar
```

## Éditer `scripts/train.sbatch`

- Remplacer `aip-CHANGEME` par ton allocation (`aip-<nom_du_prof>`).
- Vérifier le nombre de GPU/nœud sur la page tamIA et ajuster `--gres` si besoin.

## Run de TEST d'abord (fortement conseillé)

Avant un run de plusieurs heures, valider que tout tourne sur GPU en ~minutes.
En interactif :

```bash
salloc --account=aip-<prof> --gres=gpu:h100:1 --cpus-per-task=12 --mem=64G --time=0:30:0
source $SCRATCH/csf-venv/bin/activate
tar -xf $SCRATCH/hi-ucd.tar -C $SLURM_TMPDIR
cd $HOME/csf-mamba
python -m scripts.train \
    --data-root $SLURM_TMPDIR/hi-ucd --dataset hi_ucd \
    --encoder vmamba_mini \
    --encoder-pretrained $SCRATCH/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth \
    --backend auto --epochs 1 --limit-batches 20 --output $SCRATCH/csf-test
```

**À vérifier :** le kernel se charge (pas d'erreur `selective_scan_cuda`), la loss
`total` descend d'un step à l'autre, une ligne `[val]` s'affiche à la fin. Si oui,
la plomberie GPU est bonne.

## Le vrai run

```bash
cd $HOME/csf-mamba
sbatch scripts/train.sbatch
squeue -u $USER              # suivre l'état (PD = en file, R = en cours)
tail -f logs/csf-mamba-*.out # suivre les logs
```

Sorties dans `$SCRATCH/csf-mamba-runs/<jobid>/` : checkpoints par époque +
`best.pt` (meilleur SeK sur validation).

## Ce qu'on regarde

- Pendant l'entraînement : la loss `total` doit **descendre**.
- Après chaque époque, la ligne `[val]` : **c'est le SeK qu'on cherche à maximiser**
  (+ Fscd, mIoU, OA). C'est le chiffre à comparer au SOTA.
