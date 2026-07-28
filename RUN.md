# Lancer csf-mamba sur Narval (Alliance Canada)

Cluster GPU **A100**. Allocation classique **`def-hervete`** (Prof. Eric Hervet).

**Layout retenu :**
| Quoi | Où |
|---|---|
| Code (repo git) | `$HOME/csf-mamba` |
| venv + kernels compilés | `$SCRATCH/csf-venv-cu12` (persistant) |
| Dataset | `/scratch/<user>/hi-ucd/` + `$SCRATCH/hi-ucd.tar` |
| Poids ImageNet | `$SCRATCH/pretrained_weight/` |

---

## Installation (une fois, sur un nœud de CONNEXION — internet requis)

```bash
cd $HOME && git clone <url-du-repo> csf-mamba && cd csf-mamba
bash scripts/setup_third_party.sh                       # VMamba (ChangeMamba)
bash scripts/setup_env.sh                               # venv cu12 + kernels
bash scripts/download_pretrained.sh $SCRATCH/pretrained_weight
```

`setup_env.sh` installe la stack **torch 2.5.1 / CUDA 12** (le wheelhouse par défaut
est en CUDA 13, incompatible avec le kernel `selective_scan`), compile
`selective_scan`, installe `mamba_ssm` et corrige son `__init__.py` cassé. Tout est
automatique — voir les commentaires du script pour le détail des pièges.

### Dataset

Le zip officiel Hi-UCD-S se dézippe en `train/ val/ test/` (paire annotée
**2018→2019** ; `test/` **sans masque** → on valide sur `val/`). Transfert + archive :

```bash
# depuis ton PC :
rsync -avP Hi-UCD.zip <user>@narval.alliancecan.ca:/scratch/<user>/
# sur le cluster :
cd /scratch/<user> && unzip Hi-UCD.zip -d hi-ucd
# archive pour le staging rapide (train+val seulement, ~15 Go) :
tar -cf $SCRATCH/hi-ucd.tar -C /scratch/<user>/hi-ucd train val
```

### Dataset SECOND (benchmark de référence)

Téléchargement direct sur un nœud de connexion (3,84 Go, version prétraitée
ChangeMamba : cartes sémantiques mono-canal, changement binaire déjà généré) :

```bash
bash scripts/download_second.sh          # -> $SCRATCH/SECOND
```

Vérifier le loader sur les vraies données (attendu : **2 968** train / **1 694**
test, soit le split officiel) :
```bash
python -c "
from csf_mamba.datasets import DATASETS
cls, n = DATASETS['second']
for s in ['train','test']:
    ds = cls('$SCRATCH/SECOND', s); print(s, len(ds), 'paires')
"
```

---

## Run de test rapide (QOS debug, quelques minutes)

`scripts/test.sbatch` (`--qos=cc-debug`, 10 batches) vérifie que le kernel + le
pipeline tournent sur GPU. `sbatch scripts/test.sbatch`, puis lire `csf-test-*.out` :
on veut voir `epoch 0 step 0 {...}` puis une ligne `[val]`.

## Le vrai entraînement

```bash
cd $HOME/csf-mamba && sbatch scripts/train.sbatch          # Hi-UCD  (~14 h)
cd $HOME/csf-mamba && sbatch scripts/train_second.sbatch   # SECOND  (~3-4 h)
squeue -u $USER
```

⚠️ **Les deux recettes ne sont pas interchangeables.** Sur SECOND : `--val-split test`
(pas de split `val`), `--sek-warmup-iters 0` (comme Mamba-FCS), et
`--lambda-sem-change 0` (redondant, la sémantique y est déjà restreinte au
changement). Détails et justifications dans `scripts/train_second.sbatch`.

Recette (dans `train.sbatch`) : backbone VMamba-mini pré-entraîné, crops 256, batch 8,
AMP bf16, LR cosine+warmup, warmup SeK, **loss BCD pondérée + Dice** (contre le
déséquilibre : ~2,5 % de pixels changés seulement), 100 époques (~14 h).

**Reprise automatique** : si le job atteint la limite de 24 h, resoumets le même
`sbatch` — il repart depuis `last.pt` (modèle + optimiseur + scheduler + step).
Sortie dans `$SCRATCH/csf-mamba-runs/<config>/` : `best.pt`, `last.pt`, `metrics.csv`.

**Suivre la courbe** (le CSV survit à un `rm` du `.out`) :
```bash
tail -f $SCRATCH/csf-mamba-runs/<config>/metrics.csv
```
On veut le **SeK monter au-dessus de 0** (le modèle détecte enfin les changements).

## Évaluer un checkpoint

Tableau complet des métriques + diagnostic collapse + visualisations, dans un job GPU :

```bash
python -m scripts.evaluate \
    --data-root /scratch/<user>/hi-ucd \
    --checkpoint $SCRATCH/csf-mamba-runs/<config>/best.pt \
    --encoder vmamba_mini --output eval_<config>
```
Affiche SeK/Fscd/mIoU/OA/kappa, le % de changement prédit vs vérité (repère un
collapse « aucun changement »), et sauve des panneaux T1|T2|changement|sémantique.

---

## Pièges connus (déjà gérés par les scripts)

| Symptôme | Cause | Où c'est réglé |
|---|---|---|
| `CUDA version mismatch 12/13` | wheelhouse torch = CUDA 13 | `setup_env.sh` : torch 2.5.1 cu12, `cuda/12.2` |
| `torchvision::nms does not exist` | torch/torchvision dépareillés | installés appariés en `--no-index` |
| `GreedySearchDecoderOnlyOutput` | `mamba_ssm/__init__` tire transformers | patch sed dans `setup_env.sh` |
| step qui ne finit jamais | backend `ref` (scan Python) | `--backend mamba` (kernel) |
| `CUDA out of memory` | 512² batch 8 | crops 256 + `PYTORCH_CUDA_ALLOC_CONF` |
| SeK bloqué à 0 | déséquilibre (2,5 % de changement) | loss BCD pondérée + Dice |
