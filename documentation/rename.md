# Renommage CSF-Mamba → DistilSCD (13 octobre 2026)

**Pourquoi.** Le résultat du projet est désormais la distillation de PerASCD dans
un élève compact (README §10, article JSTARS en préparation), pas l'architecture
CSF-Mamba elle-même — dont les blocs propres (C²S², FFT, CGA) ne contribuent pas
de façon mesurable. Le code prend donc le nom de la méthode : **DistilSCD**.

**Principe.** On renomme tout ce qui est *code* (paquet, classes, imports,
docstrings, titres) ; on garde tout ce qui *désigne des données existantes sur
Narval* (chemins, runs, caches, venv, noms de jobs) ; on ne réécrit pas l'histoire
(documents datés, `logs/`). Les anciens imports et les anciens pickles restent
lisibles grâce à une cale de compatibilité `csf_mamba/`.

---

## 1. Correspondance ancien → nouveau

| Ancien | Nouveau | Remarque |
|---|---|---|
| paquet `csf_mamba/` | **`distilscd/`** | déplacement à l'identique (36 fichiers, mêmes sous-paquets) |
| `import csf_mamba.X` / `from csf_mamba.X import …` | `distilscd.X` | 37 fichiers réécrits dans `distilscd/`, `scripts/`, `tests/` ; `RUN.md` |
| classe `CSFMamba` (`model.py`) | **`DistilSCDStudent`** | `CSFMamba = DistilSCDStudent` conservé (alias déprécié) |
| classe `CSFMambaLoss` (`losses/composite.py`) | **`DistilSCDLoss`** | `CSFMambaLoss = DistilSCDLoss` conservé (alias déprécié) |
| « CSF-Mamba » dans docstrings, messages, libellés | « DistilSCD » | p. ex. `train.py`, `benchmark_latency.py`, `audit_gmacs.py`, `archive_results.sh` |
| `echo "== csf-mamba <commit> …"` (`cache_teacher.sbatch`, `eval_perascd.sbatch`) | `echo "== distilscd <commit> …"` | simple étiquette de log ; aucun script ne la lit |
| `scripts/narval_audit.sh` : `grep … "$r"/csf_mamba` | `grep … "$r"/distilscd "$r"/csf_mamba` | couvre les clones d'avant et d'après |
| `scripts/check_second_classes.py` : chemin `csf_mamba/datasets/second.py` | `distilscd/datasets/second.py` | lecture du source sans import |
| README : titre `# CSF-Mamba` | `# DistilSCD` | description recentrée sur la distillation ; §1–11 et tous les chiffres inchangés ; libellés de tableaux « CSF-Mamba — … » → « DistilSCD — … » ; arborescence `distilscd/` + cale ; « 8 fichiers de tests » → 12 (compte réel) |
| RUN.md : titre, `from csf_mamba.datasets …`, URL de clone | DistilSCD, `distilscd`, `github.com/mathismagron/distilscd` | note en tête sur les chemins conservés |
| dépôt GitHub `mathismagron/csf-mamba` | `mathismagron/distilscd` | **à faire par toi**, §5a |

Documents datés (`journal-de-bord.md`, `distillation.md`, `landsat-scd.md`,
`hybride.md`, `plan_recap_CSF-Mamba2.md`) : **une seule ligne ajoutée en tête**
(note du 13 oct. 2026), texte d'origine intact.

## 2. Ce qui est gardé volontairement

| Gardé | Où | Pourquoi |
|---|---|---|
| `$HOME/csf-mamba` (clone Narval) | `REPO=` / `cd` de 23 sbatch et scripts shell, et RUN.md | le renommer casserait tous les sbatch et les jobs en file ; le nom du dossier local n'a pas à suivre celui du dépôt GitHub |
| `$SCRATCH/csf-mamba-runs/` | 12 scripts (`train*.sbatch`, `evaluate.py/.sbatch`, `analyse.sbatch`, `aggregate_seeds`, `check_narval`, `archive_results.sh`, `narval_audit.sh`, `inspect_layerscale`) et RUN.md | ~150 runs existants, référencés tels quels dans le journal et `logs/` |
| `$SCRATCH/csf-distill/` | 44 occurrences dans `scripts/` (caches de logits du professeur, runs KD, `checks/`) | données de la distillation ; les chemins des tables finales y renvoient |
| `$SCRATCH/csf-venv-cu12` | 29 occurrences dans `scripts/` et RUN.md | venv compilé (kernels `selective_scan`, mamba_ssm) : rien à réinstaller |
| `$SCRATCH/csf-test` | `test.sbatch` | dossier de sortie du test GPU |
| noms de jobs SLURM `csf-*` (`csf-mamba`, `csf-second`, `csf-kd`, `csf-kd-lsat`, `csf-check-P`, …) et noms de logs `logs/csf-*-%j.out` | en-têtes `#SBATCH` | `scripts/resubmit.sh` relit `logs/csf-second-<job>.out` ; `logs/` et le journal citent ces noms |
| tags de runs (`second_mini_chess_*`, `hiucd_mini_chess_*`, `landsat_*`, `hyb-*`, …) | sbatch | noms des dossiers de runs existants |
| URL du miroir Gitea `git.couturier.prof/mmagron/csf-mamba.git` | `.github/workflows/mirror.yml` | dépôt externe ; à changer seulement si tu renommes aussi le miroir (§5a, optionnel) |
| `documentation/plan_recap_CSF-Mamba2.md`, `documentation/csf_mamba_architecture_overview.png` | noms de fichiers | documents historiques, cités sous ce nom dans le journal |
| `logs/`, `third_party/`, `paper/` | — | hors périmètre (logs = archive ; `paper/` = autres chantiers) |
| dossier local `/home/mathis/Projects/csf-mamba` | — | facultatif ; le garder évite de casser les chemins d'outils locaux |

## 3. Cale de compatibilité `csf_mamba/`

`csf_mamba/__init__.py` ne contient aucun code du modèle. À l'import :

1. il émet un `DeprecationWarning` (« importer 'distilscd' ») ;
2. il place en tête de `sys.meta_path` un *finder* qui résout tout
   `csf_mamba.X.Y` en **le module `distilscd.X.Y` lui-même** (même objet dans
   `sys.modules`, pas une copie ; `distilscd.X.__spec__` est préservé) — de
   façon paresseuse, seuls les sous-modules demandés sont importés ;
3. il enregistre aussitôt les alias des sous-modules `distilscd.*` déjà chargés.

Conséquence : un pickle qui référence `csf_mamba.model.CSFMamba` se charge —
pickle importe `csf_mamba.model`, reçoit `distilscd.model`, y lit `CSFMamba`
(l'alias), donc obtient `DistilSCDStudent`. Pareil pour toute classe d'un
sous-module (`csf_mamba.modules.c2s2.C2S2Block`, …), dont le nom n'a pas changé.

**Ce que contiennent réellement les checkpoints.** Vérifié dans le code :

- `best.pt` = `state_dict()` nu (`scripts/train.py`, `torch.save(eval_model.state_dict(), …)`) ;
- `last.pt` = dict `{model, optimizer, scheduler, epoch, best_sek, global_step[, ema][, kd_feat]}`,
  uniquement des `state_dict` et des scalaires. Le scheduler est un `LambdaLR` dont
  la lambda est une fonction locale : `LambdaLR.state_dict()` ne la sérialise pas.
- les chargements (`train.py --resume`, `evaluate.py`, `check_narval.py`) font
  `load_state_dict`, jamais d'objet picklé.

Les checkpoints existants ne référencent donc **aucun** chemin de module : ils se
chargent après le renommage même sans la cale, et même en `weights_only=True`
(défaut de `torch.load` à partir de torch 2.6 ; Narval est en 2.5.1). La cale est
le filet de sécurité pour tout objet complet sauvé à la main
(`torch.save(model)`), les notebooks et le vieux code `import csf_mamba`. Un tel
objet complet exige `weights_only=False` (fichiers de confiance uniquement).

À retirer quand plus rien n'importe `csf_mamba` (un `grep -rn csf_mamba` sur le
clone Narval et les notebooks suffit à le vérifier).

## 4. Tests

```bash
cd ~/Projects/csf-mamba
KMP_AFFINITY=disabled PYTHONPATH=$PWD python -m pytest -q tests
```

Résultat local (CPU, torch 2.5.1, Python 3.11, env `cm-cpu`, 13 oct. 2026) :
**60 passed, 0 failed** — les 55 tests existants (55 passed aussi avant le
renommage) + 5 nouveaux dans `tests/test_compat_csf_mamba.py` :

| Test | Vérifie |
|---|---|
| `test_shim_aliases_same_modules` | `csf_mamba.model is distilscd.model`, `CSFMamba is DistilSCDStudent`, `CSFMambaLoss is DistilSCDLoss`, `__spec__` du vrai module intact |
| `test_shim_emits_deprecation_warning` | `DeprecationWarning` à l'import (interpréteur neuf) |
| `test_unpickle_old_module_path_after_rename` | un modèle complet sauvé par `torch.save` **sous l'ancien chemin** (le flux ne contient que `csf_mamba.*` / `CSFMamba`, aucun `distilscd`) se recharge dans un interpréteur **neuf** où rien n'est importé ; type `DistilSCDStudent`, sorties identiques bit à bit |
| `test_handwritten_protocol0_pickle` | pickle protocole 0 écrit à la main (`csf_mamba.losses.composite CSFMambaLoss`) → `DistilSCDLoss` |
| `test_state_dict_checkpoint_unaffected` | `best.pt` / `last.pt` au format de `train.py` : aucune référence de module, chargement en `weights_only=True` |

`python tests/test_compat_csf_mamba.py` fonctionne aussi sans pytest (repli de
`check_narval.sbatch`).

## 5. Commandes à lancer

### 5.0 Commit local (avant tout le reste)

```bash
cd ~/Projects/csf-mamba
git add -A distilscd csf_mamba scripts tests README.md RUN.md \
    documentation/rename.md documentation/journal-de-bord.md documentation/distillation.md \
    documentation/landsat-scd.md documentation/hybride.md documentation/plan_recap_CSF-Mamba2.md
git status --short | head -50      # attendu : R  csf_mamba/... -> distilscd/... (renommages détectés)
git commit -m "Renomme le paquet csf_mamba en distilscd (DistilSCD) ; cale de compatibilité csf_mamba"
git push
```

(`paper/`, `documentation/refs/` et `yuan2022.txt` ne sont volontairement pas dans
ce commit.)

### 5a. Renommer le dépôt GitHub et mettre à jour le remote local

Sur GitHub : *Settings → General → Repository name* → `distilscd` → *Rename*.
Ou en ligne de commande (si `gh` est installé et authentifié) :

```bash
gh repo rename distilscd --repo mathismagron/csf-mamba
```

GitHub redirige ensuite l'ancienne URL (clone, fetch, push) vers la nouvelle —
tant que personne ne crée un nouveau dépôt `mathismagron/csf-mamba`. Mettre quand
même le remote à jour :

```bash
cd ~/Projects/csf-mamba
git remote set-url origin git@github.com:mathismagron/distilscd.git
git remote -v && git fetch origin && git status -sb
```

Le superviseur, qui a accès au dépôt, garde son accès (le renommage ne touche pas
aux collaborateurs) ; prévenir pour qu'il fasse le même `git remote set-url`.

Miroir Gitea (facultatif) : si tu renommes aussi `mmagron/csf-mamba` sur
git.couturier.prof, remplacer dans `.github/workflows/mirror.yml`
`…/mmagron/csf-mamba.git` par `…/mmagron/distilscd.git`. Sinon ne rien toucher :
le miroir continue de pousser sous l'ancien nom.

### 5b. Mettre à jour le clone Narval

Le dossier reste `~/csf-mamba` (cf. §2). Sur un nœud de connexion :

```bash
cd ~/csf-mamba
git status --short                    # doit être vide ; sinon commit/stash d'abord
git remote -v                         # noter le schéma (https ou ssh)
git remote set-url origin git@github.com:mathismagron/distilscd.git
#   (ou https://github.com/mathismagron/distilscd.git si le clone est en https)
git fetch origin && git pull --ff-only
ls distilscd csf_mamba                # distilscd/ complet ; csf_mamba/ = __init__.py seul
git clean -ndX csf_mamba              # aperçu des __pycache__ orphelins de l'ancien paquet
git clean -fdX csf_mamba              # les supprimer (fichiers ignorés uniquement)
```

`git pull` fonctionne même sans `set-url` grâce à la redirection GitHub ; le
`set-url` évite de dépendre de cette redirection. Les `__pycache__` orphelins sont
inoffensifs (le *finder* de la cale passe avant eux), le nettoyage est cosmétique.

### 5c. Réinstaller / vérifier sur Narval

Il n'y a **rien à réinstaller** : le dépôt n'est pas un paquet pip installé, les
scripts tournent depuis `~/csf-mamba` avec `PYTHONPATH=.` (ou `python -m
scripts.…` depuis la racine), et le venv `csf-venv-cu12` ne contient que les
dépendances. Vérification rapide sur le nœud de connexion :

```bash
module load python/3.11 cuda/12.2
source $SCRATCH/csf-venv-cu12/bin/activate
cd ~/csf-mamba
python -c "import pytest" 2>/dev/null || pip install --no-index pytest
python -c "import distilscd.model as m, csf_mamba.model as o; assert o.CSFMamba is m.DistilSCDStudent; print('OK')"
KMP_AFFINITY=disabled PYTHONPATH=. python -m pytest -q tests     # attendu : 60 passed
```

Vérification complète sur GPU (tests + réévaluation stricte des 7 `best.pt` du
point d'efficience avec le code renommé, critère |ΔSeK| ≤ 1e-4 en bf16 ; ≈ 30–60
min A100, sortie dans `$SCRATCH/csf-distill/checks/`) :

```bash
cd ~/csf-mamba && mkdir -p logs && sbatch scripts/check_narval.sbatch
```

Si un `venv` devait être recréé un jour, la procédure reste `scripts/setup_env.sh`
(module load + `virtualenv --no-download` + `pip install --no-index`, pas de uv ni
de conda sur le cluster) ; le renommage n'y change rien.
