# CSF-Mamba

*Change-aware Spatio-Frequency Mamba* — architecture Mamba **efficiente** (cible
~15M paramètres) pour la **détection sémantique de changements** (SCD), visant à
battre le SOTA (Mamba-FCS, 189M) sur Hi-UCD et SECOND.

Le raisonnement de conception complet est dans `documentation/plan_recap_CSF-Mamba2.md`.
Ce README ne couvre que la mise en route du code.

## Idée directrice

Garder les *idées* de Mamba-FCS (qui coûtent ~0 paramètre) et remplacer sa
*machinerie* (qui coûte les 189M) :

| Bloc | Provenance | Statut |
|---|---|---|
| Encodeur VMamba siamois | ChangeMamba | ✅ branché (mini 13M / tiny 28M) |
| C²S²-Block (chessboard + MCA-SF + S6) | ChessMamba + CSSM | ✅ implémenté |
| Récurrence CSSM-L1 (ablation) | CSSM | ✅ implémenté (2 détails à confirmer) |
| Injection FFT2 + CGA résiduelle | Mamba-FCS | ✅ implémenté |
| Décodeur SCD partagé + embedding τ | ChessMamba | ✅ implémenté |
| DySample | ChessMamba | ✅ implémenté |
| Loss composite (CE+mIoU+SeK+L_sc) | Mamba-FCS + AtrousMamba | ✅ implémenté (SeK à valider) |

## Décision : code propre + références isolées

On **ne forke pas** Mamba-FCS. Le code propre vit dans `csf_mamba/`. Les dépôts
de référence (VMamba, baselines, SeK-loss verbatim, module L1) sont clonés dans
`third_party/` (git-ignoré) par `scripts/setup_third_party.sh` et servent
uniquement de source à reproduire / lever des briques vérifiées.

## Le point qui dé-risque tout : backend SSM interchangeable

`mamba-ssm` exige une compilation CUDA, et sa présence dans le wheelhouse
d'Alliance Canada n'est **pas garantie**. Donc **rien n'impose `mamba_ssm` à
l'import** :

- `backend="ref"` — scan PyTorch pur, tourne sur CPU (tests, debug). Lent.
- `backend="mamba"` — kernel rapide, exige `mamba_ssm` (erreur claire sinon).
- `backend="auto"` — kernel si disponible, sinon `ref`.

Conséquence : le modèle complet est instanciable et différentiable sur un laptop
sans GPU. L'entraînement réel se fait sur Alliance Canada.

## Structure

```
csf_mamba/
  modules/     chessboard, mca_sf, ssm (+fallback), fusion (FFT/CGA), c2s2, cssm
  backbone/    encoder (ConvEncoder CPU + VMambaTinyEncoder cluster)
  decoders/    dysample, binary (Y_BCD + {CM_i}), semantic (partagé + τ)
  losses/      composite (CE + mIoU + SeK + L_sc)
  datasets/    hi_ucd (PNG 3 canaux, décalage −1, ignore_index)
  model.py     assemblage CSF-Mamba + count_parameters
scripts/       setup_env.sh, setup_third_party.sh, train.py, train.sbatch
```

## Ordre des opérations (prérequis §12.2, risque décroissant)

1. **Reproduire Mamba-FCS (189M)** et matcher SeK 25,50 sur SECOND via les poids
   HF — prérequis absolu de comparabilité. *(à faire dans `third_party/`)*
2. **Figer le protocole unique** (split SECOND 2968/1694, crops, itérations,
   seed=42, métriques) ; ré-entraîner ChangeMamba + ChessMamba dedans.
3. **Intégrer Hi-UCD** — dataloader fait ; valider sur un dump réel.
4. **Baseline C²S²-cœur qui tourne** (VMamba-Tiny + chessboard + S6).
5. **Lire `method/` de CSSM** pour confirmer les 2 détails du portage L1
   (voir `csf_mamba/modules/cssm.py`).

## Choix de backbone : mini vs tiny (impacte la cible 15M)

Le backbone VMamba est branché sur ChangeMamba, en deux variantes (commutateur =
`mlp_ratio`, mesuré) :

| `--encoder` | backbone | modèle complet | verdict |
|---|---|---|---|
| `vmamba_mini` | 13,1 M | **19,8 M** | tient la Piste A (§11-5) — **défaut** |
| `vmamba_tiny` | 28,0 M | 34,8 M | hors cible |

⚠️ Le « VMamba-Tiny ~14M » du plan correspond en fait à la config **mini** (branche
MLP désactivée). Le forward VMamba exige le **kernel CUDA `selective_scan`** : il
ne tourne pas sur CPU. Les tests CPU utilisent donc `--encoder conv`.

Dépendances backbone (au-delà du cœur) : `einops timm fvcore triton`.

## SeK-loss : reproduction Mamba-FCS (fait)

Enseignement de la repro : Mamba-FCS **ne construit pas de carte SCD « from-to »
unique**. Sa `SeK_Loss` différentiable opère sur les deux branches sémantiques
restreintes aux zones changées par le `change_mask`, avec le mIoU **déjà inclus**
dans le terme SeK (pas de terme mIoU séparé). Reproduit dans
`losses/sek_mambafcs.py` (portage **verbatim**), validé numériquement identique à
l'original (`tests/test_sek_port.py`). La loss composite est recâblée en
conséquence — plus besoin de cible `scd`.

## Convention d'index : A (index 0 réservé) — tranché

Retenue pour n'avoir **qu'une seule config de loss** entre SECOND et Hi-UCD.
Sémantique : classes réelles **1..9**, `unlabeled (0) → 255`, têtes à **10 canaux**
(index 0 réservé, jamais une cible). La SeK exclut `non_change_class=0` exactement
comme sur SECOND. `NUM_SEMANTIC_CLASSES = 10` dans `datasets/hi_ucd.py`.

## Évaluation : métriques SCD (fait)

`evaluation/metrics.py` : SeK / Fscd / mIoU / OA, maths portées **verbatim** de
ChangeMamba, validées numériquement identiques (`tests/test_metrics.py`).
Accumulation par histogramme (tient les 40k images), gestion de l'ignore, cartes
SCD par date (0 = no-change = notre index 0 réservé). La boucle de validation est
câblée dans `scripts/train.py` (`validate()`), appelée à chaque époque, avec suivi
du meilleur SeK (`best.pt`).

## Poids pré-entraînés ImageNet (fait)

`scripts/download_pretrained.sh` récupère le backbone VMamba-Tiny ImageNet
(`vssm_tiny_0230_ckpt_epoch_262.pth`, Zenodo, ~123 Mo). **Un seul checkpoint pour
les deux variantes** (shape-matching) — vérifié en le chargeant réellement :

| variant | poids chargés | mismatch | ignorés |
|---|---|---|---|
| tiny | 218 | 0 | tête classif ImageNet (normal) |
| mini | 152 | 0 | poids MLP absents de mini (normal) |

Les seuls poids frais sont les `outnorm*` (normes d'extraction, hors backbone
ImageNet). Config alignée sur le checkpoint : `depths=[2,2,5,2]`, MLP présent.
Passer `--encoder-pretrained <chemin>` à `scripts/train.py`.

## Ce qui reste à câbler (marqué TODO dans le code, pas masqué)

- **Confirmer les 2 détails de portage L1** (axe de réduction, RMSNorm) sur le
  dépôt CSSM, avant l'ablation chess vs L1.

## Démarrage rapide (laptop, CPU)

```bash
pip install -e .          # torch CPU suffit
# (test de bout en bout : voir tâche « smoke test », à définir ensemble)
```

## Sur Alliance Canada

```bash
bash scripts/setup_third_party.sh          # clone les références
sbatch scripts/train.sbatch                # adapter --account et le dataset
```
