# CSF-Mamba

*Change-aware Spatio-Frequency Mamba* — architecture Mamba **efficiente** (cible
~15M paramètres) pour la **détection sémantique de changements** (SCD), visant à
battre le SOTA (Mamba-FCS, 189M) sur Hi-UCD et SECOND.

- Conception et raisonnement d'architecture : `documentation/plan_recap_CSF-Mamba2.md`
- **Journal de bord** (chronologie, décisions, résultats des runs) :
  `documentation/journal-de-bord.md` — matière première du rapport
- Lancer un entraînement / une évaluation : `RUN.md`

## État actuel (juillet 2026)

- **Pipeline complet validé de bout en bout sur GPU** (Narval, A100). Modèle
  ~20,8 M params, backbone ImageNet chargé, kernels CUDA opérationnels.
- **Run n°1 (100 époques) terminé.** Résultat : SeK = 0 → le modèle collapse vers
  « aucun changement », à cause du **déséquilibre extrême** (2,45 % de pixels
  changés sur Hi-UCD). Diagnostic clair, pas un bug.
- **Correctif en cours (run n°2)** : loss BCD **pondérée + Dice** contre le
  déséquilibre. Puis itérations et **ablations** (chess vs L1, ±FFT, ±L_sc).

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
| Loss composite (CE+SeK+L_sc+Dice) | Mamba-FCS + AtrousMamba | ✅ SeK validé verbatim ; +Dice/pondération BCD |

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

## Feuille de route (mise à jour)

La priorité est **csf-mamba sur Hi-UCD** (reproduire Mamba-FCS sur SECOND n'est
plus prioritaire pour l'instant). Étapes :

1. ✅ **Pipeline qui tourne sur GPU** (fait).
2. 🔄 **Baseline qui apprend** : régler le déséquilibre pour sortir le SeK de 0
   (loss pondérée + Dice), calibrer poids/époques via `metrics.csv`.
3. **Optimiser** : Lovász, EMA, résolution, LR — selon la courbe de validation.
4. **Ablations** (la contribution) : chess vs L1, ±FFT, ±L_sc, mini vs tiny.
5. **Comparaison efficience/SOTA** : params/FLOPs/temps vs ChangeMamba, Mamba-FCS.

Reste à confirmer avant l'ablation L1 : les 2 détails du portage CSSM (axe de
réduction, RMSNorm) — voir `csf_mamba/modules/cssm.py`.

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

## Entraînement & évaluation

Recette (`scripts/train.sbatch`) : crops 256, batch 8, AMP bf16, LR cosine+warmup,
warmup SeK, loss BCD pondérée + Dice, 100 époques, reprise auto. Métriques
persistées dans `metrics.csv`. Évaluation d'un checkpoint + visualisations :
`python -m scripts.evaluate --checkpoint <run>/best.pt ...`.

**Marche à suivre complète (installation, run, éval, pièges) : `RUN.md`.**

## Démarrage rapide (laptop, CPU)

```bash
pip install torch numpy pillow scipy      # CPU suffit pour les tests
PYTHONPATH=. python tests/test_smoke.py   # forward/backward + formes (encodeur conv)
```

Le forward VMamba (kernel CUDA) ne tourne que sur GPU ; en local on teste la
plomberie avec `--encoder conv` / `--backend ref`.
