# CSF-Mamba

*Change-aware Spatio-Frequency Mamba* — architecture Mamba **efficiente** (cible
~15M paramètres) pour la **détection sémantique de changements** (SCD), visant à
battre le SOTA (Mamba-FCS, 189M) sur Hi-UCD et SECOND.

- Conception et raisonnement d'architecture : `documentation/plan_recap_CSF-Mamba2.md`
- **Journal de bord** (chronologie, décisions, résultats des runs) :
  `documentation/journal-de-bord.md` — matière première du rapport
- Lancer un entraînement / une évaluation : `RUN.md`

## État actuel (3 août 2026)

Pipeline complet validé sur GPU (Narval, A100), **20,8 M paramètres**, deux datasets
supportés et validés sur données réelles.

**Résultat de référence — SECOND** (split officiel, code de métriques verbatim).
GMACs mesurés en 512×512 avec fvcore, handlers SSM inclus — même convention que
ChangeMamba, qui étiquette ses sorties fvcore « GFLOPs » alors qu'il s'agit de MACs :

| Méthode | Params | GMACs | OA | Fscd | mIoU | **SeK** |
|---|---|---|---|---|---|---|
| Mamba-FCS | 189,54 M | 263,15 | 88,62 | 65,78 | 74,07 | **25,50** |
| MambaSCD-Base | 89,99 M | 211,55 | — | — | — | *22,92* |
| **MambaSCD-Tiny** | 21,51 M | 73,42 | — | — | — | *22,08* |
| **CSF-Mamba** | **20,80 M** | **41,30** | 87,57 | 62,10 | 72,00 | **21,04 ± 1,06** |
| CSF-Mamba, décodeur élargi | 24,27 M | 53,45 | 87,24 | 61,67 | 71,70 | 20,89 † |

*(SeK en italique = checkpoints publiés par ChangeMamba, évalués par eux.
† une seule graine — non comparable au ± de la ligne CSF-Mamba.)*

**Notre SeK est une moyenne sur 4 graines**, pas un run isolé. L'écart-type
run-à-run vaut **0,0089**, mesuré sur 8 entraînements ne différant que par la
graine. Un chiffre unique de cette distribution peut s'écarter de ±0,02 : toute
comparaison portant sur moins de 0,013 d'écart, à 4 graines, est indécidable.

**Le résultat d'efficience** — face à MambaSCD-Tiny, seul modèle de taille comparable :
**−3 % de paramètres, −44 % de calcul, pour −4,7 % de SeK**. Face à Mamba-FCS :
**6,4× moins de calcul et 9,1× moins de paramètres** pour −17 % de SeK.
L'écart à MambaSCD-Tiny (0,0104) est du même ordre que notre écart-type
run-à-run (0,0089) : les deux modèles sont proches à la précision de la mesure.

Répartition du coût (512²) : convolutions 63 %, `MambaInnerFn` (C²S²) 12 %,
matmul 12 %, einsum 9 %, scan sélectif du backbone 3 %. Le modèle est dominé par
ses parties convolutionnelles, non par la machinerie SSM.

**Ablations — ⚠️ statut révisé le 10 août 2026.** L'écart-type run-à-run vaut
**0,0089** de SeK, mesuré sur 8 entraînements ne différant que par la graine, et
non ±0,004 comme supposé jusque-là — ce plancher venait d'un unique réplicat
accidentel. Toutes les comparaisons sont refaites ci-dessous **appariées** (un seul
facteur change) et testées sur `t = Δ / SE`, avec `SE = σ·√(1/n₁+1/n₂)` ; le seuil
à 95 % est |t| > 2,45.

| Facteur | Dataset | Δ SeK | t | Statut |
|---|---|---|---|---|
| **Retirer la compensation de déséquilibre** | SECOND, c256 | **+0,032** | +2,56 | ✅ **établi** |
| Crops 256 → 512 | SECOND, mini | +0,022 | +2,22 | limite |
| Backbone mini → tiny | SECOND, c256 | +0,018 | +1,41 | ? |
| Lovász | SECOND, c256 | −0,006 | −0,50 | ? |
| Poids 2 + Lovász | SECOND, c256 | +0,005 | +0,38 | ? |
| Backbone mini → tiny | SECOND, c512 | −0,001 | −0,12 | ? |
| Décodeur élargi (+3,47 M, +12,15 GMACs) | SECOND, c512 | −0,002 | −0,15 | ? |
| LR cosine → constant | SECOND, c512 | −0,004 | −0,65 | ? |
| **Sur-échantillonnage ×3** | Hi-UCD | **+0,037** | +2,96 | ✅ **établi** |
| Crops 512 (vs 256) | Hi-UCD | −0,024 | −1,88 | ? |
| Supervision sémantique ciblée | Hi-UCD | +0,019 | +1,53 | ? |
| Sur-échantillonnage ×10 (vs ×3) | Hi-UCD | −0,015 | −1,15 | ? |
| Backbone mini → tiny | Hi-UCD | −0,009 | −0,74 | ? |
| Sur-échantillonnage ×5 (vs ×3) | Hi-UCD | −0,003 | −0,25 | ? |

**« ? » ne signifie pas « effet nul »** mais « indécidable à ce nombre de graines ».
La plupart de ces runs n'en ont qu'une : détecter 0,010 en demanderait 7 de chaque
côté, détecter 0,020 en demande 2. Le σ n'a été mesuré que sur SECOND en crops 512 ;
son application à Hi-UCD est une **hypothèse**, faute de réplicat sur ce dataset.

**Deux effets seulement résistent** — et ce sont les deux plus gros, tous deux
liés aux **données** plutôt qu'à l'architecture ou à l'optimisation : calibrer la
compensation de déséquilibre sur le taux de changement du dataset, et
sur-échantillonner les tuiles porteuses de signal.

Le même réglage anti-déséquilibre est **décisif sur SECOND et neutre sur Hi-UCD** :
il doit être calibré sur le taux de changement du dataset.

**Le plafond de l'IoU du changement.** Reconstruit depuis `IoU_fg = 1 + ln(SeK/κ)`,
il vaut **0,546 à 0,562 sur les sept configurations SECOND** — un étalement de
0,016, deux fois moindre que celui du SeK (0,032). Trois modèles pourtant très
différents y convergent au même point : référence 20,6 M → **0,5621**, encodeur
élargi 36,9 M → **0,5594**, décodeur élargi 24,1 M → **0,5597**. Le plus petit est
le meilleur. Ce qui sépare une bonne configuration d'une mauvaise sur SECOND n'est
donc **pas la délimitation du changement** mais la qualité sémantique (κ, Fscd).
Cinq familles de leviers — loss, données, résolution, capacité d'encodeur, capacité
de décodeur — laissent ce plafond intact : il n'est imputable ni à l'optimisation,
ni à la capacité, où qu'on la place.

**Constat transversal :** les seuls leviers efficaces touchent aux **données**
(densité du signal) et à la **résolution** — jamais à la loss. Cohérent avec le fait
que la somme des erreurs de localisation (FN+FP) reste constante (~100 M sur SECOND)
quelle que soit la loss : celles-ci déplacent le point de fonctionnement
précision/rappel sans améliorer la courbe.

**Hi-UCD — conclusion.** Les quatre familles de leviers (loss, données, capacité,
résolution) sont épuisées : le meilleur SeK y plafonne à **0,053** contre 0,214 sur
SECOND. Doubler le backbone *dégrade*, la résolution 512 *dégrade*, et l'époque du pic
recule dès qu'on augmente la pression sur les données. **Le plafond est une propriété
du jeu de données** — 1 130 tuiles porteuses de signal sur 12 000 — non du modèle.
Hi-UCD est clos comme terrain d'optimisation, conservé comme dataset d'ablation et
résultat de caractérisation.

Chronologie détaillée, décisions et diagnostics : `documentation/journal-de-bord.md`.

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

## Feuille de route

1. ✅ **Pipeline sur GPU** — fait.
2. ✅ **Détection de changements fonctionnelle** sur Hi-UCD (Fscd 0,227).
3. 🔄 **Sémantique des transitions** : sortir le kappa du négatif (run 3).
4. **Premier chiffre sur SECOND** — le seul terrain de comparaison non ambigu.
5. **Ablations** (la contribution) : damier vs CSSM-L1, ± FFT, ± L_sc, ± loss SeK,
   mini vs tiny, crops 256 vs 512.
6. **Comparaison efficience/SOTA** : params, FLOPs, temps d'inférence.

Reste à confirmer avant l'ablation L1 : les 2 détails du portage CSSM (axe de
réduction, RMSNorm) — voir `csf_mamba/modules/cssm.py`.

## Choix de backbone : mini vs tiny (impacte la cible 15M)

Le backbone VMamba est branché sur ChangeMamba, en deux variantes (commutateur =
`mlp_ratio`, mesuré) :

| `--encoder` | backbone | modèle complet | verdict |
|---|---|---|---|
| `vmamba_mini` | 13,84 M | **20,80 M** | **défaut** — meilleur SeK, 41,30 GMACs |
| `vmamba_tiny` | 28,0 M | 36,90 M | +0,018 de SeK en crops 256, −0,005 en crops 512 |

*(Comptes mesurés par fvcore. Les estimations initiales — 13,1 / 19,8 / 34,8 M —
étaient légèrement basses.)*

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
