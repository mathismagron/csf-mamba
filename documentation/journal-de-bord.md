# Journal de bord — CSF-Mamba

Chronologie du projet : ce qui a été fait, **pourquoi**, et ce que ça a donné.
Destiné à servir de matière première au rapport de stage.

Contexte : stage de recherche été 2026, Université de Moncton (Prof. Eric Hervet,
Prof. Andy Couturier). Objectif : une architecture Mamba **efficiente** (~20M
paramètres) pour la détection sémantique de changements (SCD), à comparer au SOTA
(Mamba-FCS, 189M) sur Hi-UCD.

---

## Phase 1 — Conception et implémentation (17–21 juillet 2026)

### Point de départ
Un plan de conception détaillé (`plan_recap_CSF-Mamba2.md`) et un dépôt vide.
L'architecture visée assemble des briques issues de la littérature :

| Bloc | Origine |
|---|---|
| Encodeur VMamba siamois | ChangeMamba |
| C²S²-Block (damier + MCA-SF + S6) | ChessMamba + CSSM |
| Injection FFT2 + Change-Guided Attention | Mamba-FCS |
| Décodeur SCD partagé + DySample | ChessMamba |
| Loss composite (CE + SeK + L_sc) | Mamba-FCS + AtrousMamba |

### Décisions structurantes

**D1 — Code propre plutôt que fork.** Le plan suggérait de forker Mamba-FCS. Choix
retenu : écrire notre code dans `csf_mamba/`, et isoler les dépôts de référence dans
`third_party/` (git-ignoré). *Raison :* garder un code lisible et défendable, tout en
pouvant reprendre des briques vérifiées à l'identique.

**D2 — Backend SSM interchangeable.** `backend = ref | mamba | auto`. Le scan SSM peut
tourner soit via un kernel CUDA rapide, soit via une implémentation PyTorch pure.
*Raison :* ne pas dépendre d'une compilation CUDA pour instancier le modèle → le
squelette est testable sur un laptop sans GPU. **Ce choix s'est révélé décisif** : il a
permis de développer et tester toute la plomberie avant d'avoir un environnement
cluster fonctionnel.

**D3 — Backbone `mini` plutôt que `tiny`.** En mesurant les paramètres réels :

| Variante | Backbone | Modèle complet |
|---|---|---|
| VMamba-tiny (MLP actif) | 28,0 M | 34,8 M |
| **VMamba-mini (MLP désactivé)** | **13,8 M** | **~20,8 M** |

Le « VMamba-Tiny ~14M » du plan correspondait en fait à la config **mini**. Sans cette
vérification, le budget aurait été dépassé de 75 %.

**D4 — Convention d'index A.** Sémantique en classes 1..9, `unlabeled (0) → 255`, têtes
à 10 canaux (index 0 réservé). *Raison :* une seule configuration de loss et de métrique
valable pour Hi-UCD comme pour SECOND, et l'index 0 réservé coïncide exactement avec le
« no-change » de l'évaluation SCD standard.

**D5 — SeK et métriques portées verbatim.** La SeK-loss vient de Mamba-FCS et les
métriques (SeK/Fscd/mIoU/OA) de ChangeMamba, reprises **telles quelles** puis
**validées numériquement identiques** aux originales (`tests/test_sek_port.py`,
`tests/test_metrics.py`). *Raison :* garantir que nos chiffres sont comparables à la
littérature.

### Enseignement de la reproduction de Mamba-FCS
En lisant leur code, on découvre que **Mamba-FCS ne construit pas de carte SCD
« from-to »** : sa loss SeK opère directement sur les deux branches sémantiques
restreintes aux zones changées, et le mIoU y est **déjà inclus**. Le terme mIoU séparé
prévu au plan était donc redondant — supprimé.

---

## Phase 2 — Mise en production sur cluster (21–22 juillet 2026)

Cible initiale : tamIA. Basculé sur **Narval** (A100, allocation `def-hervete`) car
l'espace `$SCRATCH` n'était pas provisionné sur tamIA.

Layout : code dans `$HOME`, environnement + données + poids dans `$SCRATCH`.

### Obstacles techniques résolus
Cette phase a été de loin la plus coûteuse en temps. Chaque problème et sa cause :

| Symptôme | Cause | Résolution |
|---|---|---|
| `CUDA version mismatch (12.2 vs 13.2)` | module CUDA ≠ version de torch | charger `cuda/13.2` explicitement |
| `nvcc fatal: Unsupported gpu architecture 'compute_70'` | CUDA 13 a retiré Volta/V100 | retirer `compute_70` (A100 = `sm_80`) |
| `cub has no member 'LaneId' / 'CTA_SYNC'` | **CUDA 13 a supprimé ces primitives CUB**, utilisées par le kernel `selective_scan` | **redescendre sur torch CUDA 12** |
| `torchvision::nms does not exist` | torch et torchvision de builds différents | les installer **appariés** et en premier |
| `GreedySearchDecoderOnlyOutput` introuvable | `mamba_ssm/__init__.py` importe un modèle LM incompatible avec `transformers` récent | neutraliser cet import |
| `CUDA out of memory` | 512² en batch 8 sature l'A100 40G | crops 256 + `expandable_segments` |
| Step qui ne se termine jamais | backend `ref` (boucle Python) sur séquences de 32k tokens | installer `mamba_ssm`, forcer `--backend mamba` |

**Décision notable (D6) :** face au problème CUB, deux options — patcher le code CUDA du
kernel, ou redescendre torch en CUDA 12. **Choix : redescendre torch.** *Raison :* 5 des
9 appels cassés étaient des primitives de *warp shuffle* touchant la logique numérique du
scan ; une erreur y aurait produit des calculs faux **sans planter**, corrompant
silencieusement l'entraînement. Le risque n'était pas acceptable pour un projet dont le
livrable est un chiffre de performance.

**Stack finale validée :** torch 2.5.1 + torchvision 0.20.1 + mamba_ssm 2.2.4 +
causal_conv1d, tout depuis le wheelhouse Alliance, avec `cuda/12.2`.

### Jalon — 22 juillet 2026
**Pipeline validé de bout en bout sur GPU.** Poids ImageNet chargés (152 tenseurs,
0 incompatibilité), 20,8 M paramètres, forward + backward + validation + sauvegarde.

---

## Phase 3 — Expérimentation (22 juillet 2026 → en cours)

### Le dataset
Hi-UCD-S : paire annotée **2018→2019**, tuiles 512×512, masques RGB 3 canaux
(sémantique T1 / sémantique T2 / changement), 9 classes sémantiques. Le split `test/`
n'a **pas** de masques → validation sur `val/`.

**Mesure déterminante : seulement 2,45 % des pixels sont des changements.** Ce
déséquilibre extrême conditionne tout ce qui suit.

### Run n°1 — baseline (22–23 juillet, 13 h 45, 100 époques)

Recette : VMamba-mini pré-entraîné, crops 256, batch 8, AMP bf16, LR cosine + warmup,
warmup SeK, loss CE + SeK + L_sc.

| SeK | Fscd | mIoU | OA |
|---|---|---|---|
| **0.000** | 0.000 | ~0.50 | 0.9995 |

**Diagnostic : collapse.** Le modèle prédit « aucun changement » partout. Comme 97,5 %
des pixels sont effectivement inchangés, il obtient 99,95 % de justesse (OA) **sans
jamais détecter un seul changement**. C'est le défi classique du déséquilibre de classes
en détection de changements — pas un bug.

### Correctif → Run n°2 (23 juillet, 100 époques)

Deux leviers ajoutés contre le déséquilibre :
1. **Cross-entropy pondérée** sur la classe changement (poids 20)
2. **Loss Dice** sur le changement — plus robuste qu'une CE pondérée en déséquilibre
   extrême, car elle optimise le recouvrement plutôt que la justesse pixel par pixel

| Métrique | Run n°1 | **Run n°2** | Lecture |
|---|---|---|---|
| Fscd | 0.000 | **0.227** | ✅ le modèle détecte enfin les changements |
| mIoU | ~0.50 | **0.693** | ✅ progrès net |
| SeK | 0.000 | **−0.019** | ❌ toujours nul |
| kappa | 0.000 | −0.034 | ❌ négatif |

**Le collapse est corrigé.** Le modèle localise correctement *où* ça change. Mais il
échoue à dire *de quoi vers quoi* : un kappa négatif signifie que, dans les zones
changées, la classification sémantique est **moins bonne que le hasard**.

### Découverte — 28 juillet : la loss SeK produit des NaN

Le journal du run n°2 montre `'sek': nan` de façon récurrente, et sinon des valeurs qui
oscillent entre `0.0` et `~11.5` sans rien entre les deux.

**Cause identifiée** (`sek_mambafcs.py`) :
```python
sek_value = kappa * torch.exp(beta * miou)
log_sek   = (sek_value + eps).log()      # log d'un NÉGATIF quand kappa < 0 → NaN
```
Et le `clamp(min=0)` final annule alors le gradient. Conséquence : **la loss SeK n'a
jamais rien appris** — soit gradient nul, soit pic énorme.

**Pourquoi Mamba-FCS n'a pas ce problème** (deux raisons vérifiées dans leur code) :

1. **Nature des labels.** Sur SECOND, la sémantique n'est annotée que dans les zones
   changées (`label_clf[label==0] = 255`). Leur tête sémantique s'entraîne donc
   exactement sur la population que SeK mesure → kappa positif rapidement. Sur Hi-UCD,
   la sémantique est en **pleine scène** : la tête optimise 97,5 % de pixels inchangés,
   alors que SeK ne regarde que les 2,5 % changés.
2. **Ils contournent le problème.** Leur code contient
   `SEK_START_ITER = 0 if dataset == 'SECOND' else 150000` : sur tout dataset autre que
   SECOND, ils retardent la loss SeK de 150 000 itérations — soit, à notre échelle, la
   durée d'un run entier.

**Conclusion :** cette loss suppose un kappa déjà positif. Elle n'est pas transposable
telle quelle à un dataset à sémantique pleine-scène.

**Distinction importante pour le rapport :** corriger la *loss* SeK ne compromet en rien
la comparabilité avec le SOTA. Ce qui est comparé, c'est la **métrique** SeK
(`evaluation/metrics.py`), portée verbatim et validée — et elle reste intacte. La loss
est un choix de méthode interne, propre à chaque travail.

### Run n°3 — supervision sémantique ciblée (28 juillet, en cours)

**Hypothèse testée :** si le kappa est négatif parce que la tête sémantique
n'optimise pas les zones changées, alors ajouter une supervision sémantique
**restreinte à ces zones** doit faire monter le kappa, donc le SeK.

**Changement (un seul, volontairement) :** nouveau terme `--lambda-sem-change`, une
cross-entropy sémantique où tout pixel hors changement est mis à `ignore` — la CE ne
compte donc que les pixels changés. Poids 1.0, soit une sur-pondération relative d'un
facteur ~40 pour cette population (2,45 % des pixels).

**Ce qui n'a PAS changé :** la loss SeK et les métriques restent strictement
identiques (vérifié : aucune modification de `sek_mambafcs.py` ni `metrics.py`). Un
seul facteur varie → l'effet observé sera attribuable sans ambiguïté, et cette
comparaison run 2 vs run 3 constitue une **ligne d'ablation directement utilisable**.

**Note technique :** la loss SeK continuera d'émettre des `NaN` (kappa négatif). Vérifié
empiriquement que c'est sans danger : le `clamp(min=0)` annule le gradient de ce terme,
et les gradients des autres termes restent finis. La loss SeK est donc simplement
*muette*, elle ne corrompt pas l'entraînement.

Résultat : _à compléter_

---

## État au 28 juillet 2026

**Acquis :** pipeline complet et reproductible sur GPU ; modèle ~20,8 M paramètres
(vs 189 M pour Mamba-FCS) ; détection de changements fonctionnelle (Fscd 0,227) ;
outillage d'évaluation avec visualisations ; métriques garanties comparables.

**Verrou actuel :** la classification sémantique **dans les zones changées** — c'est
elle qui détermine le SeK, la métrique cible.

**Piste de travail identifiée :** la cause racine est que la tête sémantique
n'optimise quasiment pas les zones changées (2,5 % des pixels). Deux actions
envisagées :
1. ajouter un terme de supervision sémantique **restreint aux zones changées** — ce que
   SECOND obtient gratuitement par la nature de ses labels ;
2. rendre la loss SeK numériquement robuste (supprimer le `log`), pour qu'elle
   redevienne exploitable une fois le kappa positif.

**Ensuite :** études d'ablation (damier vs CSSM-L1, ± FFT, ± L_sc, mini vs tiny) et
comparaison efficience/performance avec l'état de l'art.

---

## Notes de méthode

- Chaque changement de recette part dans un **dossier de sortie distinct** pour ne pas
  reprendre un modèle entraîné avec une autre loss.
- Les métriques sont écrites dans un `metrics.csv` **à côté des checkpoints** (un log
  `.out` peut être perdu ; le CSV, non).
- Reprise sur checkpoint (`last.pt` : modèle + optimiseur + scheduler + step) : un job
  interrompu par la limite de temps se relance sans perte.
- Trois tests de non-régression (`tests/`) : formes et budget du modèle, portage SeK,
  portage des métriques.
