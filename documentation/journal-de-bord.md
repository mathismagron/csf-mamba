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

**Résultat partiel (8 premières époques) — effet marginal à ce stade.**

Comparaison époque par époque avec le run 2 (les deux `metrics.csv` existent) :

| Époque | kappa run 2 | kappa run 3 |
|---|---|---|
| 0 | −0,147 | −0,209 |
| 1 | −0,128 | −0,111 |
| 2 | −0,060 | −0,019 |
| 3 | −0,006 | **+0,002** |
| 4 | −0,076 | −0,054 |
| 5 | −0,104 | −0,103 |

Les deux trajectoires sont **quasi superposées** : même pic à l'époque 3, même
rechute ensuite. Le run 3 passe positif de justesse (première fois du projet), mais
l'écart reste dans le bruit. **L'hypothèse n'est ni validée ni infirmée** à ce stade.

Deux enseignements de méthode :
- Le pic à l'époque 3 est un **transitoire commun** aux deux runs, non prédictif :
  le run 2 finissait à −0,03 après ce même pic. Comparer le début d'un run à la fin
  d'un autre induit en erreur — d'où l'intérêt du `metrics.csv` par run.
- La chute simultanée de l'OA à l'époque 3 (0,932 contre ~0,97) révèle une
  **compétition entre les deux objectifs sémantiques** : `ce_sem` (pleine scène,
  tirée par les 97,5 % de pixels inchangés) contre `sem_ch` (zones changées). À
  poids égal (λ=1), ils se neutralisent.

Verdict attendu vers les époques 20-50, quand le LR aura décru et que la loss SeK
se sera activée (époque ~13).

---

## Phase 4 — Pivot vers SECOND (28 juillet 2026)

### Le constat qui change la stratégie

En cherchant les chiffres à battre sur Hi-UCD, découverte gênante :

1. **Mamba-FCS n'évalue pas sur Hi-UCD.** Le papier ne teste que sur **SECOND** et
   **Landsat-SCD**. L'objectif affiché depuis le début (« battre Mamba-FCS sur
   Hi-UCD ») n'avait donc **aucune cible chiffrée**.
2. Les chiffres publiés sur Hi-UCD sont **épars** (améliorations relatives plutôt
   qu'absolues) et portent sur des **variantes différentes** du dataset (Hi-UCD
   complet / Hi-UCD-mini). Comparer notre résultat à un chiffre obtenu sur *mini*
   n'aurait aucune validité.

**Sur SECOND, au contraire**, les références sont denses et sans ambiguïté :

| Méthode | Params | OA | Fscd | mIoU | **SeK** |
|---|---|---|---|---|---|
| ChangeMamba (MambaSCD) | ~90 M | 88,12 | 64,03 | 73,68 | **24,11** |
| Mamba-FCS | 189 M | 88,62 | 65,78 | 74,07 | **25,50** |
| **CSF-Mamba (nous)** | **20,8 M** | — | — | — | *à établir* |

**Décision (D7) : SECOND devient le terrain de comparaison principal**, Hi-UCD
restant le dataset d'origine du sujet et un terrain d'ablation.

Trois arguments : les chiffres y sont directement comparables ; le dataset est
petit (2 968 paires contre ~11 600 → ~3-4 h au lieu de 14 h) ; et la sémantique y
étant annotée **uniquement sur le changement**, le problème de kappa négatif
rencontré sur Hi-UCD ne devrait pas s'y poser.

### Support SECOND ajouté

Le modèle n'a **pas** eu à être modifié — la **convention A** (décision D4), choisie
un an... une semaine plus tôt précisément pour coller à SECOND, s'avère native :
SECOND encode déjà 0 = inchangé pour la sémantique. Seul le nombre de canaux des
têtes change (10 → 7), soit **291 paramètres d'écart** sur 20,8 M.

Travail réalisé : dataloader `SECONDDataset`, registre de datasets
(`csf_mamba/datasets/__init__.py`) d'où `train.py` et `evaluate.py` tirent la classe
et le nombre de canaux, script de téléchargement, script d'entraînement.

**Validé sur les vraies données** : 2 968 paires en train, 1 694 en test — soit
exactement le **split officiel** utilisé par la littérature (les listes
`train.txt`/`test.txt` du dump sont lues, plutôt qu'un parcours de dossier).

Réglages spécifiques à SECOND, et leur justification :

| Réglage | Valeur | Pourquoi |
|---|---|---|
| `--val-split` | `test` | SECOND n'a pas de split `val` |
| `--sek-warmup-iters` | `0` | comme Mamba-FCS (`SEK_START_ITER=0` sur SECOND) |
| `--lambda-sem-change` | `0` | redondant : `ce_sem` y est **déjà** restreint au changement |

Ce dernier point mérite d'être noté : le terme de supervision ciblée est un
correctif **spécifique à Hi-UCD**, sans objet sur un dataset à sémantique
change-only.

### Bug d'évaluation découvert et corrigé (28 juillet)

Le premier run SECOND affichait des métriques incohérentes entre elles : `Fscd 0,698`
et `SeK 0,380` (meilleurs que Mamba-FCS !) mais `OA 0,608` et `mIoU 0,371`
(catastrophiques). Ces quatre valeurs ne peuvent pas coexister sur un modèle sain.

**Cause :** le masque de validité de `SCDEvaluator` exigeait une sémantique annotée
sur **tous** les pixels. Or SECOND n'annote la sémantique que sur le changement →
toute la population « non changé » disparaissait de l'histogramme. Confirmé
numériquement : en posant `iu[0] = 0`, on retrouvait le SeK affiché à 1e-7 près.

**Conséquence la plus grave :** les **fausses détections** n'étaient pas comptées —
un pixel prédit « changé » à tort n'entrait pas dans le calcul.

**Correction :** un pixel inchangé porte légitimement le label 0 dans la carte SCD ;
il ne requiert aucune annotation sémantique. Seuls les pixels changés en exigent une.
Cette logique est aussi celle de l'implémentation de référence (`SCDD_eval_all`
compte tous les pixels, sans notion d'ignore sémantique).

**Impact mesuré :** structurel sur SECOND (une prédiction *parfaite* donnait
mIoU = 0,50 au lieu de 1,00) ; négligeable sur Hi-UCD (~0,6 % relatif sur le SeK),
car la sémantique y est annotée en pleine scène. Les conclusions passées sur Hi-UCD
restent donc valides.

**Garde-fou ajouté** (`tests/test_metrics_validity.py`) : une prédiction parfaite
doit donner des métriques parfaites, vérifié sur les **deux** conventions de
labellisation. Ce test aurait dû exister dès l'ajout de SECOND.

### Premiers résultats comparables — SECOND (28 juillet)

| Méthode | Params | OA | Fscd | mIoU | **SeK** |
|---|---|---|---|---|---|
| Mamba-FCS | 189 M | 88,62 | 65,78 | 74,07 | **25,50** |
| ChangeMamba | ~90 M | 88,12 | 64,03 | 73,68 | **24,11** |
| **CSF-Mamba** | **20,8 M** | 79,85 | 53,60 | 65,27 | **15,61** |

Premier chiffre du projet directement confrontable à la littérature (même split
officiel, même code de métriques). **Nettement en dessous du SOTA** : ~65 % du SeK de
ChangeMamba, avec 4,3× moins de paramètres.

**Diagnostic principal : le modèle sur-détecte le changement**, et de façon cohérente
sur les deux datasets :

| Dataset | Changement réel | Prédit | Excès | `bcd-change-weight` |
|---|---|---|---|---|
| SECOND | 20,07 % | 30,86 % | ×1,5 | 5 |
| Hi-UCD | 2,05 % | 7,35 % | ×3,6 | 20 |

L'excès suit le poids appliqué. Les mécanismes anti-déséquilibre (CE pondérée +
Dice), introduits pour sortir du collapse du run 1, sont **trop agressifs** : la
sur-détection dégrade la précision, donc Fscd, mIoU, OA et in fine le SeK.

Sur SECOND c'est particulièrement injustifié : à **20 % de changement**, le dataset
est quasi équilibré et ne nécessitait aucune compensation. Ces réglages étaient un
réflexe hérité de Hi-UCD (2,45 %).

**Prochaine expérience :** SECOND avec `--bcd-change-weight 1 --lambda-dice 0`
(aucune compensation). Un seul facteur varie → ligne d'ablation exploitable :
« effet des mécanismes anti-déséquilibre selon le taux de changement du dataset ».

---

## État au 28 juillet 2026

**Acquis :** pipeline complet et reproductible sur GPU ; modèle ~20,8 M paramètres
(vs 189 M pour Mamba-FCS) ; détection de changements fonctionnelle sur Hi-UCD
(Fscd 0,227) ; outillage d'évaluation avec visualisations ; métriques garanties
comparables ; **support des deux datasets** (Hi-UCD et SECOND) validé sur données
réelles.

**Verrou actuel :** la classification sémantique **dans les zones changées** sur
Hi-UCD — c'est elle qui détermine le SeK, la métrique cible. Le run 3 teste un
correctif ; verdict en attente.

**Chantiers en cours :**

| Chantier | État |
|---|---|
| Run 3 Hi-UCD (supervision ciblée) | en cours, effet marginal sur 8 époques |
| Entraînement SECOND | prêt à lancer (~3-4 h) |

**Ensuite :**
1. Premier chiffre sur SECOND, comparable à ChangeMamba (24,11) et Mamba-FCS (25,50).
2. Études d'ablation — la contribution scientifique : damier vs CSSM-L1, ± FFT,
   ± L_sc, ± loss SeK, mini vs tiny, crops 256 vs 512 (décalage de longueur de
   séquence entre entraînement et test, spécifique aux modèles SSM).
3. Comparaison efficience/performance (params, FLOPs, temps d'inférence). Option la
   plus rigoureuse : évaluer les checkpoints ChangeMamba publiés (SeK 22,08 / 22,92)
   avec **notre** code de métriques, pour éliminer tout doute de protocole.

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
