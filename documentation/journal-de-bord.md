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

**Mesure déterminante : le changement est très minoritaire.** Deux statistiques
distinctes, à ne pas confondre — elles motivent des correctifs différents :

| Mesure | train | val | Ce qu'elle conditionne |
|---|---|---|---|
| **% de pixels** changés | 1,36 % | 2,20 % | le **déséquilibre de classes** dans la loss |
| **% de tuiles** contenant du changement | 9,4 % | 14,7 % | la **densité du signal** d'entraînement |

Les deux se combinent : `9,4 % des tuiles × ~14,5 % de surface dans ces tuiles
= 1,36 % de la surface totale`. Le changement est donc **concentré dans peu de
tuiles**, et y occupe environ un septième de la surface.

*(Note : un premier chiffre de 2,45 % circulait, issu d'un échantillon non
aléatoire — voir plus bas l'erreur d'échantillonnage contigu. Les valeurs
ci-dessus proviennent d'un tirage aléatoire de 1 500 tuiles par split.)*

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
facteur important pour cette population (~1,4 % des pixels en train).

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
réflexe hérité de Hi-UCD (~1,4 % de pixels changés).

**Prochaine expérience :** SECOND avec `--bcd-change-weight 1 --lambda-dice 0`
(aucune compensation). Un seul facteur varie → ligne d'ablation exploitable :
« effet des mécanismes anti-déséquilibre selon le taux de changement du dataset ».

### Première série d'ablations (29 juillet)

Cinq runs de 100 époques, dont un **réplicat involontaire** (`w20` reproduit la
config du run 3) qui fournit une mesure du **bruit run-à-run : ±0,004 de SeK**.
Cette valeur sert de seuil de significativité pour tout ce qui suit.

**SECOND — effet des mécanismes anti-déséquilibre**

| Config | OA | Fscd | mIoU | SeK |
|---|---|---|---|---|
| poids 5 + Dice | 79,85 | 53,60 | 65,27 | 15,61 |
| **poids 1, sans Dice** | **87,36** | **58,94** | **70,54** | **18,84** |

Toutes les métriques progressent simultanément : +3,2 SeK (+20 % relatif) et
+7,5 OA. Sur un dataset à 20 % de changement, la compensation était non seulement
inutile mais **nuisible** — elle provoquait une sur-détection (30,9 % prédits contre
20,1 % réels) qui dégradait la précision.

**Hi-UCD — effet de la supervision sémantique ciblée** (à poids égal, même version
du code) :

| Config | `sem_change` | meilleur SeK |
|---|---|---|
| `nosemch` | 0 | −0,0034 |
| `w5` | 1,0 | **+0,0158** |

Écart **+0,019 ≈ 5× le bruit** → effet réel. **L'hypothèse du run 3 est validée.**
Un jugement antérieur la qualifiait de « marginale » sur la base de 6 époques : cette
lecture était prématurée, les premières époques étant dominées par le bruit.

**Hi-UCD — effet du poids de changement** :

| Config | meilleur SeK |
|---|---|
| poids 20 | +0,0155 |
| poids 5 | +0,0158 |

Écart 0,0003, très en dessous du bruit → **levier sans effet ici**, contrairement à
SECOND. La différence de taux de changement (~1,4 % contre 20 %) explique que le même
réglage soit décisif sur un dataset et neutre sur l'autre.

**Positionnement actuel sur SECOND :**

| Méthode | Params | OA | Fscd | mIoU | SeK |
|---|---|---|---|---|---|
| Mamba-FCS | 189 M | 88,62 | 65,78 | 74,07 | 25,50 |
| ChangeMamba | ~90 M | 88,12 | 64,03 | 73,68 | 24,11 |
| **CSF-Mamba** | **20,8 M** | **87,36** | 58,94 | 70,54 | **18,84** |

OA à 0,8 point de ChangeMamba, mIoU à 3 points ; le SeK reste l'écart principal
(78 % de ChangeMamba) avec 4,3× moins de paramètres.

**Enseignement de méthode :** disposer d'un réplicat change tout. Sans lui, +0,019
et +0,0003 auraient pu être lus de la même façon. Prévoir systématiquement un
réplicat par configuration de référence dans les ablations à venir.

### Diagnostic approfondi : où sont réellement les erreurs (29 juillet)

Le SeK plafonnait à 0,016 sur Hi-UCD contre 0,188 sur SECOND — un ordre de grandeur.
Plutôt que d'essayer des correctifs, mesure de ce que contiennent les annotations et
de ce que le modèle se trompe.

**Erreur de méthode à consigner.** Une première mesure de la distribution des classes
prenait les 300 *premières* tuiles. Or les tuiles de Hi-UCD sont numérotées
séquentiellement et couvrent Tallinn de proche en proche : cet échantillon
correspondait à **une seule zone géographique**. Il suggérait que 3 classes sur 9
seulement apparaissaient dans les zones changées, et une conclusion (« la tâche est
dégénérée sur cette paire temporelle ») en avait été tirée. **Conclusion fausse.**
Sur un tirage aléatoire, le tableau est tout autre :

| | Hi-UCD (train) | SECOND (train) |
|---|---|---|
| Classes présentes dans le changement | 8 / 9 | 6 / 6 |
| Classe dominante | **29,5 %** | 36,1 % |
| Types de transition observés | **31** (sur 48 documentés) | 30 |

Hi-UCD est en réalité **mieux réparti** que SECOND. La leçon vaut au-delà de ce cas :
un échantillon contigu n'est pas un échantillon, même pour une simple statistique
descriptive.

**La vraie anomalie — la densité du signal :**

| | Pixels changés | Tuiles contenant du changement |
|---|---|---|
| Hi-UCD train | 1,36 % | **141 / 1500 (9,4 %)** |
| Hi-UCD val | 2,20 % | 220 / 1500 (14,7 %) |
| SECOND train | 20,13 % | **1499 / 1500 (99,9 %)** |

**90 % des tuiles d'entraînement de Hi-UCD ne contiennent aucun changement.** Avec des
crops tirés uniformément, l'écrasante majorité des exemples ne porte aucun signal pour
la tâche. Sur SECOND, la totalité des tuiles en porte.

**Décomposition des erreurs** (matrices de confusion, `scripts/confusion_report.py`) :

| | Hi-UCD | SECOND |
|---|---|---|
| Changement détecté, bien classé | 56,5 % | **83,9 %** |
| Erreurs de localisation (FP+FN) | 52,0 M | 97,7 M |
| Erreurs de sémantique | 14,9 M | 18,9 M |
| **Ratio localisation / sémantique** | **3,49** | **5,17** |
| IoU du changement | **39,7 %** | **54,6 %** |

**Conclusion : la branche sémantique fonctionne ; c'est la LOCALISATION du changement
qui plafonne tout.** Sur SECOND, une fois le changement détecté, il est correctement
classé dans 84 % des cas. Et comme le SeK contient un facteur `exp(IoU_changement)`,
chaque point d'IoU se répercute directement sur la métrique cible.

Ce diagnostic invalide les pistes explorées jusque-là (pondération sémantique,
distribution dégénérée) : ce n'était pas le bon problème.

Détail complémentaire : sur SECOND, FN (60,8 M) ≫ FP (36,9 M) — le modèle **rate** du
changement (17,4 % prédits pour 20,1 % réels). La suppression de la compensation de
déséquilibre a corrigé la sur-détection mais est allée un cran trop loin.

### Loss Lovász-Softmax (29 juillet)

Correctif dicté par le diagnostic : la **Lovász-Softmax** est une extension convexe
de l'IoU — elle optimise **directement** la quantité qui bloque, là où la
cross-entropy ne travaille que pixel par pixel. Portage verbatim de Mamba-FCS,
validé numériquement identique (`tests/test_lovasz_port.py`). Appliquée à la sortie
de changement (BCD), activable par `--lambda-lovasz`.

**Runs lancés** (hypothèse posée avant résultat) : si la localisation est bien le
goulot, l'IoU du changement doit monter, et le SeK avec lui.

| Run | Dataset | Config |
|---|---|---|
| `second_..._lovasz` | SECOND | Lovász 0,5 seule |
| `second_..._w2lovasz` | SECOND | Lovász 0,5 + poids 2 (récupérer du rappel) |
| `hiucd_..._w5lovasz` | Hi-UCD | Lovász 0,5 |

**Résultat : hypothèse RÉFUTÉE.** (seuil de bruit : ±0,004 de SeK)

| Run | IoU changement | SeK | % changement prédit |
|---|---|---|---|
| SECOND `nobal` (réf.) | 54,59 % | 0,1884 | 17,38 % |
| SECOND `lovasz` | **54,64 %** | 0,1819 | 17,63 % |
| SECOND `w2lovasz` | 56,07 % | 0,1931 | **20,88 %** |
| Hi-UCD `w5` (réf.) | 39,70 % | 0,0155 | 2,13 % |
| Hi-UCD `w5lovasz` | **39,12 %** | 0,0064 | 2,29 % |

La Lovász **ne modifie pas l'IoU du changement** (54,59 → 54,64 sur SECOND) et le
dégrade légèrement sur Hi-UCD, où elle fait chuter le SeK de moitié. Le seul gain
observé (+1,5 point d'IoU) provient du **poids de changement porté à 2**, pas de la
Lovász.

**Enseignement structurel — un budget d'erreur incompressible.** Somme des erreurs de
localisation sur SECOND :

| Run | FN | FP | FN + FP |
|---|---|---|---|
| `nobal` | 60,8 M | 36,9 M | **97,7 M** |
| `lovasz` | 59,9 M | 38,3 M | **98,2 M** |
| `w2lovasz` | 47,6 M | 54,8 M | **102,3 M** |

Le poids 2 calibre parfaitement la détection (20,88 % prédits pour 20,07 % réels) en
échangeant 13 M de faux négatifs contre 18 M de faux positifs — mais la **somme reste
autour de 100 M dans les trois configurations**.

Conclusion : **les interventions au niveau de la loss déplacent le point de
fonctionnement sur la courbe précision/rappel ; elles n'améliorent pas la courbe.**
Le plafond est une limite de **capacité discriminante**, pas d'objectif
d'optimisation. Cela ferme toute la famille des correctifs par pondération et
reformulation de loss, et oriente vers : densité du signal d'entraînement, résolution,
ou capacité du modèle.

**Effet secondaire notable :** pousser la détection dégrade légèrement la sémantique
(83,9 % → 82,3 % de classification correcte sur SECOND). Les deux tâches se disputent
la capacité du décodeur — argument supplémentaire en faveur d'une limite de capacité.

**Prochains leviers, visant la capacité et non la loss :**
1. **Hi-UCD — densité du signal** : seules 9,4 % des tuiles d'entraînement contiennent
   du changement. Sur-échantillonner ces tuiles. C'est un problème de données,
   qu'aucune loss ne pouvait résoudre — cohérent avec ce qui précède.
2. **SECOND — crops 512** : la densité de signal y est bonne (99,9 %), donc la limite
   est ailleurs. Entraîner à la résolution du test supprime le décalage de longueur de
   séquence (256² → 512², ×4 de tokens), spécifique aux modèles SSM.

### Sur-échantillonnage des tuiles avec changement (29 juillet, en cours)

**Hypothèse testée.** Si le plafond de Hi-UCD vient d'un manque de signal
d'entraînement plutôt que d'une limite d'architecture, alors augmenter la proportion
d'exemples utiles doit faire monter l'IoU du changement — la mesure qui a résisté à
toutes les manipulations de loss.

L'argument reposant sur une corrélation frappante entre les deux datasets :

| | Tuiles avec changement | SeK obtenu |
|---|---|---|
| Hi-UCD | 9,4 % | 0,0155 |
| SECOND | 99,9 % | 0,1931 |

Rapport de densité ~10:1, rapport de SeK ~12:1, à architecture et recette identiques.

**Mécanisme.** Index de changement par tuile (lecture du seul masque, mis en cache et
réutilisé à la reprise), puis `WeightedRandomSampler` : les tuiles contenant du
changement reçoivent un poids `OVERSAMPLE`, tirage avec remise, taille d'époque
inchangée. Densité vérifiée conforme à la théorie :

| `OVERSAMPLE` | Densité du signal | Répétitions par tuile utile |
|---|---|---|
| 1 (uniforme) | 9,4 % | 1 |
| 3 | 24,5 % | 2,5 |
| 10 | 51,9 % | 5,3 |

**La validation reste strictement uniforme** — on ne biaise que ce que le modèle voit
à l'entraînement, jamais ce sur quoi il est jugé.

**Compromis assumé et instrumenté.** Un facteur élevé fait revoir souvent les ~1 130
mêmes tuiles (sur-apprentissage) et raréfie les exemples négatifs (faux positifs).
Deux garde-fous mesurables :
- courbe de SeK en validation qui monterait puis **redescendrait** ;
- « % de changement prédit » s'éloignant des 2,05 % réels (à 2,13 % actuellement,
  donc bien calibré).

Deux runs lancés (`ov3` et `ov10`) pour obtenir à la fois le résultat et la
sensibilité au réglage.

**Signal de succès attendu :** IoU du changement au-delà de 39,7 %.
**En cas d'échec des deux :** la limite serait bien la capacité du modèle, et il
faudrait se tourner vers la résolution ou la taille du backbone.

**Résultat : hypothèse validée, mais insuffisante.** (seuil de bruit : ±0,004)

| Config | SeK | Fscd | kappa |
|---|---|---|---|
| `w5` (uniforme) | 0,0158 | 0,321 | 0,029 |
| **`ov3`** | **0,0531** | **0,382** | **0,096** |
| `ov10` | 0,0386 | 0,380 | 0,072 |

`ov3` **triple le SeK** (+0,037, soit ~9× le bruit) — de loin l'effet le plus fort
mesuré sur Hi-UCD. Le manque de signal était donc bien **un** facteur.

Et le compromis anticipé se manifeste : **`ov10` fait moins bien que `ov3`**. À 5,4
répétitions par tuile et par époque, le sur-apprentissage coûte plus que la densité
ne rapporte. L'optimum est proche de 3.

**Mais le niveau absolu reste mauvais** : 0,053 contre 0,214 sur SECOND — un facteur 4
d'écart subsiste. Le manque de signal n'était donc pas **le** facteur. Il ne faut pas
confondre « effet statistiquement net » et « résultat exploitable ».

### Crops 512 sur SECOND — le décalage de longueur de séquence (29 juillet)

**Hypothèse.** L'entraînement se fait sur des crops 256, la validation sur des tuiles
512 : pour un modèle **SSM**, qui traite l'image comme une séquence, cela quadruple la
longueur de séquence entre les deux régimes. Anodin pour un CNN, potentiellement
coûteux ici.

**Protocole.** `CROP=512 BATCH=2 ACCUM=4` : l'accumulation de gradient conserve le
batch effectif de 8, donc **seule la résolution varie** (512² en batch 8 sature
l'A100). Équivalence des gradients vérifiée à 1e-6 près.

| Config | SeK | Fscd | mIoU | OA |
|---|---|---|---|---|
| `nobal` (crops 256) | 0,1884 | 0,593 | 0,710 | 0,869 |
| `w2lovasz` | 0,1931 | 0,592 | 0,713 | 0,859 |
| **`crop512`** | **0,2143** | **0,621** | **0,720** | **0,876** |

**+0,026 (≈6× le bruit), toutes métriques en hausse simultanément.** Hypothèse
validée : pour un modèle SSM, entraîner à la résolution d'inférence compte réellement.
C'est une observation qui dépasse le cadre de ce projet.

**Convergence :** les deux runs plafonnent vers l'époque 62-68 puis déclinent
légèrement — 100 époques suffisent, prolonger dégraderait.

**Positionnement actualisé :**

| Méthode | Params | SeK |
|---|---|---|
| Mamba-FCS | 189 M | 25,50 |
| ChangeMamba (publié) | ~90 M | 24,11 |
| ChangeMamba (checkpoint réel) | ~37 M | 22,08 |
| **CSF-Mamba** | **20,8 M** | **21,43** |

**0,65 point du checkpoint réel de ChangeMamba, avec 56 % de ses paramètres.**

**Constat transversal :** les deux seuls leviers qui ont fonctionné touchent aux
**données** (densité du signal) et à la **résolution** — jamais à la loss. Cohérent
avec le budget d'erreur incompressible observé plus haut.

### Série suivante (lancée le 29 juillet)

Toutes partent de `OVERSAMPLE=3`, la meilleure configuration connue, en ne faisant
varier qu'un facteur :

| Run | Hypothèse testée |
|---|---|
| `ov5` | l'optimum du sur-échantillonnage est entre 3 et 10 |
| `ov3_tiny` | **la capacité du modèle est le verrou restant** (backbone 13,8 M → 28 M) |
| `ov3_crop512` | le décalage de longueur de séquence coûte aussi sur Hi-UCD |

`ov3_tiny` porte le modèle à 34,8 M, hors cible d'efficience : c'est une **ablation
diagnostique**, pas un modèle candidat.

**Ces trois runs épuisent les pistes identifiées.** Si aucun ne porte le SeK de
Hi-UCD au-delà de ~0,08, la conclusion défendable sera que le plafond vient du
**dataset lui-même** — étayée par les mesures accumulées (9,4 % de tuiles utiles,
budget d'erreur constant sous toute variation de loss, capacité). Ce serait une
caractérisation, pas un échec, et elle expliquerait pourquoi la littérature publie si
peu sur cette paire temporelle.

**Résultats — les trois hypothèses sont tranchées, et Hi-UCD peut être conclu.**
(seuil de bruit : ±0,004)

| Run | Params | SeK | Époque du pic | Écart vs `ov3` |
|---|---|---|---|---|
| **`ov3`** | 20,8 M | **0,0531** | 39 | — |
| `ov5` | 20,8 M | 0,0500 | 51 | −0,003 (dans le bruit) |
| `ov10` | 20,8 M | 0,0386 | 33 | −0,015 |
| `ov3_tiny` | **36,9 M** | 0,0438 | **12** | −0,009 |
| `ov3_crop512` | 20,8 M | 0,0295 | **15** | −0,024 |

**1. L'optimum du sur-échantillonnage est un plateau entre 3 et 5.** L'écart `ov3`
vs `ov5` (0,003) est sous le seuil de bruit ; au-delà (×10), la dégradation est nette.

**2. La capacité n'est PAS le verrou.** Doubler le backbone (20,8 → 36,9 M) **dégrade**
le SeK de 0,009. C'était le dernier candidat non écarté — il est éliminé.

**3. La résolution 512 dégrade sur Hi-UCD** (−0,024) alors qu'elle **améliorait**
SECOND (+0,026). Explication cohérente : sur Hi-UCD, les crops 256 fournissent 4× plus
de fenêtres distinctes par tuile, soit une augmentation de données précieuse quand
seules ~1 130 tuiles portent du signal. Passer à 512 la supprime. Sur SECOND
(2 968 tuiles toutes utiles), l'augmentation compte moins et l'alignement des longueurs
de séquence l'emporte. **Le même levier a donc un signe opposé selon la richesse du
dataset.**

**L'époque du pic est le fil conducteur.** Plus on ajoute de capacité ou plus on retire
d'augmentation, plus le modèle culmine tôt : 51 (`ov5`) → 39 (`ov3`) → 15 (`crop512`)
→ 12 (`tiny`). C'est la signature d'un sur-apprentissage **contraint par la quantité de
signal disponible**, non par la capacité du modèle.

---

## Conclusion sur Hi-UCD (30 juillet 2026)

**Les quatre familles de leviers sont épuisées :**

| Levier | Effet sur le SeK |
|---|---|
| Loss (pondération, Dice, Lovász, supervision ciblée) | déplace le point de fonctionnement, **pas la courbe** |
| Données (sur-échantillonnage ×3) | **+0,037**, plafonne à 0,053 |
| Capacité (backbone ×1,8) | **−0,009** |
| Résolution (crops 512) | **−0,024** |

**Le plafond de Hi-UCD est une propriété du jeu de données, pas du modèle.** Avec
1 130 tuiles porteuses de signal sur 12 000 (paire 2018→2019), chaque tentative
d'extraire davantage se paie immédiatement en sur-apprentissage.

Trois observations convergentes l'étayent :
1. l'IoU du changement plafonne à ~40 % quelle que soit l'intervention ;
2. l'époque du pic recule dès qu'on augmente la pression sur les données ;
3. le même modèle atteint SeK 0,214 sur SECOND — dont **99,9 %** des tuiles portent du
   signal contre **9,4 %** ici.

Ce n'est pas un échec expérimental mais une **caractérisation mesurée**. Elle éclaire
d'ailleurs deux faits de la littérature : Mamba-FCS n'évalue pas sur Hi-UCD, et les
rares chiffres publiés portent sur la variante *mini*, non comparable.

**Décision : Hi-UCD est clos comme terrain d'optimisation.** Il reste utile comme
dataset d'ablation (les effets y sont mesurables) et comme résultat de
caractérisation dans le rapport.

**Le résultat à défendre reste SECOND :**

| Méthode | Params | OA | Fscd | mIoU | SeK |
|---|---|---|---|---|---|
| Mamba-FCS | 189 M | 88,62 | 65,78 | 74,07 | 25,50 |
| ChangeMamba (publié) | ~90 M | 88,12 | 64,03 | 73,68 | 24,11 |
| ChangeMamba (checkpoint réel) | ~37 M | — | — | — | 22,08 |
| **CSF-Mamba** | **20,8 M** | **87,57** | 62,10 | 72,00 | **21,44** |

**0,64 point du checkpoint réel de ChangeMamba, avec 56 % de ses paramètres.**

### Ce que les gains ont réellement amélioré

Contre-intuitif au vu du diagnostic : les deux leviers efficaces (sur-échantillonnage,
résolution) ont fait progresser le **kappa** (qualité sémantique) bien plus que l'IoU
du changement (localisation) :

| | kappa | IoU changement |
|---|---|---|
| `ov3` vs `w5` (Hi-UCD) | **×3,4** | +1,4 pt |
| `crop512` vs `w2lovasz` (SECOND) | **+11 %** | **+0,1 pt** |

Sur SECOND, `crop512` et `w2lovasz` ont le **même IoU** (56,2 % vs 56,1 %) mais des SeK
très différents (0,214 vs 0,193) : tout l'écart vient de la sémantique.

**Le verrou de localisation n'a jamais cédé** — IoU du changement bloqué à ~40 %
(Hi-UCD) et ~56 % (SECOND) sous *toutes* les interventions testées. C'est le sujet
naturel d'une suite éventuelle.

---

## Phase 5 — Efficience et ablation croisée sur SECOND (1–3 août 2026)

### Ablation croisée backbone × résolution

Quatre configurations, toutes sur 100 époques, SECOND (`WEIGHT=1 DICE=0 LOVASZ=0`) :

| Config | Backbone | Crop | SeK | Époque du pic |
|---|---|---|---|---|
| `nobal` | mini (20,8 M) | 256 | 0,1884 | 68 |
| `tiny` | **tiny (36,9 M)** | 256 | 0,2062 | 52 |
| **`crop512`** | mini (20,8 M) | **512** | **0,2143** | 62 |
| `crop512_tiny` | tiny (36,9 M) | 512 | 0,2092 | 86 |

**Effets isolés** (seuil de bruit ±0,004) :

| Facteur | Contexte | Effet |
|---|---|---|
| Backbone mini → tiny | crops 256 | **+0,018** ✅ |
| Crop 256 → 512 | backbone mini | **+0,026** ✅ |
| Crop 256 → 512 | backbone tiny | +0,003 (dans le bruit) |
| Backbone mini → tiny | crops 512 | **−0,005** ❌ |

**Les deux leviers ne s'additionnent pas.** Pris séparément ils apportent +0,018 et
+0,026 ; combinés, le résultat (0,2092) est **inférieur** au meilleur des deux
(0,2143). Ils corrigent donc la **même limitation** — vraisemblablement la quantité
de contexte spatial exploitable — et saturent.

**Conséquence pratique majeure : `mini + crop512` domine sur les trois axes à la
fois** — meilleur SeK, 44 % de paramètres en moins, 39 % de calcul en moins que la
variante tiny. La résolution est un levier **gratuit en paramètres**, contrairement à
la capacité.

**Contraste avec Hi-UCD, cohérent avec la conclusion précédente :** le passage à tiny
**améliore** SECOND (+0,018) et **dégrade** Hi-UCD (−0,009). Le même levier, deux
signes opposés — la capacité paie là où les données sont riches (99,9 % de tuiles
utiles) et sur-apprend là où elles sont pauvres (9,4 %).

### Mesure des GMACs

Demande du superviseur. Deux pièges rencontrés, tous deux consignés dans
`scripts/count_gmacs.py` :

**1. Incompatibilité fvcore / PyTorch.** Le handler `einsum` de fvcore impose
`assert len(inputs) == 2`, mais PyTorch trace désormais `aten::einsum` avec un
troisième argument (le `path` d'optimisation) : le comptage échouait. *(À noter : la
version du wheelhouse Alliance est plus stricte que celle de PyPI, l'erreur ne se
reproduit pas partout.)*

**2. Sous-estimation silencieuse de 11,7 %.** `mamba_ssm` fusionne conv1d, x_proj,
dt_proj, le scan et out_proj dans une seule `autograd.Function` (`MambaInnerFn`) que
fvcore ne sait pas décomposer. Sans handler dédié, **les quatre blocs C²S² comptaient
zéro** et les sous-modules apparaissaient « never called ». Le total passait de 41,30
à 36,45 GMACs — une sous-estimation **flatteuse**, exactement le type d'erreur qui
invaliderait un tableau d'efficience. Le script alerte désormais explicitement (`⛔`)
si un scan SSM reste non compté.

**Résultats** (entrée 512×512, paire bi-temporelle) :

| Backbone | Params | GMACs |
|---|---|---|
| **vmamba_mini** | **20,80 M** | **41,30** |
| vmamba_tiny | 36,90 M | 67,94 |

Répartition du coût (mini) : convolutions **63,2 %**, `MambaInnerFn` (nos C²S²)
11,7 %, matmul 11,7 %, einsum 8,6 %, scan sélectif du backbone 2,7 %, le reste < 2 %.
**Le modèle est dominé par ses parties convolutionnelles, non par la machinerie SSM** —
nuance importante : l'efficience obtenue vient surtout de la légèreté du décodeur.

### ⚠️ Convention GMACs / GFLOPs — piège de comparaison

ChangeMamba calcule ses chiffres avec `fvcore.flop_count` et les étiquette
« GFLOPs » (`vmamba.py`, méthode `flops()`). Or **fvcore compte des MACs**. Leurs
« GFLOPs » sont donc des **GMACs**, comme les nôtres.

**Nos chiffres et les leurs sont directement comparables ; appliquer un facteur 2
pour « convertir » serait une erreur** qui nous ferait paraître deux fois plus
coûteux. Dans le rapport : reprendre leur intitulé, ou ajouter une note de méthode.

### Positionnement final sur SECOND

| Méthode | Params | GMACs | SeK |
|---|---|---|---|
| Mamba-FCS | 189,54 M | 263,15 | 25,50 |
| ScanNet (Transformer) | 27,90 M | 264,95 | — |
| MambaSCD-Base | 89,99 M | 211,55 | 22,92\* |
| MambaSCD-Small | 54,28 M | 146,70 | — |
| **MambaSCD-Tiny** | 21,51 M | 73,42 | 22,08\* |
| **CSF-Mamba** | **20,80 M** | **41,30** | **21,44** |

\* checkpoints publiés par ChangeMamba, évalués par eux.
*(Sources : table de complexité de ChangeMamba et table VI de Mamba-FCS, toutes deux
en entrée 512×512 bi-temporelle.)*

**Comparaison à taille équivalente — MambaSCD-Tiny :**

| | MambaSCD-Tiny | CSF-Mamba | Écart |
|---|---|---|---|
| Params | 21,51 M | 20,80 M | **−3,3 %** |
| GMACs | 73,42 | 41,30 | **−43,7 %** |
| SeK | 22,08 | 21,44 | −0,64 (**97,1 %** du score) |

**Face à Mamba-FCS :** 11,0 % de ses paramètres, 15,7 % de son calcul, pour **84,1 %
de son SeK**.

### Reformulation de l'objectif

L'objectif initial — « battre Mamba-FCS » — n'était pas atteignable à ce budget :
**9,1× moins de paramètres et 6,4× moins de calcul**. Les mesures accumulées
soutiennent en revanche un énoncé solide :

> Une architecture SCD atteignant **97 % du SeK de MambaSCD-Tiny pour 56 % de son
> coût de calcul**, et **84 % du SeK de Mamba-FCS pour 16 % de son calcul et 11 % de
> ses paramètres**.

### ⚠️ Ce qui n'a jamais été fait

**Les ablations des composants propres à CSF-Mamba** — ± FFT, ± L_sc, damier vs
CSSM-L1, ± loss SeK — prévues au plan initial, n'ont **jamais été lancées**. On ne
sait donc pas si le C²S²-Block, la branche fréquentielle ou la CGA résiduelle
contribuent au résultat. C'est à la fois :
- la **contribution scientifique manquante** du projet (un modèle efficient sans
  démonstration de *pourquoi* il l'est) ;
- un **gain potentiel** : un composant pourrait nuire, et son retrait ferait gagner
  des points *et* des paramètres.

Autres pistes non explorées : EMA des poids, augmentation plus riche (rotations,
colorimétrie), redémarrages du LR, augmentation au moment du test, Lovász appliquée
aussi aux branches sémantiques (Mamba-FCS le fait, nous seulement sur le changement).

---

## Phase 6 — Le décodeur n'est pas le goulot (5 août 2026)

### L'hypothèse

Sur SECOND, l'IoU du changement plafonnait à ~56 % sous toute intervention, alors
que la sémantique atteignait 84 % **une fois le changement détecté** : le modèle
comprend ce qui a changé, mais délimite mal *où*. Or délimiter est le travail du
décodeur — et les deux décodeurs ne pesaient que **0,98 M sur 20,61 M, soit 4,8 %
du modèle**, contre 13,84 M pour le seul encodeur.

La raison de cette légèreté est le bloc de raffinement : une convolution
**depthwise** 3×3, soit 9·C paramètres au lieu de 9·C² pour une convolution
complète — un facteur 384 à C = 384. Une depthwise filtre chaque canal isolément :
**aucun mélange inter-canaux dans l'opération spatiale**. Or délimiter une
frontière demande justement de combiner texture, couleur et contexte au même
endroit. L'hypothèse était donc que le décodeur, sous-dimensionné, bornait la
qualité de la frontière quoi qu'on fasse en amont.

### Le test

`--decoder-refine {dw, full}` : `full` remplace les depthwise par des 3×3
complètes dans les deux décodeurs. Rien d'autre ne bouge — ni l'encodeur, ni les
C²S², ni les losses, ni le DySample. `dw` reste le défaut, et sa vérification a
été faite avant lancement : **compte de paramètres identique et 338 clés de
`state_dict` identiques** à la version précédente, donc les checkpoints existants
se rechargent tels quels et la référence est strictement inchangée.

Coût du mode `full` : 20,61 M → **24,08 M (+3,47 M)**, les décodeurs passant de
4,8 % à **18,5 %** du modèle. Lancé sur la meilleure configuration connue
(`mini` + crop 512), 100 époques, tout le reste identique au contrôle.

### Le résultat — hypothèse réfutée

Bilan de **toutes** les configurations entraînées sur SECOND, l'IoU du changement
étant reconstruit depuis `SeK = κ·exp(IoU_fg)/e`, soit `IoU_fg = 1 + ln(SeK/κ)` :

| Configuration | ép. | SeK | **IoU chgt** | IoU fond | Fscd |
|---|---|---|---|---|---|
| `crop512` (référence) | 62 | **0,2143** | **0,5621** | 0,8778 | 0,6210 |
| `crop512_tiny` | 86 | 0,2092 | 0,5594 | 0,8759 | 0,6155 |
| `crop512-decfull` | 70 | 0,2089 | 0,5597 | 0,8745 | 0,6167 |
| `tiny` | 52 | 0,2061 | 0,5595 | 0,8775 | 0,6120 |
| `w2lovasz` | 59 | 0,1931 | 0,5607 | 0,8650 | 0,5917 |
| `nobal` | 68 | 0,1883 | 0,5460 | 0,8732 | 0,5926 |
| `lovasz` | 70 | 0,1820 | 0,5464 | 0,8724 | 0,5821 |

Le décodeur élargi donne **SeK 0,2089 contre 0,2143** : −0,0054, soit un peu
au-delà du plancher de bruit de ±0,004. Pas un effondrement, mais aucun gain.

Et surtout, la colonne qui compte : **IoU du changement 0,5597 contre 0,5621**.
+3,47 M de paramètres injectés directement là où le problème était supposé se
situer, et la frontière n'a pas bougé.

### Ce que la colonne IoU révèle vraiment

Le tableau dit plus que le verdict sur le décodeur. Trois modèles
architecturalement très différents convergent au même endroit :

| | paramètres | IoU chgt |
|---|---|---|
| référence | 20,61 M | 0,5621 |
| encodeur élargi (`tiny`, +16 M) | 36,9 M | 0,5594 |
| décodeur élargi (`full`, +3,47 M) | 24,08 M | 0,5597 |

**0,5594 / 0,5597 / 0,5621** — trois valeurs dans un intervalle de 0,003, alors
que la capacité varie de 20,6 M à 36,9 M et que le paramètre ajouté l'est tantôt
en amont, tantôt en aval. Le plus petit modèle a même le meilleur IoU.

Sur les sept configurations, **l'IoU du changement s'étale sur 0,016** (0,546 à
0,562) quand le SeK s'étale sur **0,032**, soit deux fois plus. Autrement dit :
ce qui distingue une bonne configuration d'une mauvaise sur SECOND **n'est pas la
qualité de la délimitation du changement** — c'est la qualité sémantique et
l'équilibre précision/rappel, que capturent κ et Fscd. Les variantes de loss
(`nobal`, `lovasz`) perdent leur SeK via l'IoU du fond et le Fscd, pas via l'IoU
du changement.

### Bilan des leviers testés sur le plafond des 56 %

| Levier | Verdict |
|---|---|
| Loss (pondération, Dice, Lovász, SeK) | déplace le point de fonctionnement, pas la courbe |
| Données (sur-échantillonnage) | +0,037 sur Hi-UCD, plafonne ; sans effet sur le plafond IoU |
| Résolution (crop 256 → 512) | +0,026 de SeK, IoU chgt inchangé |
| Capacité d'encodeur (mini → tiny) | −0,005, IoU chgt inchangé |
| **Capacité de décodeur (dw → full)** | **−0,005, IoU chgt inchangé** |

Cinq familles de leviers, toutes écartées **par la mesure**. Le plafond n'est
donc imputable ni à l'optimisation, ni au signal d'entraînement, ni à la
résolution, ni à la capacité — où qu'on la place. Les explications restantes
sortent du modèle : la précision d'annotation des frontières dans SECOND, et
l'information réellement disponible dans une paire d'images bi-temporelles pour
trancher le contour exact d'un changement. Un plafond mesuré, pas supposé.

### Correction — les noms de classes SECOND étaient faux (5 août)

Le tableau de diagnostic sémantique attribuait **37,11 % des pixels changés à
« arbre »**. Invraisemblable sur des scènes industrielles chinoises : c'est ce
qui a déclenché la vérification.

Nos `CLASS_NAMES` avaient été écrits d'après l'ordre d'**énumération de l'article**
SECOND, qui n'est pas l'ordre des indices dans le dump prétraité. Contrôle par
`scripts/check_second_classes` : pour chaque indice, couleur dominante des pixels
correspondants dans les cartes `GT_T*_COLORED` que le dataloader ignore, comparée
à la palette officielle. **Pureté 100 % sur 200 tuiles**, aucune ambiguïté.

| indice | couleur | classe réelle | ce qu'on écrivait |
|---|---|---|---|
| 1 | vert foncé | **végétation basse** | ~~sol non végétalisé~~ |
| 2 | gris | **sol non végétalisé** | ~~arbre~~ |
| 3 | vert vif | **arbre** | ~~végétation basse~~ |
| 4 | bleu | eau | eau ✓ |
| 5 | rouge foncé | bâtiment | bâtiment ✓ |
| 6 | rouge | terrain de sport | terrain de sport ✓ |

**Aucune métrique n'était affectée** : SeK, Fscd, mIoU, OA et kappa sont calculés
sur des indices, de façon cohérente entre prédiction et vérité. Le portage
verbatim reste valide, aucun réentraînement n'est nécessaire. Seuls les *noms*
étaient faux — mais dans un rapport ou une figure, c'eût été une erreur factuelle.

Le diagnostic sémantique se relit ainsi, et devient cohérent :

| classe | % des pixels changés | rappel |
|---|---|---|
| sol non végétalisé | 37,1 % | 89,8 % |
| bâtiment | 35,1 % | 92,8 % |
| végétation basse | 21,5 % | 74,0 % |
| arbre | 4,5 % | 64,9 % |
| eau | 1,0 % | 55,0 % |
| terrain de sport | 0,7 % | 82,6 % |

Sol nu et bâtiment font 72 % des pixels changés : la signature attendue de
l'urbanisation. Les deux classes dominantes sont aussi les mieux reconnues.

**Réserve honnête** : le lien couleur → nom est visuellement évident pour le bleu
(eau), le gris (sol nu) et le rouge foncé (bâtiment). Il l'est moins entre les
deux verts — vert foncé pour la végétation basse, vert vif pour l'arbre — repris
de la palette de ChangeMamba. La répartition mesurée le corrobore (végétation
basse 21,5 % contre arbre 4,5 %, l'ordre attendu), sans le démontrer.

### Diagnostic complet du changement sur SECOND (5 août)

Décomposition de la matrice de confusion du meilleur modèle, indexée
[prédiction, vérité], sur 888,1 M pixels :

| | pixels |
|---|---|
| changement correctement détecté (TP) | 122,1 M |
| changement **raté** (FN) | **56,1 M** |
| **fausse alarme** (FP) | **40,0 M** |

**Rappel 68,5 %, précision 75,3 %**, IoU 0,5596 — identique à la valeur
reconstruite depuis le SeK, les deux chemins de calcul concordent. FN+FP =
96,1 M, la constante ~100 M observée depuis juillet sous toute intervention.

Le modèle **rate plus qu'il ne sur-détecte**. Les visualisations montrent deux
régimes distincts : des objets trouvés mais à la forme dégradée (angles droits
arrondis, bords déchiquetés), et des objets entièrement manqués ou inventés. Le
second régime n'a rien à voir avec la délimitation — ce qui explique que le
décodeur élargi n'ait rien donné.

La sémantique, elle, fonctionne : **85,9 % d'exactitude** sur les pixels de
changement correctement détectés.

**Ce qui est en jeu**, à kappa constant (0,3245) :

| IoU changement | SeK |
|---|---|
| 0,56 (actuel) | 0,2089 |
| 0,70 | 0,2404 |
| **0,80** | **0,2657** |
| 1,00 | 0,3245 |

Porter le seul IoU du changement à 0,80, sans toucher à la sémantique,
dépasserait Mamba-FCS (0,2550). Tout le déficit restant tient dans cette unique
quantité.

### Incidents d'infrastructure (5 août)

Deux jobs tués par la limite de temps **sans produire une seule ligne Python**,
malgré `python -u` : `csf-gmacs` (190707, 5 h, nœud GPU) et `csf-classes`
(194247, 20 min, nœud CPU). Aucun n'a atteint son premier `print`. Le même jour,
`csf-eval` s'est terminé normalement en 1 h 03 avec le même venv.

Point commun des deux jobs bloqués : l'import de torch depuis le venv sur Lustre.
`csf-classes` a été réécrit pour ne plus dépendre de torch du tout — il lit
`CLASS_NAMES` par analyse syntaxique du source et n'utilise que numpy et Pillow,
fournis par `scipy-stack`. Relancé sans venv, il a terminé en **3 min 30**
(job 196374). Cela **incrimine le venv sur Lustre**, sans le prouver formellement.

Leçon générale retenue : instrumenter avec des traces horodatées et flushées
avant de supposer où part le temps. Sur `csf-gmacs`, l'hypothèse « le traçage
fvcore est lent » était fausse — le traçage n'avait jamais commencé.

### Incident — TIMEOUT du comptage GMACs (job 173649)

Le job de comptage a fini en **TIMEOUT à 1 h**. Cause : le traçage fvcore d'un
modèle SSM est lent, et le script bouclait sur **les deux backbones**. Deux
correctifs : walltime porté à 3 h, et variable `ENCODERS` pour ne mesurer que la
variante manquante au lieu de tout reprendre. Prévision analytique en attente de
confirmation : les trois convolutions élargies par décodeur coûtent 9·C²·H·W à
128²/64²/32², soit **+8,1 GMACs** — 41,30 → ~49,4, toujours loin sous les 73,42
de MambaSCD-Tiny.

---

## État au 29 juillet 2026

**Acquis :** pipeline complet et reproductible sur GPU ; modèle ~20,8 M paramètres
(vs 189 M pour Mamba-FCS) ; détection de changements fonctionnelle sur Hi-UCD
(Fscd 0,227) ; outillage d'évaluation avec visualisations ; métriques garanties
comparables ; **support des deux datasets** (Hi-UCD et SECOND) validé sur données
réelles.

**Verrou identifié : la LOCALISATION du changement.** La branche sémantique
fonctionne (84 % de classification correcte sur SECOND une fois le changement
détecté) ; ce sont les faux positifs et faux négatifs de détection qui pèsent 3,5×
(Hi-UCD) à 5,2× (SECOND) plus lourd. IoU du changement : 39,7 % et 54,6 %.

**Chantiers en cours :**

| Chantier | État |
|---|---|
| Loss Lovász (optimise l'IoU) | ❌ réfutée — n'améliore pas l'IoU |
| Sur-échantillonnage des tuiles avec changement (Hi-UCD) | 2 runs lancés (`ov3`, `ov10`) |
| Crops 512 sur SECOND (décalage de longueur de séquence) | à faire |
| Évaluation des checkpoints ChangeMamba avec notre code | à faire |

**Meilleur résultat à ce jour** — SECOND, `w2lovasz` : SeK **19,31** avec 20,8 M
paramètres, détection parfaitement calibrée (20,88 % prédits pour 20,07 % réels).
À comparer aux checkpoints publiés par ChangeMamba eux-mêmes : SeK 22,08 pour
~37 M paramètres — soit **87 % de leur score avec 56 % de leurs paramètres**.

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
