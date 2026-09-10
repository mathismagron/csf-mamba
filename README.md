# CSF-Mamba

Architecture Mamba **efficiente** pour la **détection sémantique de changements**
(SCD) sur imagerie aérienne : à partir de deux images d'une même zone prises à
deux dates, produire la carte des zones changées **et** la classe d'occupation du
sol avant et après.

Stage de recherche, Université de Moncton — Mathis Magron, encadré par
Prof. Eric Hervet et Prof. Andy Couturier. Entraînements sur Narval
(Alliance Canada, A100).

- **Journal de bord** : `documentation/journal-de-bord.md` — chronologie complète,
  décisions, résultats de tous les runs. Matière première du rapport.
- **Conception de l'architecture** : `documentation/plan_recap_CSF-Mamba2.md`
- **Lancer un entraînement ou une évaluation** : `RUN.md`

---

## 1. Où en est le projet (10 septembre 2026)

Le pipeline est complet et validé sur GPU, sur deux jeux de données réels. La
campagne compte **~150 entraînements**, tous à plusieurs graines depuis le
7 août.

**Le résultat tient en une phrase :** à performance égale ou supérieure, le
modèle coûte **quatre à sept fois moins de calcul** que les architectures Mamba
publiées pour cette tâche.

Deux configurations sont retenues, aux deux extrémités du compromis :

| | Paramètres | GMACs | SeK |
|---|---|---|---|
| **Point d'efficience** | **16,48 M** | **31,42** | **0,2387** |
| **Point de performance** | 32,58 M | 58,06 | **0,2484** |

Le premier **dépasse MambaSCD-Base** avec 18 % de ses paramètres. Le second
atteint **97 % du SeK de Mamba-FCS** avec 17 % des siens.

---

## 2. Tableau comparatif — SECOND

**Deux travaux servent de référence, et leurs noms prêtent à confusion.**
*ChangeMamba* est un article qui propose trois modèles ; celui qui traite notre
tâche s'appelle **MambaSCD**, décliné en Tiny, Small et Base. *Mamba-FCS* est un
article **différent**, plus récent et bien plus gros (189 M). Leurs SeK sont ceux
qu'ils publient, obtenus par leur propre code sur le même split officiel.

SECOND est le jeu de référence du domaine : 4 662 paires 512×512, 6 classes
sémantiques, split officiel. **SeK** (Separated Kappa) est la métrique consacrée
de la SCD ; elle combine la qualité de localisation du changement et celle de la
classification sémantique à l'intérieur des zones changées. Plus haut = mieux.

### Positionnement face à l'état de l'art

| Modèle | Params | GMACs | **SeK** | Origine des chiffres |
|---|---:|---:|---:|---|
| Mamba-FCS | 189,54 M | 263,15 | **0,2550** | article Mamba-FCS, table VI |
| MambaSCD-Base | 89,99 M | 211,55 | **0,2292** | article **ChangeMamba** |
| MambaSCD-Tiny | 21,51 M | 73,42 | **0,2208** | article **ChangeMamba** |
| **CSF-Mamba — performance** | **32,58 M** | **58,06** | **0,2484** | mesuré, n = 3 |
| **CSF-Mamba — efficience** | **16,48 M** | **31,42** | **0,2387** | mesuré, n = 4 |

**Lecture en pourcentages du modèle de référence** (100 % = à égalité) :

| | vs Mamba-FCS | vs MambaSCD-Base | vs MambaSCD-Tiny |
|---|---|---|---|
| **efficience** (16,48 M) | 94 % SeK · **9 % params** · **12 % calcul** | **104 % SeK · 18 % params · 15 % calcul** | **108 % SeK · 77 % params · 43 % calcul** |
| **performance** (32,58 M) | **97 % SeK · 17 % params · 22 % calcul** | 108 % SeK · 36 % params · 27 % calcul | 113 % SeK · 152 % params · 79 % calcul |

Les deux cases en gras sont les énoncés à retenir :

- **le point d'efficience bat MambaSCD-Base**, un modèle 5,5 fois plus gros, pour
  **15 % de son calcul** ;
- **le point de performance approche Mamba-FCS à 2,6 % près**, pour **22 % de son
  calcul**. L'objectif initial du stage — le battre — n'est pas atteint, mais
  l'écart s'est réduit de 17,5 % (recette de juillet, SeK 0,2103) à 2,6 %.

### Toutes nos configurations, par ordre de SeK

| Configuration | Params | GMACs | SeK (max) | SeK (finale) | n |
|---|---:|---:|---:|---:|---:|
| **performance** = efficience + encodeur `tiny` + supervision profonde | 32,58 M | 58,06 | **0,2484 ± 0,0017** | 0,2397 | 3 ‡ |
| **efficience** = `lean` + échange temporel + jitter + EMA | **16,48 M** | **31,42** | **0,2387 ± 0,0007** | 0,2325 | 4 |
| `lean` + encodeur `tiny` | 32,58 M | 58,06 | 0,2367 ± 0,0016 | 0,2258 | 4 |
| `lean` + échange temporel + EMA | 16,48 M | 31,42 | 0,2348 ± 0,0017 | 0,2303 | 4 |
| `lean` + échange temporel + jitter | 16,48 M | 31,42 | 0,2331 ± 0,0010 | 0,2239 | 6 ‡ |
| `lean` + échange temporel | 16,48 M | 31,42 | 0,2285 ± 0,0015 | 0,2216 | 4 |
| `lean` + EMA | 16,48 M | 31,42 | 0,2275 ± 0,0009 | 0,2238 | 4 |
| `best` = `nosek` + supervision profonde + LR constant | 20,80 M | 41,30 | 0,2264 ± 0,0020 | 0,2214 | 7 |
| `lean` = `nosek` sans C²S², LR constant | 16,48 M | 31,42 | 0,2230 ± 0,0018 | 0,2164 | 7 |
| `nosek` = recette initiale sans la loss SeK | 20,80 M | 41,30 | 0,2228 ± 0,0019 | — | 7 |
| Recette initiale (juillet) | 20,80 M | 41,30 | 0,2103 ± 0,0105 | — | 8 |

*(‡ deux entraînements interrompus par un blocage du système de fichiers ont été
relancés ; ces deux groupes passeront à 4 et 7 graines. Le σ de 0,0007 du point
d'efficience est estimé sur 3 degrés de liberté et n'est donc pas fiable en
lui-même : tous les intervalles de confiance de ce document utilisent le σ **mis
en commun** sur l'ensemble des configurations, plus prudent.)*

---

## 3. Comment ces chiffres ont été obtenus

Trois précautions de protocole, adoptées après avoir constaté que les
comparaisons du premier mois n'étaient pas fiables. Elles expliquent pourquoi ce
projet conclut moins souvent mais plus solidement.

**1. Plusieurs graines par configuration, jamais un run isolé.** Deux
entraînements identiques ne diffèrant que par leur graine aléatoire donnent des
SeK qui s'écartent de ±0,002 typiquement — et jusqu'à ±0,010 dans le régime
défectueux de juillet. Un écart de 0,005 entre deux configurations à une graine
ne veut donc rien dire. Tous les chiffres ci-dessus sont des **moyennes**.

**2. Deux tests statistiques qui doivent concorder.** Un écart n'est déclaré
« établi » que si le test à variance mise en commun **et** le test de Welch — qui
ne suppose pas les variances égales — franchissent tous deux le seuil de 95 %. Le
second est indispensable ici : certaines configurations sont cinq fois plus
dispersées que d'autres.

**3. Deux métriques qui doivent concorder.** SECOND n'a pas de split de
validation : l'époque retenue est celle du meilleur SeK **sur le test**, ce qui
est légèrement optimiste. Cet optimisme a été **mesuré** — il vaut 0,004 à 0,011
selon la configuration — et chaque conclusion est rejouée sur la **dernière
époque**, insensible à ce biais. Une conclusion qui ne tient que sur une des deux
métriques n'est pas retenue.

Ce troisième point n'est pas théorique : l'augmentation par rotations et jitter
photométrique donnait **+0,0034 « établi »** sur la métrique du maximum et
**−0,0010 non établi** sur l'époque finale. Sans cette vérification, un résultat
faux aurait été publié.

⚠️ **ChangeMamba sélectionne également son époque sur le test** — vérifié dans
leur code (`changedetection/tasks/metadata.py`, tâche `scd` : l'unique *eval
loader*, pourtant nommé « Validation », lit `test_data_name_list`). La
comparaison du tableau ci-dessus est donc **appariée et licite**. L'asymétrie qui
subsiste, et qu'il faut énoncer : nous mesurons notre optimisme, nous ignorons le
leur.

---

## 4. Ce qui produit le résultat

Effets **établis**, mesurés sur l'époque finale (sans biais de sélection) et
confirmés sur la métrique du maximum. Par ordre d'ampleur :

| Levier | Δ SeK | Nature | Coût en inférence |
|---|---:|---|---|
| Retirer la compensation de déséquilibre (sur SECOND) | **+0,032** | loss | aucun |
| Crops 512 plutôt que 256 | **+0,022** | résolution | aucun |
| **Retirer la loss SeK** | **+0,0125** | loss | aucun |
| LR constant sur 200 époques | +0,0098 | optimisation | aucun |
| **Échange temporel T1↔T2 à l'entraînement** | **+0,0096** | données | **aucun** |
| Encodeur `mini` → `tiny` | +0,0094 | capacité | +16,1 M, +26,6 GMACs |
| **EMA des poids** | **+0,0074 à +0,0087** | optimisation | **aucun** |
| DySample plutôt qu'un rééchantillonnage bilinéaire | +0,0075 | architecture | négligeable |
| Supervision profonde des cartes de changement | +0,0056 | loss | aucun |

**Six des neuf leviers ne coûtent rien à l'inférence.** C'est ce qui rend le point
d'efficience possible : il pèse exactement le même nombre de paramètres et de
GMACs que la variante `lean` de départ, pour +0,0157 de SeK.

### Trois enseignements qui dépassent ce projet

**La loss SeK, reprise verbatim de Mamba-FCS, était nuisible.** La retirer
rapporte +0,0125, divise l'écart-type par cinq et supprime un mode de défaillance
qui frappait **une graine sur quatre**. Elle produit des NaN dès que le kappa
passe négatif. C'est le résultat principal du stage.

**Un balayage d'hyperparamètre mené dans un régime défectueux mesure la
défaillance, pas l'hyperparamètre.** La supervision profonde valait +0,0089
au-dessus de la recette initiale, mais seulement +0,0056 une fois la loss retirée :
les trois quarts de son bénéfice n'étaient qu'un rattrapage des dégâts d'un terme
de loss cassé. Plusieurs verdicts de juillet ont dû être rejoués pour cette raison.

**L'EMA des poids est le seul levier additif.** +0,0074 seule, +0,0087 avec
l'échange temporel, +0,0086 avec échange et jitter : le même effet à 0,0007 près
dans trois contextes. Elle n'agit pas sur ce que le modèle apprend, seulement sur
la façon dont ses poids sont lus en fin d'entraînement — donc elle ne recouvre
aucun autre levier. Tous les autres se recouvrent partiellement.

---

## 5. Ce qui ne contribue pas — résultats négatifs

Ces mesures ont autant de valeur que les précédentes, et elles sont inhabituelles
dans cette littérature où les ablations sont rarement répliquées.

Convention : la colonne donne ce que le composant **apporte** — donc l'opposé de
l'effet mesuré en le retirant. Un signe positif signifie « le garder aide ».

| Composant | Ce qu'il apporte | IC 95 % | Verdict |
|---|---:|---|---|
| **C²S² entier** (damier + MCA-SF + scan S6) | +0,0016 | [−0,0004 ; +0,0036] | non détectable |
| MCA-SF seul | +0,0015 | [−0,0024 ; +0,0052] | non détectable |
| CGA (« Change-aware ») | −0,0001 | — | non détectable |
| Branche fréquentielle FFT | −0,0046 | — | non établi |
| Décodeur élargi (+3,5 M, +12 GMACs) | −0,0007 | — | non détectable |
| Rotations 90° + jitter photométrique | −0,0010 à +0,0023 | — | non établi |

*(Cette distinction n'est pas cosmétique : ce README a porté trois semaines une
borne lue à l'envers, qui annonçait pour le C²S² « pas plus de 0,0004 » là où la
valeur juste est 0,0036 — un facteur dix. Δ mesure l'effet du **retrait**, la
contribution vaut **−Δ**, et l'intervalle se retourne avec.)*

**Trois des quatre briques que le nom du modèle revendique ne gagnent pas leur
place.** Ni le C²S² (le *Spatio*), ni la branche FFT (le *Frequency*), ni la CGA
(le *Change-aware*) ne contribuent de façon détectable. Le seul composant
architectural établi est **DySample**, une brique reprise de ChessMamba.

Le C²S² pesait **4,31 M de paramètres et 9,88 GMACs** : le retirer est ce qui a
rendu le point d'efficience possible. Sa contribution réelle est bornée à
+0,0036, compatible avec zéro.

L'explication de l'efficience n'est donc **pas** l'architecture revendiquée, mais :
le backbone VMamba-mini, un décodeur léger à DySample, le retrait d'un terme de
loss nuisible, les crops 512, le LR constant, et l'augmentation par échange
temporel.

---

## 6. Les trois réserves à connaître

**1. Un seul jeu de données porte l'énoncé.** L'efficience n'est démontrée que sur
SECOND. Sur Hi-UCD, le SeK plafonne à 0,054 quel que soit le levier — c'est une
propriété du jeu (1 130 tuiles porteuses de signal sur 12 000), pas du modèle,
mais cela signifie qu'une confirmation sur un troisième jeu manque. Le retrait de
la loss SeK, résultat principal, **ne transfère pas** à Hi-UCD : il y est neutre
et y multiplie l'écart-type par quatre. Le résultat est **spécifique à SECOND**.

**2. La latence n'est pas encore mesurée.** Toute la thèse d'efficience repose sur
les paramètres et les GMACs. Or les noyaux SSM ont un mauvais rapport calcul →
temps, et `grid_sample`, au cœur de DySample, ne compte presque rien en MACs mais
coûte du temps réel. La mesure est **en cours** (`scripts/benchmark_latency.py`).

**3. Les deux configurations retenues comptent 3 et 4 graines**, contre 7 pour les
lignes consolidées. Les relances sont lancées.

---

## 7. Ce qui reste à faire

| | Chantier | État |
|---|---|---|
| C1 | Latence et mémoire crête, face à MambaSCD au même protocole | **en cours** |
| C2 | Évaluer le checkpoint MambaSCD publié avec **notre** code de métriques | **en cours** |
| — | Consolider les deux configurations retenues à 7 graines | **en cours** |
| D | Un troisième jeu de données (Landsat-SCD, rapporté par ChangeMamba) | à décider |
| — | Trancher l'apport du jitter photométrique (+0,0022, non établi) | à faire |
| — | Split de validation propre, pour un chiffre sans biais de sélection | à arbitrer |

C2 réglera la dernière réserve de protocole : le tableau comparatif deviendrait
« évalué avec le même code » au lieu de « d'après les chiffres publiés ». Le
blocage — leur dépôt refactorisé après publication — est levé, l'évaluation tourne
sur le commit contemporain des poids.

---

## 8. L'architecture

Idée directrice : garder les *idées* de Mamba-FCS (qui coûtent ~0 paramètre) et
remplacer sa *machinerie* (qui coûte ses 189 M).

| Bloc | Provenance | Contribue ? |
|---|---|---|
| Encodeur VMamba siamois (`mini` 13,8 M / `tiny` 29,9 M) | ChangeMamba | oui — c'est le socle |
| **DySample** (rééchantillonnage appris) | ChessMamba | **oui, +0,0075 établi** |
| Décodeur SCD partagé + embedding temporel τ | ChessMamba | oui — ÷2 paramètres |
| C²S²-Block (damier + MCA-SF + scan S6) | ChessMamba + CSSM | **non — retiré** |
| Injection FFT2 + CGA résiduelle | Mamba-FCS | **non détectable** |
| Loss composite (CE + Dice + L_sc) | Mamba-FCS + AtrousMamba | oui, **sans** le terme SeK |

Répartition du coût en 512² pour le point d'efficience : convolutions 82 %,
einsum 11 %, scan sélectif 4 %, `grid_sample` 1 %. **Le modèle est dominé par ses
parties convolutionnelles, non par la machinerie SSM** — ce qui nuance
l'étiquette « Mamba ».

**Comptage des GMACs audité** (13 août, revérifié le 10 septembre avec témoin) :
une seule opération non comptée porte des MACs, la FFT2, bornée à 0,3 % du total.
Même convention que ChangeMamba, qui étiquette ses sorties fvcore « GFLOPs »
alors qu'il s'agit de MACs — attention en comparant à la littérature.

---

## 9. Utilisation

**Marche à suivre complète — installation, entraînement, évaluation, pièges :
`RUN.md`.**

```
csf_mamba/
  modules/     chessboard, mca_sf, ssm (+fallback), fusion (FFT/CGA), c2s2, cssm
  backbone/    encoder (ConvEncoder CPU + VMambaEncoder cluster)
  decoders/    dysample, binary (Y_BCD + cartes de changement), semantic (partagé + τ)
  losses/      composite (CE + Dice + SeK + L_sc + Lovász)
  datasets/    second, hi_ucd, transforms (augmentations), oversample
  ema.py       moyenne mobile exponentielle des poids
  model.py     assemblage CSF-Mamba
scripts/       train, evaluate, aggregate_seeds, count_gmacs, benchmark_latency, …
tests/         6 tests de non-régression
```

**Le point qui dé-risque tout : le backend SSM est interchangeable.** `mamba-ssm`
exige une compilation CUDA dont la disponibilité n'est pas garantie ; rien ne
l'impose à l'import. `backend="ref"` donne un scan PyTorch pur qui tourne sur CPU
(lent, pour les tests) ; `backend="mamba"` le noyau rapide ; `backend="auto"`
choisit. Le modèle complet est donc instanciable et différentiable sur un portable
sans GPU.

```bash
pip install torch numpy pillow scipy       # CPU suffit pour les tests
PYTHONPATH=. python tests/test_smoke.py    # formes, forward/backward, budget
```

Le forward VMamba exige le noyau CUDA `selective_scan` et ne tourne pas sur CPU :
les tests locaux utilisent `--encoder conv --backend ref`.

**Reproductibilité.** Chaque `sbatch` affiche au démarrage une ligne `== config`
énumérant ses 24 paramètres, et chaque run écrit un `metrics.csv` à côté de ses
checkpoints. `scripts/aggregate_seeds.py` regroupe les runs par configuration,
calcule les deux tests et les deux métriques, et signale les groupes hétérogènes.
