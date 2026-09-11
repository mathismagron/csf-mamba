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

**Le résultat tient en une phrase.** Évalué avec **le même code de métriques**
sur les mêmes images, notre modèle **dépasse le checkpoint publié de MambaSCD
avec 44 % de ses paramètres, 27 % de son calcul et 1,65 fois moins de temps**.
Face aux chiffres que la littérature cite, il dépasse aussi MambaSCD-Base en
consommant 6,7 fois moins de calcul.

Deux configurations sont retenues, aux deux extrémités du compromis :

| | Paramètres | GMACs | SeK |
|---|---|---|---|
| **Point d'efficience** | **16,48 M** | **31,42** | **0,2390** |
| **Point de performance** | 32,58 M | 58,06 | **0,2485** |

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
| **MambaSCD-Tiny, leur checkpoint publié** | **37,13 M** | **115,44** | **0,2334** | **mesuré par nous, même code** |
| **CSF-Mamba — performance** | **32,58 M** | **58,06** | **0,2485** | mesuré, n = 6 |
| **CSF-Mamba — efficience** | **16,48 M** | **31,42** | **0,2390** | mesuré, n = 7 |

⚠️ **Les deux lignes MambaSCD-Tiny ne décrivent pas le même modèle.** La première
reprend leur table publiée. La seconde est **leur propre checkpoint publié**,
chargé et évalué par nous : 802 tenseurs, 0 manquant, 0 inattendu — aucune
ambiguïté sur l'identité du modèle — et il compte 37,13 M paramètres, pas 21,51.
Nous mesurons son SeK à **0,2334** là où ils annoncent 0,2208.

Nous ne savons pas expliquer cet écart de +0,0126 avec certitude, et **son sens
compte** : notre chaîne leur donne un score *plus élevé* que le leur, ce qui exclut
l'hypothèse d'un modèle mal alimenté par notre code. L'explication la plus
économique est celle déjà établie deux fois — leur dépôt public a évolué après
publication, ce qui a rendu irreproductibles leur `state_dict` puis leur table de
complexité. Les deux chiffres sont donc donnés côte à côte.

**Lecture en pourcentages du modèle de référence** (100 % = à égalité) :

| | vs Mamba-FCS | vs MambaSCD-Base | vs MambaSCD-Tiny |
|---|---|---|---|
| **efficience** (16,48 M) | 94 % SeK · **9 % params** · **12 % calcul** | **104 % SeK · 18 % params · 15 % calcul** | **108 % SeK · 77 % params · 43 % calcul** |
| **performance** (32,58 M) | **97 % SeK · 17 % params · 22 % calcul** | 108 % SeK · 36 % params · 27 % calcul | 113 % SeK · 152 % params · 79 % calcul |

### L'énoncé principal : la comparaison à code identique

C'est la plus rigoureuse dont nous disposions — mêmes images, même split, **même
code de métriques des deux côtés** — et elle ne dépend d'aucun chiffre publié.

| | SeK | Params | GMACs |
|---|---:|---:|---:|
| Leur checkpoint publié, évalué par nous | 0,2334 | 37,13 M | 115,44 |
| **CSF-Mamba — efficience** | **0,2390** | **16,48 M** | **31,42** |
| | **102,4 %** | **44,4 %** | **27,2 %** |

C'est aussi la comparaison **la moins favorable pour nous** : face à leur chiffre
publié, la marge serait de 108 % au lieu de 102,4 %. Nous retenons la plus stricte.

L'IC 95 % de notre moyenne est **[0,2377 ; 0,2403]** sur 7 graines : **sa borne
basse reste au-dessus de leur 0,2334**.

Les autres énoncés, appuyés sur les chiffres que cite la littérature :

- **le point d'efficience bat MambaSCD-Base**, un modèle 5,5 fois plus gros, pour
  **15 % de son calcul** ;
- **le point de performance approche Mamba-FCS à 2,6 % près**, pour **22 % de son
  calcul**. L'objectif initial du stage — le battre — n'est pas atteint, mais
  l'écart s'est réduit de 17,5 % (recette de juillet, SeK 0,2103) à 2,6 %.

### Toutes nos configurations, par ordre de SeK

| Configuration | Params | GMACs | SeK (max) | SeK (finale) | n |
|---|---:|---:|---:|---:|---:|
| **performance** = efficience + encodeur `tiny` + supervision profonde | 32,58 M | 58,06 | **0,2485 ± 0,0011** | 0,2389 | 6 ‡ |
| **efficience** = `lean` + échange temporel + jitter + EMA | **16,48 M** | **31,42** | **0,2390 ± 0,0015** | 0,2330 | 7 |
| `lean` + encodeur `tiny` | 32,58 M | 58,06 | 0,2367 ± 0,0016 | 0,2258 | 4 |
| `lean` + échange temporel + EMA | 16,48 M | 31,42 | 0,2348 ± 0,0017 | 0,2303 | 4 |
| `lean` + échange temporel + jitter | 16,48 M | 31,42 | 0,2331 ± 0,0010 | 0,2239 | 6 ‡ |
| `lean` + échange temporel | 16,48 M | 31,42 | 0,2285 ± 0,0015 | 0,2216 | 4 |
| `lean` + EMA | 16,48 M | 31,42 | 0,2275 ± 0,0009 | 0,2238 | 4 |
| `best` = `nosek` + supervision profonde + LR constant | 20,80 M | 41,30 | 0,2264 ± 0,0020 | 0,2214 | 7 |
| `lean` = `nosek` sans C²S², LR constant | 16,48 M | 31,42 | 0,2230 ± 0,0018 | 0,2164 | 7 |
| `nosek` = recette initiale sans la loss SeK | 20,80 M | 41,30 | 0,2228 ± 0,0019 | — | 7 |
| Recette initiale (juillet) | 20,80 M | 41,30 | 0,2103 ± 0,0105 | — | 8 |

*(‡ une septième graine du point de performance tourne encore. Les deux
configurations retenues sont désormais à 7 et 6 graines, comme les lignes
consolidées.)*

**Les deux chiffres de tête ont résisté au doublement des graines** — c'est le
contrôle le plus utile de cette consolidation. L'efficience passe de 0,2387 (n=4)
à **0,2390** (n=7) et la performance de 0,2484 (n=3) à **0,2485** (n=6) : moins de
quatre dix-millièmes d'écart dans les deux cas.

En revanche l'écart-type affiché à 4 graines, 0,0007, **valait bien le double** :
0,0015 une fois mesuré sur 7. La réserve inscrite ici — un σ estimé sur 3 degrés
de liberté n'est pas fiable en lui-même — était fondée. Tous les intervalles de
confiance de ce document utilisent le σ **mis en commun** sur l'ensemble des
configurations, plus prudent.

### Vitesse et mémoire à l'inférence

Mesuré le 10 septembre sur A100, entrée 512×512, lot de 8 paires, médiane sur
50 itérations après 10 de chauffe.

*Deux précisions numériques sont rapportées.* **fp32** code chaque nombre sur
32 bits (~7 chiffres significatifs) : c'est le format historique et la référence
neutre que n'importe qui peut reproduire. **bf16** n'en utilise que 16 tout en
gardant la **même plage de valeurs** que le fp32 — mêmes bits d'exposant, mais
~3 chiffres significatifs au lieu de 7. Pour un réseau de neurones, ne pas
déborder compte plus que la précision décimale, d'où ce compromis : moitié moins
de mémoire, et des multiplications de matrices bien plus rapides sur les *tensor
cores* de l'A100. C'est le format utilisé à l'entraînement comme à l'inférence,
donc le **régime réel** ; n'en rapporter qu'un des deux cacherait quelque chose.

| Modèle | fp32 | bf16 | Mémoire crête |
|---|---:|---:|---:|
| **CSF-Mamba — efficience** (16,48 M) | **129 ms** · 62 paires/s | **114 ms** · 70 paires/s | 1,9–2,0 Go |
| **CSF-Mamba — performance** (32,58 M) | 144 ms · 56 paires/s | 125 ms · 64 paires/s | 1,9–2,1 Go |
| MambaSCD, variante publiée (19,57 M) ‖ | 252 ms · 32 paires/s | 187 ms · 43 paires/s | 4,0–5,3 Go |

Le point d'efficience est **1,95× plus rapide en fp32, 1,65× en bf16**, pour **2,1
à 2,7× moins de mémoire**. Sur une heure : 253 000 paires traitées contre 154 000.

**Le fait le plus parlant** : notre modèle de **32,58 M** (125 ms) est plus rapide
que le leur de **19,57 M** (187 ms) — 1,50× à taille supérieure.

**L'avantage rétrécit en bf16, et il faut le dire.** Leur architecture, plus
dominée par du calcul dense, profite mieux des *tensor cores* ; la nôtre est
davantage limitée par les accès mémoire (`grid_sample` en tête). La latence ne
suit donc pas les GMACs à l'identique.

‖ Variante à branche MLP désactivée, celle qui correspond à leur table publiée.
Elle mesure **19,57 M**, à 9 % de leurs 21,51 M annoncés — reconstruction établie
en août et reproduite ici à l'identique. Une première mesure, faite contre la
variante à 37,13 M du dépôt actuel, donnait 1,84× en bf16 : elle **surestimait**
l'avantage et a été remplacée.

**Reproductibilité.** Nos propres temps, remesurés deux jours plus tard sur
d'autres nœuds, retombent à **0,05 % près**.

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
est légèrement optimiste. Cet optimisme a été **mesuré** — il vaut 0,0037 à 0,0110
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

**Sept des neuf leviers ne coûtent rien à l'inférence** — seuls l'encodeur
`tiny` et, marginalement, DySample se paient. C'est ce qui rend le point
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
SECOND. *(Chantier en cours : le support de **Landsat-SCD** est implémenté et
testé — voir `documentation/landsat-scd.md`. Il ne manque que le dump.)* Sur Hi-UCD, le SeK plafonne à 0,054 quel que soit le levier — c'est une
propriété du jeu (1 130 tuiles porteuses de signal sur 12 000), pas du modèle,
mais cela signifie qu'une confirmation sur un troisième jeu manque. Le retrait de
la loss SeK, résultat principal, **ne transfère pas** à Hi-UCD : il y est neutre
et y multiplie l'écart-type par quatre. Le résultat est **spécifique à SECOND**.

**2. Leur modèle publié ne correspond pas à leur table publiée.** Trois mesures
indépendantes le montrent : leur checkpoint charge exactement dans un modèle de
**37,13 M**, leur variante à branche MLP coupée mesure **19,57 M** contre 21,51
annoncés, et nous mesurons son SeK à **0,2334** contre 0,2208 annoncé. La cause
est établie depuis le 13 août : **leur dépôt public a évolué après publication**.
**Les auteurs n'ont ni menti ni fait d'erreur** ; c'est le cas ordinaire d'un
dépôt qui continue de vivre, et c'est précisément pourquoi une table publiée est
difficile à revérifier deux ans plus tard.

Ce document donne donc **les deux** : leurs chiffres publiés, que cite la
littérature, et nos mesures à code identique, qui sont les plus rigoureuses et
aussi les moins favorables pour nous.

**3. Il reste une septième graine à venir** sur le point de performance (6 sur 7
terminées). Le point d'efficience est consolidé à 7 graines, et les deux moyennes
ont bougé de moins de 0,0004 en doublant les graines — le chiffre est stable.

---

## 7. Ce qui reste à faire

| | Chantier | État |
|---|---|---|
| C1 | Latence et mémoire crête, face à MambaSCD au même protocole | ✅ **fait** (§2) |
| C2 | Évaluer le checkpoint MambaSCD publié avec **notre** code de métriques | ✅ **fait** (§2) |
| — | Consolider les deux configurations retenues à 7 graines | ✅ **fait** (1 run restant) |
| — | Élucider l'écart de +0,0126 entre leur SeK publié et notre mesure | ouvert |
| D | **Troisième jeu de données — Landsat-SCD** | 🔧 **implémenté**, dump à vérifier |
| — | Trancher l'apport du jitter photométrique (+0,0022, non établi) | à faire |
| — | Split de validation propre | ⏸️ **écarté** (voir ci-dessous) |

**C2 est abouti.** Le checkpoint publié de MambaSCD ne se chargeait plus dans
leur propre dépôt — 558 poids manquants, 590 inattendus, leur décodeur ayant été
renommé après publication. Sorti le commit contemporain des poids dans un
*worktree* git, il se charge **exactement : 802 tenseurs, 0 manquant, 0
inattendu**, et l'évaluation aboutit. Le tableau comparatif porte désormais une
ligne « évaluée avec le même code » et non plus seulement « d'après les chiffres
publiés ».

**Décision sur le split de validation (11 septembre) : on conserve la convention
du domaine.** SECOND ne fournit pas de split de validation, l'époque est donc
choisie sur le test — et **ChangeMamba procède de même**, vérifié dans leur code.
Adopter unilatéralement un split de validation ferait baisser notre chiffre
d'environ 0,005 sans faire bouger le leur : nous paraîtrions moins bons **en étant
plus honnêtes**, pour une raison sans rapport avec le modèle. Le biais est donc
**mesuré et rapporté** (§3 et §6) plutôt que corrigé d'un seul côté. Si la
question devait être rouverte, la bonne forme serait d'**ajouter** une ligne
conservatrice, pas de remplacer le chiffre comparable.

Reste ouvert l'écart de **+0,0126** entre leur SeK publié et notre mesure de leur
propre checkpoint. Il ne remet pas en cause la comparaison — il la rend plus
stricte pour nous — mais son origine mériterait d'être identifiée avant
publication.

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

## 9. Piste annexe close : hybride Mamba / Transformer

⚗️ **Hors du cadre initial**, tenue à l'écart et sans effet sur ce qui précède.
Une attention bi-temporelle jointe aux stages profonds, conçue pour attaquer la
localisation du changement — le goulot identifié au §6. 8 entraînements, 4 graines
par configuration.

**Résultat : elle n'apporte rien, et nuit au petit modèle.**

| Base | Δ sur le maximum | Δ sur l'époque finale |
|---|---:|---:|
| efficience (16,48 → 25,95 M) | −0,0028 *(partiel)* | **−0,0051 (établi)** |
| performance (32,58 → 42,05 M) | −0,0000 | +0,0012 |

Et le critère secondaire, inscrit d'avance, réfute le mécanisme supposé :
l'**IoU du changement baisse** de 0,0064 sur la base efficience — l'attention a
dégradé exactement ce qu'elle devait améliorer.

**C'est le quatrième bloc architectural testé dans ce projet, et le quatrième à
ne rien rapporter.** Face à cela, les leviers de données et d'optimisation
donnent +0,0074 à +0,032. Le motif mérite d'être énoncé : *sur ce jeu de données
et à cette échelle, les ajouts de conception architecturale ne déplacent pas le
SeK ; les données, l'optimisation et la capacité brute d'encodeur le déplacent.*

Réserve principale, écrite sans l'atténuer : les blocs Transformer ont été
entraînés avec le réglage du reste du modèle (LR constant, pas de warmup dédié),
alors qu'ils y sont réputés plus sensibles. C'est la seule objection qui pourrait
renverser ce résultat, et elle n'a pas été testée.

Le code vit dans `csf_mamba/experimental/`, se lance par
`scripts/train_hybrid.sbatch`, tague ses runs `hyb-*` et se documente en détail
dans **`documentation/hybride.md`**. Sans `--attn-stages`, le modèle de référence
est inchangé à l'octet près — `tests/test_hybride.py` le vérifie.

---

## 10. Utilisation

**Marche à suivre complète — installation, entraînement, évaluation, pièges :
`RUN.md`.**

```
csf_mamba/
  modules/     chessboard, mca_sf, ssm (+fallback), fusion (FFT/CGA), c2s2, cssm
  backbone/    encoder (ConvEncoder CPU + VMambaEncoder cluster)
  decoders/    dysample, binary (Y_BCD + cartes de changement), semantic (partagé + τ)
  losses/      composite (CE + Dice + SeK + L_sc + Lovász)
  datasets/    second, hi_ucd, landsat_scd, transforms (augmentations), oversample
  ema.py       moyenne mobile exponentielle des poids
  model.py     assemblage CSF-Mamba
scripts/       train, evaluate, aggregate_seeds, count_gmacs, benchmark_latency, …
tests/         8 fichiers de tests de non-régression
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
