# Plan de recherche — architecture Mamba efficiente pour la détection sémantique de changements

### Document de synthèse pour relecture et avis

> Objet : proposer une architecture SCD (Semantic Change Detection) **efficiente** dérivée de Mamba-FCS, en intégrant les idées de quatre travaux récents (CSSM, CDMamba, AtrousMamba, ChessMamba). Ce document résume l'analyse, le raisonnement de conception, l'architecture retenue, la spécification technique, le plan d'ablation, et les points ouverts qui appellent votre avis.
>
> Nom de travail de l'architecture : **CSF-Mamba** (*Change-aware Spatio-Frequency Mamba*).

> **Mises à jour depuis la première version** (résolution de trois points ouverts) :
> 1. **Gate CGA tranché** : le format d'annotation de Hi-UCD (sémantique pleine-scène) impose le gate **résiduel** `X·(1+σ(CM))` — voir §5.1.
> 2. **Faisabilité de la variante L1 relevée** : le code CSSM est en PyTorch pur (pas de kernel CUDA) → risque d'implémentation **faible** (seul coût : la vitesse) — voir §6 et §8.
> 3. **Décision d'implémentation** : **forker Mamba-FCS** (code public, bâti sur ChangeMamba) plutôt que repartir de zéro — voir §12 et §14.

---

## 1. Contexte et objectif

Le stage vise une architecture Mamba **spécialisée SCD multi-classe** et **efficiente** (cible : ~15M paramètres, ordre de grandeur), évaluée sur SECOND et surtout **Hi-UCD** (grand dataset SCD urbain à 0,1 m, 9 classes, 48 transitions, 3 dates), qu'aucune méthode Mamba-SCD n'utilise à ce jour.

Rappel des trois tâches du domaine :
- **BCD** (binary change detection) : *où* y a-t-il un changement.
- **SCD** (semantic change detection) : *où* ET *de quoi vers quoi* (« from-to »). Métrique de référence : **SeK** (Separated Kappa), qui mesure l'accord sémantique **uniquement dans les régions changées**.
- **BDA** (building damage assessment) : cas particulier multi-classe du SCD (niveaux de dommage).

---

## 2. Le problème de départ : Mamba-FCS va dans le mauvais sens pour l'efficience

Mamba-FCS est l'état de l'art SCD, mais il pèse **189,54M paramètres** — soit **~12× la cible**. Partir de Mamba-FCS et « lui ajouter des modules » aboutirait à ~200M. Il faut donc **renverser la logique** :

> **Garder les *idées* de Mamba-FCS (qui coûtent quasiment 0 paramètre) et remplacer sa *machinerie* (qui coûte tout le budget).**

Décomposition du budget de Mamba-FCS et gains d'ablation (sur SECOND, en partant du Base Model à SeK 23,70) :

| Composant | Coût en paramètres | Gain rapporté (SeK) |
|---|---|---|
| Encodeur VMamba-**Base** | ~90M | — (backbone) |
| 3 décodeurs (VSS + CBAM upsampling à C=1024) | ~100M | — |
| **CGA** (change-guided attention) | ≈ 0 (sigmoïde + produit) | **+1,43** |
| **Loss SeK** | **0** | +0,79 |
| **Branche FFT2** | 0 (la FFT n'a pas de poids appris) | +1,47 |

Conclusion : **les trois contributions revendiquées de Mamba-FCS coûtent ~0 paramètre ; les 189M sont dans le backbone et les décodeurs.** On peut donc conserver les contributions et supprimer ~90 % du modèle.

---

## 3. Synthèse des six articles étudiés

### 3.1 ChangeMamba (Chen et al., IEEE TGRS 2024) — la fondation
Premier Mamba appliqué au CD. Encodeur siamois **VMamba** (cross-scan 4 directions, complexité O(N)). *Change decoder* avec **trois mécanismes spatio-temporels** (sequential / cross / parallel) combinés dans un bloc STSS (3 VSS blocks en parallèle). Ablation : le *cross* seul atteint presque le score des trois combinés (F1 82,24 vs 82,83 sur SYSU) → **un seul mécanisme suffit presque**, argument clé pour un modèle léger. Variantes Tiny 17,13M / Small 49,94M / Base 84,70M. Sur SECOND : MambaSCD-Base SeK 22,92 ; **Tiny 22,08** (l'écart Tiny→Base ≈ −0,8 SeK est notre seule calibration fiable, voir §10).

### 3.2 Mamba-FCS (Wijenayake et al., IEEE JSTARS 2025) — l'état de l'art SCD, mais lourd
Part de MambaSCD et corrige trois faiblesses :
1. **Joint Spatio-Frequency Fusion** : branche FFT2 (features fréquentielles log-amplitude) pour accentuer les bords et atténuer les artefacts d'illumination.
2. **Change-Guided Attention (CGA)** : `X̂ = X ⊙ σ(CM)`, injecte la carte de changement du décodeur binaire dans les décodeurs sémantiques → couple BCD et SCD.
3. **SeK-inspired loss** : rend la métrique SeK différentiable via une *soft confusion matrix* de probabilités softmax → optimise directement le déséquilibre de classes.
Résultat SECOND : **SeK 25,50 / mIoU 74,07 / Fscd 65,78 / OA 88,62** — meilleur du domaine. **Coût : 189,54M params, 263,15 GFLOPs.**

### 3.3 CSSM (Ghazaei & Aptoula, IEEE GRSL 2025) — l'ultra-léger change-aware
Modifie la **récurrence du SSM elle-même** pour la rendre bi-temporelle :
```
h_t = A·h_{t−1} + ‖ B'_t·z^post_t − B_t·z^pre_t ‖₁
y^pre_t  = C_t·h_t  + D_t·z^pre_t
y^post_t = C'_t·h_t + D'_t·z^post_t
```
Le changement n'est plus calculé *après* le SSM — il **pilote l'état caché** via la distance L1 entre les deux projections temporelles. Ablation : **L1 > L2 > cosinus > Chebyshev** ; 5 blocs CSSM optimal. Résultat : **4,34M params, 5,10 GFLOPs** (21× moins que ChangeMamba), meilleur IoU sur LEVIR-CD+ (86,63). **Limite : BCD uniquement ; jamais testé en multi-classe.**

### 3.4 CDMamba (Zhang et al., 2025) — le local compte
Thèse : Mamba est aveugle au local, rédhibitoire en prédiction dense.
- **SRCM** : une 3ᵉ branche conv2D dans le bloc Mamba pour réinjecter le local.
- **AGLGF** : interaction bi-temporelle guidée (global + local, gating softmax).
Deux ablations décisives : (a) **+SRCM = +5,52 F1** sur WHU-CD (le plus gros gain de module de toute la littérature étudiée) ; (b) **l'AGLGF n'aide qu'aux stages 1–2 (haute résolution) et dégrade aux stages 3–4** — justification directe pour placer nos modules lourds uniquement en haute résolution. 11,90M params.

### 3.5 AtrousMamba (Wang et al., Information Fusion 2025) — le scanning local/global
**AWVSS** : au lieu du cross-scan, on partitionne l'image en 4 groupes de fenêtres dilatées (rates 2/5/7/9), traités en parallèle par 4 modules S6, puis recalibration canal (SE). Ablation : atrous scan vs cross-scan → **IoU +2,0** sur SYSU. Fait BCD **et** SCD. Apporte aussi une **loss de cohérence sémantique** `L_sc` (contrastive/cosinus) que nous reprenons. ⚠️ **Protocole SECOND différent** (crops 256, 11 872 patchs) → chiffres non comparables à Mamba-FCS.

### 3.6 ChessMamba (Ding et al., 2025) — le plus proche de notre cible
- **Chessboard interleaving + snake scan** : masque damier binaire M, `X^a = M⊙F1+(1−M)⊙F2`, `X^b` complémentaire, entrelacés en une **séquence 1D unique** parcourue en serpent. Chaque pixel est entouré de voisins *cross-phase* → comparaison bi-temporelle directe, topologie 2D préservée. **Coût : 0 paramètre**, SSM standard.
- **MCA-SF** : pré-agrégation locale par convolutions depthwise multi-dilatées, avec un noyau qui pour d=1 n'a de poids qu'au **centre et aux coins** — soit exactement les voisins de même phase dans le damier.
- Décodeur : DySample (upsampling appris) au lieu d'interpolation.
Ablation Levir : baseline IoU 83,26 → +chessboard 83,97 → +MCA-SSM 84,30 → **les deux : 85,20**, gain de rappel **+2,09**. Robustesse au **décalage spatial** : −10,84 % IoU à 16 px vs −15,77 % pour la baseline. Résultat SECOND : **SeK 24,80 / mIoU 73,62 / OA 90,08 en 22,65M et 60,47 GFLOPs.**

---

## 4. L'état de l'art réel sur SECOND (référence de comparaison)

| Méthode | Params (M) | GFLOPs | OA | Fscd | mIoU | **SeK** |
|---|---|---|---|---|---|---|
| ChangeMamba (MambaSCD) | 89,99 | 211,6 | 88,12 | 64,03 | 73,68 | 24,11 |
| AtrousMamba (AWMambaSCD_S)* | 54,63 | 36,9 | — | 64,24 | 73,66 | 24,95 |
| ChessMamba | **22,65** | 60,5 | **90,08** | 65,32 | 73,62 | 24,80 |
| Mamba-FCS | 189,54 | 263,2 | 88,62 | **65,78** | **74,07** | **25,50** |

\* protocole différent (non directement comparable).

**Lecture** : Mamba-FCS mène sur SeK, mais avec **8× plus de paramètres** que ChessMamba pour +0,70 SeK. Une part importante de son avance vient probablement de la **capacité brute**, pas de ses modules. Il existe donc un **trou dans le plan (paramètres, SeK)** que personne n'occupe : c'est notre créneau.

---

## 5. Architecture proposée : CSF-Mamba

![Architecture globale CSF-Mamba](figures/csf_mamba_overview.svg)

Trois blocs :
1. **Encodeur** : VMamba-Tiny siamois pré-entraîné ImageNet. N'invente rien — c'est le budget paramètres. On s'attend à ~−0,8 SeK vs Base (calibration ChangeMamba, §10), que les modules doivent regagner.
2. **Décodeur binaire (BCD)** : lieu de la contribution. À chaque stage, un **C²S²-Block** fusionne la paire bi-temporelle ; aux **stages 1–2 (haute résolution)** uniquement, on ajoute la branche **FFT2** + l'interaction bi-temporelle ; upsampling **DySample + depthwise 3×3**. Sorties : `Y_BCD` et les cartes de changement intermédiaires `{CM_i}`.
3. **Décodeurs sémantiques (SCD)** : **un seul décodeur à poids partagés** + embedding temporel `τ₁/τ₂` pour distinguer les dates (÷2 sur les paramètres vs deux décodeurs indépendants). Chaque stage est guidé par `CM_i` via une **CGA résiduelle** `X·(1+σ(CM))`.

**Loss composite** (0 paramètre ajouté) :
```
L = CE_BCD + ½(CE_T1 + CE_T2) + λ₁·mIoU + λ₂·SeK + λ₃·L_sc
```
Le terme `L_sc` (cohérence sémantique, repris d'AtrousMamba/ChessMamba/Bi-SRNet) comble un trou de Mamba-FCS : la loss SeK ne supervise que *l'intérieur* des zones changées (elle exclut le no-change par construction) ; **rien ne supervise la cohérence sur les pixels inchangés.** `L_sc` s'en charge, gratuitement.

Provenance des briques : C²S²-Block = **ChessMamba + CSSM** ; FFT2, CGA, loss SeK = **Mamba-FCS** ; `L_sc` = **AtrousMamba** ; DySample + décodeur partagé + embedding temporel = **ChessMamba**.

### 5.1 Traitement des données Hi-UCD (résout le choix de gate CGA)

Format d'annotation Hi-UCD : chaque masque est un **PNG à 3 canaux**, 512×512 :
- **canal 1** = prédiction sémantique T1 (pleine-scène, 9 classes utiles),
- **canal 2** = prédiction sémantique T2 (pleine-scène, 9 classes utiles),
- **canal 3** = prédiction de changement (`unchanged` / `change`).

Classes sémantiques (index → classe) : 0 unlabeled, 1 water, 2 grass, 3 building, 4 green house, 5 road, 6 bridge, 7 others, 8 bare land, 9 woodland. Classes de changement : 0 unlabeled, 1 unchanged, 2 change. **Les index sémantiques et de changement doivent être décalés de −1** pour correspondre aux valeurs prédites (`unlabeled` devient donc l'index à ignorer).

**Implication décisive** — contrairement à SECOND (où la sémantique n'est étiquetée que dans les zones changées), **Hi-UCD étiquette la sémantique partout, à T1 comme à T2**, avec un canal de changement *séparé*. Il faut donc prédire une sémantique correcte même dans les régions inchangées. Un gate CGA **multiplicatif** `X⊙σ(CM)` y annulerait le signal → **le gate résiduel `X·(1+σ(CM))` est obligatoire** (il amplifie les zones changées sans détruire les zones stables). Ce point est désormais tranché.

Conséquences pour la loss :
- **CE sémantique** sur les deux cartes pleine-scène, avec `ignore_index` pour `unlabeled` (index 0 → −1 après décalage, ou mappé à 255).
- **SeK** : on construit la carte SCD en masquant la sémantique par le canal 3 (comme Mamba-FCS) ; la SeK-loss se transpose sans modification.
- **`L_sc`** : le masque « inchangé » vient directement du canal 3 — plus propre que sur SECOND où il faut le dériver.
- **Déséquilibre réel** (green house, bridge, others, bare land rares) → SeK + `L_sc` bien motivés sur ce dataset.

---

## 6. La décision de conception centrale : L1 (CSSM) vs chessboard (ChessMamba)

C'est le point le plus délicat, et il faut être explicite car les deux mécanismes visent le **même** objectif (mettre le contraste bi-temporel au bon endroit) par des moyens **incompatibles**.

- **CSSM-L1** veut deux flux propres et alignés `F1`, `F2` : le « pré » passe toujours par la projection `B`, le « post » toujours par `B'`. C'est cette **asymétrie apprise B/B'** qui porte la sensibilité au changement.
- **Chessboard** fusionne les deux dates en composites damier `X^a`, `X^b` et scanne une séquence unique.

**Piste de réconciliation envisagée puis écartée.** Comme les composites sont complémentaires, on a exactement `‖X^a − X^b‖₁ = ‖F1 − F2‖₁` en tout point — ce qui semble marier les deux. **Mais cela casse la sémantique de CSSM** : dans le damier, aux cases blanches `F1` passe par `B` et `F2` par `B'`, aux cases noires c'est **inversé**. Les projections asymétriques pré/post reçoivent donc des dates permutées une case sur deux → le signal de changement devient spatialement incohérent. Le mariage est correct sur le papier, faux en pratique.

**Décision : ne pas les combiner dans le même scan. Les faire concourir.**

| | Chessboard + MCA-SF | CSSM-L1 |
|---|---|---|
| Prouvé en SCD **multi-classe** | ✅ (ChessMamba, SECOND 24,80 SeK, 22,65M) | ❌ (BCD seulement) |
| SSM standard | ✅ `mamba_ssm` off-the-shelf | ⚠️ récurrence custom, **mais PyTorch pur** (voir ci-dessous) |
| Partenaire local co-conçu | ✅ (noyau centre-coins de MCA-SF *fait pour* le damier) | ➖ (locality par depthwise ordinaire) |
| Nouveauté | moyenne | forte |

**Recommandation :**
- **Cœur du C²S²-Block = chessboard interleaving + MCA-SF + S6 standard.** Seul mécanisme *change-aware prouvé en multi-classe*, sans risque de kernel, immédiatement buildable.
- **CSSM-L1 = branche d'ablation**, implémentée sur flux propres `F1/F2` (sans damier), A/B-testée contre le cœur en multi-classe (ablation §9, ligne 9). S'il gagne → on le promeut, la nouveauté monte d'un cran. S'il perd (probable) → résultat négatif propre et publiable, le cœur tient.

**Faisabilité de la variante L1 — mise à jour.** Le dépôt public de CSSM (accepté IEEE GRSL, code publié nov. 2025) est **100 % Python** et ne dépend **ni de `mamba_ssm`, ni de `causal-conv1d`, ni de Triton, ni d'une chaîne CUDA**. La récurrence L1 est donc implémentée en **PyTorch pur** (boucle de scan manuelle). Conséquence :
- Le risque « kernel CUDA à écrire » **disparaît** : le code de sélection L1 est lisible et directement portable dans le fork.
- **Le seul coût réel est la vitesse** : un scan séquentiel Python est lent. Acceptable pour le petit modèle BCD de CSSM (256²) ; potentiel goulot pour notre SCD sur Hi-UCD (512²). Si — et seulement si — la variante L1 prouve sa valeur en ablation, on portera le terme L1 dans le kernel `mamba_ssm` à ce moment-là.

**Conséquence de dé-risquage majeure** : avec le S6 standard au cœur ET la variante L1 garantie implémentable (PyTorch pur), **plus aucun risque kernel ne bloque le projet.** On a un modèle complet et entraînable sans jamais toucher à un kernel CUDA.

![Flux tensoriel du C²S²-Block](figures/c2s2_block.svg)

---

## 7. Spécification tensorielle (vérification de compatibilité)

Flux de bout en bout, pour une entrée 256×256 (le pipeline Hi-UCD est en 512×512 — mêmes règles, résolutions doublées) :

| Étape | Entrée | Sortie | Note |
|---|---|---|---|
| Encodeur stages 1→4 | `(B,3,256,256)` | `X_i ∈ (B,C_i,H_i,W_i)`, C=96/192/384/768, H=64/32/16/8 | par date, poids partagés |
| C²S²-Block (par stage) | `X^T1_i, X^T2_i ∈ (B,C_i,H_i,W_i)` | `Z_i ∈ (B,C_i,H_i,W_i)` | la fusion bi-temporelle **est** ce bloc |
| Injection FFT (stages 1–2) | `Z_i` + `\|Ff1−Ff2\|` | `(B,C_i,H_i,W_i)` | concat 2C → conv 1×1 → C |
| DySample ↑ + addition | `(B,C_i,H_i,W_i)` | `(B,C_{i-1},H_{i-1},W_{i-1})` | fusion top-down |
| Sorties BCD | dernier stage | `Y_BCD ∈ (B,2,256,256)`, `{CM_i}` | CM_i extrait à chaque stage |
| CGA (décodeur SCD) | `X^Tj_i`, `CM_i` | `X̂ = X·(1+σ(CM_i))` | gate **résiduel** (confirmé Hi-UCD, §5.1) |
| Sorties SCD | décodeur partagé + τ_j | `Y^Tj ∈ (B,N,256,256)` | N classes (9 utiles sur Hi-UCD) |

**Point de clarification que la spec révèle** : le C²S²-Block **remplace entièrement** le mécanisme de fusion « 5C » de Mamba-FCS. Il n'y a pas de concaténation 5C séparée *puis* un bloc — le bloc consomme directement la paire `(X^T1, X^T2)`. La branche FFT vient *après* le scan, aux stages 1–2 seulement, via `[Z_i, |Ff1−Ff2|] → conv 1×1`. C'est la « fusion 3C » d'origine, correctement recâblée en 2C→C.

Détail interne du C²S²-Block (cœur par défaut) :
1. Composites damier : `X^a = M⊙F1 + (1−M)⊙F2`, `X^b` complémentaire — `(B,C,H,W)` chacun.
2. MCA-SF : agrégation depthwise multi-dilatée (noyau centre-coins pour d=1, plein pour d=3,5), spatial préservé.
3. Interleave `X^a, X^b` → `(B,C,H,2W)`, ordre serpent.
4. Scan **S6 standard** (`mamba_ssm`).
5. Dé-interleave → fusion additive → `Z ∈ (B,C,H,W)`.

Variante d'ablation (remplace 3–5) : deux flux `F1,F2` → récurrence CSSM-L1 → double sortie → recombinaison.

---

## 8. Checklist de compatibilité des interfaces

| Interface | Risque | Résolution |
|---|---|---|
| Canaux encodeur → décodeur (96/192/384/768) | faible | conv 1×1 d'alignement, standard |
| Masque damier `M (H,W)` broadcast sur `(B,C,H,W)` | faible | `H,W` pairs (512→16..128 ✓) ; sinon padding à ×32 |
| MCA-SF depthwise dilaté | faible | `groups=C`, `padding=dilation`, spatial préservé |
| Interleave → 2W + S6 | faible | `mamba_ssm` off-the-shelf, dé-interleave après |
| **Récurrence L1 (variante d'ablation)** | **faible** *(révisé)* | code CSSM en **PyTorch pur**, portable ; seul coût = **vitesse** (scan séquentiel) |
| Branche FFT2 | faible | `torch.fft.fft2`, `abs`→`log`, sortie réelle, channel-wise |
| Injection fréquentielle (stages 1–2) | faible | `[Z, \|Ff1−Ff2\|]` 2C→C, conv 1×1 |
| **CGA gate résiduel** | **faible** *(confirmé)* | Hi-UCD = sémantique pleine-scène → `X·(1+σ(CM))` obligatoire (§5.1) |
| Décodeurs partagés + τ | faible | ajouter embedding temporel `τ₁/τ₂` en entrée |
| Loss `L_sc` (cosinus) | faible | 0 paramètre, masque « inchangé » = canal 3 de Hi-UCD |

**Aucune interface n'est en risque élevé.** Les deux points auparavant incertains (récurrence L1, gate CGA) sont désormais résolus.

---

## 9. Plan d'ablation

Baseline = Mamba-FCS reproduit dans notre pipeline. Sur SECOND **et** Hi-UCD, une ligne par ajout (ablation **additive**) :

1. `Mamba-FCS (Base)` — 189M — référence
2. `+ encodeur Tiny` — mesure la part de l'avance qui vient de la simple **capacité** (ablation jamais faite dans la littérature, et gênante pour les auteurs de Mamba-FCS)
3. `+ décodeurs sémantiques partagés + embedding temporel`
4. `+ chessboard/snake` (remplace le VSS bi-temporel)
5. `+ MCA-SF` (locality)
6. `+ récurrence CSSM-L1` ← **le cœur candidat**
7. `+ fusion résolution-adaptative` (FFT/interaction stages 1–2 seulement)
8. `+ L_sc`
9. **Choix de distance dans la récurrence** : L1 / L2 / cosinus **en multi-classe** (rejoue l'ablation CSSM Table 3, qu'ils n'ont faite qu'en binaire) — **potentiellement l'invalidation du cœur, à faire tôt**
10. **Robustesse au décalage spatial** (ε = 4/8/16 px, protocole ChessMamba)

**Complément méthodologique important** — ablations *leave-one-out*. L'ablation additive confond « ce module aide-t-il ? » et « ce module aide-t-il *en présence des autres* ? ». Comme les trois idées centrales (chessboard, MCA-SF, L1) visent le même objectif, l'ordre d'ajout fausse la lecture (le dernier ajouté paraît toujours faible). **Ajouter, pour les trois composants du C²S²-Block, des ablations « modèle complet moins un module »** — seul moyen de savoir si les trois sont nécessaires ou si l'un est un passager clandestin.

**Sur le test de robustesse au shift (ligne 10)** : Hi-UCD est à 0,1 m sur 3 dates, le mésalignement y est un problème réel, et **aucun papier SCD ne rapporte cette métrique**. C'est un axe d'évaluation à la fois pertinent et inédit.

---

## 10. Avertissement méthodologique : protocole unique obligatoire

Les protocoles des articles diffèrent (splits SECOND, tailles de crop, nombre d'itérations, seeds). **Preuve chiffrée de l'ampleur du problème** : ChangeMamba obtient **24,11 SeK** quand Mamba-FCS le reproduit, mais **22,83–22,92** dans son propre papier — un écart de ~1,2 SeK sur la même méthode, du seul fait du protocole. C'est plus grand que les gains qu'on vise.

**Conséquence** : figer un **protocole unique** (split, crops, itérations, seed=42, métriques) et **ré-entraîner toutes les baselines dedans**. Sans cela, aucun tableau comparatif n'a de sens. *Le fork de Mamba-FCS (§12) résout ce point structurellement : baseline et modèle proposé partagent le même harnais.*

**Calibration Tiny vs Base** : la seule estimation transférable est l'écart intra-famille ChangeMamba sur SECOND — **Tiny 22,08 vs Base 22,92 SeK, soit ≈ −0,8**. C'est le déficit approximatif que le passage à VMamba-Tiny nous impose *a priori*, et que nos modules doivent regagner. **Attention** : ce n'est PAS une addition naïve « −0,8 puis +X » — les gains d'ablation des autres papiers ont été mesurés sur d'autres backbones/protocoles/datasets (le +5,52 F1 du SRCM est sur un encodeur maison en BCD ; le +1,94 IoU de ChessMamba est en BCD sur Levir). **Nos gains sont inconnus jusqu'à notre propre ablation.**

---

## 11. Risques et points ouverts (honnêteté sur l'incertitude)

1. **Compatibilité MCA-SF ↔ snake scan.** Le noyau centre-coins de MCA-SF est co-conçu avec le damier de ChessMamba, mais sur *leur* sérialisation exacte. Qu'il reste pertinent avec notre agencement précis est à **valider empiriquement** dès que la baseline tourne. Premier point de vérification. *(reste le principal inconnu)*
2. **CSSM-L1 en multi-classe.** Une norme L1 scalaire sur l'écart de projection discrimine changé/inchangé (prouvé en BCD) ; qu'elle discrimine *le type de transition* en multi-classe est non prouvé. L'ablation ligne 9 en décide — d'où l'importance de la faire tôt.
3. **Vitesse de la variante L1.** *(mis à jour)* Le code CSSM étant en PyTorch pur, l'implémentation ne pose aucun risque, mais le scan séquentiel peut ralentir l'entraînement sur Hi-UCD (512²). À surveiller ; portage kernel seulement si la variante s'avère gagnante.
4. **Annotation Hi-UCD.** *(résolu, §5.1)* Sémantique pleine-scène → gate CGA résiduel, `ignore_index` sur `unlabeled`, masque `L_sc` = canal 3.
5. **Tenue de la cible 15M.** L'encodeur VMamba-Tiny seul pèse ~14M. La Piste A (Tiny + décodeurs allégés, ~18–22M) satisfait un « ordre de grandeur 15M » mais dépasse la barre stricte. Une Piste B (encodeur Mamba étroit maison C=64/128/256/512 + pré-entraînement sur FSC-180k, ~6–10M) tiendrait la barre — mais c'est un **bonus optionnel**, pas un prérequis. Recommandation : faire la Piste A solidement d'abord.

---

## 12. Décision d'implémentation et prérequis de dé-risquage

### 12.1 Forker Mamba-FCS plutôt que repartir de zéro
Le code de Mamba-FCS est **public** (dépôt Buddhi19/MambaFCS ; poids pré-entraînés sur Hugging Face ; notebook d'évaluation fourni) et il est **bâti sur le dépôt de ChangeMamba** (ChenHongruixuan/MambaCD), qui fournit VMamba, les configs VMamba-Tiny, et les scripts train/infer de MambaBCD/SCD/BDA. **Recommandation : forker Mamba-FCS et l'étendre avec des modules propres.**

Raisons (contraintes, pas préférences) :
1. On doit **reproduire Mamba-FCS comme baseline** de toute façon → le fork rend cette reproduction quasi gratuite (les poids HF permettent de vérifier SeK 25,50 sur SECOND sans réentraîner).
2. On **conserve les briques réutilisées** — SeK-loss (non triviale à réimplémenter correctement), CGA, branche FFT2 — vérifiées plutôt que réécrites.
3. **Même harnais** pour baseline et modèle → comparaison à protocole identique (résout §10 structurellement).

Ce qu'on **remplace** (modules neufs et isolés) : encodeur Base→Tiny (changement de config, le yaml `vssm_tiny` existe), bloc de fusion/décodeur (le C²S²-Block), décodeur sémantique partagé + embedding temporel, upsampling DySample. Le **module L1** est levé depuis le dépôt CSSM.

**Garde-fou** : avant de s'engager, cloner Mamba-FCS, le faire tourner, reproduire **un seul chiffre** (SeK sur SECOND avec les poids HF). Si ça tourne proprement → fork-and-extend. Si le code est inutilisable → prendre ChangeMamba (plus propre) comme squelette, mais **récupérer verbatim la SeK-loss et les métriques** de Mamba-FCS pour garantir la comparabilité. Vérifier les licences si publication prévue.

### 12.2 Prérequis, par ordre de risque décroissant
1. **Reproduire Mamba-FCS (189M)** et matcher son SeK 25,50 sur SECOND (via le fork + poids HF). Prérequis absolu.
2. **Figer le protocole unique** (split SECOND 2968/1694, crops, itérations, seed=42, métriques) dans le fork ; ré-entraîner ChangeMamba + ChessMamba dedans.
3. **Intégrer Hi-UCD** au pipeline (dataloader 3-canaux, décalage d'index −1, `ignore_index`, dérivation carte SCD depuis le canal 3).
4. **Baseline C²S²-cœur qui tourne** (VMamba-Tiny + chessboard + S6 standard, sans FFT/CGA/SeK) pour un point de départ mesurable.
5. **Lire `method/` du dépôt CSSM** pour extraire les 4 détails du portage L1 : (a) projections `B`/`B'` séparées ? (b) scan en boucle Python ou vectorisé ? (c) sorties `y^pre`/`y^post` via `C/C'`, `D/D'` ? (d) normalisation d'entrée (RMSNorm).

---

## 13. Questions ouvertes pour discussion

1. **Cible de paramètres** : viser strictement < 15M (implique la Piste B, encodeur maison + FSC-180k) ou accepter ~18–22M (Piste A, VMamba-Tiny pré-entraîné) comme « ordre de grandeur » satisfaisant ?
2. **Cœur du bloc** : valider le choix chessboard-comme-cœur / L1-comme-ablation, ou préférez-vous investir d'emblée dans la récurrence L1 comme contribution principale (plus risqué, plus original) ?
3. **Dataset principal** : Hi-UCD comme cible d'évaluation prioritaire (inédit) ou consolider d'abord sur SECOND + Landsat-SCD (comparaison directe à Mamba-FCS/ChessMamba) ?
4. **Périmètre** : rester strictement SCD, ou prévoir dès le départ la compatibilité BCD/BDA (comme ChangeMamba et ChessMamba) pour élargir la portée du travail ?

---

## 14. Ressources et code (dépôts publics)

| Élément | Lien | Usage |
|---|---|---|
| Mamba-FCS (code) | github.com/Buddhi19/MambaFCS | **fork de base**, baseline SCD |
| Mamba-FCS (poids) | huggingface.co/buddhi19/MambaFCS | vérifier SeK 25,50 sans réentraîner |
| ChangeMamba (code) | github.com/ChenHongruixuan/MambaCD | socle VMamba, configs Tiny, MambaSCD |
| CSSM (code) | github.com/Elman295/CSSM | lever le module de récurrence L1 (PyTorch pur) |

*Statut de publication vérifié : Mamba-FCS (IEEE JSTARS 2026, code + poids en ligne), CSSM (accepté IEEE GRSL, code publié nov. 2025), ChangeMamba (IEEE TGRS 2024, code de référence).*

---

*Document préparé comme support de discussion. Les chiffres proviennent des six articles cités ; les écarts de protocole signalés au §10 impliquent que toute comparaison finale devra être re-mesurée dans un pipeline unifié.*
