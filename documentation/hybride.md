# Piste hybride Mamba / Transformer

⚗️ **Hors du cadre initial du stage.** Ce document est séparé du journal de bord,
et le code correspondant vit dans `csf_mamba/experimental/`. Rien n'est actif par
défaut : sans `--attn-stages`, le modèle de référence construit exactement les
mêmes tenseurs qu'avant l'ajout de cette piste, et `tests/test_hybride.py` le
vérifie.

**Objectif** : dépasser le meilleur SeK publié sur SECOND — Mamba-FCS, **0,2550**.

---

## 1. Pourquoi une brique d'attention, et pourquoi celle-là

Trois mesures du projet dictent la conception, plutôt qu'une intuition.

**Le goulot est la localisation, pas la sémantique.** Diagnostic du 14 août : IoU
du changement **0,577**, mais **0,864** de justesse sémantique *à l'intérieur* des
zones détectées. Et `SeK = κ·exp(IoU_fg)/e` : tout point de SeK passe par l'IoU.
Une brique qui n'améliore pas la localisation ne sert à rien ici.

**La machinerie SSM ne pèse que 4 % du calcul**, contre 82 % pour les convolutions
(comptage du 10 septembre). Le raisonnement global du modèle est bien plus mince
que son étiquette « Mamba » ne le suggère — il y a de la place.

**Un scan SSM est ordonné.** Deux positions éloignées dans l'ordre de balayage
n'interagissent qu'à travers un état compressé. Or détecter un changement, c'est
**apparier** une zone à T1 avec la même zone à T2 : une opération par paires, sans
ordre. C'est exactement ce que fait l'attention.

## 2. Ce que fait la littérature

| Travail | Enseignement retenu |
|---|---|
| **MambaVision** (CVPR 2025) | Placer quelques blocs d'attention **aux derniers stages** est le placement le plus rentable — testé contre « au début », « au milieu », « tous les *l* blocs ». Le hybride y est plus rapide que le Mamba pur **et** que le ViT pur. |
| **RCDT** (2022) | « *Change detection tasks are cross attention driven because the nature of the work is to compare bi-temporal information.* » L'attention croisée entre dates est le mécanisme naturel de la tâche. |
| **DCAT**, **STeInFormer**, **TiBT-Net** | Confirment la même direction : interaction explicite entre jetons des deux dates, plutôt que soustraction ou concaténation. |
| **Jamba / Samba** (NLP) | Un ratio faible d'attention (1 bloc pour 6 à 8 blocs SSM) suffit ; l'attention est un complément, pas un remplacement. |

## 3. Conception retenue

**Une attention jointe sur les deux dates**, aux stages profonds, insérée entre
l'encodeur et la fusion.

```
encodeur siamois → {X^T1_i}, {X^T2_i}
                       ↓  (stages profonds seulement)
        [ jetons T1 | jetons T2 ]  →  Transformer  →  jetons enrichis
                       ↓
                 fusion → décodeurs
```

Cinq décisions, chacune avec sa raison :

**Séquence jointe plutôt que self + cross séparées.** On concatène les jetons des
deux dates en une seule séquence de 2·H·W. L'attention apprend seule à mélanger
dans une date et entre les dates, là où deux blocs séparés imposeraient la
répartition — et coûteraient près du double.

**Aux stages profonds seulement.** L'attention est quadratique en nombre de
jetons. Le calcul tranche sans appel :

| Stage | Grille (entrée 512²) | Jetons (2 dates) | GMACs/bloc | Params/bloc |
|---|---|---:|---:|---:|
| 1 | 128×128 | 32 768 | **208,57** | 0,07 M |
| 2 | 64×64 | 8 192 | **28,19** | 0,30 M |
| 3 | 32×32 | 2 048 | 5,64 | 1,18 M |
| **4** | **16×16** | **512** | **2,82** | **4,72 M** |

Les stages 1 et 2 sont inabordables — 208 GMACs pour un seul bloc, contre 31,42
pour le modèle entier. Le stage 4 coûte 2,82. Le choix n'est pas une préférence,
c'est le seul point abordable. MambaVision y arrive par une autre route.

**Position par convolution depthwise**, non par table apprise. Une table est liée
à une résolution et casserait au changement de taille de crop (256 en juillet,
512 aujourd'hui). La convolution 3×3 depthwise en est indifférente — c'est le
*conditional positional encoding* de CPVT. Sans position, l'attention serait
invariante par permutation et ne pourrait rien localiser ; un test le vérifie.

**Embedding de date**, même rôle que le τ du décodeur sémantique : sans lui les
deux moitiés de la séquence seraient indiscernables.

**⚠️ LayerScale initialisé à 1e-5 — le choix le plus important.** Les deux résidus
sont multipliés par un gamma appris, initialisé quasi nul : **à l'initialisation,
le bloc est l'identité**. Mesuré : l'attention change les sorties de **8·10⁻⁶**.
Le modèle hybride ne peut donc pas partir plus bas que la ligne de base ; il
apprend s'il le veut à se servir de l'attention. Sans cela, on injecterait du
bruit dans un modèle qui marche, et un résultat négatif ne dirait pas si l'idée
est mauvaise ou si l'initialisation l'a tuée.

**Le bloc d'attention est construit en dernier.** Chaque module consomme le
générateur aléatoire à sa création : le placer plus haut décalerait
l'initialisation de tous les modules suivants. En dernier, un run hybride et un
run de référence à graine égale partagent **exactement** les mêmes poids de départ
pour l'encodeur, la fusion, la FFT et les deux décodeurs — vérifié sur les
132 tenseurs partagés. La comparaison est appariée jusque dans l'initialisation.

## 4. Coût

| Configuration | Params | GMACs |
|---|---:|---:|
| Point d'efficience (référence) | 16,48 M | 31,42 |
| **+ attention, stage 4, profondeur 2** | **25,92 M** | **37,06** |
| Point de performance (référence) | 32,58 M | 58,06 |
| **+ attention, stage 4, profondeur 2** | **42,02 M** | **63,70** |
| *pour mémoire : MambaSCD publié, mesuré* | *37,13 M* | *115,44* |
| *pour mémoire : Mamba-FCS* | *189,54 M* | *263,15* |

Même la variante la plus lourde reste **sous la moitié du calcul de MambaSCD** et
au quart de celui de Mamba-FCS.

## 5. Pré-enregistrement

Écrit avant tout lancement, selon la méthode de la phase 7.

**Deux bases, deux questions distinctes.**

| Base | Départ | Ce qu'il faut trouver | Question |
|---|---|---|---|
| `eff` | 0,2387 (16,48 M) | **+0,0163** pour dépasser Mamba-FCS | L'attention aide-t-elle ? |
| `perf` | 0,2484 (32,58 M) | **+0,0066** pour dépasser Mamba-FCS | Peut-on prendre le record ? |

**Réserve honnête, écrite d'avance.** Le plus gros effet jamais établi dans ce
projet est **+0,0125** (retrait de la loss SeK). Demander +0,0163 à une seule
brique architecturale serait sans précédent ici, et les trois briques
architecturales déjà testées ont donné entre −0,0001 et +0,0016. La base `perf`,
qui ne demande que +0,0066, est le pari réaliste ; la base `eff` sert à savoir si
l'attention apporte quelque chose, indépendamment du record.

**Critères de lecture, fixés d'avance.**

- Chaque configuration hybride se lit contre **son propre témoin** — `eff` contre
  `crop512-lean-augswap-ema`, `perf` contre `crop512-max` — jamais contre l'autre
  base. Un seul facteur doit varier.
- Verdict sur **les deux métriques**, maximum et époque finale, comme tout le
  reste du projet.
- Avec σ ≈ 0,0018 et 4 graines contre 7, **tout effet ≥ 0,0023 est détectable**.
- **Un gain qui ne se voit que sur le maximum sera rejeté** : c'est exactement le
  piège qui a produit le faux positif de l'augmentation le 8 septembre.

**Signal secondaire à regarder.** Si l'attention agit bien sur la localisation
comme prévu, l'**IoU du changement** doit monter davantage que la justesse
sémantique. Un gain de SeK sans mouvement de l'IoU voudrait dire qu'on a gagné
autrement que par le mécanisme supposé — à noter, pas à masquer.

## 6. Lancer

```bash
# Question scientifique : l'attention aide-t-elle ? (base efficience)
for S in 1 2 3 4; do
  BASE=eff SEED=$S sbatch --export=ALL,BASE,SEED scripts/train_hybrid.sbatch
done

# Tentative de record (base performance)
for S in 1 2 3 4; do
  BASE=perf SEED=$S sbatch --export=ALL,BASE,SEED scripts/train_hybrid.sbatch
done
```

Les deux blocs ne diffèrent que par `BASE=eff` / `BASE=perf` — c'est voulu, un
seul facteur varie. Le reste est identique parce qu'il doit l'être.

Trois protections contre le mélange, chacune couvrant ce que la précédente laisse
passer :

1. **Les tags sont forcés en `hyb-*`** — le lanceur refuse tout autre préfixe. Les
   runs de la campagne commencent tous par `crop512-`, ils ne peuvent donc pas se
   retrouver dans un même groupe d'agrégation.
2. **`BASE` entre dans le tag** (`hyb-eff-…` contre `hyb-perf-…`), donc les deux
   bases écrivent dans des dossiers distincts.
3. **Une empreinte de configuration est gravée** dans `config.txt` au premier
   passage. Si une variable ne se propageait pas — préfixe oublié dans
   `--export`, faute de frappe — le tag retomberait sur son défaut et deux
   configurations viseraient le même dossier ; `--resume auto` y reprendrait
   alors les checkpoints de l'autre run **sans le moindre message**. Le lanceur
   refuse de démarrer dans ce cas. Une reprise légitime retrouve la même
   empreinte et passe.

```bash
# Dépouillement, sur les deux métriques
for MODE in best final; do
  python -m scripts.aggregate_seeds --min-epochs 195 --on $MODE \
      --ref second_mini_chess_crop512-lean-augswap-ema \
      $SCRATCH/csf-mamba-runs/second_mini_chess_crop512-lean-augswap-ema-s* \
      $SCRATCH/csf-mamba-runs/second_mini_chess_crop512-max-s* \
      $SCRATCH/csf-mamba-runs/second_mini_chess_hyb-*
done
```

## 7. Résultat — l'attention n'apporte rien, et nuit sur le petit modèle

8 entraînements, 200 époques chacun, 4 graines par base. Contrôle d'intégrité
passé : deux lignes de configuration à quatre occurrences, et le bloc d'attention
mesure **9 466 368 paramètres** contre 9 440 000 prévus analytiquement — totaux
25,95 M et 42,05 M contre 25,92 et 42,02 annoncés. Le modèle entraîné est bien
celui conçu.

**Lecture selon les critères fixés au §5, chaque base contre son propre témoin.**

| Base | Δ sur le maximum | Δ sur l'époque finale | Verdict |
|---|---:|---:|---|
| **efficience** (16,48 → 25,95 M) | −0,0028 *(partiel)* | **−0,0051 (ÉTABLI)** | ⛔ **nuit** |
| **performance** (32,58 → 42,05 M) | −0,0000 | +0,0012 | ⛔ **aucun effet** |

**Sur la base efficience, l'attention dégrade le modèle**, et c'est établi sur la
métrique sans biais de sélection. Sur la base performance, elle ne fait
strictement rien : −0,0000 sur le maximum, les deux moyennes sont identiques au
dix-millième.

**Le record n'est pas pris.** L'écart à Mamba-FCS reste de +0,0065, inchangé.

### Le signal secondaire réfute le mécanisme supposé

C'était le critère inscrit d'avance : *si l'attention agit bien sur la
localisation, l'IoU du changement doit monter davantage que le reste.*

| Base | IoU du changement (époque finale) | Δ |
|---|---|---:|
| efficience | 0,5665 contre **0,5729** | **−0,0064** |
| performance | 0,5784 contre 0,5777 | +0,0007 |

Sur la base efficience, **l'IoU baisse** — l'attention a dégradé précisément ce
qu'elle était censée améliorer. Ce n'est pas un gain obtenu par un autre chemin :
c'est une réfutation directe de l'hypothèse de départ.

### Ce que le garde-fou d'initialisation permet de conclure

Le LayerScale garantissait qu'à l'initialisation le bloc est l'identité — mesuré à
8·10⁻⁶ — et les 132 tenseurs partagés étaient identiques à ceux du témoin. Le
modèle hybride **est donc bien parti de la ligne de base**, puis a appris à se
servir de l'attention d'une façon qui l'a dégradé.

C'est exactement ce que ce garde-fou devait établir : le résultat négatif porte
sur **l'idée**, pas sur une initialisation malheureuse. Sans lui, on n'aurait pas
pu trancher.

### Trois réserves, écrites sans les atténuer

**Une seule configuration a été testée** — stage 4, profondeur 2, 8 têtes,
mlp_ratio 2. Ce n'est pas un balayage. Une autre profondeur ou un autre stage
pourrait se comporter autrement, même si le pré-enregistrement prévoyait de ne
balayer qu'en cas de gain.

**Les blocs Transformer ont été entraînés avec le réglage du reste du modèle** :
LR constant 1e-4, AdamW à weight decay 0,01, sans warmup propre. Les blocs
d'attention y sont réputés plus sensibles que les convolutions ou les SSM. Le
LayerScale atténue ce risque sans l'annuler — c'est la réserve la plus sérieuse
contre ce résultat négatif.

**Sur la base efficience, l'attention ajoute +57 % de paramètres** à un modèle
entraîné sur 2 968 paires. Le surapprentissage est une explication plausible, mais
les époques de pic ne le montrent pas : [50, 57, 62, 69] pour l'hybride contre
[47, 62, 62, 66, 72, 94, 100] pour le témoin — pas de pic nettement plus précoce.
L'hypothèse reste non vérifiée.

### Ce que ça ajoute au tableau d'ensemble

C'est le **quatrième** bloc architectural testé dans ce projet, et le quatrième à
ne rien rapporter :

| Composant | Ce qu'il apporte |
|---|---:|
| C²S² entier | +0,0016 |
| MCA-SF | +0,0015 |
| CGA | −0,0001 |
| Branche FFT | −0,0046 |
| Décodeur élargi | −0,0007 |
| **Attention bi-temporelle** | **−0,0051 à +0,0012** |
| *DySample* | *+0,0075 — la seule exception* |

Face aux leviers de **données et d'optimisation** : retrait de la loss SeK
+0,0125, échange temporel +0,0096, LR constant +0,0098, EMA +0,0074 à +0,0087,
crops 512 +0,022, compensation de déséquilibre +0,032.

**Le motif est net et vaut d'être énoncé dans le rapport : sur ce jeu de données
et à cette échelle, les ajouts de conception architecturale ne déplacent pas le
SeK ; les données, l'optimisation et la capacité brute d'encodeur le déplacent.**
La seule exception, DySample, n'est pas un module de raisonnement mais un
opérateur de rééchantillonnage — la façon de décoder, pas la façon de penser.

**Décision : la piste est close.** Le code reste en place, désactivé par défaut,
documenté. Rien n'est repris dans le modèle de référence.

## 8. Ce qui resterait à faire pour rouvrir la piste

### ⚠️ Le diagnostic à faire AVANT de relancer quoi que ce soit

`gamma` part de 1e-5. Si l'entraînement ne l'a pas fait grandir, **le bloc n'a
jamais été allumé** et le résultat négatif ne dit rien de l'attention — il dit
que le bloc est resté éteint.

    python -m scripts.inspect_layerscale $SCRATCH/csf-mamba-runs/second_mini_chess_hyb-*/best.pt

Deux issues, et elles ne mènent pas au même endroit :

| valeur moyenne de gamma | Lecture |
|---|---|
| ≈ 1e-5, soit inchangé | **Le bloc est resté éteint.** Le résultat ne porte pas sur l'attention. Régler l'optimisation est indispensable avant de conclure. |
| ≥ 1e-2, soit ×1000 | **Le bloc a été utilisé** et a quand même dégradé le modèle. Le résultat porte sur l'idée, et l'optimisation n'y changera pas grand-chose. |

Deux minutes, et cela décide si les huit entraînements suivants valent la peine.

### Résultat du diagnostic (11 septembre) — le bloc s'est bien allumé

| Run | valeur moyenne de gamma | facteur depuis l'init |
|---|---:|---:|
| `hyb-eff-s3-d2-s1` | **1,78 · 10⁻²** | ×1 780 |
| `hyb-perf-s3-d2-s1` | **7,30 · 10⁻³** | ×730 |

Le bloc n'était **pas** gelé : l'optimiseur a multiplié `gamma` par mille. Deux
des trois hypothèses « il n'a pas eu les moyens de s'allumer » tombent, et il faut
le dire d'autant plus clairement que j'en avais désigné une comme la plus suspecte.

**⚠️ Le weight decay n'a rien bridé.** J'avais écrit qu'il « pousse gamma vers zéro
pendant que le bloc essaie de s'allumer ». Le calcul dit l'inverse : AdamW
découplé applique `gamma ← gamma·(1 − lr·wd)` = `×0,999999` par pas, soit
**×0,93 sur les 74 200 pas de l'entraînement entier**. Sept pour cent. Négligeable.
L'hypothèse était fausse.

**⚠️ L'initialisation non plus.** Atteindre 1,78 · 10⁻² demande environ 178 pas
d'Adam, sur 74 200 disponibles — **0,2 % du budget**. `gamma` avait tout le temps
de monter bien plus haut et ne l'a pas fait. Et l'initialisation à 1e-1 que je
proposais est **5,6 à 13,7 fois au-dessus** de la valeur retenue par
l'entraînement : la forcer reviendrait à imposer au bloc de contribuer plus que ce
que l'optimisation a choisi.

**Ce qu'il reste.** Une seule hypothèse cohérente : **l'attention elle-même était
mal conditionnée, et le petit `gamma` en est le symptôme, pas la cause.** Si la
sortie du bloc est bruitée ou inutile, l'optimiseur apprend justement un `gamma`
faible pour l'étouffer — c'est exactement ce qu'on observe. Un taux
d'apprentissage plus doux sur les seuls paramètres d'attention pourrait produire
une attention mieux conditionnée, donc un `gamma` utile plus grand.

C'est le dernier test défendable. S'il échoue, la piste reste close.

### Les réglages câblés, et pourquoi

Quatre leviers, ajoutés le 11 septembre. **Tous inertes par défaut** : sans eux,
l'optimiseur est littéralement l'appel d'origine sur `model.parameters()`, et un
test le vérifie — les 150 entraînements de la campagne restent reproductibles.

| Réglage | Défaut | Statut après diagnostic |
|---|---|---|
| **`--attn-lr-scale`** | 1.0 | ✅ **seul levier encore défendable.** Agit sur les poids de l'attention (qkv, mlp), pas sur la liberté de `gamma`. Une attention mieux conditionnée donnerait un `gamma` utile plus grand. |
| `--attn-dropout` | 0.0 | ➖ plausible sur la base efficience seulement (+57 % de paramètres sur 2 968 paires), mais les époques de pic ne montrent pas de surapprentissage. |
| `--attn-no-wd` | non | ⛔ **hypothèse réfutée** — 7 % d'effet sur tout l'entraînement. Conservé parce qu'il ne coûte rien et retire un confondant mineur. |
| `--attn-layer-scale` | 1e-5 | ⛔ **hypothèse réfutée** — `gamma` a grimpé ×1 780 en 0,2 % du budget, et l'init proposée dépasse la valeur choisie par l'entraînement. |

### Autres pistes, non engagées

- Profondeur 1 au lieu de 2 : moitié moins de paramètres ajoutés.
- Stage 3 au lieu du 4 : 1,18 M par bloc au lieu de 4,72, pour 5,64 GMACs au lieu
  de 2,82, et une résolution où la localisation se joue davantage.

Le résultat tel qu'il est suffit à conclure que **l'attention bi-temporelle, telle
qu'entraînée ici, n'apporte rien**. Ce qu'il ne tranche pas encore, c'est si elle
a seulement eu l'occasion d'essayer.

## 9. Sources

- [MambaVision: A Hybrid Mamba-Transformer Vision Backbone](https://arxiv.org/abs/2407.08083) — CVPR 2025
- [RCDT: Relational Remote Sensing Change Detection with Transformer](https://arxiv.org/pdf/2212.04869)
- [DCAT: Dual Cross-Attention-Based Transformer for Change Detection](https://doi.org/10.3390/rs15092395)
- [STeInFormer: Spatial-Temporal Interaction Transformer for Change Detection](https://arxiv.org/pdf/2412.17247)
- [Cross attention is all you need: relational remote sensing change detection with transformer](https://www.tandfonline.com/doi/full/10.1080/15481603.2024.2380126)
- [Awesome-Mamba-in-Remote-Sensing](https://github.com/BaoBao0926/Awesome-Mamba-in-Remote-Sensing) — veille du domaine
