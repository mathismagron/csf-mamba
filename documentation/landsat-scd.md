# Landsat-SCD — troisième jeu de données

**Pourquoi.** L'énoncé d'efficience du projet ne repose que sur SECOND, et le
résultat principal — le retrait de la loss SeK — **ne transfère pas** à Hi-UCD.
C'est la réserve n° 1 du README et la seule qui bloque une publication. Un
troisième terrain est ce qui manque.

**Pourquoi celui-là.** **Mamba-FCS rapporte Landsat-SCD**, la comparaison est
donc directe. *(Correction d'une erreur antérieure : je l'avais attribué à
ChangeMamba, qui ne le traite pas.)*

---

## 1. Le jeu

| | |
|---|---|
| Fichiers livrés | **8 468** de 416×416, résolution 30 m |
| Tuiles sources | ~2 425, le reste étant des variantes augmentées hors ligne |
| Répartition | **aucune liste de split dans le dump** |
| Période | Landsat, 1990–2020 |
| Zone | Tumushuke, Xinjiang, Chine — bordure du Taklamakan |
| Classes | **4 réelles** — farmland, desert, buildings, water — plus « sans changement » |
| Transitions | 10 types |
| Téléchargement | <https://figshare.com/articles/19946135> |

⚠️ **La littérature ne correspond pas au dump, et c'est le dump qui fait foi.**
Elle annonce 2 425 paires et « 1 908 pour l'entraînement, 477 pour le test » —
chiffres déjà incohérents entre eux, puisque 1 908 + 477 = 2 385. Le dump figshare
contient **8 468 fichiers** par dossier. Le rapport 8 468 / 2 425 ≈ 3,5 suggère
que 2 425 compte les **tuiles sources** et 8 468 les fichiers livrés, variantes
augmentées comprises. `check_landsat` rapporte les comptes réels plutôt que de les
vérifier contre des chiffres publiés qui ne décrivent pas ce fichier.

## 2. Format, repris de leur code et non deviné

Source : `third_party/MambaFCS/changedetection/datasets/make_data_loader.py`
(classe `SemanticChangeDetectionDatset_LandSat`) et `configs/train_LANDSAT.yaml`.

```
root/A/<id>.png          image T1 (RGB)
root/B/<id>.png          image T2 (RGB)
root/labelA/<id>.png     sémantique T1, carte d'indices 0..4
root/labelB/<id>.png     sémantique T2, carte d'indices 0..4
root/{train,val,test}_list.txt
```

Trois points de convention, tous repris à l'identique :

**Il n'y a pas de carte de changement binaire.** Mamba-FCS la dérive :
`cd_label[(t1_label > 0) | (t2_label > 0)] = 1`. On fait pareil.

**La sémantique vaut 0 hors changement**, donc `0 → ignore_index` — même
convention que SECOND, et **`num_classes: 5`** dans leur config confirme notre
convention A : index 0 réservé, classes réelles 1..4.

**L'entraînement concatène `train_list` et `val_list`.** C'est ce que fait leur
config ; s'en écarter rendrait notre chiffre non comparable au leur.

## 3. Télécharger

⚠️ **Sur un nœud de connexion** : les nœuds de calcul d'Alliance Canada n'ont pas
d'accès réseau, et les nœuds de connexion sont précisément faits pour ça.

```bash
cd $HOME/csf-mamba
scripts/download_landsat.sh              # vers $SCRATCH/Landsat-SCD
```

Le script télécharge `Landsat-SCD_dataset.zip` (figshare 19946135, **4,1 Gio**,
CC BY 4.0), **vérifie la taille exacte** (4 400 617 661 octets) avant d'extraire,
regarde la structure de l'archive, et **normalise l'arborescence** : beaucoup de
dumps enferment tout dans un dossier racine, et l'extraire à l'aveugle donnerait
`A/` un niveau trop bas — le dataloader échouerait. Les trois cas — archive à
plat, archive imbriquée, archive au mauvais format — sont couverts et testés.

Le téléchargement **reprend là où il s'est arrêté** : une session de nœud de
connexion coupée ne coûte pas les 4 Gio déjà transférés. Prévoir ~9 Gio libres,
l'archive et son contenu coexistant un moment ; `diskusage_report` donne le quota.

## 3bis. ⚠️ Le dump ne suit pas le format attendu — état du chantier

Exploration du 11 septembre. **Trois obstacles**, dont deux ouverts.

| Constat | Statut |
|---|---|
| **8 468 fichiers**, pas 2 425 | ✅ compris — 2 425 tuiles sources, le reste étant des variantes augmentées |
| **Un seul dossier `label/`**, valeurs 0..9, mode L | 🔧 ce sont des **transitions**, pas de la sémantique par date |
| **Aucune liste de split** | ⛔ **ouvert** |

### `label/` encode des transitions, pas des classes

La description figshare est sans ambiguïté : *« each "from-to" change type is a
separate class representing land-cover transitions »*. Les valeurs 0..9 codent
donc des **transitions**, quand tout notre pipeline — loss, SeK, décodeurs —
travaille sur **deux cartes sémantiques**, une par date. Il faut décoder
`v → (classe à T1, classe à T2)`.

La table n'est publiée nulle part : ni dans les métadonnées figshare, ni dans le
dépôt de Mamba-FCS, qui a prétraité le jeu sans publier son prétraitement.

**Elle est récupérable depuis les données.** Si `v` code « X → Y », les pixels
portant `v` montrent l'aspect de X dans A et celui de Y dans B. Donc les valeurs
partageant une classe de départ ont la même couleur moyenne dans A, et avec
4 classes de terrain les couleurs doivent former **4 groupes** — pas 9, pas 2.
La prédiction est falsifiable : si la structure n'apparaît pas, on ne décode pas.

```bash
python -m scripts.decode_landsat_label --data-root $SCRATCH/Landsat-SCD
```

Le script regroupe les couleurs de A **et** de B ensemble — les séparer donnerait
deux numérotations indépendantes et fabriquerait de fausses transitions vers
elles-mêmes, défaut qu'un test sur une table connue a révélé. Il vérifie ensuite
qu'aucune transition ne pointe vers sa propre classe, et que la valeur 0 présente
un écart A→B **plus faible** que les transitions — contrôle indépendant de son
statut « sans changement ».

Validé sur un jeu synthétique dont la table était connue : **les neuf transitions
sont retrouvées à l'identique**. Et il refuse correctement sur un jeu sans
structure, comme sur un seuil de regroupement absurde dans les deux sens.

**⚠️ Première tentative : hypothèse réfutée par les données.** Les couleurs
moyennes absolues donnent **deux** groupes au lieu de quatre. La raison est
visible dans les chiffres : (124, 120, 118), (65, 63, 61), (94, 88, 86) —
**R ≈ G ≈ B partout**, les images du dump sont quasi monochromes. Seule la
luminosité sépare désert, cultures et bâti, et elle varie d'une tuile et d'une
date à l'autre — saison, capteur, atmosphère. Moyenner sur 400 tuiles écrase donc
le contraste entre classes sous la variation d'éclairement. Seule l'eau ressort,
à (35, 42, 42).

Le script travaille désormais sur des couleurs **relatives à chaque tuile**, ce
qui supprime l'éclairement et ne laisse que le contraste entre classes. Il affiche
en outre la **matrice des distances** et **balaie le seuil** : s'il existe un
palier large où le regroupement donne 4 groupes, la structure est réelle ; s'il
n'existe aucun seuil qui la produise, elle ne l'est pas et le script refuse.

**Si le décodage échoue malgré tout**, la table est publiée dans l'article source
— Table 3 de [Yuan *et al.*, *Int. J. Digital Earth*
2022](https://www.tandfonline.com/doi/full/10.1080/17538947.2022.2111470), fermé
en accès libre mais accessible depuis un réseau universitaire. C'est la voie sûre,
et elle dispense de toute reconstruction.

### Les splits restent à trouver

Aucune liste dans le dump, aucune dans le dépôt de Mamba-FCS. Deux conséquences :

- sans les leurs, notre chiffre ne porterait pas sur le même découpage — or c'est
  la raison même d'utiliser ce jeu ;
- **un découpage fabriqué au hasard créerait de la fuite**, puisque les variantes
  augmentées d'une même tuile source se retrouveraient des deux côtés. Tout
  découpage devra se faire **par tuile source**, jamais par fichier.

## 4. ⚠️ Vérifier le dump AVANT d'entraîner

Le dataloader a été écrit d'après leur code, **pas d'après les fichiers**.
Personne n'a encore regardé le dump.

```bash
module load python/3.11 cuda/12.2
source $SCRATCH/csf-venv-cu12/bin/activate
python -m scripts.check_landsat --data-root $SCRATCH/Landsat-SCD
```

Le script contrôle l'arborescence, les listes, la taille des tuiles, le fait que
les labels soient mono-canal, la plage des indices, et **mesure le taux réel de
pixels changés**. Il ne dépend ni de torch ni du GPU.

Ce contrôle n'est pas une formalité : le projet a porté pendant deux semaines de
**faux noms de classes** sur SECOND, parce que l'ordre d'énumération de l'article
ne correspondait pas aux indices du dump. Les noms retenus ici — *farmland,
desert, buildings, water* — sont **explicitement marqués non vérifiés** dans le
code, et un test échoue si l'on oublie de lever le drapeau après vérification.

### ⚠️⚠️ Le jeu contient des variantes augmentées hors ligne

Découvert en listant l'archive, pas dans la littérature :

```
From1990To1993_01.png              tuile source
From1990To1993_01CropResize0.png   recadrage
From1990To1993_01ZheDang1.png      occlusion (遮挡)
From1990To1993_01rotate180.png     rotation
```

Chaque tuile source existe en plusieurs variantes déjà présentes dans le dump.
**Deux conséquences, et la première est grave.**

**Risque de fuite entre splits.** Si une tuile et sa rotation se trouvent de part
et d'autre du découpage train/test, le test contient des variantes d'images
d'entraînement, et tout chiffre publié sur ce jeu en est gonflé — **le nôtre
comme celui de Mamba-FCS**. `check_landsat` regroupe les noms par tuile source et
signale les recouvrements. Si fuite il y a, la bonne réponse est de **la
signaler**, pas de refaire le split : changer de découpage romprait la
comparaison, qui est la raison même d'utiliser ce jeu.

**Notre augmentation fait doublon.** Les rotations 90° et l'échange temporel
s'ajouteraient à des rotations déjà présentes dans les données. L'effet mesuré sur
SECOND n'a donc aucune raison de se reproduire ici — c'est une raison de plus de
faire tourner la base `lean` en témoin.

## 5. Deux réglages qui ne se transposent pas

**Le nombre d'époques se transpose en PAS, pas en époques.** Le résultat « LR
constant sur 200 époques » a été établi sur SECOND, qui compte 2 968 paires
d'entraînement : 371 pas d'optimiseur par époque, **74 200 au total**. Landsat en
compte ~1 948, soit 243 pas par époque. Reprendre « 200 époques » donnerait
48 600 pas — **35 % de budget en moins**, sans l'avoir décidé. D'où **305 époques**
par défaut.

**⚠️ La compensation de déséquilibre est le réglage le plus risqué.** Le défaut du
lanceur (`WEIGHT=1 DICE=0`) est celui de SECOND, où la retirer a rapporté +0,032.
Sur Hi-UCD, où 1,4 % des pixels changent, elle est **indispensable** — sans elle
le modèle s'effondre à SeK 0,000. Le journal en a tiré une règle : *un réglage
anti-déséquilibre ne se recopie pas d'un jeu de données à l'autre.*

Landsat annonce ~19 % de changement, proche des 20,1 % de SECOND, d'où ce défaut.
Mais c'est `check_landsat` qui mesure le taux réel, et il avertit s'il est faible.
Le cas échéant : `WEIGHT=20 DICE=1`.

## 6. Lancer

```bash
# 1. Vérifier le dump (2 minutes, sans GPU)
python -m scripts.check_landsat --data-root $SCRATCH/Landsat-SCD

# 2. La configuration retenue du projet, 4 graines
for S in 1 2 3 4; do
  BASE=eff SEED=$S sbatch --export=ALL,BASE,SEED scripts/train_landsat.sbatch
done

# 3. Le témoin sans les leviers d'augmentation, pour savoir s'ils transfèrent
for S in 1 2 3 4; do
  BASE=lean SEED=$S sbatch --export=ALL,BASE,SEED scripts/train_landsat.sbatch
done
```

`BASE` vaut `lean`, `eff` ou `perf` — les trois configurations du README. Les
runs sont tagués `landsat-*` et écrivent dans `landsat_<tag>`, donc ils ne
peuvent pas se mêler aux groupes de SECOND lors de l'agrégation.

```bash
for MODE in best final; do
  python -m scripts.aggregate_seeds --min-epochs 290 --on $MODE \
      --ref landsat_landsat-lean $SCRATCH/csf-mamba-runs/landsat_*
done
```

## 7. Pré-enregistrement

Écrit avant tout lancement, selon la méthode de la phase 7.

**La question n'est pas « quel SeK atteint-on ? » mais « l'énoncé transfère-t-il ? »**
Trois résultats possibles, tous publiables, mais qui ne racontent pas la même
histoire :

| Issue | Lecture |
|---|---|
| `eff` dépasse Mamba-FCS sur Landsat comme sur SECOND | L'efficience est une propriété du modèle. C'est le meilleur cas, et il lève la réserve n° 1. |
| `eff` reste compétitif sans dépasser | L'énoncé devient « compétitif sur deux jeux, supérieur sur un ». Plus faible mais honnête et publiable. |
| `eff` s'effondre | Comme Hi-UCD. Il faudra alors chercher **pourquoi**, et la réponse — plafond du jeu ou fragilité du modèle — fait l'intérêt du papier. |

**Critères de lecture fixés d'avance**, identiques au reste du projet :

- verdict sur **les deux métriques**, maximum et époque finale ;
- **σ mesuré sur Landsat**, jamais celui de SECOND transposé. Hi-UCD a montré
  que le σ dépend du jeu ; l'appliquer par hypothèse était une réserve qui a
  traîné trois semaines ;
- `eff` se lit contre `lean` — un seul facteur, les leviers d'augmentation ;
- 4 graines par configuration.

**Réserve honnête, écrite d'avance.** Trois des leviers établis sur SECOND —
retrait de la loss SeK, échange temporel, EMA — n'ont **aucune garantie** de
transférer. Le premier ne transfère déjà pas à Hi-UCD. S'ils ne transfèrent pas
ici non plus, la conclusion du projet devient : *ces leviers sont spécifiques au
jeu de données*, ce qui est un résultat en soi mais affaiblit l'énoncé général.
