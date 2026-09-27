# Distillation PerASCD → CSF-Mamba — état des lieux et plan

Rédigé le 25 septembre 2026, **avant toute ligne de code**. Même méthode que
`landsat-scd.md` et `hybride.md` : ce qui a été vérifié, ce qui ne l'a pas été,
le plan chiffré, et les critères de lecture fixés d'avance.

Unités : SeK en **points de pourcentage** (26,11 = 0,2611 dans le reste du dépôt).

---

## 1. Ce que dit le dépôt, et où la note de cadrage est en retard

La note de cadrage (texte collé le 25 septembre) décrit une version antérieure du
modèle. Le code et le README du 11 septembre font foi :

| Note de cadrage | Dépôt (`main`, commit `ab8b1fe`) | Conséquence pour la distillation |
|---|---|---|
| VMamba-Tiny **tronqué à 3 étages** (1/4, 1/8, 1/16) | `out_indices=(0,1,2,3)` : **4 étages**, canaux 96/192/384/768, aucune trace de troncature ni de l'ablation « −59 % » | l'échelle 1/32 existe → distillation de features possible aux 4 échelles |
| C²S², FFT, CGA au cœur du modèle | C²S² **retiré** du point d'efficience (`fusion=concat`) ; FFT et CGA non détectables | l'élève est essentiellement *VMamba-mini + décodeur DySample partagé* |
| Loss CE + mIoU + terme SeK + L_sc | terme SeK **retiré** (+1,25 pt, résultat principal du stage) | ne pas le réintroduire via la KD |
| ~15–22 M | **16,48 M / 31,42 GMACs** (efficience, `mini`) ou **32,58 M / 58,06 GMACs** (performance, `tiny` + supervision profonde) | choisir l'élève (question Q3) |
| Calcul sur Vulcan, estimations en « GPU-heures A100 », compatibilité « avec Narval » | le code cible Narval (`def-hervete`, `gpu:a100`), environnement `$SCRATCH/csf-venv-cu12` | **décision du 25 septembre : ce projet reste sur Narval.** Scripts et environnement existants réutilisables tels quels ; heures directement en A100 |
| — | SeK élève : **23,90 ± 0,15** (meilleure époque test, n=7) ; **23,30** (époque finale) | écart au professeur : **2,2 pt** (efficience), 1,3 pt (performance) |

**Tranché le 25 septembre (Q5)** : l'élève « 3 étages » n'existe nulle part — ni dans
les 139 commits locaux, ni sur Narval (clone unique `~/csf-mamba` à `ab8b1fe`, aucune
modification non commitée, aucun stash, `out_indices=(0, 1, 2, 3)`). **L'élève a 4 étages.**

---

## 2. Vérifications faites sur le professeur

Sources : branches `legacy` (`a4d808a`) et `master` (`dfdcc1e`) de
`github.com/SathShen/PerASCD`, dépôt HF `SathShen/PerASCD-Checkpoint`, dépôt
`github.com/SathShen/PerA`. **Aucun poids téléchargé** : seuls le répertoire des zip
(requêtes HTTP partielles) et les journaux TensorBoard (30 Mo) ont été lus.

### Vérifié

| Point | Résultat |
|---|---|
| Branche | Il n'y a pas de `main` : les branches sont **`master`** (refactorisée) et **`legacy`** (reproduction du papier). |
| Checkpoint SECOND | Publié : `PerAChain_40e_mIoU74.33_Sek26.11_Fscd66.41_OA88.70.pth`, **4 386 Mo** (poids + moment SGD), dans un zip HF de 4,1 Go, avec son journal TensorBoard. Aussi sur BaiduDisk. |
| Checkpoint LandsatSCD | **Non publié.** Distiller sur Landsat-SCD suppose de réentraîner le professeur. |
| Second professeur publié | `vmambaB_42e_…_Sek25.31_Fscd65.61…pth` (905 Mo) : VMamba-B + CG-Decoder, SECOND. |
| Taille réelle | **548,17 M paramètres** (comptés sur le code `legacy` instancié sur `meta`) : blocs ViT 503,85 M, interactions ViT-Adapter 27,71 M, décodeur 9,04 M. Pas ~300 M. Rapport élève/professeur : **33×** (efficience). |
| Architecture | « ViT-G/16/1024 » = largeur 1024, **profondeur 40**, 16 têtes, MLP ×4 (pas un ViT-g DINOv2 de largeur 1536). |
| Contrat encodeur (legacy) | 4 sorties **toutes à 1024 canaux** (`CascadeGatedDecoder([1024]*4, 128, …)`), pas [128, 256, 512, 1024] — cette liste est le gabarit générique de `master`. |
| **448 / 512** | Le code `legacy` ne redimensionne pas l'image : `input_size=448` fixe seulement la grille de l'embedding positionnel, **interpolée en bicubique** à la taille réelle. Le professeur traite donc **nativement du 512** : grilles 128/64/32/16, **identiques à celles de l'élève**. Le problème d'interpolation annoncé disparaît. |
| Sélection du checkpoint | Le « val loader » lit **`test`** et sauvegarde sur le **meilleur Fscd**. Même pratique que ChangeMamba et que nous, mais sur une autre métrique. |
| Biais de sélection du professeur | Mesuré sur son journal : SeK 26,11 à l'époque 40 (max Fscd), max 26,13 (époque 42), **finale 25,85**, moyenne des 10 dernières 25,85 ± 0,25. Optimisme ≈ **+0,26 pt**. Une seule graine (3701). |
| Recette du professeur | SGD lr 0,1, momentum 0,9, poly 1,5, warmup 10 %, 50 époques, batch 4 × accumulation 2, fp16 autocast, clip 1,5, drop-path 0,3. Augmentations : flip/rot D4 **et** jitter (b 0,2, c 0,2, s 0,1, h 0,1). 742 itérations/époque → **les 2 968 paires de train** (pas de split de validation). |
| Loss | CE `ignore_index=0` sur A et B (×0,5) + BCE pondérée sur le changement + SSCLoss sur les canaux 1..6. |
| Classes | 7 canaux, 0 = « non-changé ». Le canal 0 **n'est jamais une cible** de la CE → à exclure de toute KL. |
| Normalisation | PerA (0,3585, 0,3741, 0,3155) / (0,1483, 0,1283, 0,1198), bien dans `legacy` (`DataPerAAUG`). |
| Code de métrique | `SCDD_eval_from_hist` (legacy) et `csf_mamba/evaluation/metrics.py` implémentent **la même formule** (histogramme global, κ sur `hist_n0`, SeK = κ·e^(IoU_fg−1)). Le `mIoU` rapporté est le mIoU **binaire** changé/non-changé. |
| Licences | Code PerASCD : MIT. Code PerA : MIT. |
| Poids PerA pré-entraînés | Publiés (ViT-G/16-1024) sur **Google Drive et Baidu**, pas sur HF. Papier PerA : doi 10.1080/10095020.2026.2628435. |
| Données du professeur | `SECONDbi.zip` (3,78 Go) et `LandsatSCD512.zip` (1,61 Go) sur HF `SathShen/PerASCD-datasets`. |

![Courbes test des deux professeurs et puissance statistique](distillation_teacher.png)

*(a) SeK test par époque, relu dans les journaux TensorBoard publiés ; cercle = checkpoint publié (meilleur Fscd test). En orange, l'élève d'efficience (moyenne n = 7). (b) Plus petit écart de SeK détectable entre deux bras (test t bilatéral, α = 0,05, puissance 0,8, σ = 0,18 pt, σ mis en commun du projet).*

### Non vérifié

1. **Licence des poids.** Aucune carte de modèle sur HF, aucune mention de licence pour les poids PerA (Google Drive) ni pour le corpus RSRSD-5m. La licence MIT du code ne couvre pas explicitement les poids. Pour publier un élève distillé, écrire aux auteurs (`shenht@whu.edu.cn`) est la voie propre.
2. **Compilation de `MultiScaleDeformableAttention` sur Narval** (A100, sm_80, torch 2.5.1 cu12 de `setup_env.sh`, module `cuda/12.2`). `legacy` épingle torch 2.8 + cu129 + xformers 0.0.32 ; xformers est facultatif (repli explicite). Le noyau dispose d'une implémentation PyTorch de référence qui suffit à l'évaluation si la compilation échoue. Environnement séparé de `csf-venv-cu12`, pour ne pas toucher celui de l'élève.
3. **Identité SECONDbi ↔ notre SECOND** (version prétraitée ChangeMamba). Même split officiel très probable (2 968 / 1 694), codage des labels à comparer pixel à pixel.
4. **Reproduction du 26,11** : non faite (aucun calcul GPU lancé).
5. **Paramètres du professeur VMamba-B** : ~113 M estimés à partir de la taille du checkpoint (poids + moment SGD = 8 octets/paramètre), non comptés.
6. **Débit professeur sur A100 40 Go** : ~35 ms/paire estimées, non mesurées ; mémoire en inférence (1,1 Go de poids en demi-précision) sans risque, mais à confirmer pour la KD de features en ligne (professeur + élève sur la même carte).
7. **Accès SSH depuis Claude Science** : Narval exige aussi Duo ; les jobs restent lancés par toi.
8. ~~Checkpoints de l'élève sur Narval~~ → **vérifié le 25 septembre**, voir §2bis.

---

## 2bis. État de Narval (audit du 25 septembre, `scripts/narval_audit.sh`)

**Les 7 graines du point d'efficience sont intactes.** Dossiers
`$SCRATCH/csf-mamba-runs/second_mini_chess_crop512-lean-augswap-ema-s{1,2,3,4,5,6,42}` :
200 époques chacune, `best.pt` (66 Mo) et `last.pt` (264 Mo). SeK à l'époque 199 :
moyenne **0,23305 ± 0,0010** (n = 7), identique au 0,2330 du README. Aucune liste de
purge à mon nom.

| Point | Constat | Conséquence |
|---|---|---|
| Date d'accès des `.pt` | 8–10 septembre | purge du scratch (~60 jours sans accès, règle Alliance) **vers début novembre** : archiver maintenant |
| `config.txt` | **absent** des 7 dossiers : lancés avant le garde-fou (`cc37672`, 10 sept. 21 h) | configuration exacte à reconstituer depuis les lignes `== config` des logs Slurm, avant de coder la KD |
| `$SCRATCH/SECOND` | format ChangeMamba (`T1 T2 GT_T1 GT_T2 GT_CD`, 2 968 / 1 694) | le professeur attend `im1 im2 label1 label2` : adaptateur de chargement ou liens symboliques pour l'étape 0 |
| Environnement `csf-venv-cu12` | torch 2.5.1, mamba_ssm 2.2.4, causal_conv1d 1.5.0.post8, selective_scan 0.0.2, **transformers 5.14.1, triton 3.6.0** | fonctionne (≈150 runs) grâce au correctif de `setup_env.sh`, mais ne pas y installer le professeur : environnement séparé |
| Quotas | `/home` 6,2/50 Go ; `/scratch` 169 Go/20 To ; `/project def-hervete` 53 Go/1 To mais **475 k/500 k fichiers (95 %, quota de groupe)** ; nearline `def-hervete` et `aip-hervete` vides (1 To chacun) | archiver en **une seule archive tar**, jamais en fichiers dépliés sur `/project` ; le cache du professeur (≈10 Go) va sur `/scratch` en un seul fichier (npz/HDF5 ou tar) |
| Archives existantes | `~/csf-checkpoints-20260819.tar` (4,9 Go) et `~/csf-archive-20260819.tar.gz` (89 Mo) | antérieures aux 7 graines de septembre : elles ne les couvrent pas |

**Marge GPU.** `def-hervete` est une allocation par défaut, **sans plafond d'heures**
(`GrpTRESMins` vide) : la contrainte est la priorité, pas un solde.

- Groupe `def-hervete_gpu` : **LevelFS 2,86** (> 1, le groupe consomme moins que sa part : bonne priorité).
- Toi, dans le groupe : 85 % de sa consommation effective (LevelFS 0,20). Cela ne pèse que sur l'arbitrage entre membres du groupe.
- Consommation depuis juillet (`sacct`) : 135 GPU-h (juillet), 1 007 (août), 763 (septembre) ; **1 905 GPU-h au total**. Au rythme d'août–septembre (≈32 GPU-h/jour : 1 770 GPU-h en 56 jours), le plan à 120 époques (≈255 A100-h) représente **environ 8 jours** de calcul (≈435 A100-h, deux semaines, à 200 époques).

**Limites d'ordonnancement.** Un job GPU dure **24 h au plus** (`gpubase_bygpu_b3`) ;
≤ 12 h ouvre aussi les partitions `b2`, ≤ 3 h les `b1`. À 200 époques, un run élève (12,8 h) et un run
KD de features en ligne (~18 h) tiendraient sous 24 h ; à 120 époques (adopté), tous passent sous 12 h — mais un nœud lent peut faire dépasser (cf.
`max-s6`, 3,3× trop lent) : la reprise automatique (`last.pt`, `resubmit.sh`) reste
indispensable. Des tranches MIG A100 (`a100_3g.20gb`, `a100_4g.20gb`, …) existent :
utiles pour l'évaluation du professeur et la génération du cache, si 20 Go suffisent.

### La baseline, relue dans ses `metrics.csv` (26 septembre)

**Configuration exacte**, reconstituée depuis les lignes `== config` des logs Slurm
(identique pour les 7 graines, seule la graine change) :

```
weight=1 dice=0 lovasz=0 crop=512 batch=2 x accum=4 enc=vmamba_mini dec=dw
fusion=concat cga=1 mcasf=1 up=dysample core=chess deep=0 sek=0 sc=0.1 fft=[0,1]
lr=constant rot90=1 photo=0.2 tswap=0.5 ema=0.9998 epochs=200   seeds 1 2 3 4 5 6 42
```

À retenir pour la KD : **la branche FFT est active** (`fft=[0,1]`), les rotations
90° aussi (`rot90=1` → groupe D4 complet, d'où les 8 variantes du cache), et le
micro-batch vaut **2** (batch effectif 8) — le professeur en ligne traitera donc 2
paires par passe.

| | max SeK test (époque) | SeK époque finale | SeK au max Fscd |
|---|---:|---:|---:|
| moyenne ± σ, n = 7 | **23,90 ± 0,15** (47–100) | **23,31 ± 0,10** | 23,90 ± 0,15 |

![Courbes des 7 graines de la baseline](distillation_baseline.png)

*(a) SeK test par époque des 7 graines (gris) et leur moyenne (bleu) ; cercles = meilleure époque de chaque graine. (b) Moyenne sur 7 graines du SeK à la meilleure époque et à la dernière époque, si l'entraînement s'arrêtait après E époques (préfixes des mêmes courbes).*

Trois constats, dont un corrige le journal :

1. **L'écart max – finale n'est pas du bruit de sélection, c'est un déclin.** Avec
   l'EMA, le bruit d'une époque à l'autre après l'époque 150 ne vaut que
   **0,03 pt** (σ des résidus d'une droite) : prendre un maximum sur une série
   aussi lisse ne rapporte que quelques centièmes. Or l'écart est de **0,60 pt**.
   La moyenne culmine à l'époque 63 (23,81) puis **décroît de 0,45 pt par
   100 époques** (±0,21) sous LR constant. Le journal (11 septembre) attribuait
   l'essentiel de l'écart au « maximum d'une série bruitée » avec un bruit de
   0,18 pt : c'était vrai sans EMA, plus avec.
2. **Choisir l'époque sur le Fscd ou sur le SeK ne change rien** : 23,895 contre
   23,899. Même constat chez le professeur (26,11 contre 26,13). La règle n° 1
   (« ne pas mélanger Fscd et SeK ») est satisfaite à 0,02 pt près des deux côtés.
3. **Un chiffre sans sélection sur le test existe.** Époque fixée à l'avance par
   validation croisée entre graines (pic de la moyenne des 6 autres, soit l'époque
   63, 64 ou 79) : **23,77 ± 0,17**. C'est 0,13 pt sous le maximum — bien moins que
   l'écart « maximum – finale » ne le laissait craindre.

**Conséquence pour la KD : la distillation régularise souvent, et ici la
régularisation se verrait surtout sur l'époque finale.** Si la KD freine le déclin
sans relever le pic, elle gagnera beaucoup sur la finale et peu sur le maximum. La
règle du projet (« les deux métriques doivent concorder ») classerait alors un vrai
effet en « non établi ». Lecture fixée d'avance : gain sur le maximum = **meilleur
pic** ; gain sur la finale seule = **stabilité**, rapportée comme telle, sans être
présentée comme une amélioration du SeK.

### Budget de 120 époques au lieu de 200 — **adopté le 26 septembre**

Avec un LR constant, **rien dans l'entraînement ne dépend du nombre d'époques
prévu** (`total_iters` ne sert qu'au cosinus, vérifié dans `_make_scheduler`) : un
run de 120 époques est exactement le préfixe d'un run de 200, au non-déterminisme
GPU près. Les 7 graines de 200 époques **fournissent donc déjà la baseline à 120
époques**, sans rien relancer : maximum **23,90 ± 0,15** (inchangé, tous les pics
sont avant l'époque 100), finale 23,65 ± 0,22.

- Coût d'un run : 12,8 h → **7,7 h** ; KD de features en ligne ~10,8 h : **tout passe sous 12 h**, donc dans les partitions `b2` en plus de `b3`, avec une attente plus courte.
- Plan complet : ≈435 → **≈255 A100-h** ; chemin court ≈275 → **≈160 A100-h**.
- Règle n° 3 respectée : KD et baseline au même budget (120).
- Garde-fou : si une graine KD culmine après l'époque 100, sa configuration est prolongée à 200 époques par reprise sur `last.pt` (possible précisément parce que le LR est constant ; il faudra alors autoriser le changement d'`epochs` dans l'empreinte `config.txt`).

## 3. Protocole unifié

**Décisions du 25 septembre** : chantier **parallèle** à `robustcd`, **sur Narval** (A100, `def-hervete`) ; **sélection

**Décisions du 26 septembre** : budget de **120 époques** (§2bis) ; criblage sur val puis confirmation sur test (ci-dessous) ; pas d'échéance ni de revue visée — le critère est la solidité, donc le **plan complet** plutôt que le chemin court.
d'époque sur le test conservée** (convention du projet, du 11 septembre) ; élève =
**point d'efficience** (16,48 M) ; professeur = **checkpoint PerASCD publié** seul.

| Élément | Choix | Justification |
|---|---|---|
| Split | split officiel : train 2 968 / test 1 694 | continuité avec les 7 graines existantes et avec le professeur (entraîné sur les mêmes 2 968) |
| Sélection d'époque | **meilleur SeK test**, poids EMA ; **époque finale** et **époque fixée d'avance** rapportées systématiquement | convention du projet ; la double métrique (phase 12, « C3 ») reste le garde-fou |
| Résolution | tuile entière 512, entraînement et évaluation, pour tous | le professeur est nativement en 512 (§2) |
| Code de métrique | `csf_mamba/evaluation/metrics.py` pour tous, professeur compris | formule identique vérifiée ; on l'exécute quand même sur le professeur |
| Budget | **120 époques**, LR constant, recette d'efficience, **identique avec et sans KD** ; baseline = préfixe des 7 runs de 200 époques | règle n° 3 ; §2bis |
| Graines | **5** pour les bras de conclusion, **2** pour le criblage | 3 graines ne détectent qu'un Δ ≥ 0,55 pt (fig. b) ; 5 → 0,36 pt |
| Tests | variance mise en commun **et** Welch, sur **les deux** métriques | règle du projet (phase 7) |

**Fscd contre SeK chez le professeur : l'écart est négligeable, et mesuré.** Son
checkpoint a été choisi sur le Fscd du test ; selon notre critère (max SeK test) il
vaudrait **26,13** au lieu de 26,11 (époque 42, journal publié). On rapporte
26,11 (le checkpoint qui existe), avec la note « +0,02 sous notre critère ; 25,85 en
fin d'entraînement ». La règle n° 1 tient à 0,02 pt près.

### ⚠️ Le risque que cette décision laisse ouvert, et une parade qui la respecte

Chaque balayage de KD (λ, T, masque, échelles) ajoute une **sélection entre
configurations** au-dessus de la sélection d'époque. Choisir λ sur le test, c'est
régler un hyperparamètre sur le test ; avec 12 configurations criblées, le gain de
KD serait gonflé d'un biais que la colonne « époque finale » ne capte pas (elle
corrige l'époque, pas la configuration).

Parade **retenue le 26 septembre**, sans changer la convention des chiffres rapportés :

1. **criblage** (étapes 2–4, 2 graines) entraîné sur 2 671 et jugé sur les 297
   paires de val du split `robustcd` (blake2b `robustcd-second-val-v1`) — le test
   n'est jamais regardé pour choisir λ, T, masque ou échelles ;
2. **confirmation** (étape 5) : la configuration choisie est réentraînée sur les
   2 968 paires, 5 graines, et rapportée selon la convention (max SeK test + finale).

Coût : aucun run supplémentaire (le criblage a lieu de toute façon).

**Pourquoi, en termes simples.** Ce n'est pas une question propre à la
distillation, c'est celle de tout réglage d'hyperparamètres. Si l'on essaie
12 réglages et qu'on garde celui qui fait le meilleur score *sur le test*, une
partie de son avance vient de la chance de ce réglage-là sur ces 1 694 images
précises : le test a servi à choisir, il ne mesure plus de façon neutre. Avec des
écarts attendus de quelques dixièmes de point et un bruit entre graines de
0,15 pt, cette part de chance n'est pas négligeable. Choisir sur 297 images que
le modèle final ne verra jamais à l'évaluation, puis mesurer **une seule fois**
sur le test, supprime ce biais pour le prix d'un léger bruit supplémentaire au
criblage.

Split utilisé : `robustcd/splits/SECOND/{train,val}.txt` (2 671 / 297 identifiants,
sha256 `b3fec730…` et `87306dcb…`, ratio de changement 0,200 / 0,193), à copier
tel quel dans `csf-mamba/splits/SECOND/` pour que les deux projets partagent le
même découpage.

---

## 4. Plan de travail

Coût d'un run élève à **120 époques** : **7,7 h A100** sur 2 968 paires (3,85 min/époque mesurées sur Narval),
**6,9 h** sur 2 671. KD en cache +5 % ; KD de features en ligne +40 % (≈10,8 h sur 2 968). Professeur en avant seul : ~1,3 TMAC par paire
(blocs ViT : 1,20 TMAC, estimé), ~35 ms/paire estimées sur A100.

| Étape | Objectif | Livrable | Critère de réussite | A100-h |
|---|---|---|---|---:|
| **P** | Vérifier l'existant sur Narval | `git pull`, `tests/`, un checkpoint d'efficience réévalué ; `--account=def-hervete`, `gpu:a100` inchangés | SeK réévalué à ±1e-4 de la valeur du journal | 1 |
| **0** | Reproduire le professeur sous notre protocole | env `legacy` séparé + op compilée, éval test fp32/fp16/bf16, code legacy **et** le nôtre, params/GMACs/latence ; comparaison des labels SECONDbi / notre SECOND | SeK **26,11 ± 0,05**, Fscd 66,41 ± 0,05 ; écart entre les deux codes < 1e-4 ; labels identiques ou différence documentée | 2 |
| 0-cache | Cache des sorties du professeur | logits à 1/4 (128², canaux 1..6 de A et B + changement), 8 variantes D4, fp16, sur les 2 968 paires | cache relu = sortie en ligne à la précision fp16 près | 1 |
| **1** | Élève sans KD | **les 7 graines Narval existantes**, tronquées à 120 époques (max 23,90 ± 0,15 ; finale 23,65 ± 0,22) + 2 graines sur 2 671 (témoin du criblage, 120 époques) | le chemin sans KD du code modifié reste identique à l'octet près (test à la manière de `tests/test_hybride.py`) | 14 |
| **2** | KD du **changement** seul (cache) | BCE à cibles douces, λ ∈ {0,5 ; 1 ; 2} × 2 graines, criblage sur val | Δ ≥ +0,36 pt contre le témoin sur val | 44 |
| **3** | + KD **sémantique** (cache) | KL sur canaux 1..6, T ∈ {2 ; 4} × masque {changé GT ∪ professeur ; toute l'image}, 2 graines | incrément ≥ +0,36 pt sur l'étape 2, sur val | 58 |
| **4** | + KD de **features** (en ligne) | adaptateurs 1×1 élève → 1024, cosinus après LayerNorm ; {1/8+1/16} vs {1/8+1/16+1/32}, 2 graines | incrément ≥ +0,36 pt sur l'étape 3, sur val, sinon non retenu | 50 |
| **5** | Confirmation, convention du projet | meilleure KD logits seuls et meilleure KD logits+features, **5 graines chacune sur 2 968, 120 époques**, max SeK test + finale ; params, GMACs, latence fp32/bf16 sur A100, **mêmes conditions que la latence du README** (lot de 8, 512², médiane sur 50) | fraction d'écart comblée (S_KD − S_1)/(S_prof − S_1) avec IC 95 %, les deux tests et les deux métriques concordants | 110 |
| **Total** | | | | **≈ 250** (durées `sacct` de septembre : ≈3,4 min/époque) |

**Chemin court (~160 A100-h).** λ = 1 sans balayage à l'étape 2, un seul masque à
l'étape 3 (toute l'image, le plus informatif), une seule variante de features. On
perd la courbe de sensibilité, pas la conclusion principale.

*Non retenus le 25 septembre* : réentraînement du professeur (36–60 A100-h) et
professeur VMamba-B publié (coût marginal, test de l'écart de capacité). Ce
dernier reste la première option à rouvrir si la KD du 548 M déçoit (risque 9).

### Pourquoi cet ordre

Le journal a établi que **la localisation du changement est le goulot** (IoU de
changement 0,58 ; la sémantique conditionnelle au changement est déjà bonne). La KD
du changement vise ce goulot, elle est supervisée partout (la carte binaire existe
sur toute l'image), et elle est la moins chère. Si elle ne rapporte rien, les étapes
suivantes deviennent douteuses — c'est une porte de décision.

### Hors ligne ou en ligne — chiffré

| Cible | Taille par paire (fp16) | 2 968 × 8 D4 | Coût en ligne par run |
|---|---:|---:|---:|
| Logits à 1/4 (13 canaux, 128²) | 0,43 Mo | **10,2 Go** | ≈ +5 h A100 |
| Logits à 512 (inutile : le professeur suréchantillonne en bilinéaire depuis 1/4) | 6,8 Mo | 162 Go | — |
| Features encodeur, 4 échelles × 1024 canaux × 2 dates | 89 Mo | 2,1 To | ≈ +5 h A100 |

Générer le cache coûte ~15 min de GPU. **Logits : hors ligne. Features : en ligne**
(+40 % sur le run, ~10,8 h A100 au lieu de 7,7 à 120 époques).

- **D4** : 8 variantes pré-calculées → cohérence exacte, pas seulement approchée.
- **Échange temporel** : échanger predA ↔ predB du cache ; le changement est symétrique. Exact.
- **Jitter photométrique** : le professeur voit la vue propre, l'élève la vue altérée. Ce n'est pas un défaut à contourner : c'est une contrainte d'invariance radiométrique, et PerASCD revendique précisément cette robustesse. Cela reste une différence avec la KD « cohérente » (Beyer et al., 2022) : on l'isole par un bras jitter-off si l'étape 2 est positive.
- **Crop** : sans objet, tuile entière 512.

### La question des zones non-changées

Sur SECOND, ~80 % des pixels n'ont **aucune** supervision sémantique, ni pour
l'élève ni pour le professeur : ses prédictions y ont été façonnées par la seule
SSCLoss (cohérence A/B), pas par des étiquettes. C'est donc une extrapolation, pas
une connaissance vérifiée. Deux raisons de la tester quand même : elle multiplie
par ~5 les pixels porteurs de signal sémantique, et le SeK compte la sémantique de
tout pixel **prédit** changé — donc des faux positifs de l'élève en zone
non-changée. Le masque est un facteur explicite de l'étape 3. À noter : les auteurs
avaient écrit un pseudo-étiquetage des zones non-changées, **commenté** dans
`legacy/train.py` — signe qu'ils l'ont essayé sans le retenir.

---

## 4bis. Lancer les étapes P et 0 (code écrit le 26 septembre)

Fichiers : `scripts/check_narval.{py,sbatch}` (P), `scripts/teacher/setup_teacher.sh`,
`scripts/teacher/eval_perascd.{py,sbatch}` (0). Sorties dans
`$SCRATCH/csf-distill/{teacher,checks}/`.

```bash
# portable : commit + push, puis sur Narval (nœud de connexion)
cd ~/csf-mamba && git pull && mkdir -p logs
sbatch scripts/check_narval.sbatch                 # P  : ~1 h GPU
bash scripts/teacher/setup_teacher.sh              # 0a : clone, checkpoint 4,1 Go, venv (connexion)
sbatch scripts/teacher/eval_perascd.sbatch         # 0b : ~1 h GPU (compile l'op au 1er passage)
# retour (portable) : scp 'magron13@narval.alliancecan.ca:/scratch/magron13/csf-distill/checks/*.json' ~/Projects/csf-mamba/logs/
```

**Journal d'exécution.**

- 26 sept. : `setup_teacher.sh` OK sur Narval. PerASCD `legacy` @ `a4d808a` (2026-05-29, « init »). Checkpoint `PerAChain_40e_mIoU74.33_Sek26.11_Fscd66.41_OA88.70.pth`, 4 386 148 239 octets, sha256 `a820956553bf89bac4b48f60be4c0cbc1cc080998f4dc83a257777a403c0e4f5` (zip `6f6de82c…c35dea`). venv `$SCRATCH/perascd-venv` : torch 2.5.1 (CUDA 12.2), torchvision 0.20.1, timm 1.0.29, numpy 2.4.2, scipy 1.17.1. Jobs soumis : P = 4023058, 0b = 4023491.
- 26 sept., **étape P : PASS** (job 4023058, A100-SXM4-40GB, torch 2.5.1). Les 7 `best.pt` réévalués en bf16 retrouvent le SeK de leur `metrics.csv` à **4,1e-6 près au pire** (seed 2), sous l'arrondi du CSV (5e-6) : environnement, code et données inchangés depuis septembre. En fp32, écarts de −3,7e-5 à +9,7e-5 (moyenne +1,8e-5), soit ≤ 0,01 pt : la précision de validation ne biaise pas la sélection. 16 483 643 paramètres, comme le README. ~40 s par évaluation de 1 694 paires.
- 26 sept., **étape 0b, 1er essai (job 4023491) : arrêt au chargement.** L'opérateur `MultiScaleDeformableAttention` s'est compilé sans erreur (setuptools 82, sm_80). Le `load_state_dict(strict=True)` a refusé 6 clés : le checkpoint porte `decoder.blocks.{0,1,2}.cagm.conv2.*`, le code `legacy` construit `…cagm.conv_local.*`. Le dépôt contient deux copies de `ChangeAwareGatingModule` (`models/Encoders.py` : `conv2` ; `models/PerAChain.py` : `conv_local`) au calcul **identique ligne à ligne** et aux tenseurs de même forme — le checkpoint vient de la première. Correctif : renommage borné à ce motif, avec contrôle de forme, `strict=True` conservé, clés renommées listées dans le JSON. Vérifié localement : un state_dict renommé se recharge à l'identique (tous les tenseurs égaux). La reproduction du 26,11 validera le renommage de bout en bout : une erreur d'appariement effondrerait le score.
- 26 sept., **étape 0 : PASS** (job 4026049, A100-SXM4-40GB). Le checkpoint publié, réévalué sur **notre** SECOND test en fp32 :

  | | publié (journal TB) | notre réévaluation | écart |
  |---|---:|---:|---:|
  | SeK | 26,1087 | **26,1079** | −0,0008 pt |
  | Fscd | 66,4138 | **66,4104** | −0,0034 pt |
  | mIoU | 74,33 | 74,331 | — |

  - leur code et le nôtre (`SCDEvaluator` complet, alimenté par les sorties du professeur) concordent à **1e-8** ;
  - données : **0 pixel** où `GT_CD` ≠ (label T1 > 0), 0 où label T1 > 0 ≠ label T2 > 0 sur 444 M pixels ; ordre des classes identique (l'appariement optimal est l'identité, précision sémantique 88,6 % sur les pixels changés). Notre SECOND et leur SECONDbi sont équivalents pour l'évaluation : **télécharger SECONDbi est inutile** ;
  - opérateur déformable CUDA ↔ référence PyTorch : écart max 0,003 sur les logits, argmax identique à 99,9994 % ;
  - précision : SeK fp16 = 26,1082, bf16 = 26,1049 (≤ 0,003 pt du fp32) → **fp16 retenu** pour le cache et la KD en ligne ;
  - le renommage `cagm.conv2 → conv_local` est validé par la reproduction elle-même.

  **Coût du professeur, mesuré** (lot de 8, 512², médiane de 50, protocole du README ; élève : README, 10 sept.) :

  | | professeur | élève (efficience) | rapport |
  |---|---:|---:|---:|
  | Paramètres | 548,17 M | 16,48 M | 33× |
  | GMACs / paire | 1 509,7 ¹ | 31,42 ² | ≈48× |
  | Latence fp32 | 1 613 ms · 5,0 paires/s | 129 ms · 62 paires/s | 12,5× |
  | Latence bf16 | 563 ms · 14,2 paires/s | 114 ms · 70 paires/s | 4,9× |
  | Latence fp16 | 490 ms · 16,3 paires/s | — | — |
  | Mémoire crête | 27–28 Go | 1,9–2,0 Go | ≈14× |

  ¹ `torch.utils.flop_counter`, opérateur déformable non compté. ² compteur du projet (fvcore). Les deux compteurs diffèrent : recompter l'élève avec le même outil à l'étape 5 avant de publier le rapport.

  **Conséquence sur le budget.** Mon estimation de 1,3 TMAC était basse (1,51 mesuré). À 16,3 paires/s, le professeur en ligne ajoute ≈2,7 min par époque de 2 671 paires, soit **+80 %** et non +40 % : un run de KD de features passe à ≈12,4 h (2 671) / ≈13,8 h (2 968), donc en partition 24 h. Étapes 4 et 5 : +11 h et +15 h ; **plan complet ≈280 A100-h** au lieu de 255. Réserve : avec le micro-batch de 2 de l'élève, le débit du professeur peut être inférieur à celui mesuré en lot de 8 — à mesurer au premier run de l'étape 4.
- 26 sept. : split `robustcd` copié à l'octet près dans `splits/SECOND/` (2 671 / 297 / 1 694). Les sha256 de `README.json` portent sur `"\n".join(ids)` sans saut de ligne final (convention de `robustcd/scripts/check_dataset.py`), d'où leur différence avec `sha256sum` des fichiers — vérifié, les listes sont identiques.
- 26 sept. : `scripts/teacher/cache_teacher.{py,sbatch}` écrit. 8 fichiers `d4_{0..7}.npy` (2 968, 15, 128, 128) fp16, `ids.txt`, `meta.json`. Testé localement (CPU, ViT-B aléatoire) : la correspondance entre les transforms de l'élève (flips puis rot90) et l'index de cache est vérifiée sur 200 tirages de la vraie chaîne `csf_mamba.datasets.transforms`, et le suréchantillonnage du cache redonne exactement la sortie 512 (écart 0,0). Le job mesure aussi l'écart d'équivariance du professeur et son SeK sur le train depuis le cache relu.
- 26 sept., **cache : OK** (job 4027028, 26 min, 15,2 paires·vue/s). 8 × 1 458 831 488 octets = 2 968 × 15 × 128² × 2 o + en-tête : complet. Contrôles de `meta.json` :
  - SeK du professeur **sur le train**, relu depuis le cache fp16 : **33,24** (Fscd 71,50, mIoU 77,77) — au-dessus du test (26,11), loin de la saturation : les cibles douces ne recopient pas les étiquettes ;
  - **écart d'équivariance** : entre T(g·x) et g·T(x), la décision de changement diffère sur **2,2–2,5 % des pixels** et la classe sur 2,7–3,0 % des pixels changés. Rapporté aux ~20 % de pixels changés, c'est de l'ordre de 12 % de la zone changée : **les 8 vues sont justifiées**, un cache à vue unique aurait donné des cibles incohérentes avec les augmentations de l'élève ;
  - piste ouverte par ce chiffre, notée pour l'étape 3 : un professeur **moyenné sur D4** (« TTA ») est gratuit à partir de ce cache et probablement meilleur que chaque vue seule.
- 26 sept., **code des étapes 1–3** :
  - `SECONDDataset(ids_file=, image_split=)` : sous-ensembles lus dans `train/` (val = 297 ids) ; comportement inchangé sans ces arguments ;
  - les transforms enregistrent leurs tirages (hflip, vflip, k, swap, crop) **seulement** si l'échantillon porte la clé `_aug` — tirages aléatoires strictement identiques sinon (testé) ;
  - `csf_mamba/distill/d4.py` : convention D4 unique, désormais importée par le script de cache ;
  - `TeacherCacheDataset` : applique la chaîne d'augmentation, lit la vue D4 tirée, échange T1/T2 si l'échange temporel a été tiré, refuse tout crop partiel ;
  - `DistillLoss` : BCE douce sur le changement (T², entropie du professeur retranchée pour des logs lisibles) ; KL sur les classes 1..6 renormalisées, masque `changed` ou `all` ;
  - `train.py` : `--train-ids`, `--val-ids`, `--kd-cache`, `--lambda-kd-change`, `--kd-t-change`, `--lambda-kd-sem`, `--kd-t-sem`, `--kd-sem-mask` ; tout à 0 par défaut, garde-fous si λ > 0 sans cache ou cache sans λ ;
  - `scripts/train_kd_second.sbatch` : recette d'efficience figée (ligne `== config` des 7 graines), `MODE=screen|confirm`, empreinte `config.txt`, cache et SECOND copiés dans `$SLURM_TMPDIR`.
  - Tests (`tests/test_distill.py`, CPU) : transforms inchangés sans `_aug` ; sous-ensemble d'ids ; **alignement de bout en bout** avec un faux professeur « oracle » dans un cache au format réel — 200 tirages de la vraie chaîne (flips, rot90, jitter, échange) sans une erreur, et le témoin « toujours la vue 0 » échoue bien ; perte nulle et gradient nul quand l'élève reproduit le professeur. Les tests existants passent. Entraînement CPU de 2 époques avec KD : termes `kd_change` et `kd_sem` présents et sommés au total.
- 26 sept., **durée d'un run : tranchée par `sacct`.** Les 28 jobs `csf-second` des 7–8 septembre (lot de reprise, 200 époques, recette `lean` et variantes, validation sur le test à chaque époque) ont duré **11 h 04 à 11 h 30** pour la plupart (4 jobs à 12 h 36–12 h 43, sans doute les variantes à encodeur `tiny`), soit **≈3,4 min/époque**. Le « 7 h 45 pour 100 époques » de l'en-tête de `train_second.sbatch` datait d'une recette antérieure. Un run de 120 époques dure donc ≈6,8 h (2 968 paires) / ≈6,1 h (2 671, validation sur 297). Avec le professeur en ligne (+≈2,7 min/époque) : ≈11,6 h / ≈12,9 h. **Plan complet ≈250 A100-h.**
- 26 sept. : `/scratch` très lent dans la journée — un `tar` des 23 310 PNG de SECOND a pris 47 min 31 s (0,4 s de calcul : attente d'E/S), et le premier run d'essai (job 4030416) a épuisé son heure dans `cp -r`. Deuxième essai (job 4039676) : l'archive `SECOND_stage.tar`, qui listait 23 310 PNG juste après sa création, a été retrouvée **tronquée à 16 711 680 octets** (3 650 entrées, date de modification 6 s avant la soumission du job) ; l'extraction n'a rien produit d'utilisable et le chargement a échoué avant l'entraînement. Une coupure au niveau du système de fichiers est plausible ; intégrité des autres fichiers de `/scratch` à vérifier. **Staging désormais par `unzip` de `$SCRATCH/SECOND.zip`** (archive d'origine, 1 fichier, CRC vérifié par fichier), garde-fou 2 968 / 1 694 paires avant l'entraînement, temps journalisés. Ce même essai a montré que la copie du cache (12 Go) prend 35 s.

Ce que `eval_perascd.py` mesure en un passage, pour fp32, fp16 et bf16 :

- SeK / Fscd / mIoU par **trois** calculs sur les mêmes prédictions : leur code (`legacy`), notre formule sur leur histogramme, et notre `SCDEvaluator` complet ;
- **contrôles de données** : pixels où `GT_CD` ≠ (label T1 > 0), où label T1 > 0 ≠ label T2 > 0 ;
- **diagnostic de permutation** des classes (appariement hongrois sur la matrice de confusion des pixels changés) : repère un ordre de classes différent entre notre SECOND et SECONDbi ;
- accord de l'opérateur déformable CUDA avec sa référence PyTorch ;
- latence (lot de 8, médiane de 50, protocole du README), mémoire crête, FLOPs d'une paire.

Vérifié localement (CPU, poids aléatoires, ViT-B, 3 paires synthétiques) : la chaîne
tourne de bout en bout et **les deux codes de métrique concordent à 4e-19**. Mesure
locale en passant : ViT-B + CG-Decoder = **409 GMACs par paire** hors opérateur
déformable, soit bien plus que les blocs ViT seuls (212). Mon estimation de
~1,3 TMAC pour le ViT-G ne comptait que ses blocs : le vrai coût est plus élevé,
et le surcoût de la KD en ligne (+40 %) est à recalculer sur la mesure de l'étape 0.

## 5. Risques qui pourraient invalider la comparaison

1. **Réglage sur le test.** Avec 12 à 18 configurations criblées, choisir λ/T/masque sur le test gonflerait le gain de KD. Parade : criblage sur val, confirmation sur test (§3). Le plus grave des risques listés.
2. **Puissance insuffisante.** 3 graines → Δ détectable ≥ 0,55 pt, soit le quart de l'écart. Un gain réel de 0,3–0,4 pt serait déclaré « non établi ». Parade : 5 graines sur les bras de conclusion.
3. **Professeur à graine unique.** Aucun σ disponible pour lui ; son 26,11 est un point, pas une moyenne. Le dénominateur de la « fraction d'écart comblée » porte donc une incertitude que l'on ne peut pas chiffrer : l'énoncer, ou donner la fraction pour les deux valeurs 26,11 (checkpoint) et 25,85 (fin d'entraînement).
4. **Baseline d'une autre époque du code.** Les 7 graines de référence datent d'août–septembre ; les bras KD tourneront sur un code modifié. Parade : test de non-régression à l'octet près du chemin sans KD, et, si `setup_env.sh` a changé de versions entre-temps, 2 graines de contrôle (+26 h).
5. **Données différentes.** SECONDbi et notre SECOND peuvent différer de codage ; un décalage d'indices de classes fausserait silencieusement la KL. Parade : comparaison pixel à pixel à l'étape 0.
6. **Canal 0 du professeur.** Jamais cible de la CE ; l'inclure dans une KL transmettrait du bruit. Restreindre aux canaux 1..6 et renormaliser.
7. **Normalisations différentes.** PerA pour le professeur, ImageNet pour l'élève : deux normalisations appliquées au même tenseur augmenté.
8. **Précision du professeur.** Entraîné en fp16 ; le passer en bf16 peut déplacer son SeK. Mesurer les trois à l'étape 0 et cacher dans celle qui reproduit 26,11.
9. **Écart de capacité (33×).** Un professeur trop gros transfère parfois moins bien qu'un intermédiaire (Mirzadeh et al., 2020). Le professeur VMamba-B publié (25,31, même famille que l'élève, normalisation ImageNet, pas d'op déformable) permet de le tester à coût marginal. Non retenu le 25 septembre ; première option à rouvrir.
10. **Transfert du « pourquoi ».** Les leviers du point d'efficience ne transfèrent pas tous à Hi-UCD. Un gain de KD sur SECOND seul reste un énoncé sur un jeu. Landsat-SCD exige un professeur réentraîné (checkpoint non publié).
11. **Matériel.** Les latences doivent toutes être mesurées sur la même carte (A100 Narval, protocole du README). Narval a des A100 **40 Go** : vérifier que professeur + élève tiennent ensemble pour la KD de features en ligne (étape 4).
12. **Licence des poids** (§2, non vérifié) : bloquante pour la publication d'un modèle dérivé, pas pour l'expérience.

---

## 6. Questions

Tranchées le 25 septembre : **Q1** parallèle à `robustcd` ; **Q2** sélection sur le
test conservée ; **Q3** point d'efficience ; **Q4** checkpoint PerASCD publié seul.

Encore ouvertes :

- ~~Q2bis~~ retenu : criblage sur val, confirmation sur test (§3).
- ~~Q5~~ tranchée : 4 étages (§1). ~~Q6~~ tranchée : les 7 graines sont là (§2bis).
- ~~Q7~~ pas de plafond d'heures, priorité de groupe bonne ; pas d'échéance ni de revue visée.
- ~~Q8~~ adopté : 120 époques.

Toutes les questions de cadrage sont tranchées. Prochaine étape : **P** (vérification sur Narval) puis **0** (reproduction du professeur).
