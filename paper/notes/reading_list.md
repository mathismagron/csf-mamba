# Bibliographie de DistilSCD : audit des citations et plan de lecture

Fichiers associés : `paper/notes/reading_list.csv` (une ligne par clé citée, 88 lignes : phrase(s) citante(s), affirmation soutenue, rôle, recommandation, quoi lire, temps estimé).
Périmètre : `paper/main.tex` et `paper/supplement.tex`. Le supplément ne contient aucune citation ; les 139 appels de citation (88 clés distinctes) sont tous dans `main.tex`. `refs.bib` contient 108 entrées, dont 20 ne sont pas citées (BibTeX les ignore ; elles n'apparaissent pas dans le PDF).
**Appliqué le 14 oct. 2026** : les 17 modifications (E8 et E17 adaptées au texte déjà corrigé) ; `main.tex` cite désormais 48 clés, le PDF compte 48 références et 14 pages. Le tableau des résultats publiés cite maintenant chaque ligne. Les 40 clés retirées restent dans `refs.bib` (non imprimées) : on peut en remettre une si une lecture le justifie.

## (a) Réponse courte à ta question

Non, garder les 88 références telles quelles n'est pas une bonne idée, et ton inquiétude est fondée. La norme est simple :

1. On cite ce sur quoi le texte s'appuie : une méthode, des données ou une métrique qu'on utilise, des chiffres auxquels on se compare, un travail proche dont on se distingue, ou une affirmation non évidente qu'on ne démontre pas soi-même.
2. Pour chaque citation, il faut avoir lu au moins la partie de l'article qui soutient la phrase où elle apparaît. Pour une affirmation précise (« Recht et al. montrent que... »), il faut avoir vérifié qu'ils le montrent. Lire le résumé suffit pour une citation générique ; ça ne suffit pas pour une affirmation chiffrée ou une page citée.
3. Le remplissage se trouve presque toujours dans les listes d'exemples de l'état de l'art (« ... sont pré-entraînés par MIM ou par contraste [13 références] »). Ces listes n'apportent rien au lecteur et c'est là qu'un relecteur repère les citations non lues.

L'audit confirme ce diagnostic : 40 des 88 clés sont des éléments de listes qu'on peut retirer sans affaiblir le texte. Il a aussi trouvé deux citations qui soutiennent mal la phrase où elles sont :

- `recht2019imagenet` est cité pour « réutiliser un jeu de test pour la sélection de modèles biaise la précision rapportée ». Or son résumé conclut que les baisses de précision observées « are not caused by adaptivity », c'est-à-dire qu'elles ne viennent pas de la réutilisation du jeu de test. La citation va contre la conclusion de l'article.
- `zhao2022decoupled` (DKD) est rangé parmi les « variantes basées sur les features », alors que DKD est une méthode de distillation des logits (son résumé reformule la perte KD classique).

Deux autres points relèvent de la même règle : `olofsson2014good` porte sur les bonnes pratiques d'estimation de surface et d'évaluation de la précision, pas spécifiquement sur la dépendance spatiale entre entraînement et test, et `wang2025bi` n'a été vérifié que sur son titre (related_work.md).

Combien de références pour JSTARS ? J'ai interrogé CrossRef (revue JSTARS, requête « semantic change detection remote sensing », articles publiés depuis 2025, 40 premiers résultats par pertinence). Sur les 23 articles de détection de changement dont le nombre de références est renseigné, la médiane est de 59 références (quartiles 52,5 et 67,5, min 43, max 90). Trois articles que nous citons : Mamba-FCS (JSTARS 2026) 62, CDKD (JSTARS 2024) 69, Liu et al. 2025 (JSTARS) 43. Ce n'est pas un échantillon systématique, mais il montre qu'une fourchette d'environ 45 à 70 références est normale pour un article régulier de CD dans JSTARS. 48 références se situent dans le bas de cette fourchette, ce qui est approprié pour un article qui ne propose pas d'architecture nouvelle.

## (b) Liste réduite proposée

Rien n'est retiré dans les catégories suivantes : méthodes et données utilisées, chiffres comparés, travaux les plus proches (BFD, CDKD, GUIDE, SCDistill, Vemulapalli et al.), et affirmations précises (capacity gap, born-again, label smoothing, validation spatiale, fuites). Toutes les clés retirées ne sont citées qu'une fois, dans une liste ; aucune autre phrase n'est touchée.

### Clés proposées pour suppression (40)

`bastani2023satlaspretrain`, `chen2024time`, `cui2022shuffle`, `dong2024changeclip`, `fuller2023croma`, `guo2024skysense`, `hong2024spectralgpt`, `huang2022knowledge`, `jakubik2023foundation`, `li2023lightweight`, `li2024new`, `liu2023structured`, `liu2025gstm`, `liu2025novel`, `liu2026jl1`, `lu2024weakly`, `lyons2018comparison`, `marsocci2024pangaea`, `meyer2021predicting`, `meyer2022machine`, `olofsson2014good`, `pang2024hicd`, `park2019relational`, `recht2019imagenet`, `reed2023scale`, `sun2023ringmo`, `tian2020contrastive`, `tong2026semdino`, `wang2023advancing`, `wang2024mtp`, `wang2025bi`, `wu2024lightweight`, `xie2020self`, `xiong2024neural`, `yang2022cross`, `yun2021re`, `zagoruyko2016paying`, `zhao2022decoupled`, `zheng2022changemask`, `zheng2024segment`

### Modifications de phrases (avant / après), toutes dans `main.tex`

Les numéros de ligne correspondent à la version actuelle de `main.tex`.

**E1 — l. 159-161 (Related Work, SCD).**
Avant : `Bi-SRNet \cite{ding2022bi} and SCanNet~\cite{ding2024joint} added bi-temporal semantic reasoning and spatio-temporal Transformer modelling, and ChangeMask~\cite{zheng2022changemask} decoupled semantic and change branches.`
Après : `Bi-SRNet~\cite{ding2022bi} and SCanNet~\cite{ding2024joint} added bi-temporal semantic reasoning and spatio-temporal Transformer modelling.`

**E2 — l. 162-165.**
Avant : `State-space models entered the field with ChangeMamba \cite{chen2024changemamba}, built on Mamba~\cite{gu2023mamba} and VMamba \cite{liu2024vmamba}, and were followed by Mamba-FCS~\cite{wijenayake2026mamba} and GSTM-SCD \cite{liu2025gstm}.`
Après : `State-space models entered the field with ChangeMamba~\cite{chen2024changemamba}, built on Mamba~\cite{gu2023mamba} and VMamba~\cite{liu2024vmamba}, and were followed by Mamba-FCS~\cite{wijenayake2026mamba}.`

**E3 — l. 165-167.**
Avant : `The strongest recent results use foundation-model encoders: PerASCD \cite{shen2026foundation}, DBTANet~\cite{li2026dual} with a SAM branch, and SemDINO \cite{tong2026semdino}.`
Après : `The strongest recent results use foundation-model encoders, for example PerASCD~\cite{shen2026foundation} and DBTANet~\cite{li2026dual}, which adds a SAM branch.`

**E4 — l. 167-168.**
Avant : `Lightweight change detection has mostly been pursued by architecture design for binary change~\cite{codegoni2023tinycd,li2023lightweight,cui2022shuffle}; Corley \emph{et al.}~\cite{corley2024change} caution that well-trained simple baselines remain competitive.`
Après : `Lightweight change detection has mostly been pursued by architecture design for binary change, for example TinyCD~\cite{codegoni2023tinycd}; Corley \emph{et al.}~\cite{corley2024change} caution that well-trained simple baselines remain competitive.`

**E5 — l. 174-177 (Foundation Models).**
Avant : `Remote-sensing foundation models are pre-trained by masked image modelling or contrastive objectives on large archives~\cite{cong2022satmae,reed2023scale,sun2023ringmo, wang2023advancing,manas2021seasonal,bastani2023satlaspretrain,fuller2023croma, hong2024spectralgpt,jakubik2023foundation,guo2024skysense,xiong2024neural,wang2024mtp, shen2026contrastive}.`
Après : `Remote-sensing foundation models are pre-trained on large archives, by masked image modelling as in SatMAE~\cite{cong2022satmae} or by contrastive objectives as in SeCo~\cite{manas2021seasonal} and PerA~\cite{shen2026contrastive}.`

**E6 — l. 178-180.**
Avant : `For change detection they are usually kept inside the deployed network, either fine-tuned~\cite{shen2026foundation,ding2024adapting,li2024new,chen2024time, dong2024changeclip,li2026dual} or used zero-shot~\cite{zheng2024segment}.`
Après : `For change detection they are usually fine-tuned and kept inside the deployed network~\cite{shen2026foundation,ding2024adapting,li2026dual}.`

**E7 — l. 181-182.**
Avant : `Benchmarking studies report that such comparisons are sensitive to evaluation details~\cite{marsocci2024pangaea, corley2024revisiting}; we encountered the same sensitivity when reproducing a published teacher (Section~\ref{sec:teachers}).`
Après : `A benchmarking study reports that such comparisons are sensitive to preprocessing, in particular resizing and normalisation~\cite{corley2024revisiting}; we encountered the same sensitivity when reproducing a published teacher (Section~\ref{sec:teachers}).`
(Plus précis qu'avant : c'est exactement le problème de normalisation du teacher VMamba-B.)

**E8 — l. 187-189 (Knowledge Distillation).**
Avant : `Output distillation~\cite{hinton2015distilling} and its feature-based variants \cite{romero2014fitnets,zagoruyko2016paying,park2019relational,tian2020contrastive, zhao2022decoupled} are surveyed in~\cite{gou2021knowledge}.`
Après : `Output distillation~\cite{hinton2015distilling} and feature-based variants such as FitNets~\cite{romero2014fitnets} are surveyed in~\cite{gou2021knowledge}.`
(Corrige aussi le classement erroné de DKD.)

**E9 — l. 190-191.**
Avant : `...larger teachers do not always make better teachers for small students~\cite{cho2019efficacy,mirzadeh2020improved,huang2022knowledge}, although...`
Après : `...larger teachers do not always make better teachers for small students~\cite{cho2019efficacy,mirzadeh2020improved}, although...`

**E10 — l. 193-194.**
Avant : `...akin to stored per-crop soft labels~\cite{shen2022fast,yun2021re};`
Après : `...akin to the stored per-crop soft labels of FKD~\cite{shen2022fast};`

**E11 — l. 196-197.**
Avant : `Second, students can outperform their teachers \cite{furlanello2018born,xie2020self,yuan2020revisiting}, and...`
Après : `Second, students can outperform their teachers~\cite{furlanello2018born,yuan2020revisiting}, and...`

**E12 — l. 201-202.**
Avant : `For dense prediction, structured and channel-wise distillation transfer spatial structure for segmentation \cite{liu2019structured,liu2023structured,shu2021channel,yang2022cross}.`
Après : `For dense prediction, structured and channel-wise distillation transfer spatial structure for segmentation~\cite{liu2019structured,shu2021channel}.`

**E13 — l. 209-211 (Distillation in RS and CD).**
Avant : `..., and several works align features between large and small binary change detectors~\cite{wang2025bi,yang2026intrascale, wu2024lightweight}.`
Après : `..., and feature alignment between large and small binary change detectors has also been explored~\cite{yang2026intrascale}.`

**E14 — l. 214-217.**
Avant : `Other teacher--student work targets accuracy or robustness rather than compression: multi-teacher distillation~\cite{liu2026jl1}, self-distillation~\cite{liu2025novel,qu2026semantic,pang2024hicd}, weak supervision \cite{lu2024weakly}, and GUIDE~\cite{huang2026guide}, a teacher--student framework for SCD under cross-temporal domain shift.`
Après : `Other teacher--student work targets robustness rather than compression: SCDistill~\cite{qu2026semantic} uses self-distillation to make change detectors robust to semantic perturbations, and GUIDE~\cite{huang2026guide} is a teacher--student framework for SCD under cross-temporal domain shift.`
(SCDistill est nommé, puisqu'il fait partie des travaux les plus proches ; vérifier le nom et la formulation à la lecture.)

**E15 — l. 224-225 (Evaluation Leakage).**
Avant : `...blocked or spatial cross-validation is the remedy~\cite{roberts2017crossvalidation,ploton2020spatial, meyer2021predicting,meyer2022machine}.`
Après : `...blocked or spatial cross-validation is the remedy~\cite{roberts2017crossvalidation,ploton2020spatial}.`

**E16 — l. 226-227.**
Avant : `In remote sensing, spatial dependence between training and test samples is a documented pitfall of accuracy assessment~\cite{lyons2018comparison, karasiak2022spatial,kattenborn2022spatially,olofsson2014good}.`
Après : `In remote sensing, spatial dependence between training and test samples is a documented pitfall of accuracy assessment~\cite{karasiak2022spatial,kattenborn2022spatially}.`

**E17 — l. 232-233.**
Avant : `Reusing a test set for model selection biases reported accuracy~\cite{dwork2015reusable,recht2019imagenet}, which motivates...`
Après : `Reusing a test set for model selection can bias reported accuracy~\cite{dwork2015reusable}, which motivates...`

Après ces modifications : relancer `python paper/scripts/check_numbers.py` (aucun nombre n'est touché, il doit rester à 0 problème) puis `tectonic main.tex` et vérifier qu'aucune citation n'est indéfinie.

### Autres points relevés (hors suppression)

- Le Tableau 3 (`tables_tex/table3_published.tex`) compare des chiffres publiés (Mamba-FCS, MambaSCD, SCanNet ré-entraîné par PerASCD, PerASCD) sans citation dans les lignes ; la colonne « Provenance » renvoie à « the cited paper ». Ajouter `\cite{...}` par ligne dans le générateur `make_tables_tex.py`, sans modifier le `.tex`. Cela n'ajoute aucune clé nouvelle.
- Tu as lu plusieurs travaux qui ne sont pas cités : CDMamba, CSSM, AtrousMamba, ChessMamba, FUSU, et Hi-UCD (`tian2020hi`, présent dans refs.bib mais non cité). Ne les ajoute pas simplement parce que tu les as lus ; ajoute-les seulement si une phrase en a besoin. La seule place naturelle serait la phrase E2, si tu veux mentionner les détecteurs de changement binaires à base de Mamba (+4 références, à vérifier dans CrossRef comme les autres).
- Certaines entrées de refs.bib pointent vers la version arXiv alors qu'une version publiée existe probablement (FitNets, Bouthillier et al.) ; à vérifier, sans effet sur le décompte.

## (c) Plan de lecture par priorité

Les temps de lecture sont des estimations pour une lecture ciblée (pas une lecture complète de chaque article). Le détail en anglais, avec l'affirmation exacte que chaque citation soutient, est dans `reading_list.csv`.

### Priorité 1 — avant soumission : ce que le papier utilise ou dont il se distingue (15 h + 2 h de relecture)

| clé | référence | quoi lire | h |
|---|---|---|---|
| `shen2026foundation` | Shen, 2026, arXiv 2602.13780 | Article complet. Emplacements vérifiés (arXiv v3) : split p. 8, Tableau I p. 11 (SECOND, Landsat-SCD, SCanNet ré-entraîné), Tableau III p. 13 (paramètres/FLOPs), Tableau VII p. 15. Section méthode : encodeur PerA, ViT-Adapter, décodeur cascade gated. | 2 |
| `yuan2022transformer` | Yuan, 2022, Int. J. Digital Earth | Section jeu de données + p. 11 (split aléatoire 6053/1729/686) + p. 19 (extension multi-régions en perspective). Les deux pages ont été vérifiées dans le texte intégral (related_work.md). | 1.5 |
| `ding2022bi` | Ding, 2022, IEEE TGRS | Résumé + méthode (définition de la perte de cohérence sémantique reprise comme L_sc) + protocole expérimental (split de SECOND). | 1.5 |
| `hinton2015distilling` | Hinton, 2015, arXiv 1503.02531 | Article court, en entier : objectif de distillation et température. | 1 |
| `wang2025burden` | Wang S., 2025, IEEE TGRS | Article complet : méthode (appariement de features bi-temporel, distillation contrastive par patch) et expériences. Un relecteur demandera la comparaison. | 2 |
| `wang2024knowledge` | Wang G., 2024, IEEE JSTARS | Article complet : méthode (termes de perte sur les probabilités de sortie) et expériences (jeux de données, tailles d'élèves). | 2 |
| `vemulapalli2024knowledge` | Vemulapalli, 2024, ICML | Article complet : méthode et comparaison principale transfert orienté tâche vs transfert agnostique. | 2 |
| `huang2026guide` | Huang, 2026, IEEE TGRS | Article complet : méthode (guidage par le teacher, mélange de distributions) et expériences SCD (jeux de données, split, présence ou non d'un modèle de fondation). | 1.5 |
| `qu2026semantic` | Qu, 2026, ECCV (arXiv 2607.19000) | Résumé + méthode : vérifier qu'il s'agit d'auto-distillation pour la robustesse et non de compression ; confirmer que « SCDistill » est bien le nom donné par les auteurs. | 1.5 |

Relecture rapide des articles déjà lus (vérifier les emplacements cités) :

| clé | référence | quoi lire | h |
|---|---|---|---|
| `yang2022asymmetric` | Yang, 2022, IEEE TGRS | Déjà lu. Revérifier la description du jeu de données et la définition du SeK implémentée. | 0.5 |
| `chen2024changemamba` | Chen, 2024, IEEE TGRS | Déjà lu. Revérifier les Tableaux VII-VIII p. 13 (arXiv v7, emplacements vérifiés) et le passage qui justifie « meilleur SeK test sur les époques ». | 0.5 |
| `wijenayake2026mamba` | Wijenayake, 2026, IEEE JSTARS | Déjà lu. Revérifier Tableau V p. 15 et Tableau VI p. 18 (arXiv v2, emplacements vérifiés). | 0.5 |
| `liu2024vmamba` | Liu, 2024, NeurIPS | Déjà lu. Revérifier la configuration VMamba-T (profondeurs, largeurs) par rapport à notre variante « mini ». | 0.5 |

### Priorité 2 — théorie de la distillation et des fuites invoquée dans l'introduction et la discussion (7 h)

| clé | référence | quoi lire | h |
|---|---|---|---|
| `cho2019efficacy` | Cho, 2019, ICCV | Résumé + expériences sur la taille du teacher + leur remède (arrêt précoce du teacher). Central pour la discussion à deux teachers. | 1.5 |
| `beyer2022knowledge` | Beyer, 2022, CVPR | Résumé + définition de l'enseignement « consistant » et comparaison des stratégies (teacher fixe, bruit indépendant, consistant). Nécessaire pour écrire « pas entièrement consistant au sens de Beyer et al. ». | 1.5 |
| `furlanello2018born` | Furlanello, 2018, ICML | Résumé + méthode (auto-distillation séquentielle) + résultats principaux. | 1 |
| `yuan2020revisiting` | Yuan, 2020, CVPR | Résumé + expériences avec teacher faible / « reversed KD » + analyse label smoothing. Notre argument d'attribution en dépend. | 1.5 |
| `kapoor2023leakage` | Kapoor, 2023, Patterns | Taxonomie des fuites, en particulier L1.4 (doublons) et L3.2 (non-indépendance) et la réserve citée dans related_work.md. | 1.5 |

### Priorité 3 — lectures ciblées (KEEP-SKIM : résumé + la section qui soutient la phrase) (15.5 h)

| clé | référence | h |
|---|---|---|
| `daudt2019multitask` | Caye Daudt, 2019, CVIU | 0.5 |
| `ding2024joint` | Ding, 2024, IEEE TGRS | 0.75 |
| `shen2026contrastive` | Shen, 2026, Geo-spatial Inf. Sci. | 0.75 |
| `chen2023vision` | Chen, 2023, ICLR | 0.5 |
| `mirzadeh2020improved` | Mirzadeh, 2020, AAAI | 0.5 |
| `stanton2021knowledge` | Stanton, 2021, NeurIPS | 0.5 |
| `muller2019label` | Muller, 2019, NeurIPS | 0.5 |
| `roberts2017crossvalidation` | Roberts, 2017, Ecography | 0.75 |
| `ploton2020spatial` | Ploton, 2020, Nature Comm. | 0.5 |
| `kattenborn2022spatially` | Kattenborn, 2022, ISPRS Open J. | 0.5 |
| `gu2023mamba` | Gu, 2023, arXiv 2312.00752 | 0 |
| `li2026dual` | Li, 2026, ICASSP | 0.75 |
| `codegoni2023tinycd` | Codegoni, 2023, Neural Comput. Appl. | 0.5 |
| `corley2024change` | Corley, 2024, arXiv 2402.06994 | 0.5 |
| `cong2022satmae` | Cong, 2022, NeurIPS | 0.5 |
| `manas2021seasonal` | Manas, 2021, ICCV | 0.5 |
| `ding2024adapting` | Ding, 2024, IEEE TGRS | 0.5 |
| `corley2024revisiting` | Corley, 2024, CVPRW | 0.5 |
| `romero2014fitnets` | Romero, 2014, arXiv 1412.6550 | 0.5 |
| `gou2021knowledge` | Gou, 2021, IJCV | 0.5 |
| `shen2022fast` | Shen, 2022, ECCV | 0.5 |
| `szegedy2016rethinking` | Szegedy, 2016, CVPR | 0.25 |
| `liu2019structured` | Liu, 2019, CVPR | 0.5 |
| `shu2021channel` | Shu, 2021, ICCV | 0.5 |
| `himeur2025applications` | Himeur, 2025, Information Fusion | 0.75 |
| `yang2026intrascale` | Yang, 2026, IEEE GRSL | 0.5 |
| `karasiak2022spatial` | Karasiak, 2022, Machine Learning | 0.5 |
| `dwork2015reusable` | Dwork, 2015, Science | 0.5 |
| `bouthillier2021accounting` | Bouthillier, 2021, arXiv 2103.03098 | 0.5 |
| `liu2023upsample` | Liu, 2023, ICCV | 0.5 |

### Total

- Lecture obligatoire (KEEP-MUST-READ, 18 articles) : 24 h, dont 22 h de nouvelles lectures et 2 h de relecture des 4 articles déjà lus.
- Lectures ciblées (KEEP-SKIM, 30 articles) : 15.5 h.
- Total : 39.5 h, soit environ cinq jours de travail. La priorité 1 seule (17 h) couvre ce qu'un relecteur vérifiera en premier.

Méthode conseillée : pour chaque article, noter dans le CSV (colonne à ajouter, par ex. `verified_by_author`) la date de lecture et une phrase disant si l'affirmation est confirmée. Si une affirmation n'est pas confirmée, corriger la phrase du papier plutôt que garder la citation.

## (d) Nombre de références visé

- Actuellement : 88 références citées (139 appels de citation).
- Après réduction : 48 références (40 supprimées). C'est dans la fourchette observée pour JSTARS (médiane 59, quartiles 52,5-67,5 sur l'échantillon CrossRef ci-dessus).
- Si tu ajoutes les travaux Mamba déjà lus dans la phrase E2 : 52 au maximum.
- Répartition des 48 : 9 méthodes/données/métriques utilisées, 3 comparaisons chiffrées directes, 5 travaux les plus proches, 27 affirmations précises, 4 exemples uniques gardés dans des listes réduites.
