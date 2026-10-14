# Révision de l'article — vague A d'expériences (décidée le 13 oct. 2026)

Réponse aux trois relecteurs internes (`paper/notes/internal_review_round1.md`), qui demandent
tous des contrôles montrant que le gain de la KD vient du **savoir du professeur** et non
d'une simple régularisation. Coût prévu : ≈ 77 A100-h (+ 19 en option).

## 1. Expériences

| # | Bras | Données | Graines | Question | A100-h |
|---|---|---|---|---|---|
| 0 | caches « élève professeur » | SECOND + plis 1–3 | — | prérequis de 1a/1b | ≈ 2 |
| 1a | `confirm_selfkd` : KD (λ_chg 8, λ_sem 1, T 2, mask all), professeur = élève **sans KD** graine 1 | SECOND (2968 / test) | 2, 3, 4 | auto-distillation (born-again) : la KD d'un professeur *non meilleur* que l'élève donne-t-elle le même gain ? | ≈ 19 |
| 1b | `landsat_f<F>_selfkd` : idem, professeur = `landsat_f<F>_base-s1` | plis 1–3 (100 ép.) | 2, 3 | idem, sur données jamais vues pendant la mise au point | ≈ 14 |
| 2 | `landsat_f<F>_ctl-gt` : mêmes pertes, λ, T, masque ; cibles = vérité lissée (changement, `gt_smooth`) + lissage d'étiquettes apparié (sémantique, `gt_ls`) | plis 1–3 (100 ép.) | 1, 2, 3 | le gain survit-il quand la cible ne contient plus que la vérité terrain (adoucie comme celle du professeur) ? | ≈ 22 |
| 3 | `confirm_nokd-cos` : sans KD, LR cosinus (min 1e-6) | SECOND | 1, 2, 3 | le LR constant de la recette désavantage-t-il la baseline ? | ≈ 19 |
| 3b | *(option)* `confirm_kdlogits-cos` : KD complète, LR cosinus | SECOND | 1, 2, 3 | le gain subsiste-t-il sous un LR décroissant ? | ≈ 19 |
| 4 | cartes qualitatives (Fig. 7) | SECOND test + pli 1 test | — | inférence seule | < 1 |

Choix et justifications :

* **Professeur élève = graine 1, élèves = autres graines.** Un élève de même graine que son
  professeur partirait de la même initialisation et verrait les mêmes lots : on l'évite.
  Comparaisons appariées par graine avec les bras existants (sans KD : `…-lean-augswap-ema-s*`
  sur SECOND, `landsat_f<F>_base-s*` ; KD : `confirm_kdlogits-s*`, `landsat_f<F>_kdlogits-s*`).
* **Sélection du checkpoint professeur** : sur SECOND, `best.pt` = meilleur SeK test (époque 100,
  23,87), la convention du checkpoint publié de PerASCD ; sur Landsat, meilleur SeK de
  validation du pli, comme les professeurs ViT-G de pli.
* **Résolution** : l'élève prédit à 512 ; ses logits sont moyennés 4 × 4 vers 128 (résolution
  native du professeur), puis la KD les resuréchantillonne comme d'habitude. `meta.json`
  enregistre le SeK du train avant / après cette réduction.
* **Contrôle `gt_ls`** : sur un pixel changé étiqueté, (1 − ε)·one-hot + ε/(C − 1) ; ailleurs,
  uniforme ; puis même lissage spatial que `gt_smooth`. ε est apparié au professeur : la classe
  vraie reçoit la probabilité moyenne que lui donne le professeur à T = 2 sur le train du pli
  (`scripts/kd_control_eps.py`, écrit `kd_control_eps.json` dans le dossier du run). Seul ce
  scalaire vient du professeur ; le cache PerASCD est lu mais n'est jamais une cible (vérifié par
  `tests/test_revision_controls.py`).
* **Contrôle sur les plis 1–3** plutôt que sur SECOND : ces données n'ont servi à aucun choix de
  la recette ; le contrôle de criblage d'origine (λ = 2, terme de changement seul, 2 graines,
  validation) est ainsi remplacé par un contrôle à la configuration retenue, sur données propres.

## 2. Code (testé localement : 63 tests passent ; entraînement de bout en bout sur données synthétiques)

* `scripts/teacher/cache_student.py` + `cache_student.sbatch` : cache au format exact du cache
  PerASCD, lu sans modification par `TeacherCacheDataset`.
* `distilscd/losses/distill.py` : `sem_target="gt_ls"`, `sem_eps` ; `scripts/train.py` :
  `--kd-sem-target`, `--kd-sem-eps`.
* `scripts/kd_control_eps.py` : ε apparié.
* `scripts/train_kd_second.sbatch` : `KD_TEACHER=student`, `LR_SCHED=cosine`, `KD_SEM_TGT/EPS`.
  `scripts/train_kd_landsat.sbatch` : `KD_TEACHER=student`, `KD_TGT`, `KD_SEM_TGT`,
  `KD_SEM_EPS=auto`. Aux valeurs par défaut, les `config.txt` des runs existants sont inchangés
  (aucune collision).
* `tests/test_revision_controls.py` (3 tests).

## 3. Commandes Narval

Après `git pull` dans `~/csf-mamba` (voir `documentation/rename.md` §5b si le dépôt a été renommé) :

```bash
cd ~/csf-mamba && git pull && git log -1 --oneline
# vérifier que les checkpoints professeurs existent encore (purge de /scratch !)
ls -l $SCRATCH/csf-mamba-runs/second_mini_chess_crop512-lean-augswap-ema-s1/best.pt \
      $SCRATCH/csf-distill/runs/landsat_f{1,2,3}_base-s1/best.pt \
      $SCRATCH/csf-distill/cache/perascd_landsat_f{1,2,3}_train/meta.json

# 0. caches de l'élève professeur (≈ 15–30 min chacun)
C0=$(sbatch --parsable scripts/teacher/cache_student.sbatch)
for f in 1 2 3; do eval "C$f=$(FOLD=$f sbatch --parsable --export=ALL,FOLD scripts/teacher/cache_student.sbatch)"; done
echo $C0 $C1 $C2 $C3

# 1a. auto-distillation SECOND (après le cache SECOND)
for s in 2 3 4; do
  MODE=confirm SEED=$s LKD_CHG=8 LKD_SEM=1 KD_T_SEM=2 KD_MASK=all KD_TEACHER=student TAG=selfkd \
    sbatch --dependency=afterok:$C0 --export=ALL,MODE,SEED,LKD_CHG,LKD_SEM,KD_T_SEM,KD_MASK,KD_TEACHER,TAG \
    scripts/train_kd_second.sbatch
done

# 1b. auto-distillation, plis 1–3 (après le cache du pli)
for f in 1 2 3; do dep=$(eval echo \$C$f); for s in 2 3; do
  FOLD=$f SEED=$s EPOCHS=100 LKD_CHG=8 LKD_SEM=1 KD_T_SEM=2 KD_MASK=all KD_TEACHER=student TAG=selfkd \
    sbatch --dependency=afterok:$dep --export=ALL,FOLD,SEED,EPOCHS,LKD_CHG,LKD_SEM,KD_T_SEM,KD_MASK,KD_TEACHER,TAG \
    scripts/train_kd_landsat.sbatch
done; done

# 2. contrôle « vérité seule », plis 1–3 (aucune dépendance : cache PerASCD existant)
for f in 1 2 3; do for s in 1 2 3; do
  FOLD=$f SEED=$s EPOCHS=100 LKD_CHG=8 LKD_SEM=1 KD_T_SEM=2 KD_MASK=all KD_TGT=gt_smooth KD_SEM_TGT=gt_ls TAG=ctl-gt \
    sbatch --export=ALL,FOLD,SEED,EPOCHS,LKD_CHG,LKD_SEM,KD_T_SEM,KD_MASK,KD_TGT,KD_SEM_TGT,TAG \
    scripts/train_kd_landsat.sbatch
done; done

# 3. baseline sans KD, LR cosinus, SECOND
for s in 1 2 3; do
  MODE=confirm SEED=$s LR_SCHED=cosine TAG=nokd-cos sbatch --export=ALL,MODE,SEED,LR_SCHED,TAG scripts/train_kd_second.sbatch
done
# 3b. (option) KD complète, LR cosinus
# for s in 1 2 3; do MODE=confirm SEED=$s LR_SCHED=cosine LKD_CHG=8 LKD_SEM=1 KD_T_SEM=2 KD_MASK=all TAG=kdlogits-cos \
#   sbatch --export=ALL,MODE,SEED,LR_SCHED,LKD_CHG,LKD_SEM,KD_T_SEM,KD_MASK,TAG scripts/train_kd_second.sbatch; done

# 4. cartes qualitatives (Fig. 7)
sbatch paper/scripts/qualitative.sbatch
```

Contrôles rapides une fois les jobs partis :

```bash
squeue -u $USER
cat $SCRATCH/csf-distill/cache/student_second_train/meta.json | python -m json.tool | sed -n '/checks/,/}/p'
cat $SCRATCH/csf-distill/runs/landsat_f1_ctl-gt-s1/kd_control_eps.json
grep -m1 "Distillation" logs/csf-kd-*.out | tail -5      # cible=gt_smooth / gt_ls ε=… ; cache student_*
```

Rapatriement (local) quand tout est fini :

```bash
ssh magron13@narval.alliancecan.ca 'cd $SCRATCH/csf-distill && tar cf - --exclude="*.pt" runs/confirm_selfkd-s* runs/confirm_nokd-cos-s* \
    runs/landsat_f?_selfkd-s* runs/landsat_f?_ctl-gt-s* cache/student_*/meta.json qualitative' > logs/revision_a.tar
mkdir -p logs/revision_a && tar xf logs/revision_a.tar -C logs/revision_a
```

## 4. Lecture prévue des résultats (fixée avant de les voir)

* **1a/1b** : gain auto-KD = selfkd − sans KD (apparié par graine) ; part attribuable au savoir du
  professeur = KD PerASCD − selfkd. Si selfkd ≈ KD PerASCD, l'article doit présenter la KD
  comme une régularisation et non comme un transfert depuis le modèle de fondation.
* **2** : ctl-gt − sans KD et KD − ctl-gt, appariés par pli × graine (9 paires).
* **3** : nokd-cos vs sans KD constant (SeK max et final) ; si l'écart dépasse ≈ 0,5, refaire 3b.
* Rapport : moyennes ± écart-type, IC à 95 % (t apparié), cohérence de signe ; pas de Wilcoxon
  sur paires pli × graine (pseudo-réplication).

## 5. Résultats (14 oct. 2026)

21 runs, 74 A100-h (+ caches ≈ 1 h). Analyse : `paper/scripts/analyze_revision_a.py` →
`paper/tables/revision_a_{runs,arms,stats,caches,misc,cost}.csv`. La règle de sélection Landsat
recalculée sur les logs bruts reproduit la table existante à 4·10⁻¹⁵ près.

| Comparaison | Données | Δ SeK [IC 95 %] | p | signes |
|---|---|---|---|---|
| auto-KD − sans KD | SECOND (graines 2–4) | +0,75 [0,41 ; 1,08] (final +1,05) | 0,011 | 3/3 |
| KD PerASCD − auto-KD | SECOND | +1,52 [1,35 ; 1,70] | 0,0007 | 3/3 |
| auto-KD − son professeur (23,87) | SECOND | +0,75 | 0,005 | 3/3 |
| sans KD cosinus − sans KD constant | SECOND (1–3) | −0,33 [−0,68 ; 0,03] (final −0,82) | 0,06 | 0/3 |
| KD − sans KD cosinus | SECOND | +2,70 | < 10⁻⁴ | 3/3 |
| auto-KD − sans KD | plis 1–3 (graines 2–3) | −0,04 [−0,86 ; 0,77] | 0,85 | 1/6 |
| KD PerASCD − auto-KD | plis 1–3 | +2,61 [1,73 ; 3,49] | 0,006 | 6/6 |
| vérité seule − sans KD | plis 1–3 (1–3) | −0,21 [−1,04 ; 0,61] | 0,38 | 4/9 |
| KD PerASCD − vérité seule | plis 1–3 | +2,75 [1,55 ; 3,94] | 0,010 | 9/9 |

(Landsat : IC sur les 3 moyennes de pli ; signes = paires pli × graine, simple comptage.)

Lecture, selon la règle fixée au §4 :
* **Le gain dépend des cibles du modèle de fondation.** Sur les plis tenus à l'écart de toute
  mise au point, ni l'auto-distillation ni la vérité terrain adoucie (ε apparié : 0,21–0,28) ne
  font gagner quoi que ce soit ; les cibles PerASCD donnent +2,6 à +2,8 de plus.
* **Sur SECOND, l'auto-distillation récupère environ un tiers du gain** (0,75 sur 2,27 aux
  mêmes graines), effet « born-again » classique : l'élève auto-distillé dépasse même son
  professeur de 0,75. Les deux tiers restants exigent PerASCD.
* **Le LR constant ne désavantage pas la baseline** : le cosinus fait moins bien, surtout à la
  dernière époque ; 3b (KD cosinus) inutile.
* Réserve : moyenner les logits de l'élève 512 → 128 coûte 0,22 SeK (train) sur SECOND mais
  2,7–3,4 sur les plis Landsat (changements petits et fragmentés) ; l'auto-distillation y est donc
  désavantagée. Le contrôle « vérité seule » n'a pas ce défaut et conduit à la même conclusion.

Article mis à jour : sous-section IV-G (bras de contrôle), section VII (contrôles + Fig. 7
qualitative), table VIII, discussion VIII-A, limites, conclusion, résumé ; supplément : tables
des runs et des caches de contrôle. `check_numbers.py` : 0 problème.
