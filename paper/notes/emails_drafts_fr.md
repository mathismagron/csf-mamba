# Brouillons de mails (à relire avant envoi)

## 1. Aux encadrants (Prof. Eric Hervet, Prof. Andy Couturier) — une fois l'article finalisé

**Objet :** Article issu du stage — distillation d'un modèle de fondation pour la SCD (soumission JSTARS envisagée)

Bonjour Professeur Hervet, bonjour Professeur Couturier,

J'espère que vous allez bien. Depuis la fin du stage, j'ai poursuivi le travail sur Narval
(compte def-hervete) et il a abouti à un article que je souhaiterais soumettre à IEEE JSTARS,
avec vous comme co-auteurs si vous l'acceptez.

En résumé :
- en distillant uniquement les sorties de PerASCD (548 M paramètres, meilleur résultat publié sur
  SECOND) dans le modèle compact du stage (16,5 M), l'élève égale son professeur sur SECOND
  (SeK 26,20 contre 26,11) avec 51 fois moins de calcul ;
- le découpage standard de Landsat-SCD place toutes les images de test dans l'entraînement ; nous
  proposons un protocole disjoint par emplacement, sous lequel la distillation améliore l'élève sur
  chaque pli et l'élève dépasse son professeur ;
- le modèle a été renommé DistilSCD (les blocs propres à CSF-Mamba n'apportaient pas de gain
  mesurable ; c'est expliqué en matériel supplémentaire).

Vous trouverez ci-joints le manuscrit et le matériel supplémentaire. J'aurais besoin de votre avis
sur trois points avant la soumission :
1. accepteriez-vous d'être co-auteurs, et dans quel ordre / avec quelles affiliations ?
2. le choix de la revue (JSTARS, accès ouvert, frais de publication d'environ 1 800 à 2 160 USD) et
   la prise en charge éventuelle de ces frais ;
3. le mail aux auteurs de PerASCD (brouillon ci-dessous), en particulier la question de la licence
   des poids.

Le code, les journaux et toutes les tables sont dans le dépôt GitHub (lien). Merci d'avance pour
votre relecture.

Bien cordialement,
Mathis Magron

## 2. Aux auteurs de PerASCD (en anglais, à valider par les encadrants)

**Subject:** PerASCD checkpoints — licence question and two reproducibility notes

Dear Dr. Shen and co-authors,

Thank you for releasing the PerASCD code, data and checkpoints. We used the released SECOND
checkpoint as a teacher for knowledge distillation into a compact change detector, and we
reproduced its score exactly (SeK 26.108 with our evaluation code).

1. **Licence.** The code is under the MIT licence, but we could not find a licence for the PerA
   pre-trained weights or the PerASCD checkpoints. May we use the checkpoint as a teacher in a
   publication, and may we release student weights trained from its outputs? We do not plan to
   redistribute your weights.
2. **VMamba-B checkpoint.** The released VMamba-B + CG-Decoder checkpoint reproduces its stored
   score (SeK 25.314) only with ImageNet normalisation; with the PerA statistics used by the
   released evaluation script we obtain 23.73. This may help other users.
3. **Landsat-SCD split.** In the LandsatSCD512 archive, the 2 385 pairs are built from 333 distinct
   images (12 tiles × about 28 dates), so that every validation and test pair has both of its
   images in the training split; a learning-free lookup reaches SeK 77.9 on the test split. This
   follows the random protocol of the original dataset paper, which targets a single-area time
   series; we mention it in case it is useful, together with our verification script (attached).
   For our study we used a location-disjoint cross-validation.

Best regards,
Mathis Magron (with E. Hervet and A. Couturier, Université de Moncton)
