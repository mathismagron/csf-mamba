# Cover letter (draft, to adapt before submission)

Dear Editor,

We submit the manuscript "DistilSCD: Foundation-Model Accuracy at Lightweight Cost for Semantic
Change Detection through Output Distillation, with a Location-Disjoint Landsat-SCD Protocol" for
consideration as a regular paper in the IEEE Journal of Selected Topics in Applied Earth
Observations and Remote Sensing [option: for the special issue "Foundation Models and Agents in
Earth Observation"].

Remote-sensing foundation models now hold the best published accuracy for semantic change
detection, but their size (548 M parameters and about 1.5 TMACs per image pair for the strongest
SECOND model) makes large-scale deployment costly. We show that distilling only the outputs of
such a model into a 16.5 M-parameter student, with an offline, augmentation-consistent logit
cache, lets the student match its teacher on SECOND at 51 times lower compute, without any change
to the student or to its training cost. The study is designed to separate the effect of the
teacher from seed noise and from test-set selection: hyper-parameters were fixed on an internal
validation split, every comparison is paired over five seeds, and ground-truth target controls
and a second teacher test where the gain comes from.

The work also contributes an evaluation finding of direct use to the JSTARS readership. The
standard random pair split of Landsat-SCD places both images of every test pair in the training
set; a learning-free lookup exceeds all published results. We document this at the pixel level,
explain why it follows from the dataset's original design as a single-area time series, and
release a location-disjoint four-fold protocol with its verification script. Under that protocol,
distillation improves the student on every fold and seed, and the distilled student outperforms
its teacher on all folds.

Code, splits, logs and student weights are publicly available. The manuscript has not been
published and is not under consideration elsewhere. All authors approved the submission.
[Suggested reviewers / conflicts, if requested by the submission system.]

Sincerely,
Mathis Magron, on behalf of the authors
