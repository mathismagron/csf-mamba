# Cover letter (draft, to adapt before submission)

Dear Editor,

We submit the manuscript "DistilSCD: Output Distillation of a Remote-Sensing Foundation Model into
a Compact Semantic Change Detector" for consideration as a regular paper in the IEEE Journal of
Selected Topics in Applied Earth Observations and Remote Sensing [option: for the special issue
"Foundation Models and Agents in Earth Observation"].

Remote-sensing foundation models now hold the best published accuracy for semantic change
detection, but their size (548 M parameters and about 1.5 TMACs per image pair for the strongest
SECOND model) makes large-scale deployment costly. We show that distilling only the outputs of
such a model into a 16.5 M-parameter student, through an offline, augmentation-consistent logit
cache, brings the student to its teacher's accuracy on SECOND (equivalent within ±0.25 SeK over five
seeds) at 51 times fewer MACs, without changing the student or its training cost.

The study is designed to show where the gain comes from. Hyper-parameters were fixed on an internal
validation split and comparisons are paired over seeds. Controls that keep the losses, cache and
schedule fixed and change only the source of the targets show that the gain depends on the
foundation model's predictions. On held-out data, neither self-distillation nor ground-truth targets
smoothed to the teacher's confidence improve the student. On SECOND, self-distillation recovers
about one third of the gain. A decaying learning rate does not improve the undistilled baseline.

The work also contributes an evaluation finding of direct use to JSTARS readers. The released
random pair split of Landsat-SCD places both images of every test pair in the training set, so it
measures recognition at locations already seen in training. We document this at the pixel level
and release a location-disjoint four-fold protocol with its verification script. Under that
protocol, distillation improves the student on every fold and seed.

Code, splits, run logs and analysis scripts are publicly available; student weights will be
released once the licence of the teacher weights has been clarified with their authors. The
manuscript has not been published and is not under consideration elsewhere. All authors approved
the submission.
[Suggested reviewers / conflicts, if requested by the submission system.]

Sincerely,
Mathis Magron, on behalf of the authors
