"""One-off revision of main.tex after the first internal review round (13 Oct 2026).
Whitespace-tolerant replacements; aborts without writing if any passage is missing."""
import re
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "main.tex"
st = {"m": P.read_text()}


def R(a, b):
    pat = r"\s+".join(re.escape(tok) for tok in a.split())
    mm = re.search(pat, st["m"])
    if not mm:
        sys.exit(f"MISSING: {a[:100]!r}")
    st["m"] = st["m"][:mm.start()] + b + st["m"][mm.end():]


def block(start, end, new):
    m = st["m"]
    i = m.index(start); j = m.index(end, i + len(start))
    st["m"] = m[:i] + new + m[j:]


# ---------------------------------------------------------------- title, abstract, keywords
block(r"\title{", r"% TODO (author list", r"""\title{DistilSCD: Output Distillation of a Remote-Sensing Foundation Model into a Compact
Semantic Change Detector}

""")
block(r"\begin{abstract}", r"\begin{IEEEkeywords}", r"""\begin{abstract}
Remote-sensing foundation models hold the best published accuracy for semantic change detection
(SCD), at a cost that is hard to deploy at scale: the strongest SECOND model, PerASCD, has
\TeacherParams\,M parameters and needs about 1.5 TMACs per image pair. We ask how much of that
accuracy a compact model inherits from the teacher's outputs alone. DistilSCD trains a
\StudentParams\,M-parameter siamese VMamba student with two extra loss terms, soft change targets
and temperature-scaled class distributions, read from an offline cache of the teacher's logits
under the eight D4 views that the student's augmentation produces. Inference cost is unchanged.
Distillation hyper-parameters were selected on an internal validation split before any
distillation run was evaluated on the test set. On SECOND (five seeds), distillation raises the
student from \SecBaseSek\ to \SecFullSek\ SeK (+\SecFullGain\ [\SecFullGainLo, \SecFullGainHi]),
equivalent to the teacher within $\pm\TostMargin$ SeK (\TeacherVitgSek; difference \SecFullVsT),
with \RatioParams$\times$ fewer parameters, \RatioGmacs$\times$ fewer MACs and
\RatioLatBf--\RatioLatFpThirtyTwo$\times$ lower GPU latency. We also show that the released random
pair split of Landsat-SCD places both images of every test pair in the training set, so that it
measures recognition at already-seen locations, and we introduce a location-disjoint four-fold
protocol. Under it, distillation improves the student on every fold and seed (+\LPrimGain\ SeK on
the three folds whose teacher passed a gate specified in advance), and the distilled students
exceed these single-run fold teachers. The gain remains between 1.8 and 3.3 SeK across six
teacher--dataset configurations, including one where the teacher does not outscore the student.
\end{abstract}

""")
R("efficient deep learning, evaluation protocol, data leakage, Landsat.",
  "efficient deep learning, evaluation protocol, spatial cross-validation, Landsat.")
R(r"""\thanks{M. Magron is with INP--ENSEEIHT, Toulouse, France; this work started during a research
internship at the Universit\'e de Moncton, Moncton, NB, Canada.}%""",
  r"""\thanks{M. Magron is with INP--ENSEEIHT, Toulouse, France; this work started during a research
internship at the Universit\'e de Moncton, Moncton, NB, Canada (corresponding author; e-mail:
TODO).}%""")

# ---------------------------------------------------------------- introduction
R(r"""with \TeacherParams\,M parameters, about \TeacherGmacs\ GMACs per pair,""",
  r"""with \TeacherParams\,M parameters, about 1.5 TMACs per pair,""")
R(r"""the training and test sets, so that test accuracy largely measures recall of seen locations.
This is a well-known pitfall of spatially structured data~\cite{roberts2017crossvalidation,
ploton2020spatial,kattenborn2022spatially,kapoor2023leakage}, and it changes what a
distillation experiment on this dataset can conclude.""",
  r"""the training and test sets, so that the test set measures recognition at locations and dates
already seen in training rather than generalisation to new locations. Random splits of spatially
structured data are known to answer this narrower question~\cite{roberts2017crossvalidation,
ploton2020spatial,kattenborn2022spatially,kapoor2023leakage}, and it changes what a distillation
experiment on this dataset can conclude.""")
R(r"""\item \emph{A controlled evaluation on SECOND.} With hyper-parameters fixed on an internal
validation split and five seeds on the test set, a \StudentParams\,M-parameter student gains
\SecFullGain\ SeK and matches the \TeacherParams\,M-parameter teacher at \RatioGmacs$\times$
lower compute. A second, smaller teacher confirms the effect and shows no capacity-gap penalty
at a $33\times$ size ratio (Section~\ref{sec:second}).""",
  r"""\item \emph{A controlled evaluation on SECOND.} With distillation hyper-parameters selected on an
internal validation split and five seeds on the test set, a \StudentParams\,M-parameter student
gains \SecFullGain\ SeK and is equivalent to the \TeacherParams\,M-parameter teacher within
$\pm\TostMargin$ SeK at \RatioGmacs$\times$ fewer MACs. A second, smaller and ImageNet-pretrained
teacher yields a smaller gain, and its student exceeds it (Section~\ref{sec:second}).""")
R(r"""\item \emph{A leakage analysis and a location-disjoint protocol for Landsat-SCD.} We verify at""",
  r"""\item \emph{A train--test overlap analysis and a location-disjoint protocol for Landsat-SCD.} We verify at""")
R(r"""\item \emph{Evidence on where the gain comes from.} Under the location-disjoint protocol the
distilled student beats both its undistilled counterpart and its teacher on every fold. The
gain is uncorrelated with the teacher's own advantage, ground-truth targets of the same form
give nothing, and feature distillation adds nothing on top of output distillation
(Sections~\ref{sec:landsat}--\ref{sec:discussion}).""",
  r"""\item \emph{Evidence on where the gain comes from, and its limits.} Under the location-disjoint
protocol the distilled student improves on its undistilled counterpart on every fold and seed and
exceeds the single-run fold teachers. On SECOND, ground-truth change targets of the same form give
no gain, and our feature-distillation term adds nothing on top of output distillation. We discuss
which alternative explanations these controls leave open
(Sections~\ref{sec:landsat}--\ref{sec:discussion}).""")
R(r"""Code, splits, the leakage-verification script and the student weights are released
(Section~\ref{sec:availability}).""",
  r"""Code, splits, the overlap-verification script and the run logs are released (see Data and Code
Availability).""")

# ---------------------------------------------------------------- related work
R(r"""student see the same view) and long schedules close much of the gap; our D4 cache is a
discrete form of consistent teaching, akin to stored per-crop soft labels
\cite{shen2022fast,yun2021re}.""",
  r"""student see the same view) and long schedules close much of the gap. Our D4 cache makes teaching
geometrically consistent, akin to stored per-crop soft labels~\cite{shen2022fast,yun2021re};
photometric jitter is applied to the student only, so teaching is not fully consistent in their
sense.""")

# ---------------------------------------------------------------- method
R(r"""Several blocks inherited from the student's earlier design (a frequency branch and a
change-guided attention) have no measurable effect on SECOND; they are kept for continuity and
documented in the supplementary material.""",
  r"""Two blocks inherited from the student's earlier design (a frequency branch and a change-guided
attention) have no established effect on SECOND; they are kept unchanged and documented in the
supplementary material, together with the design experiments, which selected epochs on the
SECOND test set as is customary on this dataset.""")
R(r"""differs on 2.2--2.5\,\% of all pixels, i.e. on roughly a
tenth of the changed area, and its class decision on 2.7--3.0\,\% of changed pixels.""",
  r"""differs on 2.2--2.5\,\% of all pixels (about 20\,\% of the pixels are changed), and its class
decision on 2.7--3.0\,\% of the changed pixels.""")
R(r"""Photometric jitter is
applied to the student's input only, which adds a radiometric invariance constraint.""",
  r"""Photometric jitter is
applied to the student's input only, which adds a radiometric invariance constraint and makes
teaching photometrically inconsistent. We did not ablate the eight-view cache against a single
view.""")
R(r"""We also screened a feature-distillation term (cosine similarity
after channel-wise normalisation between $1\times1$-projected student features and the
teacher's encoder features at two or three scales), which requires running the teacher online.""",
  r"""We also screened one feature-distillation term (cosine similarity after channel-wise
normalisation between $1\times1$-projected student features and the teacher's encoder features at
two or three scales) on top of output distillation; it requires running the teacher online.""")
R(r"""With a constant learning
rate, a shorter run is exactly a prefix of a longer one;""",
  r"""The constant rate comes from the student's design campaign; with it, a shorter run is exactly a
prefix of a longer one;""")
R(r"""Distillation adds no measurable
training time, since the teacher never runs during student training.""",
  r"""Distillation adds no measurable student training time, since the teacher never runs during
student training; the teacher's cost is paid once, when the cache is computed (and, on
Landsat-SCD, when the fold teachers are trained).""")

# ---------------------------------------------------------------- protocol
R(r"""\section{Datasets and Evaluation Protocol}
\label{sec:protocol}""",
  r"""\section{Datasets and Evaluation Protocol}
\label{sec:protocol}
Table~\ref{tab:datasets} summarises the datasets, splits and model-selection rules.""")
R(r"""above
every published result (e.g. \PerascdLandsat\ for PerASCD~\cite{shen2026foundation}). Because
each image appears in about 14 pairs, any random split at the pair level has this property,
whatever its seed.""",
  r"""higher
than the published Landsat-SCD results we are aware of, which were obtained on random pair splits
(e.g. \PerascdLandsat\ for PerASCD~\cite{shen2026foundation}). Each image appears in about 14
pairs; when 60\,\% of the pairs are drawn at random for training, the probability that a test image
appears in none of the training pairs is about $0.4^{13}<10^{-5}$, so any random pair split has
this property, whatever its seed. We verified it on the released split only.""")
R(r"""scores to within $10^{-3}$ points.""", r"""scores to within 0.004 points.""")
R(r"""the unit of inference is the fold: we report a $t$ interval over fold means and,
as a non-parametric check, a Wilcoxon signed-rank test over the fold--seed pairs. GPU training
is not bit-reproducible; replays of the same configuration differ by up to 0.5 SeK on
Landsat-SCD, which the paired design absorbs.""",
  r"""the unit of inference is the fold: we report a $t$ interval over the fold means. As a
consistency check, not a test, we also count the fold--seed pairs with a positive difference
(seeds within a fold are not independent). To compare the distilled student with the single-run
teacher on SECOND we use an equivalence test (two one-sided tests, margin $\pm\TostMargin$ SeK),
which treats the teacher's score as fixed. GPU training is not bit-reproducible; replays of the
same configuration differ by up to 0.5 SeK on Landsat-SCD, which adds variance to each paired
difference.""")
block(r"\subsection{Pre-Registered Screening}", r"\subsection{Teachers: Reproduction and Retraining}",
      r"""\subsection{Screening Specified in Advance}
\label{sec:screening}
All distillation hyper-parameters were chosen on SECOND before any distillation run was evaluated
on the test set. The 2\,968 training pairs were split into 2\,671 for training and 297 for
validation (a fixed, hashed split). With two seeds per arm and 120 epochs, three waves fixed in
turn the change weight $\lambda_{\mathrm{chg}}$, the semantic term ($T_{\mathrm{s}}\in\{2,4\}$, two
masks) and the feature term (two sets of scales, two weights). Before each wave, the decision rule
(``best mean validation SeK; an increment counts if it exceeds 0.36 points'') was written in the
project's lab journal, which is versioned with the code; it is not a public registration. The
0.36-point threshold was derived for five test seeds and serves here as a ranking threshold, not
as a test. The $\lambda_{\mathrm{chg}}$ grid started at $\{0.5,1,2\}$ and was extended to $\{4,8\}$,
then to 16, because the best value lay at the edge of the grid. Two controls, run at
$\lambda_{\mathrm{chg}}=2$, replaced the teacher's change target by the ground truth, either binary
or average-pooled to the teacher's resolution and upsampled like it. The SECOND teacher was trained
by its authors on all 2\,968 training pairs, including our 297 validation pairs, which can favour
distillation over the controls on validation. The retained configuration was then retrained on all
2\,968 pairs with five seeds and evaluated on the test set with the field's protocol
(Section~\ref{sec:stats}).

""")
R(r"""selected by their authors on the test set and are single runs.""",
  r"""selected by their authors on the test set and are single runs; the training log of the ViT-G
checkpoint reads 26.13 at its best epoch and 25.85 at its last epoch.""")
R(r"""Before caching each fold teacher, a pre-registered
gate required""", r"""Before caching each fold teacher, a gate specified in advance required""")

# ---------------------------------------------------------------- results SECOND
R(r"""Change
distillation alone raises the validation SeK monotonically with its weight up to
$\lambda_{\mathrm{chg}}=8$ (\ScrLEight\ against \ScrWit\ without distillation) and plateaus
beyond (\IncLSixteen\ at $\lambda_{\mathrm{chg}}=16$).""",
  r"""Change
distillation alone raises the validation SeK with its weight up to $\lambda_{\mathrm{chg}}=8$
(\ScrLEight\ against \ScrWit\ without distillation; the differences between 2, 4 and 8 are within
two-seed noise) and plateaus beyond (\IncLSixteen\ at $\lambda_{\mathrm{chg}}=16$).""")
R(r"""\ScrLTwoDelta\ for the teacher's targets at the same weight: the gain does not come from adding
a change loss, nor from softened boundaries, but from the content of the teacher's targets.""",
  r"""\ScrLTwoDelta\ for the teacher's targets at the same weight. For the change term, the gain is
therefore not explained by adding a change loss of that weight or by softened boundaries; the
controls were run only at $\lambda_{\mathrm{chg}}=2$, with two seeds, on validation.""")
R(r"""The two masks are equivalent within noise; the pre-registered rule retained
all pixels.""", r"""The two masks are equivalent within noise; the rule retained all pixels.""")
R(r"""Feature
distillation on top of output distillation adds""", r"""Our feature-distillation term on top of
output distillation adds""")
R(r"""below the 0.36-point
criterion, while multiplying the
training time by 2.3 and""", r"""below the 0.36-point threshold, while multiplying the training time by
2.3--2.4 and""")
R(r"""The
distilled student's best and final epochs coincide, whereas the undistilled student peaks
between epochs 47 and 100 and then declines under the constant learning rate
(Fig.~\ref{fig:second}b); distillation removes this late decline as well as raising the peak.""",
  r"""The
distilled student's best and final SeK differ by at most \SecFullMaxFinalGap\ points (best epochs
\SecFullBestEpMin--\SecFullBestEpMax), whereas the undistilled student peaks between epochs
\SecBaseBestEpMin\ and \SecBaseBestEpMax\ and then declines under the constant learning rate
(Fig.~\ref{fig:second}b). Distillation thus raises the peak and removes the late decline; a
decaying learning-rate schedule might remove part of that decline without distillation, which we
did not test, so we base the gain on the best-epoch number.""")
R(r"""[\SecFullVsTLo, \SecFullVsTHi] SeK ($p\SecFullVsTP$): it \emph{matches} the teacher, closing
\SecFullGapClosed\,\% of the gap, without a significant difference at the pre-registered budget.""",
  r"""[\SecFullVsTLo, \SecFullVsTHi] SeK; its 90\,\% interval [\TostLo, \TostHi] lies within
$\pm\TostMargin$ SeK, so the two are equivalent at that margin (two one-sided tests, $p\TostP$),
treating the single-run teacher as fixed. Against the teacher's selection-free last-epoch value
(25.85), the student's last-epoch SeK (\SecFullSekFinal) is higher.""")
R(r"""a decision taken after seeing the curves and reported as a post-hoc extension""",
  r"""a decision taken after seeing the test curves and reported as a post-hoc extension""")
R(r"""\num{113.01}\,M parameters) yields a student at \SecVmbSek\,$\pm$\,\SecVmbSekSd, which exceeds
that teacher by""", r"""\num{113.01}\,M parameters, ImageNet-pretrained rather than a remote-sensing foundation
model) yields a student at \SecVmbSek\,$\pm$\,\SecVmbSekSd, which exceeds that single-run teacher by""")
R(r"""The ViT-G teacher,
\TeacherParams\,M parameters and $33\times$ the student's size, yields the better student:""",
  r"""The ViT-G teacher, more
accurate and $33\times$ the student's size, yields the better student:""")
R(r"""Between these two teachers there is no capacity-gap penalty; the
student follows the quality of the teacher rather than its size. The two teachers also differ
in architecture, so size and family are not separated.""",
  r"""The two teachers differ at once in size, architecture, pre-training and accuracy, so this
comparison does not test the capacity-gap hypothesis~\cite{cho2019efficacy,mirzadeh2020improved};
it shows only that a much larger teacher did not yield a worse student here.""")
R(r"""Published values are taken from the papers;
their computational costs are reported as ``GFLOPs'' but follow the fvcore convention of
counting multiply--accumulates, so we treat them as GMACs, with the caveat that counters
differ. At \StudentParams\,M parameters, the distilled student is the only sub-100\,GMAC model
at the accuracy level of the foundation-model teacher; from the weaker VMamba-B teacher, it
still exceeds Mamba-FCS (25.50 SeK, 189.54\,M parameters).""",
  r"""Published values are taken from the papers.
ChangeMamba and Mamba-FCS report fvcore counts labelled ``GFLOPs'', which count
multiply--accumulates, so we plot them as GMACs; conventions are not uniform across papers
(PerASCD reports 2853 ``GFLOPs'' for the model we measure at \TeacherGmacs\ GMACs), so cross-paper
costs are indicative only. The distilled student reaches the accuracy level of the foundation-model
teacher at the lowest compute in the table. Published accuracies are single numbers whose
selection rules are not always stated.""")

# ---------------------------------------------------------------- results Landsat
R(r"""The retrained fold teachers pass the pre-registered gate on folds 1--3""",
  r"""The retrained fold teachers pass the gate on folds 1--3""")
R(r"""As pre-registered,
folds 1--3 form the primary analysis.""", r"""Because the rule fixed before training the teachers
caches only teachers that pass the gate, folds 1--3 form the primary analysis.""")
R(r"""\LPrimPos\ fold--seed pairs positive, Wilcoxon
$p=\LPrimWilcox$)""", r"""\LPrimPos\ fold--seed pairs positive)""")
R(r"""The distilled students also reduce the
spread across seeds.""", r"""The spread across seeds is smaller with distillation on three of the four
folds.""")
R(r"""($p\LPrimKdMinusTP$ with three
folds) and \LSecKdMinusT\ over all four ($p\LSecKdMinusTP$); each fold teacher is a single run.""",
  r"""($p\LPrimKdMinusTP$ with three
folds) and \LSecKdMinusT\ over all four ($p\LSecKdMinusTP$). Each fold teacher is a single run,
trained with a recipe that its authors tuned on the random split and selected on a single
validation tile, so these teachers may be under-tuned. The comparison does not depend on how the
teachers' epoch was selected: the distilled students exceed even the teachers' best test epoch on
every fold, by at least \LKdMinusTMaxMin\ points (supplementary Table~SIV).""")

# ---------------------------------------------------------------- discussion
block(r"\subsection{The Gain Does Not Track the Teacher", r"\subsection{Outputs Suffice; Features Do Not Add}",
      r"""\subsection{Gain and Teacher Advantage}
One might expect distillation to help in proportion to how much better the teacher is than the
student. Within SECOND, the more accurate teacher (advantage \TeacherAdvVitg\ against
\TeacherAdvVmb\ SeK) does give the larger gain (+\TeacherDiff\ paired). Across our six
teacher--dataset configurations (Fig.~\ref{fig:landsat}b), however, the gain stays between 1.8 and
3.3 SeK while the teacher's advantage ranges from \FZeroTMinusB\ to 2.21, and on fold 0 a teacher
slightly worse than the student on the test tiles still yields +\FZeroGain\ points. Six
configurations from two datasets with different selection rules cannot establish that the gain is
independent of the teacher's advantage (correlation $r=\CorrR$, $n=\CorrN$); they do show that a
teacher need not outscore the student to help it. In hindsight, our gate (``the teacher must beat
the student on validation'') was conservative.

What, then, is transferred? Our evidence narrows the answer only partly. On SECOND, ground-truth
change targets with the same loss and weight, binary or with the teacher's softened boundaries,
give no gain on validation; these controls concern the change term only, at
$\lambda_{\mathrm{chg}}=2$ and with two seeds, and they smooth targets spatially, not across classes
as label smoothing does~\cite{szegedy2016rethinking,muller2019label}. The teacher's targets are not
a copy of the labels: on its own SECOND training set the teacher scores 33.24 SeK, and its semantic
predictions cover the 80\,\% of pixels that carry no label, where the semantic term acts. Our
results are consistent with reports that students benefit from teachers that are weak or no
better than themselves~\cite{furlanello2018born,yuan2020revisiting,stanton2021knowledge}, which
interpret part of distillation as learned regularisation. A self-distillation control (the
undistilled student as its own teacher, with the same cache and losses) and a class-level
label-smoothing control would separate knowledge transfer from regularisation; we did not run
them.

""")
R(r"""\subsection{Outputs Suffice; Features Do Not Add}""", r"""\subsection{Output Distillation and Our Feature Term}""")
R(r"""In our setting, once outputs are distilled, aligning the student's encoder
features with those of the ViT-G teacher adds nothing at any tested weight or set of scales,
and a larger weight hurts.""",
  r"""In our setting, once outputs are distilled, our feature term (cosine alignment of projected
student features with the ViT-G encoder features) adds nothing at the tested weights and scales,
and a larger weight hurts. We did not test feature distillation alone, other feature losses, or
the generic foundation-model features that BFD distils.""")
R(r"""Published Landsat-SCD numbers, ours included when computed on the random split, answer a
recognition question over a single, already-seen area. They remain meaningful for that
question, but they should not be read as generalisation to new areas: a lookup table exceeds
every model.""",
  r"""Landsat-SCD numbers computed on random pair splits, ours included, answer a recognition question
over a single, already-seen area. They remain meaningful for that question, but they should not be
read as generalisation to new areas: on the released split, a lookup table exceeds the published
models we compared.""")
block(r"\subsection{Limitations}", r"% =====", r"""\subsection{Limitations}
\label{sec:limitations}
Our evidence covers two datasets, one student architecture and two teachers on SECOND that differ
in several respects at once. Teachers are single runs: the published SECOND checkpoints, selected
by their authors on the test set, and one retrained teacher per Landsat-SCD fold, trained with a
recipe tuned by its authors on the random split. SECOND has no official validation split, so its
headline numbers follow the field's test-epoch convention; the selection-free last-epoch numbers
lead to the same conclusions. The student recipe was developed on SECOND, with test-epoch
selection, before this study and then frozen; it uses a constant learning rate, and we did not test
whether a decaying schedule would narrow the gap at the last epoch. The screening used two seeds
per arm; the ground-truth controls cover the change term only; self-distillation and
label-smoothing controls, other distillation losses and a single-view cache were not tested. Each
Landsat-SCD fold validates on a single tile, which makes model selection noisy, and the fold-0
analysis is exploratory. On Landsat-SCD, distillation first required training one
548\,M-parameter teacher per fold (\ComputeLandsatTeachers\ A100-hours in total). Finally, latency
gains (\RatioLatOneBf--\RatioLatFpThirtyTwo$\times$ on an A100) are much smaller than compute
gains (\RatioGmacs$\times$), because the student is limited by memory accesses.

""")
block(r"\section{Conclusion}", r"\section*{Data and Code Availability}", r"""\section{Conclusion}
\label{sec:conclusion}
Distilling only the outputs of a \TeacherParams\,M-parameter remote-sensing foundation model into a
\StudentParams\,M-parameter student brings the student to the teacher's accuracy on SECOND
(equivalent within $\pm\TostMargin$ SeK) at \RatioGmacs$\times$ fewer MACs, without changing the
student or its inference cost. On Landsat-SCD, evaluating this required a location-disjoint
protocol, because the released random split places every test image in the training set; under
that protocol, distillation improves the student on every fold and seed and the distilled students
exceed their single-run fold teachers. The gain, 1.8 to 3.3 SeK across six configurations, appears
even where the teacher does not outscore the student. Future work should separate knowledge
transfer from regularisation with self-distillation and label-smoothing controls, test other
student families, and extend location-disjoint evaluation to other change detection benchmarks.

""")
R(r"""folds, the leakage-verification script, per-run logs and the student weights are available at
\url{https://github.com/mathismagron/distilscd}.""",
  r"""folds, the overlap-verification script and the per-run logs are available at
\url{https://github.com/mathismagron/distilscd}. Student weights will be released once the licence
of the teacher weights has been clarified with their authors.""")

# ---------------------------------------------------------------- captions
R(r"""{Landsat-SCD: leakage of the random pair split and location-disjoint protocol.""",
  r"""{Landsat-SCD: train--test image overlap of the released random pair split, and location-disjoint
protocol.""")
R(r"""Grey tick: PerASCD as published on the random split (\PerascdLandsat).""",
  r"""Grey tick: PerASCD as published (\PerascdLandsat).""")
R(r"""{Pre-registered screening on the SECOND internal validation split""",
  r"""{Screening on the SECOND internal validation split""")
R(r"""failed the pre-registered teacher gate and is exploratory.""",
  r"""failed the teacher gate specified in advance and is exploratory.""")
R(r"""points above the dashed line are students that outscore their teacher.
$r=\CorrR$ ($n=\CorrN$).}""", r"""points above the dashed line are students that outscore their teacher.}""")
R(r"""95\,\% interval: \SecFullVsT\ ($p\SecFullVsTP$) with ViT-G, \SecVmbVsT\ ($p\SecVmbVsTP$) with
VMamba-B.}""", r"""95\,\% interval: \SecFullVsT\ with ViT-G, \SecVmbVsT\ with VMamba-B; teachers are single runs.}""")

P.write_text(st["m"])
print("revised", P)
