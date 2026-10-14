# Paper brief — DistilSCD (target: IEEE JSTARS)

All numbers below are read from `paper/tables/*.csv` (see `paper/tables/README.md`).
SeK in percentage points.

## One-sentence pitch

Distilling only the *outputs* of a 548 M-parameter remote-sensing foundation model into a
16.5 M-parameter siamese VMamba student lifts semantic change detection by 2–3 SeK points on
two datasets, so that the student matches its teacher on SECOND and surpasses it on
location-disjoint Landsat-SCD, at 1/33 of the parameters and 1/51 of the compute; the gain
appears even where the teacher does not outscore the student.

## Working title

DistilSCD: Foundation-Model Accuracy at Embedded Cost for Semantic Change Detection
through Output Distillation  *(alternative: "When Does a Foundation-Model Teacher Help? Output
Distillation for Lightweight Semantic Change Detection, with a Leakage-Free Landsat-SCD
Protocol")*

## Contributions (as they will be claimed)

1. **A simple, offline output-distillation recipe for SCD.** Soft binary-change targets
   (BCE, λ_chg = 8) and a temperature-scaled KL on the teacher's class distributions
   renormalised over real classes (λ_sem = 1, T = 2, all pixels), read from a cache of the
   teacher's logits under the 8 D4 views that the student's augmentation can produce. No
   architectural change, no inference cost, no measurable training overhead (cache: 26 GPU-min).
   Hyper-parameters were chosen by a pre-registered screening on an internal validation split
   (2 671 / 297); the test set was used once, for confirmation.
2. **SECOND, five seeds, paired statistics.** 23.90 ± 0.15 → 26.20 ± 0.11 SeK (+2.30
   [2.13, 2.47]); student − teacher +0.09 [−0.04, 0.23], p = 0.13 → *matches* (not beats) the
   teacher, which we re-evaluated to 26.1079 (published 26.11). Final-epoch SeK gives the same
   picture (26.20 vs 23.65). 33.3× fewer parameters, 51.2× fewer GMACs (same counter),
   12.5× lower fp32 latency (4.9× bf16), 12.8× less peak memory on the same A100.
3. **Two teachers.** Same recipe from the authors' VMamba-B teacher (113 M, 25.31): student
   25.69 ± 0.18, beats it by +0.38 [0.16, 0.59]; the stronger ViT-G teacher yields the stronger
   student (+0.51 paired, 5/5 seeds, p = 0.004). No capacity-gap penalty at 33×.
4. **A leakage-free protocol for Landsat-SCD.** The dataset holds 333 distinct images (12
   tiles × ~28 dates) recombined into 2 385 pairs; in the released random pair split, 477/477
   test pairs have both images in train, and a learning-free lookup reaches SeK 77.87 (above
   every published model). We release a 4-fold location-disjoint cross-validation (3 test
   tiles per fold) with the verification script.
5. **Under that protocol, KD helps on every fold.** +2.53 SeK on the three pre-registered
   folds (95 % CI [0.74, 4.32], 9/9 seed pairs positive), +2.51 over all four (12/12); the KD
   student beats the fold teacher on all four folds (+1.46 on folds 1–3). Across six
   teacher/dataset configurations the KD gain is uncorrelated with the teacher's own
   advantage over the student (r = 0.09); on SECOND, controls with ground-truth targets of the
   same form and weight give nothing (+0.04 / −0.02), so the gain comes from the content of the
   teacher's targets.
6. **Negative results reported.** Feature distillation on top of output KD: no gain
   (−0.45 to +0.12 on val) at 2.3× training time and 4× memory; T = 4 worse than T = 2;
   change-only vs all-pixel semantic masks equivalent; λ_chg plateaus at 8 (16 ≈ 8).

## Audience

JSTARS readers: remote-sensing practitioners who need SCD at scale (deployment cost
matters) and method developers who use SECOND / Landsat-SCD as benchmarks.

## Figure arc (main paper)

1. **Fig. 1 — the pitch as data.** SeK vs GMACs per image pair (log x) on SECOND:
   student without KD, student with KD from each teacher (arrows from the no-KD point), the two
   teachers, published models (Mamba-FCS, MambaSCD-B/T). Claim: *KD moves a 29-GMAC model to the
   accuracy of a 1 510-GMAC foundation model.*
2. **Fig. 2 — method.** Teacher (frozen, offline) → D4 logit cache; student siamese VMamba-mini
   + shared DySample decoder; three loss terms; what is discarded at inference.
3. **Fig. 3 — SECOND evidence.** (a) per-seed test SeK, max and final, four arms + teacher;
   (b) mean test SeK vs epoch (KD removes the late decline of the no-KD student); (c) two
   teachers: student − teacher.
4. **Fig. 4 — what produces the gain (validation screening).** (a) λ_chg sweep with the two
   ground-truth controls at λ = 2; (b) semantic term T × mask; (c) feature KD: increment vs cost.
5. **Fig. 5 — Landsat-SCD leakage and protocol.** (a) 4 × 3 tile grid (indices 00…32) coloured
   by test fold; (b) the same student on the random split (59.85 → 69.42, no plateau) vs a
   location-disjoint fold (plateau ~29), and the lookup oracle (77.87).
6. **Fig. 6 — Landsat-SCD results.** (a) per-fold test SeK: teacher, student, KD student;
   (b) KD gain vs teacher advantage, six configurations, with the y = x line.
7. **Fig. 7 — qualitative maps** (SECOND and one Landsat fold): images, ground truth,
   student, KD student, teacher. *Needs a short inference job on Narval (checkpoints are
   there); no training.*

## Tables (main paper)

- I. Datasets and protocols (SECOND official split + internal val; Landsat random vs
  location-disjoint folds).
- II. SECOND test results: arms × {SeK max, SeK final, Fscd, mIoU, gain [CI], vs teacher}.
- III. Comparison with published SECOND results (params, GMACs, SeK, Fscd, mIoU; provenance).
- IV. Efficiency on one A100 (params, GMACs, latency b8/b1 fp32/bf16, peak memory).
- V. Screening on the internal validation split (all arms, 2 seeds).
- VI. Landsat-SCD leakage checks.
- VII. Landsat-SCD location-disjoint folds (teacher, student, KD, gain, KD − teacher).

## Supplement

Per-seed tables; 200-epoch extension (26.32, +0.22 vs teacher, p = 0.010, declared post hoc);
teacher reproduction details (ViT-G to 8e-4 SeK; VMamba-B needs ImageNet normalisation,
whereas the released evaluation script uses PerA statistics); student recipe and the
negative results of its design (C²S², FFT, CGA not detectable); compute ledger (≈ 479 A100-h);
Landsat fold construction; equivariance gap of the teacher (2.2–2.5 % of pixels change
decision across D4 views, which motivates the 8-view cache); reproducibility instructions.

## Wording rules (binding)

- **Landsat leakage = a change of question, never "the literature is wrong".** The random pair
  split is the dataset authors' own protocol (Yuan et al. 2022, p. 11), which they designed for
  a single-area time series and for which they name multi-region generalisation as future work
  (p. 19). We measure generalisation to unseen locations instead; our numbers are not
  comparable to published random-split numbers. Do not attribute specific splits to other
  papers unless verified (GSTM-SCD 3:1:1 is NOT verified).
- **SECOND: "matches" the teacher, never "beats"** at the pre-registered 120 epochs. The
  200-epoch result (+0.22, p = 0.010) is a declared post-hoc extension.
- **Fold 0 is exploratory.** Its teacher failed the pre-registered gate; it was run with the
  gate lifted and declared; primary analysis = folds 1–3; all-fold analysis is secondary. Say
  that the gate turned out to be conservative.
- **Selection conventions stated each time.** SECOND: max test SeK (field convention, same as
  ChangeMamba and PerASCD) and final epoch; Landsat: test at best fold-val epoch.
- **Counters.** 51× uses the same (aten) counter on both models; fvcore for the student is
  31.42 GMACs. Both under-count (deformable attention, selective scan) — say so.
- **Teachers single-seed** (published checkpoint on SECOND; one run per fold on Landsat).
- **Do not claim** a new architecture. The student is a compact siamese VMamba-mini with a
  shared DySample decoder, developed earlier by the first author; the C²S², FFT and CGA blocks
  inherited from that design do not contribute measurably (state it in the supplement).
- **Student recipe provenance.** The student recipe was developed on SECOND (with the field's
  test-epoch selection) before this study and then frozen; KD gains are measured against it at
  equal budget. On Landsat it was transferred unchanged.
- GPU runs are not bit-reproducible (≤ 0.5 SeK between replays on Landsat); all comparisons
  are paired by seed (and fold).

## Limitations to state

Two datasets; one student architecture; teachers single-seed; screening at 2 seeds; SECOND has
no official validation split; Landsat folds have one validation tile each (noisy model
selection); Landsat teachers were retrained by us with the authors' recipe (tuned on the random
split); no controls with ground-truth targets on Landsat; teacher-weight licence unstated
(code MIT), so we release student code and weights only; latency gains (4–12×) are much smaller
than compute gains (51×) because the student is memory-bound.
