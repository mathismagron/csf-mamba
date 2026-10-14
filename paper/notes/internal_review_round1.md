# Internal review, round 1 (13 Oct 2026)

Three independent reviewers (JSTARS methods reviewer, knowledge-distillation specialist,
skeptical associate editor) read draft 1 (PDF, supplement, sources, tables, lab journal).
**All three recommended major revision.** Full reports: workspace `handoff/reviews.json`
(not versioned); this file is the synthesis.

## Consensus (raised by 2–3 reviewers)

| Issue | Status after revision 1 |
|---|---|
| "matches the teacher" inferred from p = 0.13 | fixed: TOST equivalence at ±0.25 SeK (90 % CI [−0.01, +0.20], p = 0.02); teacher's last-epoch 25.85 reported |
| "pre-registered" (no public registration; grid extended adaptively) | fixed: "specified in advance", grid extension and commits disclosed |
| abstract: "test set used once" false (best-test-epoch protocol, design campaign on test) | fixed |
| "gain uncorrelated with teacher advantage" (n = 6, CI [−0.78, 0.84]; within SECOND the better teacher gives the larger gain) | fixed: claim reduced to "a teacher need not outscore the student"; r removed from abstract |
| mechanism claim "content of the targets" (controls only change term, λ = 2, val, 2 seeds; spatial not class smoothing; contaminated val) | fixed in text: limited claim, contrary reading (Yuan et al., born-again) stated; **self-distillation and label-smoothing controls not run** |
| "no capacity-gap penalty" (two teachers confounded) | fixed: claim removed |
| "outperforms its fold teacher" (single runs, possibly under-tuned, p = 0.051) | fixed: qualified; added that students exceed even the teachers' best test epoch (≥ 0.22) |
| Wilcoxon over fold–seed pairs = pseudo-replication | fixed: consistency count only |
| Landsat wording ("leakage") | fixed: "train–test image overlap", change-of-question wording, PerASCD split caveat, probability argument |
| MambaSCD "F1" ≠ Fscd; counters not uniform | fixed: omitted with footnote; counter caveat |
| assertive figure titles | fixed: descriptive titles |
| constant-LR baseline may inflate the last-epoch gain | stated as limitation; **cosine-LR baseline not run** |
| D4 cache never ablated; jitter makes teaching inconsistent | stated; **ablation not run** |
| no qualitative maps | **pending** (`paper/scripts/qualitative.sbatch`, < 1 A100-h) |
| novelty is modest (Hinton KD + cached views) | framing: controlled empirical study; closest work (BFD, CDKD, Vemulapalli) cited |
| factual slips (10^-3 tolerance, "best = final epoch", "reduces spread", FFT "no effect", section ref, "no training cost" on Landsat, conclusion) | fixed |

## Experiments the reviewers would request (not run)

Priority A — asked by all three; the editor calls the first "potentially fatal":
1. **Self-distillation control**: the undistilled student as its own teacher (same 8-view cache,
   same losses). SECOND 3 seeds (≈ 19 A100-h) + Landsat folds 1–3 × 3 seeds (≈ 22 A100-h).
2. **Target controls at the retained configuration on held-out data**: smoothed-GT change target
   at λ_chg = 8 + class-level label smoothing replacing the semantic KL; Landsat folds 1–3 × 3
   seeds (≈ 22 A100-h).
3. **Decaying-LR undistilled baseline** (cosine), SECOND 3 seeds (≈ 19 A100-h).
4. **Qualitative maps** (+ optional D4-TTA teacher evaluation), inference only (≈ 1–2 A100-h).

Priority B — cheap or no GPU: student–teacher fidelity analysis (inference), CPU latency of the
student (local), published-model MACs with our counter (released code, < 1 A100-h).

Priority C — would strengthen but not decisive: single-view cache ablation (≈ 10 A100-h), one
standard dense-KD baseline such as CWD (≈ 10 A100-h), teacher seed variance, second student
family (≈ 40–60 A100-h), third dataset.
