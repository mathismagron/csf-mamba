# Related work: notes and positioning (DistilSCD, IEEE JSTARS)

Distilling a large vision model into an efficient change detector at no inference cost has
been done before. Burden-free distillation \cite{wang2025burden} (TGRS 2025) transfers features
from generic vision foundation models into binary change detectors, and CDKD
\cite{wang2024knowledge} (JSTARS 2024) distils a large teacher's output probabilities for
on-board binary change detection. We found no prior work that distils a *task-fine-tuned
remote-sensing foundation-model SCD teacher* into a compact *semantic* change detector on
SECOND or Landsat-SCD. We also found none that tests where the gain comes from (ground-truth
target controls, two teachers, location-disjoint folds). This paper's novelty therefore lies
in the setting (semantic CD, foundation-model teacher, output-only distillation) and in the
evidence: a paired multi-seed design, pre-registered screening, and a leakage-free Landsat
protocol. The general idea of "foundation model to small CD model" is not new.

Keys refer to `paper/refs.bib`. Each entry there was retrieved from CrossRef and/or the arXiv
API, and the log is in `paper/notes/bib_verification.csv`. Statements about what a paper
*does* come from its abstract (CrossRef, arXiv or Semantic Scholar) or from its full text where
stated; claims marked *(title only)* rest on the title alone and need a read before they go
into the manuscript.

## Change detection background

### A. Semantic change detection methods and benchmarks

Semantic change detection (SCD) has moved from multitask CNNs to spatio-temporal Transformers
and state-space models, while the benchmarks have stayed the same. HRSCD
\cite{daudt2019multitask} framed SCD as joint land-cover mapping and change detection. Yang et
al. \cite{yang2022asymmetric} introduced SECOND, the asymmetric siamese network and the
separated kappa (SeK) metric, which this paper inherits. Bi-SRNet \cite{ding2022bi} and
SCanNet/TED \cite{ding2024joint} added explicit bi-temporal semantic reasoning and
spatio-temporal Transformer modelling; ChangeMask \cite{zheng2022changemask} and the FCN-LSTM
framework of Papadomanolaki et al. \cite{papadomanolaki2021deep} are earlier multitask designs.
State-space models arrived with ChangeMamba/MambaSCD \cite{chen2024changemamba}, built on Mamba
\cite{gu2023mamba} and VMamba \cite{liu2024vmamba}, followed by Mamba-FCS
\cite{wijenayake2026mamba} (published in JSTARS 2026, the target journal) and the
graph-enhanced GSTM-SCD \cite{liu2025gstm}. The current top SECOND results come from
foundation-model encoders. PerASCD \cite{shen2026foundation} fine-tunes the 548 M-parameter
PerA ViT-G \cite{shen2026contrastive} with a ViT-Adapter \cite{chen2023vision}. DBTANet
\cite{li2026dual} (ICASSP 2026) combines a frozen SAM branch with ResNet-34, and SemDINO
\cite{tong2026semdino} (arXiv preprint) injects DINO-type priors. Landsat-SCD
\cite{yuan2022transformer} is the second standard SCD benchmark. Hi-UCD \cite{tian2020hi} and
the binary LEVIR-CD dataset \cite{chen2020spatial} complete the dataset landscape; FC-Siam
\cite{daudt2018fully}, BIT \cite{chen2022remote}, ChangeFormer \cite{bandara2022transformer}
and ChangeStar \cite{zheng2021change} are the usual binary-CD reference points.

Lightweight CD has mostly been pursued through architecture design. TinyCD
\cite{codegoni2023tinycd}, A2Net \cite{li2023lightweight} and Shuffle-CDNet
\cite{cui2022shuffle} target binary CD and shrink the network itself. Corley et al.
\cite{corley2024change} report that a plain U-Net remains a top performer for (binary) change
detection, a caution against reading architecture gains at face value.

*How this paper differs.* We propose no new architecture. The student is a compact siamese
VMamba-mini with a DySample decoder \cite{liu2023upsample}, developed earlier and then frozen.
The contribution is a training signal (output distillation) and an evaluation protocol. The
accuracy–cost comparison of Fig. 1 places a 29-GMAC student next to a 1 510-GMAC foundation
model, which architecture-only lightweight SCD work does not reach.

### B. Backbones and building blocks

The student and teachers are built on standard blocks: selective state spaces (Mamba
\cite{gu2023mamba}, following S4 \cite{gu2022efficiently}), the VMamba visual backbone
\cite{liu2024vmamba}, ViT \cite{dosovitskiy2020image} with ViT-Adapter \cite{chen2023vision}
(whose deformable attention \cite{zhu2021deformable} our GMAC counter under-counts), DySample
upsampling \cite{liu2023upsample}, and residual CNNs \cite{he2016deep}. Swin
\cite{liu2021swin}, MAE \cite{he2022masked}, DINOv2 \cite{oquab2023dinov2} and SAM
\cite{kirillov2023segment} are cited where they are the encoders of compared methods (SAM in
DBTANet and SAM-CD; DINO-type priors in SemDINO). No position on these blocks is claimed.

## Foundation models and distillation

### C. Remote-sensing foundation models and their use in CD

Remote-sensing foundation models are pre-trained by masked image modelling or contrastive
objectives and then fine-tuned at full size. Examples are SatMAE \cite{cong2022satmae},
Scale-MAE \cite{reed2023scale}, RingMo \cite{sun2023ringmo}, the plain-ViT RS model of Wang et
al. \cite{wang2023advancing}, SeCo \cite{manas2021seasonal}, SatlasPretrain
\cite{bastani2023satlaspretrain}, CROMA \cite{fuller2023croma}, SpectralGPT
\cite{hong2024spectralgpt}, Prithvi \cite{jakubik2023foundation}, SkySense \cite{guo2024skysense},
DOFA \cite{xiong2024neural}, MTP \cite{wang2024mtp} (JSTARS) and PerA
\cite{shen2026contrastive}. For change detection, the dominant pattern is to keep the
foundation model inside the deployed network. SAM-CD \cite{ding2024adapting}, the foundation-model
learning paradigm of Li et al. \cite{li2024new}, Time Travelling Pixels \cite{chen2024time},
ChangeCLIP \cite{dong2024changeclip}, DBTANet \cite{li2026dual}, SemDINO \cite{tong2026semdino}
and PerASCD \cite{shen2026foundation} all do so; AnyChange \cite{zheng2024segment} uses SAM
zero-shot. BFD \cite{wang2025burden} names the resulting inference burden as its motivation.
Benchmarking concerns apply here too. PANGAEA \cite{marsocci2024pangaea} reports inconsistent,
narrow GFM evaluations and compares GFMs against UNet/ViT baselines. Corley et al.
\cite{corley2024revisiting} show that preprocessing (resizing, normalisation) mismatches can
dominate reported differences. That echoes our finding that the VMamba-B teacher reproduces its
published score only under ImageNet normalisation.

*How this paper differs.* We use the foundation model only at training time, through a cached
copy of its logits, and measure the price of keeping it at inference: 33× parameters, 51×
GMACs, 4.9–12.5× latency on one A100.

### D. Knowledge distillation

Classical KD theory predicts both risks and benefits of a large teacher; our results fall on
the benign side. Output distillation \cite{hinton2015distilling} and its feature variants
(FitNets \cite{romero2014fitnets}, attention transfer \cite{zagoruyko2016paying}, relational KD
\cite{park2019relational}, CRD \cite{tian2020contrastive}, DKD \cite{zhao2022decoupled}) are
surveyed in \cite{gou2021knowledge}.

The *capacity gap* is documented. Cho and Hariharan \cite{cho2019efficacy} find that larger
models often do not make better teachers because small students cannot mimic them, and
Mirzadeh et al. \cite{mirzadeh2020improved} report that the student degrades when the gap is
large, proposing teacher assistants. Huang et al. \cite{huang2022knowledge} attribute part of
the difficulty to exact matching against stronger teachers. Against this, Beyer et al.
\cite{beyer2022knowledge} show that KD can compress large models without loss *if the teacher
is consistent*, i.e. it sees the same augmented view as the student, and patient. Our 8-view
D4 logit cache is a discrete form of consistent teaching. It is equivalent in spirit to stored
soft labels per crop (FKD \cite{shen2022fast}, ReLabel \cite{yun2021re}).

The *student-surpasses-teacher* case is also established: born-again networks
\cite{furlanello2018born}, noisy student \cite{xie2020self}, and Yuan et al.
\cite{yuan2020revisiting}, who show that even a poorly trained teacher improves a stronger
student and interpret KD as learned label smoothing \cite{szegedy2016rethinking,muller2019label}.
Stanton et al. \cite{stanton2021knowledge} show that students generalise better while matching
their teachers poorly, so "student ≥ teacher" is not by itself surprising.

For dense prediction, structured KD \cite{liu2019structured,liu2023structured}, channel-wise
distillation \cite{shu2021channel} and CIRKD \cite{yang2022cross} distil spatial structure for
segmentation. The closest general-vision recipe is Vemulapalli et al.
\cite{vemulapalli2024knowledge}, who fine-tune a vision foundation model on the target task and
then transfer it to a small task-specific model, outperforming task-agnostic VFM distillation.
Agglomerative distillation (AM-RADIO \cite{ranzinger2024am}) and SAM compression (MobileSAM
\cite{zhang2023faster}) are task-agnostic counterparts.

*How this paper differs.* Three findings speak to this literature directly.
1. No capacity-gap penalty at 33×. The stronger ViT-G teacher yields the stronger student
   (+0.51 paired, 5/5 seeds), contrary to the pattern in \cite{cho2019efficacy,mirzadeh2020improved}.
2. The gain is uncorrelated with the teacher's advantage over the student (r = 0.09, six
   configurations) and appears where the teacher does not outscore the student. This fits
   \cite{yuan2020revisiting}.
3. Unlike the label-smoothing reading, ground-truth targets of the same form and weight give
   nothing (+0.04 / −0.02 on SECOND), so on this task the *content* of the teacher's targets
   matters.

The negative result that feature KD adds nothing on top of output KD bears on the dense-KD line
\cite{liu2019structured,shu2021channel}. Numbers from `paper/brief.md`; the final text must
read them from `paper/tables/*.csv`.

### E. Knowledge distillation in remote sensing and change detection

KD in remote sensing is surveyed in \cite{himeur2025applications}, and early work compressed
scene classifiers \cite{chen2018training}. Distillation from vision foundation models has
reached RS semantic segmentation (VFM-CAKD \cite{wang2026vfm}, JSTARS 2026; semi-supervised
segmentation with FM distillation \cite{xiao2026semisupervised}, TGRS 2026 — both *(title
only)*).

For change detection, the work divides into four groups:
1. **Compression for deployment.** CDKD \cite{wang2024knowledge} combines prototypical
   contrastive distillation with channel/spatial-normalised distillation of the teacher's
   *output probabilities* for on-board binary CD, on three binary datasets (abstract). The
   bi-temporal feature relational distillation of \cite{wang2025bi} (TAES 2025, *title only*),
   ICKD-Net \cite{yang2026intrascale} (Transformer→CNN feature alignment, GRSL 2026, abstract)
   and the EfficientNet student of \cite{wu2024lightweight} (LEVIR-CD, DSIFN, abstract) belong
   here.
2. **Foundation-model distillation.** BFD \cite{wang2025burden} is architecture-agnostic and
   transfers *features* from vision foundation models through dual-temporal feature matching
   and patch contrastive distillation. It is evaluated on binary CD (S2Looking cited: −80.3 %
   parameters, +1.78 IoU versus FM-based CD methods) and removes the FM at inference
   (abstract).
3. **Ensembles and self-distillation for accuracy.** These include multi-teacher KD over
   change-area-ratio partitions (JL1-CD \cite{liu2026jl1}, TIM 2026), multibranch
   self-distillation \cite{liu2025novel} (JSTARS 2025), hierarchical correlation distillation
   for quality-varied images (HiCD \cite{pang2024hicd}), and KD for weak supervision
   \cite{lu2024weakly}.
4. **Teacher–student SCD for robustness.** GUIDE \cite{huang2026guide} (TGRS 2026) is a
   teacher–student framework for *semantic* CD that targets cross-temporal domain shift through
   distribution-mixing augmentation and temporal semantic knowledge guidance (abstract; it does
   not describe compressing a foundation model). SCDistill \cite{qu2026semantic} (arXiv
   2607.19000, accepted at ECCV 2026 per its arXiv comment) uses semantic-invariant
   *self*-distillation with diffusion-simulated perturbations for robustness, not
   compression.

**Closest prior work.** BFD \cite{wang2025burden} is closest in aim: foundation-model knowledge
into an efficient CD model, nothing extra at inference. CDKD \cite{wang2024knowledge} is
closest in mechanism, since it also uses distillation of output probabilities for lightweight
CD. GUIDE \cite{huang2026guide} is the closest teacher–student work on SCD itself. Against
these, this paper differs in four ways:
1. The task is *semantic* CD with SeK on SECOND and Landsat-SCD, where none of the three
   reports a foundation-model-to-compact-student transfer.
2. The teacher is a task-fine-tuned RS foundation model (PerASCD), not generic features. This
   is the regime of \cite{vemulapalli2024knowledge}, which had not been applied to CD.
3. The recipe is outputs-only with an offline D4 cache, and feature distillation on top is
   shown not to help.
4. The claims rest on paired multi-seed statistics, ground-truth target controls, two teachers
   and location-disjoint folds.

The comparison with BFD should be stated explicitly in the paper. A reviewer will raise it.

## Evaluation and positioning

### F. Evaluation leakage and spatial validation

Ecology and remote sensing have long documented that random splits of spatially or temporally
dependent data inflate accuracy. Roberts et al. \cite{roberts2017crossvalidation} set out
blocked cross-validation for structured data. Ploton et al. \cite{ploton2020spatial} show that
spatial validation reveals poor predictive performance of large-scale biomass maps that random
validation had hidden. Meyer and Pebesma \cite{meyer2021predicting,meyer2022machine} tie map
accuracy to the area of applicability. In remote sensing, Lyons et al. \cite{lyons2018comparison}
compare resampling designs, Karasiak et al. \cite{karasiak2022spatial} show that spatial
dependence between training and test sets is a pitfall of accuracy assessment, and Kattenborn
et al. \cite{kattenborn2022spatially} find that autocorrelated training and validation samples
inflate CNN accuracy. Olofsson et al. \cite{olofsson2014good} give the standard practice for
land-change accuracy assessment.

Kapoor and Narayanan \cite{kapoor2023leakage} give the general taxonomy. Their [L3.2]
"non-independence between training and test samples" is leakage *"unless the scientific claim
is about a distribution that has the same dependence structure"* (full text, PMC). That
qualification is exactly our wording rule: the Landsat-SCD random pair split is the dataset
authors' own design for a single-area time series (verified: Yuan et al.
\cite{yuan2022transformer}, p. 11, "we randomly divide" into 6 053/1 729/686, offline
augmentations included; p. 19, multi-region extension named as future work). We change the
question to generalisation to unseen locations. [L1.4] (duplicates in datasets) also describes
the shared images. On test-set reuse, Dwork et al. \cite{dwork2015reusable} and Recht et al.
\cite{recht2019imagenet} motivate reporting a no-selection (final-epoch) number next to the
field's max-test-SeK convention. Bouthillier et al. \cite{bouthillier2021accounting} motivate
multi-seed paired comparisons.

*How this paper differs.* No SCD paper we found measures image-level overlap on Landsat-SCD or
offers a location-disjoint protocol. Searches covered CrossRef, arXiv and the reference lists
of the SCD papers above. This is a search result, not proof of absence.

## Novelty statement (calibrated)

**Genuinely new (to our search, 13 Oct 2026):**
1. Output distillation from a task-fine-tuned RS foundation-model SCD teacher into a compact
   student for semantic CD, with the student matching the teacher on SECOND and surpassing the
   fold teachers on location-disjoint Landsat-SCD.
2. Attribution evidence for SCD KD: ground-truth target controls, two teachers, and a gain
   uncorrelated with teacher advantage (r = 0.09).
3. A measured image-level overlap in the Landsat-SCD random pair split (477/477 test pairs with
   both images in train; learning-free lookup at SeK 77.87), and a released 4-fold
   location-disjoint protocol.

**Not new, and to be credited:**
1. Distilling foundation models into efficient CD models with no inference cost (BFD
   \cite{wang2025burden}).
2. Output-probability KD for lightweight CD (CDKD \cite{wang2024knowledge}).
3. Teacher–student training for SCD (GUIDE \cite{huang2026guide}).
4. Task-oriented VFM-to-small-model transfer in general vision \cite{vemulapalli2024knowledge}.
5. Consistent-teacher and cached soft-label KD \cite{beyer2022knowledge,shen2022fast,yun2021re}.
6. Students outperforming teachers \cite{furlanello2018born,yuan2020revisiting,stanton2021knowledge}.

**Suggested sentence:** *"Prior work has distilled foundation models into efficient binary
change detectors \cite{wang2025burden,wang2024knowledge}; we study semantic change detection,
distil only the outputs of a task-fine-tuned remote-sensing foundation model, and test where
the gain comes from."*

## How published SECOND and Landsat numbers were selected

| Paper | Fact | Status | Source |
|---|---|---|---|
| SECOND \cite{yang2022asymmetric} | Official split 2 968 train / 1 694 test; no validation split | verified | arXiv 2010.05687, p. 9 |
| Bi-SRNet \cite{ding2022bi} | Only 2 968 pairs were openly available; split 4:1 into 2 375 train / 593 test, i.e. **not the official test set** | verified | arXiv 2108.06103v4, p. 7 |
| SCanNet/TED \cite{ding2024joint} | SECOND: 593-pair test carved from the training set, 2 375 train, "following [Bi-SRNet]"; Landsat: 3:1:1 = 1 455/485/485 of 2 425 originals; 50 epochs; checkpoint selection not stated | verified (arXiv v4; IEEE version not read) | arXiv 2212.05245v4, p. 5–6 |
| ChangeMamba \cite{chen2024changemamba} | Trains on train and tests on the official test set; its Table VII caption states that ChangeMask, Bi-SRNet and SCanNet "divide the training set for testing"; retrained SCanNet = SeK 21.57 | verified | arXiv 2404.03425v7, Table VII p. 13 |
| ChangeMamba | Epoch selected on the test set (the only eval loader, named "Validation", reads the test list) | verified in released code, not in paper | `README.md` §3 (project verification of `changedetection/tasks/metadata.py`) |
| Mamba-FCS \cite{wijenayake2026mamba} | SECOND official 2 968/1 694; Landsat 3:1:1 = 1 455/485/485 of 2 425 originals; fixed seed 42; selection rule not stated; copies SCanNet's 593-split 23.94 under a "standard train-test splits" caption; lists ChangeMamba at 24.11 (vs 22.92 in ChangeMamba) | verified (split, numbers); selection unverified | arXiv 2508.08232v2, p. 10, Table V p. 15 |
| PerASCD \cite{shen2026foundation} | SECOND 2 968/1 694; Landsat 3:1:1 of 2 385 filtered pairs = 1 431/477/477; headline Landsat 65.21 equals the best row of a τ ablation (Table VII) whose split is not stated; retrained SCanNet = 23.18 | verified | arXiv 2602.13780v3, p. 8, Tables I, VII |
| PerASCD | SECOND checkpoint saved on best test Fscd (the "val" loader reads `test`); the paper's "validation set" curves on SECOND are therefore test curves | verified in released code, not in paper | `documentation/distillation.md` (PerASCD audit table) |
| Landsat-SCD \cite{yuan2022transformer} | Original split: random 6 053/1 729/686 over all 8 468 pairs, including offline augmentations; single area; multi-region extension named as future work | verified | PDF in `documentation/refs/`, p. 11, p. 19 |
| GSTM-SCD \cite{liu2025gstm}, DBTANet \cite{li2026dual} | Splits and selection rules | **unverified** (full text not read; GSTM-SCD 3:1:1 attribution remains unverified per the brief) | — |

Number-level checks of `paper/tables/published_second.csv` are in
`paper/notes/published_numbers_check.csv`. The main consequence: SCanNet's 23.94 / 63.66 / 73.42
is on Bi-SRNet's 593-pair split and must be footnoted or replaced by an official-split retrain
(21.57 in ChangeMamba, 23.18 in PerASCD).
