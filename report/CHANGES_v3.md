# Report v3: changes relative to v2.2 (the submitted `report_final.pdf`)

**Base.** `report_v2_2.tex`, which produced `report_final.pdf` (the two PDFs are byte-identical; the source recompiles to the same text). v2.2 is kept unchanged in `archive/v2_2/` for comparison.
**Status.** Final; approved by the authors on 2026-10-11, including the *Team contributions* paragraph (§8). 5 pages including references, team contributions and the code link; same template, font size and margins.
**Not changed.** Models, weights, thresholds, normalisation, inference options and their default, data, published results and identifiers. Every number in v3 is either in v2.2 or checked against its source file by [`check_numbers_v3.py`](check_numbers_v3.py) (0 unsourced).

## 1. New order: research first, history second

| v2.2 | v3 |
|---|---|
| 1 Motivation · 2 Related work · 3 Data · 4 The model · 5 How the method changed (six chronological sub-sections) · 6 Results · 7 Discussion · 8 Limitations · 9 Reproducibility · Team · Code | 1 Problem and previous project (with the **task** stated once: binary detection per track, not type classification) · 2 Related work · 3 **Data and evaluation** (adds the four result roles and the decision unit and metrics) · 4 **The model** (the final system before any experiment; *one model, two ways to run it*) · 5 **Four questions behind the design** (Q1–Q3) · 6 **Results**: 6.1 final test (central), 6.2 development and validation, 6.3 **Q4 (post hoc)** · 7 **Conclusions and limitations** · Code and reproducibility · Team contributions |

Each question follows the same order: *Compared → Result → Decision → Not shown*.

| Question | Experiments it summarises | Where in v2.2 |
|---|---|---|
| Q1 Why was reconstruction error not enough, and why supervision? | autoencoder, latent score, supervised head | §5.1, §5.2 |
| Q2 How did training stability and data diversity affect detection? | seed study → ensemble, tailgating data, stop-mode diversity (balancing and frozen encoder in one clause) | §5.3, §5.4 |
| Q3 Why do more detections not justify replacing the model? | legitimate-stop diagnosis and three remedies; weak-type diagnosis and targeted data (marked *after the test was exposed*) | §5.5, §6 "W1/W2", §7 "Legitimate versus abnormal stops" |
| Q4 What does the headway summary add? (post hoc) | φ<sub>u</sub> history and active vs neutralized | §4 (part), §5.4 (part), §7 "Contribution of the headway summary", Table 3 |

## 2. Repetition removed (no finding removed)

| Topic | In v2.2 | In v3 |
|---|---|---|
| Targeted-data round | method in §5.5, numbers in §6, meaning in §7, limitation (2) and (4) | once in Q3; one clause in limitations |
| Legitimate stops | §5.5, §6, §7, limitation (1) | diagnosis and remedies once in Q3; final-test numbers in §6.1; limitation (1) |
| φ<sub>u</sub> history | §4, §5.4, §7 | definition in §4; history and check once in §6.3 (Q4) |
| Final-test independence | §6 and limitation (4) | §3 (evaluation roles), §6.1, limitation (5) |
| Release paths and verification script | §9 (release folder `DeepDAS_Release_v1.0`, `verify_release.py`) | one paragraph on the public repository and `verify_phi_modes.py` |

Moved to the documentation only (with links from `research/README.md`): study numbers (`research/01`–`08`), the 23 training legitimate stops, the PyTorch version, the encoder/head AUC ranges of the weak-type diagnosis.

## 3. Internal names replaced by descriptive names

| v2.2 | v3 |
|---|---|
| W1 | targeted training-data expansion |
| W2 | the same data with controlled sampling |
| C5 / "failed only C5" | the pre-registered limit on legitimate-stop alarms (at most three more than the frozen model) / "passed the other eight conditions" |
| "Balancing types or freezing the encoder" (EXP1 / EXP2) | type-balanced sampling, a frozen encoder |
| "two pre-registered remedies … and an epoch-selection rule" (F1, F2, J) | controlled exposure to legitimate stops, an added temporal branch, checkpoint selection by a task rule |
| "study 02", "study 06", `research/01`–`08` | removed from the text; one pointer to `research/` (indexed by the four questions) |

`val_A` and `val_B` are kept as split names, defined once in §3 with their role.

## 4. Figures (presentation only; no value, denominator or threshold changed)

| Figure | v2.2 | v3 |
|---|---|---|
| Fig. 1 architecture | G0 | G0, unchanged file; caption names the "16 learned + 1 physical summary" head input |
| Fig. 2 controlled comparisons | V5 with labels W1/W2, "limit C5" | V5 with descriptive panel titles (comparison, data set, *n*), metric on each axis, legend "targeted data expansion / same + controlled sampling", limit spelled out; data CSV byte-identical |
| Fig. 3 ROC / PR | G4, four configurations (incl. post-hoc neutralized) | **V6**, the three detectors frozen before the test; values asserted equal to G4 and T2 |
| Fig. 4 recall per type | G3, four configurations | **V7**, the three frozen detectors; counts asserted equal to G3 and T1 |
| Table 3 (φ<sub>u</sub>) | neutralized \| active | active (as tested) \| neutralized; same four rows |

The four-configuration figures G3/G4 stay in `research/08_tailgating_feature_intervention/figures/` and in the post-hoc section of the figure gallery. Figure scripts: `figures/scripts/make_report_figures.py` (V5), `figures/scripts/make_main_performance_figures.py` (V6, V7); G0 is unchanged.

## 5. Precision fixes (checked against the source, recorded here)

| v2.2 text | Source | v3 text |
|---|---|---|
| "A latent score … already detected 5 of the 9 **development** cases" | `research/01_unsupervised_autoencoder/records/unsup_release_primary_latent_maxZ_quality_P99_RELEASE_REPORT.md`: 5 of 9 **original SimStudio scenarios** at P99 (two of them through a standstill: the braking vehicle that never moves and the collision without contact); none of the realised braking, straddling or tailgating cases (C1–C3) | "detected 5 of the 9 original SimStudio scenarios without labels (two through a standstill), though none of the realised braking, straddling or tailgating cases" |
| Seed sensitivity: "with **105** positives" | The seed study (study 02) trained the single-model recipe on the **34** cross-region training anomalies; 105 is the final model's training set | "with 34 training anomalies" |
| Final test: "on straddling (5/12) and collision (6/10) **the single model** is better" | `results/final_test/T1_per_type.csv`, common subset: straddling single 8/12, Z-score 0/12; collision single 9/10, **Z-score 10/10** | "on straddling the single model is better (8/12 vs 5/12), and on collision both baselines are (9/10 and 10/10 vs 6/10)" |
| "Encoder probes of 0.91–1.0 against head AUCs of 0.80–0.93" | `research/06_weak_type_diagnosis/README.md`: across the six weak types the encoder probe is 0.81–1.0 and the score AUC 0.78–0.985; the two ranges in v2.2 cover different subsets of types | Q3 keeps the full probe range (0.81–1.0); §7 gives one exact example (overweight: probe 0.96, score 0.83) |
| Stage-1 comparison "in the test regions" | `research/01_unsupervised_autoencoder/README.md`: these are held-out regions of the normal corpus, **not** the final test | "in held-out regions of the normal corpus: 91 injected and 632 normal vehicles" |
| "−7.5 m/s² … **above** the injected 6 m/s² brake" | same source; the comparison is of magnitudes | "stronger than the injected 6 m/s² brake" |
| §9 points to the release folder and `verify_release.py` | the code link is the public repository, whose checker is `scripts/verify_phi_modes.py` | "Code and reproducibility" describes the repository |

## 6. Unchanged on purpose
Abstract numbers, Table 1 (hyper-parameters), Table 2 (final test, common subset with the Z-score's own-set values in the caption), the Klausner sentence, the references, the authors (Neta Shaboo), the GitHub link, and the *Team contributions* placeholder (only the authors can fill it in).

## 7. Revision after the authors' review (2026-10-11)
The authors chose v3 as the basis for the submission report. The structure and the figures are unchanged; only wording and one count were corrected.

| Where | Before | After |
|---|---|---|
| Q1 | "The cause is the data:" | "The measurements point to the data:" |
| Q2 (i) | "head initialisation and batch order matter as much as the data" | "the seed alone moved the result by five cases" |
| Q2 (iii) | "The new data cost the stalled case: …" | "With the new data the stalled case was missed; …" (the two measured factors are listed, not presented as the proven cause) |
| Q2, not shown | "the expanded data without φ<sub>u</sub> was never trained" | "no model without φ<sub>u</sub> was trained on the expanded, final training data" |
| Q3, legitimate stops | "A read-only diagnosis found why: … mean/std pooling lets the share of stop windows decide …" | The three measurements are listed. The pooling explanation is presented as "supported, without proving it as the only cause". |
| Q3, weak types | "the information is present … but lost in the head" | "separable in the input and in the pooled code … but less so in the final score … this points to the head, without isolating one cause" |
| §6.1 | tailgating matched controls "7/11, 0/11 and 0/12" (the Z-score on its own 12 decided controls) | "7/11, 0/11 and 0/11" on the controls decided by all three detectors (`results/final_test/vehicle_outcomes.csv`) |
| §6.3 | "so the learned representation carries most of their signal"; "a 16-input head was never trained on the final data" | "which suggests that …"; "no 16-input head was trained on the final training data (early models without φ<sub>u</sub> exist)" |
| §7 | "so the brake is inside the noise"; "which places the remaining loss in the head"; limitation (1) "cause understood, not removed" | "so the brake lies within the range of normal tracks"; "which points to the head rather than the encoder"; "diagnosed, not removed" |

The report is still 5 pages; `check_numbers_v3.py` reports 0 unsourced numbers and checks the matched-controls statement against the data.

## 8. Final version (2026-10-11)

**Team contributions.** The paragraph approved by the authors replaces the placeholder:

> Noa Sapir led the technical and experimental work, with primary responsibility for designing and generating the simulation scenarios and datasets of this project (building on the SimStudio infrastructure of the earlier project), training the deep-learning models, and executing the experiments. Asaf Zini contributed to preparing and checking simulation inputs and to technical validation of the experimental workflow. Neta Shaboo contributed to the literature review and to organizing and editing the research documentation and report.

**Two cuts to stay within 5 pages** (approved together with the paragraph; no research content, number or figure changed):
- **§1:** the closing sentence that listed what each section contains was removed.
- **Code and reproducibility:** two technical details were removed; both stay documented in the repository.
  - the platform note "(byte-identical on Linux x86-64; identical decisions and tables on aarch64)", now in `docs/CROSS_PLATFORM.md`;
  - the sentence on the raw-simulation package, now in `docs/RAW_DATA_PACKAGE.md`.

**Checks.**
- **Layout.** 5 pages, all fonts embedded, no overfull lines.
- **Numbers.** `check_numbers_v3.py` reports 0 unsourced numbers.
- **The approved preview.** The text is identical to the preview the authors approved.

## Build
```bash
cd report && pdflatex report_v3.tex && pdflatex report_v3.tex     # figures from figs_v3/
python report/check_numbers_v3.py                                   # from the repository root
```
