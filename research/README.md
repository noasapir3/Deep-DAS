# Research record

The studies that led to DeepDAS Ensemble v1, in the order they were run, followed by the post-hoc analysis of study 08.

**How each page is organised.**

- **At a glance**, at the top: the research question, what was compared, on which data, the figure and results, the conclusion and limits, and how to reproduce.
- **Full record**, below it.
- **`records/` sub-folder:** the plans, results and selection records written at the time, copied verbatim.

| # | Study | Question | Outcome | Data role | Figure |
|---|---|---|---|---|---|
| 01 | [Unsupervised autoencoder](01_unsupervised_autoencoder/) | Does a model of normal driving alone flag anomalies? | No (AUROC 0.60 vs Z-score 0.89); its encoder is reused | development | [S2, S1](../figures/README.md#7-archived-stage-1-autoencoder) |
| 02 | [Supervised single model](02_supervised_single_model/) | Does a supervised head with cross-region anomalies work? | Yes, 7/9; strong seed dependence → ensembles | development / validation | [S02](../figures/README.md#2-development-data) |
| 03 | [Tailgating data](03_tailgating_data/) | Is tailgating a data problem? | Validation tailgating recall 0.385 → 0.923 (both arms with φ<sub>u</sub>) | validation | [V5a](../figures/README.md#3-validation-data) |
| 04 | [Stall diversity](04_stall_diversity/) | Do diverse stop modes fix the stall regression? | Yes → **DeepDAS Ensemble v1** selected and frozen | validation | [V5b](../figures/README.md#3-validation-data) |
| 05 | [Legitimate stops](05_legitimate_stops/) | Can false alarms on 1–10 s stops be removed? | Not without breaking other conditions → main limitation | validation | [S05](../figures/README.md#3-validation-data) |
| 06 | [Weak-type diagnosis](06_weak_type_diagnosis/) | Where are the weak types lost? | After the encoder: in the head and its few examples | exposed final-test data (post hoc) | [S06](../figures/README.md#5-post-hoc-analyses) |
| 07 | [Targeted data (W1/W2)](07_targeted_data_study/) | Does targeted data fix the weak types? | Weak6 0.35 → 0.71 / 0.74, but fails the legitimate-stop condition → not adopted | validation (after the test was exposed) | [V5c](../figures/README.md#3-validation-data) |
| 08 | [Tailgating-feature intervention](08_tailgating_feature_intervention/) | How much does the frozen model depend on φ<sub>u</sub>? | One development case; most test tailgaters stay detected without it | development, validation, exposed final test (post hoc) | [G1–G4](../figures/README.md#5-post-hoc-analyses) |

**Also related:** the final model's [Klausner experiment](../models/deepdas_ensemble_v1/experiment_klausner/), inference on the one real recording, a stalled bus (detected).

## Reading the records

- **Internal identifiers.** The records use the identifiers of the working tree. `docs/IDENTIFIERS.md` maps them to the names used in this release. The most frequent ones:

  | Identifier | Meaning |
  |---|---|
  | `E_stall_diversity_ENSEMBLE`, `E_DIV`, `E_diversity`, `REF` (studies 05 and 07) | DeepDAS Ensemble v1 |
  | `CR_P99`, `BASELINE_frozen` | the supervised single model |
  | `C_new` | the ensemble of study 03 |
  | `stallval` | the stall validation part of the stall and stop set |
  | `stopval2` | the legitimate-stop validation set |

- **Paths.** Paths inside the records refer to the original working tree. They are historical and are not paths of this release.
- **Records are not cleaned up.** They are archived as they were written, including versioned revisions, errata and disclosed incidents.

## What is not part of the record

- Exploratory scripts and intermediate files that did not lead to a reported result.
- Logs of individual training runs.
- The raw simulator output (in the raw-data package; `docs/RAW_DATA_PACKAGE.md`).
- The superseded internal backups.

`docs/ARCHIVE_INDEX.md` lists what was left out and where it is kept.
