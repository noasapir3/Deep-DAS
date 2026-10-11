# Research: four questions behind DeepDAS Ensemble v1

The final model was not designed in one step. Each change answered one question, with a comparison whose data, single difference between arms and replacement rule were written down before training.

- **This page:** the questions, the answer to each and the study folders that hold the evidence.
- **Each study page:** a summary at the top, readable on its own, with the full record below it.
- **The report:** the same four questions appear in its §5 and §6.3 ([report v3](../report/DeepDAS_report_v3.pdf); section-by-section map: [docs/QUESTION_MAP.md](../docs/QUESTION_MAP.md)).

**The model the questions led to:** [DeepDAS Ensemble v1](../models/MODEL_CARD_DeepDAS_Ensemble_v1.md) (current model card). It is a convolutional encoder over 14 track channels, a supervised head with one physical summary (φ<sub>u</sub>), five seeds and a frozen threshold. It is evaluated once on the [**independent final test**](../results/final_test/).

**Data roles** ([evaluation protocol](../docs/EVALUATION_PROTOCOL.md)):

| Role | What it means here |
|---|---|
| **development** | the 9 diagnostic cases, used for design decisions |
| **validation** | used to select models |
| **final test** | independent only for the three detectors frozen before it |
| **post hoc** | analyses defined after the final test was exposed |

---

## Q1. Why was reconstruction error not enough, and why did we move to supervised learning?
**Answer.** The proposal's autoencoder flagged almost nothing (AUROC 0.60 against 0.89 for the SimStudio Z-score): this simulator's anomalies stay inside the normal range of a 6.4 s window. Its encoder still organised the tracks usefully, so we kept the encoder and trained a supervised head on simulated anomalies whose labels are checked against simulator truth. That head detected 7 of the 9 development cases.

| Study | What it compared | Data role | Outcome |
|---|---|---|---|
| [01 Unsupervised autoencoder](01_unsupervised_autoencoder/) | reconstruction error and a latent score against the Z-score | development (archived values) | rejected as a detector; encoder kept |
| [02 Supervised single model](02_supervised_single_model/) | a supervised head trained on anomalies from other regions | development / validation | 7/9 development cases; became the single-model baseline |

## Q2. How did training stability and data diversity affect detection?
**Answer.**

- **Seed dependence.** The original single-model recipe detected 7, 2 or 3 development cases depending on the random seed (a range of 2–7). Across all four recipes tested with three seeds, the range was 2–8. Every later model is therefore a five-seed ensemble.
- **Varied tailgating data.** With varied, verified tailgating examples added, validation tailgating recall rose from 0.385 to 0.923. Both arms already included φ<sub>u</sub>.
- **Stop modes.** With three stop modes every seed detected all 18 validation stalls. With one mode repeated, one seed fell to 0.67. The three-mode ensemble became the final model.

| Study | What it compared | Data role | Outcome |
|---|---|---|---|
| [02 Supervised single model](02_supervised_single_model/) (seed study) | four recipes (the original and three variants), each with seeds 0, 1, 2 | development | original recipe 7, 2, 3 cases (2–7); all four recipes 2–8 → ensembles |
| [03 Tailgating data](03_tailgating_data/) | the same recipe without and with varied tailgating data | validation | 0.385 → 0.923; adopted |
| [04 Stop-mode diversity](04_stall_diversity/) | 24 added stall slots: one stop mode repeated or three modes; before that, type-balanced sampling and a frozen encoder | validation | three modes selected → **DeepDAS Ensemble v1** |

## Q3. Why do more detections not necessarily justify replacing the model?
**Answer.** The final model raises many false alarms on legitimate stops shorter than 10 s.

- **Remedies before the test.** Three pre-registered remedies lowered those alarms, but each broke another condition.
- **Targeted data after the test was exposed.** Targeted training data for the six weak types roughly doubled their validation recall, but pushed legitimate-stop alarms past the pre-registered limit.

None of these candidates replaced the model. A road monitor that alarms at ordinary stops is not better, even if it detects more.

| Study | What it compared | Data role | Outcome |
|---|---|---|---|
| [05 Legitimate stops](05_legitimate_stops/) | diagnosis; controlled exposure, a temporal branch, task-rule checkpoint choice | validation | alarms 40 → 37 / 27 / 21 of 54, none within the rule → main limitation |
| [06 Weak-type diagnosis](06_weak_type_diagnosis/) | where the information about six weak types is lost | exposed final test (post hoc) | probes on the input and the pooled code separate them better than the final score; the head saw four examples per type. This points to the head and its training data, without isolating one cause. |
| [07 Targeted training data](07_targeted_data_study/) | targeted data expansion, and the same with controlled sampling | validation, after the test was exposed | weak-type recall 0.35 → 0.71 / 0.74, but too many legitimate-stop alarms → not adopted |

## Q4. What does the tailgating summary φ<sub>u</sub> add beyond the learned representation?
**Answer.** The final model is a hybrid: 16 learned features plus one physical summary, φ<sub>u</sub>.

- **What we compared.** With the same frozen weights, holding φ<sub>u</sub> at zero at inference loses the development tailgating case (8/9 → 7/9). It keeps 10 of 11 final-test tailgaters (post hoc), which suggests that the learned representation carries most of their signal.
- **What remains open.** Whether the tailgating gain came from φ<sub>u</sub>, from the data or from both is not identified. No model without φ<sub>u</sub> was trained **on the final training data**. Early single models without it do exist (study 02); like those with it, they missed the tailgating case.

| Study | What it compared | Data role | Outcome |
|---|---|---|---|
| [08 Tailgating-feature check](08_tailgating_feature_intervention/) | the frozen model with φ<sub>u</sub> active against the same weights with φ<sub>u</sub> neutralized | development, validation, exposed final test (post hoc) | one development case depends on φ<sub>u</sub>; test tailgating 11/11 vs 10/11 |

The two inference options are explained in [docs/INFERENCE_MODES.md](../docs/INFERENCE_MODES.md).

---

## The final test and the real recording

**[The independent final test](../results/final_test/).** 228 new runs in 8 held-out regions; the plan was frozen before simulation and the test was run once.

- **DeepDAS Ensemble v1:** macro recall 0.568, ROC-AUC 0.908, AUPRC 0.742.
- **SimStudio Z-score, on the same vehicles:** 0.232 / 0.819 / 0.496.
- **Figures:** [figure gallery §4](../figures/README.md#4-final-test).

**[The Klausner experiment](../models/deepdas_ensemble_v1/experiment_klausner/).** The final model, run by inference only on the one real recording: a city bus that stands for about 31 s. It is detected in both inference options. This is one case, not a statistical evaluation.

## The studies in the order they were run

| # | Study | Question | Dates (2026) | Status |
|---|---|---|---|---|
| 01 | [Unsupervised autoencoder](01_unsupervised_autoencoder/) | Q1 | Sep | superseded; encoder reused |
| 02 | [Supervised single model](02_supervised_single_model/) | Q1, Q2, Q4 (history) | 27 Sep – 7 Oct | shipped as a baseline |
| 03 | [Tailgating data](03_tailgating_data/) | Q2 | 7–8 Oct | adopted |
| 04 | [Stop-mode diversity](04_stall_diversity/) | Q2 | 8 Oct (frozen 9 Oct) | **final model selected and frozen** |
| 05 | [Legitimate stops](05_legitimate_stops/) | Q3 | 9 Oct | no replacement; main limitation |
| – | [Final test](../results/final_test/) | – | 9 Oct | run once |
| 06 | [Weak-type diagnosis](06_weak_type_diagnosis/) | Q3 | 10 Oct | post hoc, read-only |
| 07 | [Targeted training data](07_targeted_data_study/) | Q3 | 10 Oct | not adopted; development closed |
| 08 | [Tailgating-feature check](08_tailgating_feature_intervention/) | Q4 | 10 Oct | post hoc, inference only |

The full chronology is in [docs/DEVELOPMENT_HISTORY.md](../docs/DEVELOPMENT_HISTORY.md).

## Reading a study page and its records

- **Page layout.**
  - **Summary**, at the top: what we wanted to find out, what was compared and held constant, the data and its role, the main figure, the result, the decision, what it does not show, and links to the code, the data and the reproduction steps.
  - **Full record**, below it: the method and the tables in detail.
  - **`records/` sub-folder:** the plans, results and selection records written at the time, copied verbatim, including versioned revisions, errata and disclosed incidents.
- **Names.** The summaries use descriptive names. The records keep the study codes: `W1`, `C5`, `EXP1`, `F2`, `stallval`, … Their meaning is listed in [docs/IDENTIFIERS.md](../docs/IDENTIFIERS.md#experiments-and-conditions-display-names). The most frequent model identifiers:

  | Identifier | Meaning |
  |---|---|
  | `E_stall_diversity_ENSEMBLE`, `E_DIV`, `E_diversity`, `REF` (studies 05 and 07) | DeepDAS Ensemble v1 |
  | `CR_P99`, `BASELINE_frozen` | the supervised single model |
  | `C_new` | the ensemble of study 03 |

- **Paths inside the records** refer to the original working tree. They are historical, not paths of this repository.
- **What the repository regenerates and what it does not.**
  - **Regenerated by the repository scripts:**
    - the results of the final model and of the single model: development, validation and final test, in both inference options;
    - the tables of study 08;
    - the evaluation of study 07.
  - **Not regenerated:** the Z-score enters the final test as its precomputed scores. The numbers of studies 01–06 come from their archived records, and each summary says so.

## What is not part of the record
- Exploratory scripts and intermediate files that did not lead to a reported result.
- Logs of individual training runs.
- The raw simulator output (in the raw-data package; [docs/RAW_DATA_PACKAGE.md](../docs/RAW_DATA_PACKAGE.md)).
- The superseded internal backups.

[docs/ARCHIVE_INDEX.md](../docs/ARCHIVE_INDEX.md) lists what was left out and where it is kept.
