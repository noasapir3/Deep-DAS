# Documentation round v3 (2026-10-11): what changed

**What this round was.** Writing and documentation only. Nothing was trained, simulated, re-calibrated or re-thresholded.

- **Unchanged:** weights, thresholds, normalisation, inference options and their default, data, results, identifiers, code paths and folder layout.
- **How it was done:** on a local working branch (`docs-v3`), not pushed.
- **The report:** its changes are listed separately in [report/CHANGES_v3.md](../report/CHANGES_v3.md).

## 1. A navigation layer by research question

| Page | Change |
|---|---|
| [README.md](../README.md) (front page) | The task is stated once as binary detection. The final model comes before the research. *One model, two ways to run it*: `active` first, as the original model; `neutralized` as the dependence check. A table separates the trained model, the inference option, the release folders and the report versions. The results table holds only the three detectors frozen before the test; the neutralized option has its own small table. The main figures are V6/V7. The research section is organised by the four questions. |
| [research/README.md](../research/README.md) | Rewritten as the entry page. For each of Q1–Q4: a short answer, the studies, their data role and outcome. Then the final test, the Klausner experiment, the chronological order of the studies, and how to read the records. |
| `research/0*/README.md` (8 pages) | The "At a glance" table is replaced by a summary that can be read on its own, always in this order: what we wanted to find out, what was compared and what was held constant, data and its role, the main figure with an explanation, result, decision, what it does not show, and code / data / reproduction links. The **full record below each summary is unchanged**, except the few statements marked † in the revision of §7: `number_sources.py` and the figure scripts read it, and still verify. |
| [docs/QUESTION_MAP.md](QUESTION_MAP.md) (new) | Research question and report section → study folder → figure → evidence files, with their data role. |
| [docs/IDENTIFIERS.md](IDENTIFIERS.md) | New section *Experiments and conditions: display names*: `W1`, `W2`, `C1`–`C9`, `C5`, `EXP1`, `EXP2`, `F1`, `F2`, `J`, `stallval`, `stopval2`, `macro9`, `Weak6`, arms `A`–`D`, `C_old`/`C_new`, `E_quantity`/`E_diversity`, `val_A`/`val_B`. It also notes that the development cases C1–C4 and the study-07 conditions C1–C9 are different series. |
| [docs/INFERENCE_MODES.md](INFERENCE_MODES.md) | A one-paragraph summary; a table "three things that are easy to confuse" (model, option, folder version); `active` listed first everywhere. |
| [figures/README.md](../figures/README.md) | §4 (final test) shows V6/V7 with the three frozen detectors; the four-configuration G3/G4 moved to §5 (post hoc). The V5, S02 and S05 descriptions say what was compared, on which data and with which metric. |
| [results/README.md](../results/README.md) | Which report section uses which folder, and how to compare detectors on one population, with the column trap of `T2_overall.csv` explained. |
| [INSTALL.md](../INSTALL.md), [models/README.md](../models/README.md) | Example commands list `active` before `neutralized`, as everywhere else. |

## 2. Display names in the figures (presentation only)

| Figure | Before | After |
|---|---|---|
| V5 | panel titles "Tailgating (*n* = 13)", "Stalled vehicles (*n* = 18)", "Targeted data: gain vs. cost"; legend "W1: + targeted data", "W2: W1 + sampling"; "pre-registered limit C5" | each title names the comparison and the data set (with *n*); each axis names the metric; legend "targeted data expansion", "same + controlled sampling"; limit "at most 3 more legitimate-stop alarms than frozen"; panel (c) marked "after the test was exposed; not adopted" |
| S05 | "F1", "F2", "Reference, epoch by J", … | "controlled exposure", "exposure + temporal branch", "frozen recipe, task-rule epoch", … |
| S02 | "A (cleaning, no φ<sub>u</sub>)", "B (cleaning + osc)", … | "+ artefact cleaning", "+ cleaning + weaving feature", … |
| V6, V7 (new) | – | ROC/PR and recall per type with the three frozen detectors, in the style of G4/G3 |

**The data behind the figures did not change.** The data CSVs of V5, S02 and S05 are byte-identical to before. V6/V7 assert that every drawn value equals G4/G3 and the published tables T1/T2.

## 3. Inconsistencies found and corrected
Each one was checked against its source file; the historical records were not edited.

| # | Where | What was wrong | Source checked | Correction |
|---|---|---|---|---|
| 1 | `RESEARCH_OVERVIEW.md` §4.3 | The comparison table put the Z-score's **own decided set** (0.231, 0.783 / 0.394, 13/1574, 23/32) next to the deep models' **common subset** (1,008 normals) | `results/final_test/T2_overall.csv`, `figures/final_test/data/V6_roc_pr_frozen_detectors.csv` | table on the common subset (0.232, 0.77 / 0.35, 0.819 / 0.496, 7/1008, 22/28); the own-set values in a separate sentence |
| 2 | `RESEARCH_OVERVIEW.md` summary | same mixing in the "Baselines on the same test" bullet | same | common-subset values, own set stated separately |
| 3 | `RESEARCH_OVERVIEW.md` §3 | study 08 missing; study 03 described as if cleaning + φ<sub>u</sub> had made tailgating detectable | `research/03_tailgating_data/README.md` (both arms include cleaning + φ<sub>u</sub>) | row 08 added; row 03 says the data is what changed and that no model without φ<sub>u</sub> was trained on the final training data |
| 4 | `README.md` (front page) | the key-results table mixed the post-hoc neutralized option into the comparison of the detectors frozen before the test | `docs/EVALUATION_PROTOCOL.md` | three frozen detectors only; neutralized in its own table |
| 5 | `README.md`, figure gallery | "On straddling and collision the single model does better" | `results/final_test/T1_per_type.csv`: collision single 9/10, **Z-score 10/10** | straddling: single model (8/12); collision: both baselines |
| 6 | `docs/LIMITATIONS.md` §7 | "the single real bus track … was used only qualitatively in study 01", which predates the Klausner experiment of the final model | `models/deepdas_ensemble_v1/experiment_klausner/README.md` | the final model detects the stalled bus in both options; one case, not an estimate |
| 7 | `docs/EVALUATION_PROTOCOL.md` §1 | only three result kinds; the post-hoc class (studies 06–08, `results_phi_neutralized/`) missing; study 08 not named as a use of the exposed test | `research/08_tailgating_feature_intervention/README.md` | "Post hoc" row added; rule that own-set and common-subset values never share a table column |
| 8 | `docs/DEVELOPMENT_HISTORY.md` | the model was listed as frozen on 8 Oct; no rows after the release build | `models/deepdas_ensemble_v1/freeze_record.json` (`frozen_on: 2026-10-09`) | selected 8 Oct, frozen 9 Oct; rows for study 08, the Klausner experiment, the repository and this round |
| 9 | `research/03_tailgating_data/README.md`, summary | "59 verified **training** tailgaters" | the full record of the same page: 24 + 23 training, 12 validation | "59 verified tailgaters (47 for training, 12 for validation)" |
| 10 | `README_v1.1_REVIEW.md` | "the public code link [is] still a placeholder" | report v2.2 | dated update note; the review-copy folder itself is unchanged |
| 11 | `models/deepdas_ensemble_v1/MODEL_CARD.md` | its comparison table puts the Z-score's own-set values next to the ensemble; it predates the Klausner experiment | `T2_overall.csv` | **not edited** (the frozen model folder is hash-checked, `verify_phi_modes.py` P1); a correction note was added to [models/README.md](../models/README.md) |
| 12 | `docs/RELEASE_README_v1.0.md` | historical front page: own-set Z-score column, seven studies | – | kept as it was, with a short banner on top |
| 13 | report v2.2 | four imprecise statements (latent score "5 of the 9 development cases"; "105 positives" in the seed study; "the single model is better on collision"; mixed AUC ranges) | see [report/CHANGES_v3.md §5](../report/CHANGES_v3.md) | corrected in v3 only; v2.2 is kept unchanged as the submitted version |

## 4. Moved from the report into the documentation
Report v3 keeps every finding of v2.2. What it no longer states, and where that detail now lives:

| Detail in v2.2 | Where it lives now |
|---|---|
| study folder numbers (`research/01`–`08`, "study 02", "study 06") | [research/README.md](../research/README.md) and [QUESTION_MAP.md](QUESTION_MAP.md): question → study folder |
| the 23 training legitimate stops (behind the "105× fewer stop windows") | [research/05](../research/05_legitimate_stops/), step 1 |
| per-type encoder-probe and score AUC ranges | [research/06](../research/06_weak_type_diagnosis/), results table (v3 keeps one exact example) |
| the per-seed stall figures repeated in the Discussion (0.93 ± 0.15 vs 1.00 ± 0.00) | stated once in Q2; full table in [research/04](../research/04_stall_diversity/) |
| the targeted-data numbers repeated in the Discussion and limitations | stated once in Q3; full table in [research/07](../research/07_targeted_data_study/) |
| release folder names, `verify_release.py`, the PyTorch version | [INSTALL.md](../INSTALL.md), [REPOSITORY_MAP.md](REPOSITORY_MAP.md), `requirements.txt` |

The study pages also hold what the report never had room for: the tailgating plan versions v1–v4, the exposure experiments, the legitimate-stop counterfactual tables, the nine conditions of the targeted-data round and the φ<sub>u</sub> history table.

## 5. Deliberately not changed
- **Verbatim copies:** every `records/` folder, `results/**/records/`, the generated display reports (`FINAL_TEST_REPORT.md`) and `REPORT_V2_NUMBER_SOURCES.*`.
- **The frozen model folder,** including its model card, and both baseline folders.
- **Code, scripts and tests.** The exceptions are the display labels in two figure scripts and one new figure script; all three are presentation only.
- **The release folders on disk** (`DeepDAS_Release_v1.0`, `DeepDAS_Release_v1.1_review`) and **report v2.2.**
- **The default inference option** (`active`).

## 6. Checks run after the changes

| Check | Result |
|---|---|
| `research/08_tailgating_feature_intervention/code/number_sources.py` (report v2.2 numbers against their sources) | 35/35 |
| `report/check_numbers_v3.py` (every number of report v3 in v2.2 or in its source) | 0 unsourced (6 new, all sourced) |
| Report v3 page count | 5 |
| Relative links and images in all Markdown files, with anchors | 0 broken (before the round: 0 of 238; after: 0 of 411) |
| Figure data CSVs of V5, S02, S05 | byte-identical to before |
| `scripts/check_models.py` (checkpoint hashes, architecture, both inference options) | all checks passed |
| `scripts/verify_phi_modes.py` on the branch, with the processed data | P1, P3–P7 passed: frozen model files identical to v1.0; `active` reproduces the published development and final-test results byte for byte; `neutralized` reproduces `results_phi_neutralized/`; report v2 numbers 35/35. P2 (original release unchanged) was replaced by a direct before/after sha256 comparison on the authors' computer: `DeepDAS_Release_v1.0` (6,098 files), `DeepDAS_Release_v1.1_review` (6,172 files) and `DeepDAS_Report_v1.0` (112 files) are identical, and `report_final.pdf` is unchanged. |

## 7. Revision after the authors' review (2026-10-11)
The authors chose report v3 as the basis for submission and asked for the corrections below. Again nothing was trained or run: models, thresholds, results and inference options are unchanged.

| Request | What was changed |
|---|---|
| **The seed range.** "2–8" in the seed-study row of `research/README.md` | The original recipe detected 7, 2 and 3 cases (2–7); 2–8 is the range over all four recipes with three seeds each. This is now stated explicitly in `research/README.md`, in the study 02 summary and in the S02 description of the figure gallery. |
| **"Never trained without φ<sub>u</sub>."** | Every such statement now says **on the final training data**, and where relevant that early single models without φ<sub>u</sub> exist (study 02, early data). The changes are in the report (Q2, Q4), the front page, `research/README.md`, the summaries of studies 02, 03 and 08, `RESEARCH_OVERVIEW.md`, `docs/INFERENCE_MODES.md` and `MISSING_BEFORE_SUBMISSION.md`. |
| **Absolute causal wording** about the classification head and the stops | Each statement now separates **what was measured** from **the explanation it supports**, without presenting a single cause as proven. The changes are listed below. |
| **The model card** | A current model card was added outside the hash-checked folder: [models/MODEL_CARD_DeepDAS_Ensemble_v1.md](../models/MODEL_CARD_DeepDAS_Ensemble_v1.md). It has every comparison on the common decided subset (including recall per type for all three detectors), both inference options and the Klausner experiment. It is linked from the front page, `models/README.md`, `research/README.md` and the question map. The original `MODEL_CARD.md` is unchanged. |
| **Report status** | v3 is marked as the basis for submission; it becomes the main report after team approval and the contributions paragraph, and v2.2 then becomes the previous version. The steps and the content of the clean submission package are in [MISSING_BEFORE_SUBMISSION.md](../MISSING_BEFORE_SUBMISSION.md). Nothing was deleted. |

**Where the causal wording was softened:**
- **The report:** Q1, Q2 (seed effect and the stall miss), Q3 (legitimate stops; weak types), §6.3 and §7. Limitation (1) now reads "diagnosed, not removed".
- **The entry pages:** the summaries of studies 04, 05 and 06, the Q2 and Q3 answers of `research/README.md`, `RESEARCH_OVERVIEW.md` (row 06: "supported, not proven"), `docs/LIMITATIONS.md` §1 and §3, and the S06 description of the figure gallery.
- **The full records:** six statements, each marked † with a note at the top of its full record. They are in study 02 (conclusion 4), study 05 (the counterfactual row, the proposed mechanism and conclusion 1), study 06 (conclusion 1) and study 07 (conclusion 3). The text as first released stays in `DeepDAS_Release_v1.0` and in git history; `records/` was not touched.

**One more population fix found while checking.**
- **The issue.** The report and the new model card gave the Z-score's alarms on tailgating matched controls as 0/12. That count is on its own 12 decided controls.
- **The fix.** On the 11 controls decided by all three detectors it is **0/11** (`results/final_test/vehicle_outcomes.csv`). The report now reads "7/11, 0/11 and 0/11"; `report/check_numbers_v3.py` checks this statement against the data.

**Checks rerun after the revision:**

| Check | Result |
|---|---|
| Report v3 | 5 pages |
| `research/08_tailgating_feature_intervention/code/number_sources.py` | 35/35 |
| `report/check_numbers_v3.py` | 0 unsourced; matched-controls statement matches the data |
| Model card numbers | every per-type count equal to `figures/final_test/data/V7_detection_by_type_frozen_detectors.csv` |
| Links and images | 0 broken of 431 |
| Report ↔ documentation consistency (key values identical; own-set Z-score values always labelled; φ<sub>u</sub>, seed-range and causal statements qualified) | 0 problems; the same check finds 9 problems in the version before this revision |
| `scripts/check_models.py` | all checks passed |

## 8. Final version (2026-10-11)
- **The report.** v3 is the final report, approved with its *Team contributions* paragraph. v2.2 moved to `report/archive/v2_2/`; it still builds to the same text. `report/check_numbers_v3.py` now reads v2.2 from there.
- **The entry pages** now point to v3 as the final report and to the archive for v2.2: `report/README.md`, the front page, `docs/QUESTION_MAP.md`, `docs/REPOSITORY_MAP.md` and `MISSING_BEFORE_SUBMISSION.md`. The last of these is now a submission-status page.
- **The front page** lists the team with the approved contributions, and states in one sentence that software tools were used, including the AI assistant Claude.
- **The clean submission package** was assembled outside the repository (see `MISSING_BEFORE_SUBMISSION.md`).
