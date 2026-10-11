# Archive index: what is in the release, and what stays in the original project

The release is a clean, self-contained selection.

- **Nothing was moved or deleted from the original project.** All original folders remain unchanged, and they are the authoritative archive.
- **Paths below are relative to the project folder `final proj/`.**
- **Every release file that was copied from the archive is listed with its source and sha256** in `docs/BUILD_PROVENANCE.tsv`.

## In the release (copied, renamed where noted)

| Release | Source in the archive |
|---|---|
| `models/deepdas_ensemble_v1/` | `DL/FROZEN_FINAL_MODEL__E_stall_diversity_ENSEMBLE__20261010/mirror/experiments_followup/stall_diversity_20261008/runs/DIV_s0…s4/model/` and the frozen candidate record |
| `models/deepdas_ensemble_v1/autoencoder_*`, `input_normalization.json` | `…/mirror/experiments_followup/amplitude_only_20260927_203146/` |
| `models/baselines/supervised_single_p99/` | `DL_supervised_crossregion_P99/` (= `…/supervised_crossregion_20260927_200203/model/`) |
| `models/baselines/zscore/` | `anomaly_model/outputs/model_live.*` (identical copy in `DL/data/fp/zscore/`) |
| `data/processed/normal_corpus/` | `DL/data/processed/` and the consolidated amplitude of `amplitude_only_20260927_203146` |
| `data/processed/development_cases/` | the original anomaly scenarios (`DL/data/fp/anomalies`) and the controlled runs `amplitude_only_…/data/r1_runs/` |
| `data/processed/anomaly_sets/*` | `batch_processed*` and `batch_vehicles*.csv` of `supervised_crossregion_…`, `tailgating_dataset_20261007` (v3, v4), `stall_diversity_20261008`, `stop_discrimination_20261009`, `newscene_test_20261007` |
| `data/processed/final_test/` | `…/final_holdout_20261009/data/` (processed runs, vehicles, plan, realisation records, Z-score scores) |
| `data/scenarios/north_tlv/`, `data/simstudio_inputs/` | `DL/data/fp/` (SimStudio regions, anomaly scenarios, Klausner bus, Z-score features) |
| `data/simulation_plans/` | the `RUN_PLAN*`, survey and plan-check files of each batch |
| `research/07_targeted_data_study/` | `DL/experiments_followup/weak_types_W1W2_20261010/` (processed data, models, evaluation) |
| `research/*/records/` | the plan, results and selection files of each study (`DL/experiments_followup/*`, `DL/docs/`, `DL/release/`) |
| `src/deepdas/` | `DL/src/deepdas/` plus the experiment scripts, refactored into modules without logic changes. Equivalence: `RELEASE_VERIFICATION.md` |

## Not in the release (kept in the archive)

| What | Where | Why not in the release |
|---|---|---|
| Raw simulator output of every DL batch | `…/sims*/` folders of each study; final test in `…/mirror/experiments_followup/final_holdout_20261009/sims_final/` | size: the separate raw-data package (`RAW_DATA_PACKAGE.md`) |
| SimStudio normal-traffic exports (source of the normal corpus) | `maps/north_TLV/…` | size, about 11 GB; referenced with hashes in the raw-data package |
| Tailgating pilot simulations v1 and v2 | `…/tailgating_dataset_20261007/sims`, `sims_v2` | superseded plan versions; their plans are in `data/simulation_plans/tailgating_routes/earlier_plans/` |
| Feasibility dry runs of study 07 | `…/weak_types_W1W2_20261010/sims_feasibility_DISCARDED/` | never imported or used |
| Aborted launches | `…/aborted_launch_*` | never evaluated; documented in the records |
| Models of the arms that were not adopted (C_old, C_new, E_quantity, EXP1/2, F1/F2, A–D, seed study) | `DL_supervised_crossregion_P99__*` folders and `…/runs/` of each study | not needed to reproduce any reported result of the final model; their results are in the records |
| Training and evaluation logs | `logs_*.log` of each study | operational detail |
| Exploratory diagnostics without a reported conclusion | `DL/diagnostics/*` (older ones) | not part of the research path |
| Frozen-model backup and restore tools | `DL/FROZEN_FINAL_MODEL__E_stall_diversity_ENSEMBLE__20261010/` | the archive's own backup; this release is built from it |
| Model-phase closing summary | `DL/MODEL_PHASE_CLOSING_20261010/` | its content is in `RESEARCH_OVERVIEW.md` and the study READMEs |
| Earlier deliverables and report | `DL/docs/REPORT.*`, `submissions/` | the unsupervised report is in `research/01_unsupervised_autoencoder/records/` |
| Transfer folders | `DL/_transfer_*`, `_release_build_staging/` | temporary; kept unchanged, as instructed |
