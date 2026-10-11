# Question map: from the report to the evidence

This page connects every part of the report to the folder that documents it, the figure that shows it and the files that hold the numbers. Use it to check a claim of the report, or to find where an experiment is described in full.

- **Report v3** is the final report ([PDF](../report/DeepDAS_report_v3.pdf)).
- **Report v2.2** is the previous version, archived ([PDF](../report/archive/v2_2/DeepDAS_report_v2_2.pdf)).
- **Data role** follows [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md):
  - **development:** used for design;
  - **validation:** used for selection;
  - **final test:** independent only for the three detectors frozen before it;
  - **post hoc:** defined after the final test was exposed.

## The four research questions

| Research question | Report v3 | Report v2.2 | Study folders | Main figure | Evidence files (data role) |
|---|---|---|---|---|---|
| **Q1. Why was reconstruction error not enough, and why did we move to supervised learning?** | §5, Q1 | §5.1, §5.2 | [01 autoencoder](../research/01_unsupervised_autoencoder/), [02 supervised single model](../research/02_supervised_single_model/) | [S2 / S1](../figures/README.md#7-archived-stage-1-autoencoder) | `research/01_unsupervised_autoencoder/records/unsup_REPORT.md` and `unsup_release_primary_latent_maxZ_quality_P99_RELEASE_REPORT.md` (development, archived values); `research/02_supervised_single_model/records/crossregion_AD_20261007_SUMMARY.md` (development / validation) |
| **Q2. How did training stability and data diversity affect detection?** | §5, Q2; Fig. 2a, 2b | §5.3, §5.4 | [02](../research/02_supervised_single_model/) (seed study), [03 tailgating data](../research/03_tailgating_data/), [04 stop-mode diversity](../research/04_stall_diversity/) | [S02](../figures/README.md#2-development-data), [V5 a–b](../figures/README.md#3-validation-data) | `research/02_supervised_single_model/records/seedstudy_20261007_SEED_SUMMARY.md` (development); `research/03_tailgating_data/records/C_tailgating_data_20261008_RESULTS.md`, `…_per_seed_and_ensemble.csv` (validation); `research/04_stall_diversity/records/exp12_balanced_vs_frozen_20261008_RESULTS_EXP12.md`, `stall_diversity_20261008_RESULTS_STALL.md`, `…_per_seed_and_ensemble.csv` (validation) |
| **Q3. Why do more detections not necessarily justify replacing the model?** | §5, Q3; Fig. 2c | §5.5, §6 (W1/W2), §7 | [05 legitimate stops](../research/05_legitimate_stops/), [06 weak-type diagnosis](../research/06_weak_type_diagnosis/) (post hoc), [07 targeted data](../research/07_targeted_data_study/) (after the test was exposed) | [S05](../figures/README.md#3-validation-data), [S06](../figures/README.md#5-post-hoc-analyses), [V5 c](../figures/README.md#3-validation-data) | `research/05_legitimate_stops/records/legit_stop_diag_20261009_README.md`, `stop_discrimination_20261009_RESULTS_F.md`, `ckpt_selection_20261009_RESULTS_CKPT.md` (validation); `research/06_weak_type_diagnosis/records/weak_types_diag_20261010_results_information_flow_by_type.csv` (exposed final test, post hoc); `research/07_targeted_data_study/results/metrics_all.csv`, `SELECTION_W.json` (validation, after the test was exposed) |
| **Q4. What does the tailgating summary φ<sub>u</sub> add beyond the learned representation?** | §6.3; Table 3 | §4, §5.4, §7, Table 3 | [08 tailgating-feature check](../research/08_tailgating_feature_intervention/); history in [02](../research/02_supervised_single_model/) and [03](../research/03_tailgating_data/) | [G1, G2](../figures/README.md#5-post-hoc-analyses) | `research/08_tailgating_feature_intervention/tables/report_summary_table.csv`, `development_cases_comparison.csv` (development), `validation_comparison.csv` (validation), `final_test_comparison_POST_HOC.csv` (post hoc); `results_phi_neutralized/` |

## The other parts of the report

| Part | Report v3 | Folder | Figure | Evidence files |
|---|---|---|---|---|
| Problem and previous project | §1 | [docs/BACKGROUND_SIMSTUDIO.md](BACKGROUND_SIMSTUDIO.md) | SimStudio images | `data/simstudio_inputs/` (data package) |
| Data and evaluation | §3 | [DATA_PIPELINE.md](DATA_PIPELINE.md), [ANOMALY_TYPES.md](ANOMALY_TYPES.md), [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md), [LEAKAGE_AUDIT.md](LEAKAGE_AUDIT.md) | – | `data/` and the data package |
| The model and its two inference options | §4; Fig. 1; Table 1 | [current model card](../models/MODEL_CARD_DeepDAS_Ensemble_v1.md), [models/deepdas_ensemble_v1/](../models/deepdas_ensemble_v1/), [ARCHITECTURE.md](ARCHITECTURE.md), [INFERENCE_MODES.md](INFERENCE_MODES.md) | [G0](../figures/README.md#1-architecture) | `models/deepdas_ensemble_v1/ensemble.json`, `figures/architecture/data/G0_architecture_two_options_spec.json` |
| **Final test** (independent for the three frozen detectors) | §6.1; Table 2; Figs. 3–4 | [results/final_test/](../results/final_test/) | [V6, V7, V3](../figures/README.md#4-final-test) | `results/final_test/T1_per_type.csv`, `T2_overall.csv`, `T3_stops_tailgating.csv`, `T5_uncertainty.json`, `vehicle_outcomes.csv`; plan: `results/final_test/records/PLAN_FINAL_TEST_FROZEN.md` |
| Development and validation | §6.2 | [results/development/](../results/development/), [results/validation/](../results/validation/) | – | `results/development/development_cases.csv`, `results/validation/validation_metrics.csv` |
| Real recording (Klausner Street) | §7 | [Klausner experiment](../models/deepdas_ensemble_v1/experiment_klausner/) | [Klausner figure](../figures/README.md#6-real-data-observation) | `klausner_scores.csv`, `klausner_das_speed.csv`, `klausner_track_summary.json` |
| Limitations | §7 | [LIMITATIONS.md](LIMITATIONS.md) | – | as above |
| Code and reproducibility | Code and reproducibility | [INSTALL.md](../INSTALL.md), [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | – | `scripts/verify_phi_modes.py`, `scripts/check_models.py` |

## Numbers
- **Report v2.2.** Every number is traced in [REPORT_V2_NUMBER_SOURCES.md](../research/08_tailgating_feature_intervention/tables/REPORT_V2_NUMBER_SOURCES.md), which `code/number_sources.py` checks.
- **Report v3.** It reuses those numbers, and [report/check_numbers_v3.py](../report/check_numbers_v3.py) checks the six numbers that are new in v3 against their source files.
- **Names.** The study codes that appear in the records (W1, C5, EXP1, …) are mapped to the names used in the report in [IDENTIFIERS.md](IDENTIFIERS.md#experiments-and-conditions-display-names).
