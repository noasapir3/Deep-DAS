# Identifiers: internal names and release names

**Two kinds of names.**
- **Documentation and display reports** use professional names: DeepDAS Ensemble v1, Supervised single model, SimStudio Z-score, and the set names below.
- **Data files, score tables and archived records** keep the identifiers of the study. Renaming them would change file contents and break the byte-for-byte reproduction against the original outputs. They are kept so that every number can be traced and reproduced.

## Models

| Release name | Identifier in score tables, records and code | Location |
|---|---|---|
| **DeepDAS Ensemble v1** | `E_DIV`; `E_stall_diversity_ENSEMBLE`, `E_diversity`, `E_div`, `DIV_s0…s4` (seed runs); `REF` in studies 05 and 07 | `models/deepdas_ensemble_v1/` |
| Supervised single model | `CR_P99`; `BASELINE_frozen`, `DL_supervised_crossregion_P99`, `crossregion P99` | `models/baselines/supervised_single_p99/` |
| SimStudio Z-score | `Z`; `model_live` | `models/baselines/zscore/` |
| Amplitude-aware autoencoder (initialisation) | `amplitude_only` | `models/deepdas_ensemble_v1/autoencoder_*` |
| Primary 13-channel autoencoder | `cae_primary`; release `primary_latent_maxZ_quality_P99` (latent max-z detector) | `research/01_unsupervised_autoencoder/runs/cae_primary/` |
| Ensemble of study 03 | `C_new`, `C_new_ENSEMBLE` (and `C_old`, its arm without new data) | not shipped (records only) |
| Arms of studies 04, 05, 07 | `E_quantity`, `EXP1`, `EXP2`, `F1`, `F2`, `REF_J`, `F1_J`, `F2_J`, `W1`, `W2` | W1/W2 in `research/07_targeted_data_study/models/`; the others in records only |

**Score-table columns.**
- `E_DIV_s<k>_logit`: the seed logit.
- `E_DIV_ens`: the ensemble score.
- `E_DIV_decision`: ALARM / no alarm / no decision.
- `CR_P99_logit`, `CR_P99_decision`.
- `Z_score`, `Z_alarm`.
- `*_orac_*`: vehicle-unit (oracle-grouped) variant.
- `Z_noGT_*`: Z-score without its truth-derived features (sensitivity check).

## Data sets

| Release name | Release folder | Internal names | Run-id prefix |
|---|---|---|---|
| Normal driving corpus | `data/processed/normal_corpus/` | `fp_normal` (group); splits `train`, `val_A`, `val_B`, `test` | region/scenario keys |
| Development cases (DC-01…DC-09) | `data/processed/development_cases/` | `valid9`, "the 9 valid cases"; controlled C1–C4 (`C1_hard_braking`, `C2_lane_straddling`, `C3_tailgating`, `C4_collision_contact_REUSED`) | original track keys |
| Cross-region anomaly set | `anomaly_sets/cross_region/` | `p10` batch, "crossregion" | `r<region>_<type>` |
| Tailgating set on planned routes | `anomaly_sets/tailgating_routes/` | `tg3`, tailgating v3 | `tg3_` |
| Tailgating set with matched controls | `anomaly_sets/tailgating_matched_controls/` | `tg4`, tailgating v4 | `tg4_` |
| Stall and stop set | `anomaly_sets/stalls_and_stops/` | `ST01`, `stall_div`, `STALLARM`; validation part = `stallval` | `st_tr_` (train), `st_st_` (stallval) |
| Legitimate-stop validation set | `anomaly_sets/legitimate_stop_validation/` | `stopval2`, `sd01` | `sv2_` |
| New-scene check | `anomaly_sets/new_scene_check/` | `newscene` | `r<region>_<type>` |
| Final test | `data/processed/final_test/` | `final_holdout`, `H`, `h01`–`h05` | `ft_` |
| Targeted-data study | `research/07_targeted_data_study/data/` | `W`, `WX`, `w03` | `w_tr_`, `w_va_` |

**Validation group names in the score tables** (`results/validation/`):

| Group | Meaning |
|---|---|
| `validation` | original validation: val_A normals + backgrounds + anomalies |
| `normal_val` | val_A corpus normals |
| `normal_val_B` | threshold set |
| `normal_test` | test-split corpus normals (development false alarms) |
| `stallval` | stall validation |
| `newscene_anomaly`, `newscene_background` | new-scene check |
| `valid9` | development cases |

## Development cases

| Case | Display name | Internal label | Unit key |
|---|---|---|---|
| DC-01 | Speeding | `speeding` | `region_003/anomaly_speeding\|veh001` |
| DC-02 | Hard braking | `C1 real braking` | `C1_hard_braking\|veh001` |
| DC-03 | Stalled vehicle | `stalled` | `region_020/anomaly_stalled_vehicle\|veh001` |
| DC-04 | Collision | `C4 collision striker` | `C4_collision_contact_REUSED\|veh001` |
| DC-05 | Reckless driving | `reckless` | `region_120/anomaly_reckless_driver\|veh001` |
| DC-06 | Lane weaving | `weaving` | `region_037/anomaly_lane_weaving\|veh001` |
| DC-07 | Overweight vehicle | `overweight` | `region_140/anomaly_overweight_vehicle\|veh001` |
| DC-08 | Lane straddling | `C2 straddling` | `C2_lane_straddling\|veh001` |
| DC-09 | Tailgating | `C3 tailgating` | `C3_tailgating\|veh001` |

The suffix `_REUSED` in the collision case is part of its original track key. It is kept because every stored score refers to it. It means that the contact-collision recording of study 01 was reused unchanged.

## Code: original script names

| Release module | Original script(s) |
|---|---|
| `deepdas.amplitude` | `amp_lib.py`, `a02` |
| `deepdas.train.supervised` | `x_train_eval_s.py` (and its predecessors `p12_train.py`, `x_train_eval_c.py`) |
| `deepdas.ensemble.scoring` | `hs_scorer.py` |
| `deepdas.evaluation.final_test` | `h04` (scoring), `h05` (analysis) |
| `deepdas.evaluation.validation` | `x_analyze_s.py` |
| `deepdas.evaluation.development` | `a03` |
| `deepdas.batches.cross_region` | `p10` |
| `deepdas.batches.tailgating_routes` | `tg12` |
| `deepdas.batches.tailgating_controls` | `tg14` |
| `deepdas.batches.stalls` | `st01` |
| `deepdas.batches.stop_study` | `sd01` |
| `deepdas.batches.final_test` | `h01` |
| `deepdas.batches.targeted_data` | `w03` |
| `deepdas.importing` | `p11` / `h03` / `w05` import step |
| `research/07_targeted_data_study/code/evaluate_selection_rule.py` | `w07_evaluate.py` |
