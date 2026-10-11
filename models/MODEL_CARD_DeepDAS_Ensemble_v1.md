# Model card: DeepDAS Ensemble v1 (current, 2026-10-11)

**This is the current model card.** The original card, [`deepdas_ensemble_v1/MODEL_CARD.md`](deepdas_ensemble_v1/MODEL_CARD.md), was written when the model was frozen. It is part of the hash-checked model folder (`scripts/verify_phi_modes.py`, P1), so it stays unchanged.

**What this card adds:**
- every detector comparison on one population, the common decided subset;
- the two inference options;
- the Klausner experiment on the real recording.

**The model is the same.** Its weights, thresholds, normalisation and results are those of the frozen folder. Nothing was retrained or re-calibrated for this card.

| | |
|---|---|
| **Task** | Binary anomaly detection for one tracked vehicle, from fused DAS + camera + GPS tracks. The output is one score and one decision (anomalous / normal); the anomaly type is **not** predicted. |
| **Output** | Ensemble score s<sub>u</sub> per unit (one usable track). ALARM if s<sub>u</sub> > τ = 1.9726181803233118; no decision with fewer than 3 windows. |
| **Architecture** | A 1-D convolutional encoder over 14 track channels, mean/std pooling of the window codes (16 learned features) plus one physical summary, the tailgating fraction φ<sub>u</sub> (17th head input), and an MLP head 17 → 16 → 1. Five seeds, standardised with frozen per-seed statistics and averaged. 67,225 trained parameters per seed. Details: [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md). |
| **Identifier** | `E_stall_diversity_ENSEMBLE` (`E_DIV`). Selected 2026-10-08 ([study 04](../research/04_stall_diversity/)) and frozen 2026-10-09, before the final test. |
| **Status** | Final. No further training, recalibration or testing. |

## Intended use
- Research on anomaly detection from DAS-based vehicle tracks, in the SimStudio simulation setting.
- Ranking and flagging tracks for review.
- **It is not a validated real-world system.** It was trained and tested on simulation; one real recording was scored qualitatively (below).

## Inputs
- **Channels.** 14 channels per 0.1 s bin from the exported tracker output: kinematics in the lane frame, leader gap and headway, sensor-presence masks, and the DAS log-amplitude.
- **Truth.** No simulator ground truth is used at inference.
- **Usable track.** A track must have ≥ 32 valid bins and pass the label-free quality gate.

## Two inference options of the same model

| Option | 17th head input | Status |
|---|---|---|
| `--phi-mode active` (default) | standardised φ<sub>u</sub> | the model as trained, selected and tested; **every result below except the last table belongs to it** |
| `--phi-mode neutralized` | 0 | the same weights with φ<sub>u</sub> held at zero; a dependence check defined after the final test was exposed, not a model trained without φ<sub>u</sub> |

Full explanation: [docs/INFERENCE_MODES.md](../docs/INFERENCE_MODES.md).

## Training data
- **Normals:** the North-TLV normal corpus (training regions), unaffected background vehicles, matched controls and legitimate stops of the simulated sets.
- **Anomalies:** simulated anomalies of 9 types in training regions: the cross-region set, two tailgating sets, and stalls in three stop modes.
- **After the ≥ 3-window filter:** 6,761 normal and 105 anomalous vehicles.
- **Labels:** every label is verified on the simulator truth ([docs/ANOMALY_TYPES.md](../docs/ANOMALY_TYPES.md)).

## Evaluation
The data roles are kept apart ([docs/EVALUATION_PROTOCOL.md](../docs/EVALUATION_PROTOCOL.md)).

**Development and validation** (used for design and selection, so not independent):

| Data | Result |
|---|---|
| Development (9 cases, one per type) | 8 of 9 detected; overweight missed (score 1.865) |
| Validation | macro recall 0.714, AP 0.781, stalls 18/18, tailgating 12/13, false alarms on selection-region normals 1.07 %, legitimate stops 40/54 |

**Final test** (228 new runs in 8 held-out regions, plan frozen before simulation, run once).
- **Who is compared.** The three detectors frozen before the test, all on the **common decided subset**: 101 anomalies and 1,008 normal vehicles.
- **Sources.** `results/final_test/T1_per_type.csv`, `T2_overall.csv` and `T5_uncertainty.json`; the Z-score's precision and AUCs on this subset come from `figures/final_test/data/V6_roc_pr_frozen_detectors.csv`.

| | DeepDAS Ensemble v1 | Supervised single model | SimStudio Z-score |
|---|---|---|---|
| Macro recall, 9 types | **0.568** [0.49, 0.65] | 0.364 [0.30, 0.43] | 0.232 [0.22, 0.26] |
| ROC-AUC / AUPRC | **0.908 / 0.742** | 0.769 / 0.515 | 0.819 / 0.496 |
| Precision / F1 | 0.74 / 0.64 | 0.88 / 0.52 | 0.77 / 0.35 |
| False alarms, normal driving | 20 / 1,008 (2.0 %) | 5 / 1,008 (0.5 %) | 7 / 1,008 (0.7 %) |
| False alarms, legitimate stops < 10 s | 20 / 28 | 14 / 28 | 22 / 28 |
| False alarms, tailgating matched controls | 7 / 11 | 0 / 11 | 0 / 11 |

**Reading the table.**
- **Intervals.** Brackets are 95 % bootstrap intervals over runs.
- **Matched controls.** That row is on the 11 matched controls decided by all three detectors (`vehicle_outcomes.csv`: block `tg_control`, label `normal_matched_control`, `common`). `T3_stops_tailgating.csv` lists the Z-score on its own 12 decided controls (0/12).
- **Paired tests.** Exact McNemar tests on the pooled anomalies: 38 vs 4 discordant detections against the Z-score (p < 0.001), and 28 vs 8 against the single model (p = 0.0012).
- **The Z-score's own decided set** (105 anomalies, 1,574 normals) gives macro recall 0.231, ROC-AUC 0.783 and AUPRC 0.394. Those values are **not** comparable column by column with this table. The frozen card's comparison table quotes them next to the ensemble; this card does not.

**Recall per anomaly type** (final test, common decided subset):

| Type | DeepDAS Ensemble v1 | Single model | Z-score |
|---|---|---|---|
| Stalled vehicle | 12/12 | 12/12 | 12/12 |
| Tailgating | **11/11** | 0/11 | 1/11 |
| Reckless driving | **7/10** | 1/10 | 0/10 |
| Lane weaving | **7/11** | 3/11 | 0/11 |
| Collision | 6/10 | 9/10 | **10/10** |
| Lane straddling | 5/12 | **8/12** | 0/12 |
| Speeding | **4/12** | 1/12 | 0/12 |
| Overweight | **4/12** | 3/12 | 0/12 |
| Hard braking | 1/11 | 0/11 | 0/11 |

**Decision coverage.** The deep models decide 1,008 of 1,708 normal vehicles; the rest are 134 untracked, 320 without a usable track and 246 with < 3 windows. Undecided vehicles are never counted as normal.

**The neutralized option (post hoc; same weights, φ<sub>u</sub> held at 0):**

| Check | `active` | `neutralized` |
|---|---|---|
| Development cases | 8/9 | 7/9 |
| Tailgating, validation | 12/13 | 9/13 |
| Tailgating, exposed final test (post hoc) | 11/11 | 10/11 |
| Macro recall, exposed final test (post hoc) | 0.568 | 0.560 |

Source: [study 08](../research/08_tailgating_feature_intervention/).

## The one real recording: Klausner Street
- **The recording.** A real DAS channel on an existing telecom fibre, and a camera, recorded a 12 t city bus on Klausner Street. It drives in, stands, and pulls away.
- **The label.** There is no simulator truth, so the project's stall rule (a stop ≥ 10 s) was applied to the bus speed measured by DAS. It stays below 2 m/s for 31.4 s and below 0.5 m/s for 23.6 s without interruption, so the recording is a **stalled vehicle**.
- **Processing.** The track was imported with the unchanged pipeline and scored by inference only, with the frozen checkpoints, statistics and τ.

| Detector | Decision | Score |
|---|---|---|
| **DeepDAS Ensemble v1, `active`** | **detected** | s<sub>u</sub> = 4.53 > τ = 1.973; all five seeds alarm (per-seed z 3.3–6.0) |
| **DeepDAS Ensemble v1, `neutralized`** | **detected** | s<sub>u</sub> = 4.55 (φ<sub>u</sub> = 0 for this bus) |
| Supervised single model | detected | logit 18.8 > −5.52 |
| SimStudio Z-score | detected | max \|z\| = 27.5 > 15.39 |
| Stage-1 autoencoder (study 01, archived) | missed | track score 2.15 < 5.20 |

**What this result is and is not.**
- **One case.** It is not a recall or false-alarm estimate on real data.
- **The label** comes from applying the simulation's rule to the measured speed.
- **Noise.** The real tracked kinematics are much noisier than the simulated ones. With one track, the share of the score due to the stop cannot be separated from the share due to that noise.

Details and reproduction: [experiment_klausner/](deepdas_ensemble_v1/experiment_klausner/).

## Known limitations
See [docs/LIMITATIONS.md](../docs/LIMITATIONS.md).
1. **False alarms on legitimate 1–10 s stops** (20/28 in the final test).
   - **Measured:** the alarms depend on the stop windows; training saw about 105× fewer legitimate-stop windows than stall windows; the tracker shortens stops.
   - **Supported explanation** (not proven as the only cause): under mean/std pooling, the share of stop windows drives the score.
2. **False alarms on tailgating matched controls** (7/11). Within each pair the tailgater scores higher.
3. **Low recall on hard braking, speeding, overweight and straddling.**
   - **Measured** on the exposed test (post hoc): probes on the input and on the pooled code separate these types better than the final score does, and the head saw four examples per type.
   - **What this points to:** the head and its training data, without isolating one cause.
4. **Coverage.** Decision coverage is limited by the tracker and the ≥ 3-window rule.
5. **Real data.** Simulation only, except the one real recording above.
6. **φ<sub>u</sub> versus the data.** The separate contributions of φ<sub>u</sub> and of the tailgating data are not identified: no model without φ<sub>u</sub> was trained on the final training data.

## Files and reproduction
- **Files:** the checkpoints, thresholds and normalisation are in [`deepdas_ensemble_v1/`](deepdas_ensemble_v1/) (file table in [models/README.md](README.md)).
- **Checks without data:** `python scripts/check_models.py` (hashes, architecture, both options).
- **Reproducing the results:** `python scripts/verify_phi_modes.py` with the data package ([INSTALL.md](../INSTALL.md)). Scores and tables reproduce exactly on Linux x86-64; on aarch64, decisions and tables are identical ([docs/CROSS_PLATFORM.md](../docs/CROSS_PLATFORM.md)).
