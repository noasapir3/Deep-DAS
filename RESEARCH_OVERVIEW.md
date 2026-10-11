# Deep DAS: research overview

> **Note for the repository.** This overview was written for the release folder. Its Z-score numbers refer to the Z-score's own decided set: macro recall 0.231, ROC-AUC 0.783, AUPRC 0.394. On the same vehicles as the deep models (the comparison used in the report and on the front page), the Z-score has 0.232 / 0.819 / 0.496. The φ<sub>u</sub> inference options and study 08 are described in [docs/INFERENCE_MODES.md](docs/INFERENCE_MODES.md).

**Question.** Can a deep model detect anomalous driving behaviour from fused DAS (distributed acoustic sensing), camera and GPS vehicle
tracks without using simulator ground truth at inference, and does it detect more than the hand-crafted Z-score model of the SimStudio
project?

**Answer of this study.** Yes, within the limits of the simulation.

- **The model.** DeepDAS Ensemble v1 is a supervised detector built on a convolutional encoder over 14 track channels and uses no ground truth in inference.
- **Independent test result.** On a test of new scenes that was planned, frozen and run once, it reached:
  - macro recall 0.568 [95 % CI 0.486–0.646] over nine anomaly types;
  - ROC-AUC 0.908 and AUPRC 0.742.
- **Baselines on the same test:**
  - SimStudio Z-score: macro recall 0.231, ROC-AUC 0.783, AUPRC 0.394.
  - Single supervised model: macro recall 0.364, ROC-AUC 0.769, AUPRC 0.515.
- **Significance:** the paired differences are significant (McNemar p < 0.001 against the Z-score, p = 0.0012 against the single model).
- **Costs:** false alarms on legitimate short stops, false alarms on tailgating matched controls, and limited recall on four anomaly types.

---

## 1. Setting

| Element | Description |
|---|---|
| **Simulator** | The SimStudio Final Project simulator (North Tel Aviv road network, 148 map regions), used unchanged. Vehicles follow IDM car-following with stochastic speed profiles. DAS, camera and GPS sensors produce detections that a Kalman tracker fuses into tracks. |
| **Model input** | The exported tracks only: 50 Hz track rows, map-matched to the road network and binned at 0.1 s. Simulator truth is used **only** to label vehicles for training and evaluation. |
| **Unit of decision** | One usable track (≥ 32 valid bins and the label-free quality gate). It is decided if it has ≥ 3 windows of 64 bins. Validation and development tables use the simulated vehicle, meaning all its usable tracks. |
| **Splits** | Spatial. The 148 regions are split into train (98), val_A (21, model selection), val_B (21, thresholds only) and test (8, independent test), with a one-tile buffer between groups. |
| **Anomaly types (9)** | Speeding, Reckless driving, Hard braking, Stalled vehicle, Lane weaving, Lane straddling, Tailgating, Collision, Overweight vehicle (definitions in `docs/ANOMALY_TYPES.md`) |

## 2. Architecture of DeepDAS Ensemble v1

```
track (0.1 s bins, burn-in removed)
  └─ 14 channels: v_lon a_lon d_lat v_lat r_lim g_lead th_lead dv_lead | m_das m_cam m_gps m_lead m_valid | das_amp_log10
     └─ lead-artefact cleaning (merged-track "leader" < 1 m at > 5 m/s removed) → robust normalisation (input_normalization.json)
        └─ windows T = 64 (6.4 s), stride 8
           └─ ConvAE encoder (Conv 14→32 k7 → 32→64 k5 s2 → 64→64 k5 s2 → 64→128 k3 s2 → Linear 1024→8)   per window: 8-d code
              └─ pooling over the unit's windows: mean and std of the codes (16) − standardised on training normals
                 └─ + tailgating fraction (share of bins with headway < 0.8 s at > 5 m/s), standardised
                    └─ MLP head 17 → 16 (GELU) → 1 logit                                                  (67,225 parameters per seed)
5 seeds:   z_k = (logit_k − mean_valB,k) / std_valB,k ;   score = mean_k z_k ;   ALARM if score > 1.9726181803233118 (P99 on val_B)
```

**Encoder initialisation.** The encoder comes from an autoencoder pre-trained to reconstruct normal windows. The 13-channel primary autoencoder of the unsupervised study was extended with the DAS log-amplitude as a 14th input and a 7th masked target.

**Training recipe:**
- **Data.** The normal corpus (train split) plus four simulated anomaly sets: a cross-region set with the nine types, two tailgating sets and a stall/stop set.
- **Batches.** 48 normal + 16 anomalous vehicles per batch, uniform sampling with replacement.
- **Optimisation.** AdamW, BCE loss, 120 steps × up to 25 epochs, early stopping on validation AP.

Details: `docs/ARCHITECTURE.md`, `models/deepdas_ensemble_v1/MODEL_CARD.md`.

## 3. Research path (`research/`)

| # | Study | Hypothesis | Outcome |
|---|---|---|---|
| 01 | Unsupervised autoencoder | A model trained on normal driving only flags anomalies by reconstruction error or latent distance | Responds to severity, but most simulated anomalies lie inside the normal envelope. The latent max-z release detected 5 of 9 original causing vehicles (+ the collision target) at P99. **Its encoder is reused.** |
| 02 | Supervised single model | Few labelled anomalies from other regions teach a head on top of the encoder | Detects more types (7/9 development cases); misses tailgating and lane weaving. Kept as the supervised baseline. |
| 03 | Tailgating data | Tailgating fails for lack of realised, varied examples, not for lack of information | Geometry-verified tailgating sets with matched controls, plus lead-artefact cleaning and the tailgating fraction, make tailgating detectable. |
| 04 | Stall diversity | Diverse stall modes generalise better than more of one mode | The diversity arm was selected → **DeepDAS Ensemble v1** |
| 05 | Legitimate stops | False alarms on 1–10 s stops can be removed by exposure, a temporal branch or checkpoint choice | Not removed without breaking other conditions. **Documented as the main limitation.** |
| 06 | Weak-type diagnosis | The weak types fail in the head, not in the representation | Confirmed: the information is present in the input and in the encoder (probe AUC 0.81–1.0). The head saw 4 examples per type. |
| 07 | Targeted data (W1/W2) | Diverse targeted data for the six weak types improves them without losing other abilities | Weak-type recall about doubled (0.35 → 0.71 / 0.74 on validation), but legitimate-stop false alarms rose beyond the pre-registered limit (C5). **Not adopted.** |

Each folder states the hypothesis, method, results and conclusion (`research/<study>/README.md`).

## 4. Results (kept strictly separate)

### 4.1 Development (used for design and selection; not independent)
The nine development cases, one per anomaly type:
- **DeepDAS Ensemble v1 detects 8 of 9.** Overweight is missed: score 1.865 against the threshold 1.973.
- This is reproduced byte for byte by `scripts/reproduce_development_cases.py`.
- Four of the nine original SimStudio anomaly scenarios did not realise their anomaly (§5). For those types, the development case is a controlled scenario that does.

### 4.2 Validation (used to select the model; not independent)
Validation regions (val_A), with the threshold from val_B:

| Measure | DeepDAS Ensemble v1 |
|---|---|
| Validation AP | 0.781 |
| Macro recall over the 9 types (validation set) | 0.714 |
| Stall validation | 18 / 18 |
| Tailgating validation | 12 / 13 |
| False-alarm rate, val_A normal driving | 1.07 % |
| Legitimate stops (stall validation + legitimate-stop validation set) | 40 / 54 alarms |

Reproduced by `scripts/reproduce_validation.py`.

### 4.3 Independent final test (frozen plan, run once on 2026-10-09)

**Setup.** 228 runs in the 8 test regions:
- new scenes, new background traffic, new simulation seeds, and injection lanes never used before;
- the plan, the models, the thresholds and the evaluation code were all frozen before simulation.

| | DeepDAS Ensemble v1 | Supervised single model | SimStudio Z-score |
|---|---|---|---|
| Macro recall (9 types) | **0.568** [0.486–0.646] | 0.364 | 0.231 |
| Precision / F1 | 0.74 / 0.64 | 0.88 / 0.52 | 0.64 / 0.33 |
| ROC-AUC / AUPRC | **0.908 / 0.742** | 0.769 / 0.515 | 0.783 / 0.394 |
| False alarms, normal driving | 20 / 1008 (2.0 %) | 5 / 1008 | 13 / 1574 |
| False alarms, legitimate stops | 20 / 28 | 14 / 28 | 23 / 32 |

**Per type (DeepDAS Ensemble v1):**
- Stalled vehicle 12/12
- Tailgating 11/11
- Reckless driving 7/10
- Lane weaving 7/11
- Collision 6/10
- Lane straddling 5/12
- Speeding 4/12
- Overweight vehicle 4/12
- Hard braking 1/11

Reproduced byte for byte by `scripts/reproduce_final_test.py`; full tables are in `results/final_test/`.

This test was later used for the weak-type diagnosis (study 06). From then on it counts as exposed data, and it is **not** used as evidence for any later variant.

## 5. Methodological decisions worth knowing
- **No ground truth in inference.** The model sees only exported tracks. The Z-score baseline is used as shipped; 13 of its 156 features use the truth position error. A truth-free sensitivity check gives the same macro recall (0.231).
- **Development cases.** An audit of the SimStudio anomaly scenarios found that the hard-braking vehicle was frozen from the start, the lane-straddling offset was 0 m, and the tailgating vehicle never came closer than 8.9 m. The collision case was re-recorded so that the contact is present in the export. These four types are represented by controlled scenarios that realise the anomaly.
- **Realisation before labelling.** Every injected anomaly is labelled only if its effect is measured in the simulator truth (thresholds in `docs/ANOMALY_TYPES.md`). Runs that do not realise are excluded and reported, never relabelled or re-run.
- **Stop rule.** A stop of ≥ 10.0 s is a stall (anomalous); 1.0 ≤ stop < 10.0 s is a legitimate stop (normal).
- **Thresholds.** Thresholds are taken only from the val_B normals and never copied between models.
- **Pre-registration.** Every comparison round was decided by a rule fixed before training.
- **Hash-guarded freezing.** Frozen artefacts were checked by hash before and after every round.

## 6. Limitations
See `docs/LIMITATIONS.md`. In short:
1. False alarms on legitimate 1–10 s stops (final test 20/28).
2. False alarms on tailgating matched controls that follow a slow leader (7/11 in the final test).
3. Low recall on hard braking, speeding, overweight and lane straddling in the final test.
4. Decision coverage: only 1008 of 1708 normal vehicles of the final test had a decided track.
5. Mild speeding (×1.1–1.25) overlaps the simulator's normal speed envelope.
6. All results are simulation-only.
