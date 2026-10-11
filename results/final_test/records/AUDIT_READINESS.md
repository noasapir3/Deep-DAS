# Readiness audit of the three detectors (2026-10-09, before any holdout data exist)

## 1. Frozen artefacts (sha256 verified)
| detector | artefact | status |
|---|---|---|
| Z-score (final project) | `model_live.pkl` 0acc8803… (156 features, threshold 15.392 σ, n_train 13,333) | identical in `anomaly_model/outputs`, `DL/data/fp/zscore` and `DL_8_out_of_10`. It reproduces the published anomaly scores to 5e-4. |
| DL_supervised_crossregion_P99 | `checkpoint_best.pt` aa517c02…, `norm.json` fc357414…, P99 = −5.523461642265342 | the release folder and the container copy are identical, and the release fingerprint list (130 files) is unchanged |
| E_stall_diversity_ENSEMBLE | 5 checkpoints and thresholds as in `FROZEN_CANDIDATE_E_diversity_sha256.txt`; ensemble P99 = 1.9726181803233118 | verified on the device and in the container |

## 2. Inference code
- `code/hs_scorer.py` is the single frozen inference module for both DL detectors.
  - normalisation: `norm.json`;
  - CR_P99: no cleaning, no extra feature;
  - E_DIV: cleaning, tg feature, 5 seeds, per-seed val_B z-constants from the freeze file;
  - decision: < 3 windows = no decision.
- **Reproduction (`audit/a01_*`):** with the earlier vehicle units, the scorer reproduces the saved development logits.
  - CR_P99: max |Δ| 7e-15 over 3,123 vehicles.
  - E_DIV seeds: ≤ 1.5e-5.
  - E_DIV ensemble: 8e-7.

## 3. Ground truth in inference — findings
- **DL channels: no ground truth.**
  - `import_exports` keeps the oracle id only in metadata.
  - The leader channels come from other *tracks*.
  - The quality gate and burn-in are label-free.
- **DL vehicle grouping used ground truth.** Until now, the tracks of a vehicle were grouped by `vehicle_id_oracle` before scoring.
  - **For the holdout, the unit of decision is one usable track.** No oracle merge; the frozen models and thresholds are unchanged.
  - In development, only 48 of 3,123 val_A/val_B vehicles have more than one usable track.
  - At the frozen thresholds, the false-alarm rate changes by ≤ 0.08 percentage points (val_B: CR 1.05 → 1.11 %, E_DIV 1.05 → 1.03 %).
  - **Ground truth is used only in evaluation:** it maps track decisions to the true vehicle (a vehicle counts as detected / false-alarmed if any of its tracks alarms).
- **The Z-score uses ground truth in 13 of its 156 features.**
  - These are the position-error features `kf_pos_err_*`, `dis_*_err_mean_m` and `cov_err_*`, all computed from |true − Kalman| position.
  - In development they almost never decide:
    - they are the top feature in 0.34 % of normal tracks and 0 of 36 anomaly tracks;
    - the flag rate on normals is 1.48 % with them and 1.42 % without them;
    - the anomaly flags are identical.
  - The shipped model cannot be made ground-truth-free without changing it (decision requested from Noa).
- **Florentin:**
  - it does not appear in any training, normalisation, calibration or selection data, or in the Z-score training rows;
  - the only mentions are plans and a design document marking it "not used".

## 4. Sensor specification difference (Florentin tiles)
- **DAS:** identical (noise 30, 30 Hz).
- **Cameras:** identical (80 m, 70°, 10 Hz) in `tile_*_sensors.sim.json`.
- **GPS:**
  - North-TLV: σ 4.0 m, 1 Hz, radius 25 m.
  - Florentin `*_sensors.sim.json`: σ 2.5 m, 5 Hz, radius 25 m.
  - The `*_sensors.sim.sim.json` variant differs further (cameras 200 m, GPS 35 m). It is not used.
