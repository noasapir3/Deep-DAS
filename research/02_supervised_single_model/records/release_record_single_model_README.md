# DL_supervised_crossregion_P99

This is a read-only snapshot of the current best **supervised** candidate, `DL/experiments_followup/supervised_crossregion_20260927_200203`, at the **P99** operating point.

**Status: simulation / development result, not independent confirmation.** Every number here comes from simulated scenarios, and the evaluation cases
had been inspected before this model and its thresholds were chosen. The frozen unsupervised release (`DL/release/primary_latent_maxZ_quality_P99`,
backup `DL_primary_latent_maxZ_quality_P99`) remains the frozen reference. Nothing here promotes or replaces it.

**What was preserved:** the model, thresholds, labels and code are exact copies of the original experiment, with no retraining, recalibration,
re-simulation or code fixes. Nothing from the later `tgv1` experiments is included.

## How the model works

1. **Input.** Each tracked vehicle is a sequence of 0.1 s bins with 14 input channels:
   - the 13 DL channels (8 kinematic/leader values + 5 sensor masks, `deepdas.data.channels.CHANNELS`);
   - `das_amp_log10`, the DAS amplitude on the same bins.

   Burn-in bins are dropped. Values are normalized with the fixed robust normalization in `preprocessing/norm.json`.
2. **Usable tracks.** A track is used only if it has ≥ 32 valid bins and passes `quality_ok`, the label-free track-quality gate.
3. **Windows.** Windows are 64 bins (6.4 s) long with a stride of 8 bins, taken over each usable track; a short tail is zero-padded to 64.
4. **Encoder.** A 1-D convolutional encoder maps each window to an 8-number code. It was initialized from the amplitude-only encoder (`preprocessing/amplitude_only_encoder_config.json`) and fine-tuned during supervised training. The decoder is not used.
5. **Vehicle score.** The mean and standard deviation of the codes over **all** windows of the vehicle give 16 features. These are standardized with fixed statistics stored in the checkpoint (`fmu`, `fsd`) and passed through a 16 → 16 → 1 MLP. The output is one **logit per vehicle**.
6. **Decision** (`code/scoring.py: decide`):
   - fewer than 3 windows: **insufficient evidence**. There is no decision, and the vehicle is never counted as normal;
   - otherwise **ALARM** if the logit is above the threshold.

   **P99 is the primary operating point** (logit −5.523461642265342); P95 (−10.89380931854248) is an explicitly labelled alternative.
   Both thresholds are percentiles (numpy linear) of the logits of 1,242 sufficient-evidence **normal val_B** vehicles, frozen before any evaluation (`thresholds/PILOT_FROZEN_THRESHOLDS.json`).

**Training** (`code/original_experiment_scripts/p12_train.py`):
- Loss: binary cross-entropy at vehicle level.
- Each batch is 48 normal + 16 anomalous vehicles, with at most 32 random windows per vehicle.
- Optimizer: AdamW (encoder learning rate 1e-4, head 1e-3), seed 0.
- The checkpoint was selected by validation AP. The best epoch, 24 (AP 0.571, AUC 0.886), was also the last of the 25 allowed epochs.

## Data and regions

- **Region split** (`splits/splits.json`):
  - train: 98 regions;
  - val_A: 21 regions (model selection);
  - val_B: 21 regions (threshold calibration only);
  - test / evaluation: 003, 010, 020, 037, 062, 083, 120, 140.
- **Anomaly training came from regions different from the evaluation regions.** The anomaly runs were 54 new simulations: training in regions 131, 050, 090 and 065, selection in 002 and 005 (`training_manifest/`).
  - 51 of the 54 runs were verified by truth-based realization rules.
  - Tailgating failed in 050, 065 and 005; those runs were excluded.
- **Encoder regions:** the encoder and its normalization came from train and val_A normal regions only (`splits/ENCODER_REGION_CHECK.json`: no evaluation region).
- **Training set:** 34 verified anomalies and 6,169 normals.
- **Selection set:** 17 verified anomalies and 1,415 normals.

## Results against the frozen unsupervised reference (same vehicles; each model at its own frozen thresholds)

| Case | Frozen reference, P99 / P95 | **This release, P99** / P95 |
|---|---|---|
| Original targets detected (of 10) | 6 / 6 | **7** / 8 |
| Weaving (original) | no / no | **no** / yes |
| Overweight (original) | no / no | **yes** / yes |
| C1: real hard braking | no / no | **yes** / yes |
| C2: straddling on the boundary | no / no | **yes** / yes |
| C3: realized tailgating | no / **yes** | **no** / no |
| C4: contact collision (both vehicles); speeding, stall, collision, reckless (original) | all yes | all yes |
| **Normal test false alarms** (513 sufficient, 119 insufficient) | 9 / 35 | **7** / 49 |
| Normal val_A false alarms (1,313 sufficient, 273 insufficient) | 26 / 78 | 17 / 112 |

**Current normal-test result at P99:** 7 false alarms among 513 vehicles with sufficient evidence; 119 vehicles have insufficient evidence.

**Gains and losses against the frozen reference:**
- **Improved detections:** overweight, the real hard braking (C1) and the corrected straddling (C2).
- **Lost:** C3, which the reference caught at P95 only.
- **P95 is worse** than the reference on false alarms (49 against 35 on test), so it is not recommended.

Full tables are in `evaluation/COMPARISON_TABLE.csv` and `evaluation/COMPARE_*.csv`, and the reference tables in `reference_frozen_unsupervised/`.

## Known limitations (see `LIMITATIONS.md`)

- **Mild tailgating is an unresolved weakness.** C3 is missed. In later val_A checks this model also missed all 3 realized mild tailgaters at P99 (1 of 3 at P95), including an isolated, input-visible case.
- **Weaving is not fully stable.** The original weaving is caught only at P95, and 1 of 2 val_A weaving vehicles at P99.
- **Collision alarms are not real-time impact detection.** The score summarizes a vehicle's whole track after the fact. It gives no time of impact and may respond to the stop or behaviour around a collision rather than to the contact itself.
- **Evidence is thin:**
  - at most 4 training anomalies per type, 1 seed and 1 training run;
  - val_A anomalies and normals were used for checkpoint selection;
  - the evaluation cases were known when the choices were made.

## Contents

| Path | What |
|---|---|
| `model/` | Selected checkpoint (`checkpoint_best.pt`: encoder, head and feature standardization), training history and logs |
| `thresholds/` | Frozen P99 / P95 thresholds (byte-exact copy) |
| `preprocessing/` | Normalization, encoder config, and the amplitude-extraction and verification scripts used to build the inputs |
| `code/` | `scoring.py` (verbatim p13 scoring plus the quality rule), `amp_lib.py`, pinned `src/deepdas`, the original experiment scripts p10–p13, and `build_release.py` |
| `splits/` | Region split manifest and encoder region check |
| `training_manifest/` | Anomaly-run manifest and labels (`batch_vehicles.csv`, `vehicles_used.csv`, `realization_summary.csv`, `SIM_BATCH_MANIFEST.csv`), plan, base scenarios and geometry |
| `evaluation/` | Stored vehicle scores and comparison tables, and the experiment report |
| `reference_frozen_unsupervised/` | Frozen unsupervised release tables and config (the comparison basis) |
| `reproduction_inputs/` | Exact input rows for every stored vehicle score (val_A, val_B, test, the 9 originals, and the controlled batch C1–C4 / K1–K3) |
| `RELEASE_MANIFEST.json`, `MANIFEST_SHA256.txt` | sha256 and source path of every file, and the hash of the manifest itself |
| `verify_release.py` | Integrity check plus score reproduction |

**Not included** (referenced by path and hash only): the full training corpus (`DL/data/processed`) and the raw simulations
(`anomaly_model/simulations/שיפור סימולציות עבור dl/batch_crossregion_20260927/`). They are needed only to re-run training, which this release does not do.

## Verify

Requirements: python ≥ 3.10, torch, numpy, pandas, pyarrow.

```
python3 verify_release.py                  # hashes + rescoring of every stored vehicle
python3 verify_release.py --dl ../DL       # also confirms the DL/ sources still match the copies
python3 verify_release.py --out /tmp/x     # also writes the rescored tables (outside the release)
```

The script writes nothing inside the release. It passes only if:
- all hashes match and no files were added;
- every stored logit reproduces within 1e-4, with identical window counts and decisions;
- the P99 / P95 thresholds recomputed from val_B match the frozen values;
- the normal-test counts are 513 / 119 / 7.
