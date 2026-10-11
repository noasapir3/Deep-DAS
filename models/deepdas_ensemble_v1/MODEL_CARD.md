# Model card: DeepDAS Ensemble v1

| | |
|---|---|
| **Task** | Flag anomalous driving behaviour of one tracked vehicle from fused DAS + camera + GPS tracks |
| **Output** | Ensemble score per unit (one usable track); ALARM if the score > 1.9726181803233118; no decision with fewer than 3 windows |
| **Type** | 5-seed ensemble of a supervised 1-D convolutional encoder with mean/std pooling and an MLP head (67,225 parameters per seed) |
| **Development identifier** | `E_stall_diversity_ENSEMBLE` (`E_DIV`), selected 2026-10-08, frozen before the final test of 2026-10-09 |
| **Status** | Final. No further training, recalibration or testing |

## Intended use
- Research on anomaly detection from DAS-based vehicle tracks, in the SimStudio simulation setting.
- It ranks and flags tracks for review.
- **It is not a validated real-world system.** No real-world data was used for training or testing.

## Inputs
- 14 channels per 0.1 s bin from the exported tracker output: kinematics in the lane frame, leader gap and headway, sensor-presence masks, and the DAS log-amplitude. See `docs/ARCHITECTURE.md`.
- **No simulator ground truth is used at inference.**
- A track must have ≥ 32 valid bins and pass the label-free quality gate.

## Files

| File | Content |
|---|---|
| `ensemble.json` | the seeds, their checkpoints, the frozen val_B mean / std per seed, and the decision threshold |
| `seed_<k>/checkpoint.pt` | encoder + head + pooled-feature (fmu, fsd) and tailgating-feature (emu, esd) standardisation |
| `seed_<k>/thresholds.json` | per-seed val_B percentiles (reference only; the release decision is the ensemble threshold) |
| `seed_<k>/training_history.csv` | loss, validation AP and AUC per epoch |
| `input_normalization.json` | fixed robust normalisation of the 14 channels |
| `autoencoder_init.pt`, `autoencoder_config.json`, `autoencoder_training_history.csv` | the encoder initialisation |
| `freeze_record.json` | the record written when the model was frozen as the candidate (original identifiers) |

**Use from Python:**
```python
from deepdas.ensemble.scoring import Ensemble, score_units
```
Or run `scripts/score_tracks.py` on imported runs.

## Training data
- **Normals:** the North-TLV normal corpus (train regions), unaffected background vehicles, matched controls and legitimate stops of the simulated sets.
- **Anomalies:** simulated anomalies of 9 types in train regions — the cross-region set, two tailgating sets, and stalls in three stop modes.
- After the ≥ 3-window filter: **6,761 normal and 105 anomalous vehicles.**
- Every label is verified on the simulator truth (`docs/ANOMALY_TYPES.md`).

## Evaluation (kept separate; see `docs/EVALUATION_PROTOCOL.md`)

| Kind | Result |
|---|---|
| Development (9 cases, not independent) | 8 of 9 (overweight missed) |
| Validation (used for selection) | macro recall 0.714, validation AP 0.781, false alarms on val_A normals 1.07 %, stall validation 18/18, tailgating 12/13 |
| **Independent final test** (228 new runs in 8 test regions, run once) | **macro recall 0.568 [0.486–0.646], ROC-AUC 0.908, AUPRC 0.742**, precision 0.74, F1 0.64; false alarms on normal driving 2.0 % |

**Final-test comparison (same test, frozen baselines):**

| Detector | Macro recall | ROC-AUC | AUPRC |
|---|---|---|---|
| SimStudio Z-score | 0.231 | 0.783 | 0.394 |
| Supervised single model | 0.364 | 0.769 | 0.515 |

**Final-test recall per type:**

| Type | Detected |
|---|---|
| Stalled vehicle | 12/12 |
| Tailgating | 11/11 |
| Reckless driving | 7/10 |
| Lane weaving | 7/11 |
| Collision | 6/10 |
| Lane straddling | 5/12 |
| Speeding | 4/12 |
| Overweight vehicle | 4/12 |
| Hard braking | 1/11 |

## Known limitations
See `docs/LIMITATIONS.md`.
- False alarms on legitimate 1–10 s stops (20/28 in the final test).
- False alarms on tailgating matched controls (7/11).
- Low recall on hard braking, speeding, overweight and straddling.
- Decision coverage limited by the tracker (1,008 of 1,708 normal vehicles decided).
- Simulation only.

## Reproduction
- **Scores and tables.** On Linux x86-64 the release scripts reproduce them exactly. On Linux aarch64 they reproduce the same decisions and tables, with scores within 1e-5 (`docs/CROSS_PLATFORM.md`).
- **Checkpoints.** Retraining reproduces the five checkpoints exactly on the build platform.
- See `docs/REPRODUCIBILITY.md` and `RELEASE_VERIFICATION.md`.
