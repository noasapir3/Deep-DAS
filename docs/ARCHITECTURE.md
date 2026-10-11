# Architecture of DeepDAS Ensemble v1

Everything below is the frozen configuration of the released model. Code references are relative to `src/deepdas/`.

## 1. Input: 14 channels per 0.1 s bin (`data/channels.py`, `amplitude.py`)
The model reads only the exported Kalman tracks of the SimStudio tracker (DAS + camera + GPS fusion, 50 Hz rows). The tracks are map-matched to the lane geometry and averaged into 0.1 s bins. **No simulator truth is read.** See `LEAKAGE_AUDIT.md`.

| # | Channel | Meaning | Unit |
|---|---|---|---|
| 1 | `v_lon` | longitudinal speed along the matched lane | m/s |
| 2 | `a_lon` | longitudinal acceleration | m/s² |
| 3 | `d_lat` | signed offset from the lane centre-line (left > 0) | m |
| 4 | `v_lat` | lateral velocity | m/s |
| 5 | `r_lim` | v_lon / speed limit − 1 | – |
| 6 | `g_lead` | bumper gap to the nearest track ahead on the same lane (cap 60 m) | m |
| 7 | `th_lead` | time headway g_lead / max(v_lon, 0.5) (cap 10 s) | s |
| 8 | `dv_lead` | closing speed to the leader | m/s |
| 9–11 | `m_das`, `m_cam`, `m_gps` | fraction of the bin's rows that carry that sensor | – |
| 12 | `m_lead` | 1 if a leader within the cap exists | – |
| 13 | `m_valid` | 1 for real, on-road bins after the burn-in | – |
| 14 | `das_amp_log10` | mean log10 DAS detection amplitude in the bin | log10 |

- **Burn-in.** The first 1.0 s of every track is excluded.
- **Usable track.** At least 32 valid bins, and the label-free quality gate passes (`data/quality.py`). The gate rejects tracks that, for at least 1 s, reverse faster than 3 m/s, exceed 45 m/s, or lie more than 5 m from the lane centre-line. These are tracker failures, not driving behaviour.

## 2. Fixed preprocessing (`ensemble/detector.py`)
1. **Lead-artefact cleaning.** A bin with a "leader" closer than 1 m while the vehicle drives faster than 5 m/s is a merged/duplicate track. It is re-encoded exactly as "no leader": m_lead 0, gap 60 m, headway 10 s, dv 0. No vehicle or label is changed.
2. **Robust normalisation.** Fixed per-channel median and scale (`models/deepdas_ensemble_v1/input_normalization.json`), fitted once on the valid bins of the train-split normal corpus.
   - The 13 track channels use the normalisation of the primary autoencoder.
   - The amplitude scale is max(1.4826 × MAD, 0.05), and the result is clipped at ±15.
3. **Windows.** 64 bins (6.4 s), stride 8 bins (0.8 s), over every usable track of the unit. A short tail is zero-padded to 64.

## 3. Network (one seed: 67,225 trained parameters + 34 fixed standardisation constants)
```
window [14 × 64]
 ├─ Conv1d 14→32, k 7          ┐
 ├─ Conv1d 32→64, k 5, stride 2 │ encoder (initialised from the amplitude-aware autoencoder,
 ├─ Conv1d 64→64, k 5, stride 2 │ fine-tuned with the head)
 ├─ Conv1d 64→128, k 3, stride 2┘
 └─ Linear 1024 → 8                                   → 8-d code per window
unit = all windows of the unit's usable tracks
 ├─ mean and std of the codes (16)  → standardised with fmu / fsd (training normals; stored in the checkpoint)
 ├─ tailgating fraction (1)         → standardised with emu / esd (training normals; stored in the checkpoint)
 └─ MLP head: Linear 17→16, GELU, Linear 16→1         → logit
```
**Tailgating fraction.** The share of valid bins, after cleaning, that have a leader, time headway < 0.8 s and speed > 5 m/s. It is the only hand-made feature.

## 4. Ensemble and decision (`ensemble/scoring.py`, `models/deepdas_ensemble_v1/ensemble.json`)
- **Seeds.** Five seeds (0–4), identical except for the seed (head initialisation and batch order).
- **Per-seed standardisation:** z_k = (logit_k − μ_k) / σ_k. μ_k and σ_k are the mean and standard deviation (ddof 0) of seed k's logits on the 1,242 sufficient-evidence **val_B** normal vehicles. They are frozen in `ensemble.json`.
- **Ensemble score:** the mean of z_0 … z_4.
- **Decision for a unit:**
  - fewer than 3 windows → **no decision**: the unit is not counted as normal or as anomalous;
  - otherwise **ALARM** if the score > **1.9726181803233118**. This is the P99 of the ensemble score on the same val_B normals. The alternative operating point P95 is recorded in `ensemble.json`, but it is not the release decision.
- **The unit.**
  - In deployment and in the final test, **one usable track**.
  - In the validation and development tables, all usable tracks of one simulated vehicle (oracle-grouped). This unit exists only in simulation.

## 5. Training recipe (`train/supervised.py`)

| Element | Value |
|---|---|
| Initialisation | encoder from `models/deepdas_ensemble_v1/autoencoder_init.pt`; head from the seed |
| Batch | 48 normal + 16 anomalous vehicles, uniform with replacement; at most 32 random windows per vehicle |
| Steps / epochs | 120 steps per epoch, at most 25 epochs, early-stopping patience 6 |
| Optimiser | AdamW, encoder lr 1e-4, head lr 1e-3, weight decay 1e-4; gradient-norm clipping 5 |
| Loss | binary cross-entropy at vehicle level |
| Checkpoint | the epoch with the best validation AP (val_A normals + validation anomalies) |
| Determinism | `torch.set_num_threads(1)`, seeded `random` / `numpy` / `torch` |

**Training data after the ≥ 3-window filter:**
- train: 6,761 normal and 105 anomalous vehicles;
- validation: 1,473 normal and 29 anomalous vehicles;
- val_B (thresholds only): 1,242 normal vehicles.

**Training anomaly sources:**
- the cross-region set (all 9 types);
- the tailgating sets;
- the stall set, with three stop modes.

**Training normals:**
- the normal corpus;
- unaffected background vehicles of the simulated sets;
- matched controls and legitimate stops.

## 6. Baselines shipped for comparison
- **Supervised single model** (`models/baselines/supervised_single_p99/`).
  - The same encoder architecture and head without the tailgating feature, one seed, no lead-artefact cleaning.
  - Threshold: the P99 of its own val_B logits, −5.523461642265342.
- **SimStudio Z-score** (`models/baselines/zscore/`).
  - The hand-crafted model of the SimStudio Final Project, used unchanged: 156 track features and a z-score against normal statistics.
  - Shipped threshold 15.392482911229285.
  - 13 of its features use the truth position error (see `LIMITATIONS.md`).
