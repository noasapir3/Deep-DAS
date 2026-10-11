# Leakage audit

**Rule (agreed in the design review).** The new deep model must not use any quantity that
is only available because the data come from a simulator. The Z-score baseline stays
exactly as shipped, including its known truth-derived features; it is a reference point,
not a leak-free comparator. Everything the autoencoder sees is derived from the exported
Kalman trajectory of a track plus the road map. Ground-truth quantities are used for two
purposes only: to **label** vehicles (which vehicle was injected) and to **stratify or
describe** results (DAS σ bins, realised severity).

## 1. Every column of the exports, and what happens to it

### `tracking_audit.xlsx`, one sheet per track (`Track T*`), 50 Hz rows

| Column (Final-Project header → name) | Origin | Disposition in DL |
|---|---|---|
| Time (s) → `t` | simulation clock | **kept** (binning) |
| Track ID → `global_track_id` | tracker | **kept** as the track key only (never an input) |
| True Vehicle ID → `vehicle_id_oracle` | simulator truth (oracle) | **metadata only**: label key. Not in `channels.parquet` |
| Road Segment → `segment_id` | sensor event payload = true lane of the vehicle | **dropped** (DL does its own map matching from the Kalman position) |
| Update Type → `update_kind` | filter | **kept** in raw import; not a channel |
| True X/Y (m) → `true_x`, `true_y` | truth | **dropped** |
| True Distance Along Road → `true_s` | truth | **dropped** |
| GPS X/Y, Camera X/Y, DAS X/Y → `*_x`, `*_y` | raw measurements | **presence only** (`m_gps`, `m_cam`, `m_das` = fraction of the bin's rows carrying a measurement); the values are dropped |
| GPS/Camera/DAS Distance Along Road → `*_s` | projection on the **oracle** segment | **dropped** |
| DAS Fiber Position → `das_fiber_position_m` | DAS model | **dropped** |
| Kalman X/Y Estimate → `x_hat`, `y_hat` | filter | **kept**: map-matched to the lane → `d_lat` |
| Kalman Velocity X/Y → `vx_hat`, `vy_hat` | filter | **kept** → `v_lon`, `v_lat`, `r_lim`, `dv_lead` |
| Kalman Acceleration X/Y → `ax_hat`, `ay_hat` | filter | **kept** → `a_lon` |
| Kalman Distance Along Road → `hat_s` | projection on the oracle segment | **dropped** |
| Kalman Position/Velocity Uncertainty σ → `sigma_pos_m`, `sigma_v_mps` | filter, driven by the idealised σ model (which is a function of true weight and true fibre distance) | **dropped** |
| 2D Position Error → `pos_err_m` | needs truth | **dropped** |
| Pre-update residuals ν_x, ν_y; S_xx, S_yy; NIS_x | filter internals, driven by the idealised σ model | **dropped** |

### `Kalman Audit` sheet (one row per sensor measurement)

Only per-track **statistics** are taken, into `meta.parquet`, and used for evaluation
strata: number of accepted DAS / camera / GPS measurements, mean and max DAS σ
(`Position Uncertainty σ`), mean DAS SNR, mean camera σ. The DAS σ is computed by the
simulator from the true weight and the true distance to the fibre, so these strata are
truth-derived; they never enter `channels.parquet`. The columns `DAS Amplitude`,
`Fiber Distance r`, `SNR Threshold`, `Lateral Offset from Lane Centre`, `Measured X/Y/Speed`,
`Heading`, `Detection Confidence`, `Accepted/Skipped`, `Skip Reason` are not used.

### `coverage_summary.xlsx`

Not used.

### Scenario files (`*.sim.json`)

Used for (a) the road map: node/segment geometry and the segment speed limits, through the
copied `rebuild_lanes`; (b) the vehicle override fields, to derive the **labels**
(`deepdas.data.labels.injected_vehicles`); (c) the severity variants. Vehicle weights,
speeds and positions in the file are never model inputs.

### Enforcement

`tests/test_map_matching_and_channels.py::test_channel_table_has_no_truth_columns` and
`tests/test_dataset_integrity.py::test_channels_table_columns` assert that the model-input
table contains only the 13 documented channels (plus bookkeeping columns) and none of the
truth columns; `TRACK_TRUTH_DROPPED` in `deepdas.data.import_exports` lists the dropped
columns explicitly, and the raw import table has no column outside `TRACK_KEEP` +
presence flags.

## 2. What the autoencoder sees

Eight value channels — `v_lon`, `a_lon`, `d_lat`, `v_lat`, `r_lim`, `g_lead`, `th_lead`,
`dv_lead` — and five masks — `m_das`, `m_cam`, `m_gps`, `m_lead`, `m_valid` — at 10 Hz
(`docs/PIPELINE.md` §3). All are functions of the Kalman state, the map and the other
tracks' Kalman states of the same export. The leader channels use the simulator's own
gap definition (bumper gap = Δs − 4.5 m), computed between **tracks**, not between true
vehicles.

## 3. Inherited truth dependence (documented limitation, not removed)

The exported trajectories were produced by the Final Project's pipeline and are reused
as they are (design decision 2). Three places in that pipeline consume simulator truth:

1. **Association.** `TrackManager` creates and refreshes tracks from
   `world.vehicle_state` events and has an oracle `vehicle_id` fast path
   (`tracking.py`). Tracks are therefore cleaner than a real tracker would produce.
2. **Filter initialisation.** Every `LegacyKalmanFilter` is initialised at the true
   position (`audit.build_audit_and_trajectory`).
3. **Camera / GPS velocity.** The camera and GPS "velocity measurements" fed to the
   filter are the noise-free true speed and heading (`audit.py`).

Consequence: the Kalman channels are less noisy than they would be with a truth-free
tracker (Phase-1 ablation: speed RMSE 0.98 → 1.42 m/s without 2 and 3). Both detectors
are evaluated on the same trajectories, so the comparison is fair, but absolute numbers
are optimistic with respect to a real deployment. The first second of every track is
excluded (burn-in) so that the truth-seeded initial state has decayed.

## 4. The Z-score baseline (kept as shipped)

`model_live.pkl` uses 156 features of which 40 are computed directly from simulator
truth and 18 are noise-free functions of true weight, distance or range (Phase-1 audit,
`docs/DESIGN.md` §C.4). They stay, by design decision 5. Its threshold (15.39 σ) was
fitted on normal tracks from all 148 regions, so its false-alarm rate on the existing
normal corpus is in-sample; the autoencoder's false-alarm rate is measured on held-out
regions. Both are reported.

## 5. Track-quality gate (label-free)

`deepdas.data.quality` flags tracks whose fused kinematics are physically impossible for at
least one second (reversing faster than 3 m/s, faster than 45 m/s, or more than 5 m from
the matched lane centre-line). The rule reads only the model channels — no truth, no
label, no score — and is applied identically to normal and anomaly tracks (3.3 % of normal
tracks fail, concentrated in the sparse-coverage scenario types; no anomaly track fails).
Gated tracks are excluded from training and from the main evaluation, and reported
separately, because they are tracker failures rather than driver behaviour.

## 6. Labels and thresholds

* Labels come from the scenario file's override fields (and `run.json` for severity
  runs), never from any score.
* The autoencoder's threshold is the 99th percentile of the **val_B** normal-vehicle
  scores; val_B regions are never used for training or early stopping (val_A), and
  contain no anomaly scenario. Test anomaly labels are never used to pick a threshold.
* The normalisation statistics are fitted on train-split normal tracks only.
* Splits are by region with a one-tile buffer (`tests/test_dataset_integrity.py`).
