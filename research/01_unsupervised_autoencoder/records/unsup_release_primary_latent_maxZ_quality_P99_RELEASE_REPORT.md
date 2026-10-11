# Release report: primary_latent_maxZ_quality_P99

**Simulation-based development results, not independent confirmation.** Every number below comes from simulated recordings evaluated by the
people who built the detector. This is the selected development reference. It is not claimed to be the best detector on every metric; the shipped Z-score
catches more of the original and severity anomalies.

Verified run on the Mac: `release_runs/20260927T190732Z_run/`. All 24 checks passed. A copy of its tables is in `reference_results/`.

## Active detector

**Pipeline:** 13 channels → ConvAE encoder (`cae_primary`, seed 0, epoch 79, d = 8, T = 64) → window codes (stride 8) → 16 vehicle features
(mean and std of the 8 codes, v2 aggregation) → |z| against the frozen v2 train-normal statistics → vehicle score = max |z|.

**Quality rule:** fewer than 3 windows means **insufficient evidence**. There is no decision, and the vehicle is never counted as normal.

**Thresholds** (full precision), from `FROZEN_THRESHOLDS.json`. They are the pooled sufficient-evidence normal val_B percentiles, 1,242 vehicles.

| Operating point | Threshold | Role |
|---|---|---|
| **P99** | **3.7988877296447754** | default |
| P95 | 2.89949893951416 | explicitly labelled alternative |

**Not active:**

- amplitude-only candidate;
- matched-information model;
- E1 window scorer;
- reconstruction-error scorer;
- v3 length bands;
- the historical v2 thresholds (3.252 / 4.875).

## What was verified

| Check | Result |
|---|---|
| 23 pinned artifacts and 44 code files: present, not symlinks, inside the folder, sha256 as in `RELEASE_MANIFEST.json` | pass |
| `deepdas` imported from this folder's `src/`, and nowhere else | pass |
| Checkpoint identity (cae_primary, seed 0, d = 8, T = 64, epoch 79; decoder range = norm.json) | pass |
| v2 feature statistics reproduced from the saved train-normal codes (the statistics belong to this checkpoint) | pass (max difference 0) |
| Frozen thresholds equal the P95 / P99 of saved sufficient-evidence val_B scores (the thresholds belong to these features) | pass (difference ≤ 4.3e-6, a numerical-library difference; the frozen values are the ones used) |
| Re-encoding 114,424 windows (originals, severity, val_B, val_A, test) from channels, compared with saved codes | pass (max \|Δ\| 3.0e-5) |
| Re-inferred vehicle scores compared with saved v2 scores (4,099 vehicles) | pass (max \|Δ\| 7.6e-6) |
| Controlled batch: 7 runs re-encoded and compared with saved codes; 28 vehicle scores | pass (max \|Δ\| 2.3e-5 / 1.4e-6) |
| Original-scene tracks come from the R0-selected export (collision = 05-21), not the historical 05-17 export | pass |
| Expectations: 5 of 9 causing vehicles plus the collision target at P99; 9 of 513 test false alarms; 119 insufficient-evidence test vehicles | all met, none forced |

## Results by evaluation group (P99 default, P95 alternative)

### Original 9 scenarios (R0 recording-aligned)

| Target | Score | Responsible feature | P99 | P95 | Z-score on the same recording |
|---|---|---|---|---|---|
| speeding | 4.09 | mean_z1 | ALARM | ALARM | 21.6 flag |
| hard braking¹ | 4.73 | mean_z1 | ALARM | ALARM | 383.5 flag |
| stalled | 4.73 | mean_z1 | ALARM | ALARM | 376.1 flag |
| collision striker² | 4.22 | mean_z7 | ALARM | ALARM | 49.7 flag |
| collision target (involved)² | 4.71 | mean_z1 | ALARM | ALARM | 160.0 flag |
| reckless | 4.27 | mean_z1 | ALARM | ALARM | 25.1 flag |
| weaving | 1.49 | mean_z1 | no | no | 5.1 |
| tailgating³ | 1.40 | mean_z1 | no | no | 2.1 |
| straddling³ | 1.32 | std_z4 | no | no | 6.2 |
| overweight | 1.28 | std_z0 | no | no | 19.0 flag |

**Totals:**

- Causing vehicles detected: **5 of 9** at P99 and P95, plus the collision target.
- Z-score on the same recordings: 6 of 9, plus the target.
- Affected background vehicles: 2 of 3 at P99.
- Normal background vehicles: 0 alarms, 3 with insufficient evidence.

The historical published Z-scores (05-17 collision export) are kept in a separate column and are not used for comparison.

¹ The original hard-braking vehicle never moves (a stall), so this is detection of a stall, not of braking.
² The original collision has no contact: the striker stops about 2 m short. Detection comes from the prolonged stationary period, and is not collision-specific.
³ The event was not realized in the original recording.

### Corrected scenarios (C1–C3, realized; seed 1)

| Target | Score | P99 | P95 | Z-score |
|---|---|---|---|---|
| C1 real hard braking (1.9 s, 12.5 m/s → 0) | 2.02 | no | no | 2.61 no |
| C2 straddling on the lane boundary | 1.34 | no | no | 160.0 flag (via a DAS amplitude–distance feature) |
| C3 tailgating (0.49 s median headway for 10.5 s) | 3.01 | no | **ALARM** | 6.32 no |

### Uncorrected controls (K1–K3, event not realized)

| Target | Score | P99 | P95 |
|---|---|---|---|
| K1 (a stall) | 4.72 | ALARM | ALARM |
| K2 | 1.33 | no | no |
| K3 | 2.28 | no | no |

The two affected background vehicles in K1 are flagged at P99.

### Verified contact collision (C4)

| Vehicle | Score | P99 | P95 | Z-score |
|---|---|---|---|---|
| Striker | 5.03 | ALARM | ALARM | 277 |
| Stationary target | 4.67 | ALARM | ALARM | 338 |
| Affected veh003 | 3.69 | no | ALARM | 129 |

Contact is verified, but the alarm is explained by the long stationary period after contact, **not by the contact itself**.

### Severity study (81 runs, 9 types × 3 levels × 3 seeds; injected vehicles)

| Type | P99 | P95 | Z-score |
|---|---|---|---|
| collision | 9 | 9 | 9 |
| speeding | 5 | 6 | 3 |
| reckless | 3 | 6 | 3 |
| tailgating | 0 | 2 | 0 |
| hard braking / stalled / straddling / weaving / overweight | 0 each | 0 each | 0 / 6 / 3 / 0 / 5 |
| **Total** | **17 of 81** | **23 of 81** | **29 of 81** |

The severity collision runs also stop without contact (a prolonged stop).

### Normal vehicles (false alarms among sufficient-evidence vehicles; no-decision counts)

| Split | Normal vehicles | Insufficient evidence | Sufficient | Alarms P99 | Alarms P95 | Z-score alarms |
|---|---|---|---|---|---|---|
| val_B (calibration) | 1,537 | 295 | 1,242 | 13 (1.0 %) | 63 (5.1 %) | 25 |
| val_A | 1,586 | 273 | 1,313 | 26 (2.0 %) | 78 (5.9 %) | 16 |
| test | 632 | **119** | **513** | **9 (1.8 %)** | 35 (6.8 %) | 2 |

## Recording alignment

- **Originals:** the DL input uses the R0-selected export of each scene. The Z-score comparison uses the R0 *same-recording* recomputation
  (`r0/zscore_vehicles_same_recording.csv`). The historical published Z-scores are shown separately only.
- **Controlled batch and severity:** DL and Z-score were both computed from the pinned export of each run (`sims/MANIFEST.csv` with sha256; `run.json` of each severity run).
- **Normals:** every val/test normal scenario has exactly one export folder, so both methods read the same recording. The 37 tracks from
  scenarios with 2–5 export folders are all in the train split and are not part of any comparison.

## Known limitations

- The data are simulated. Several "detections" are detections of prolonged stops (original braking, stalled, collision, severity collision).
- Real hard braking, straddling, weaving, overweight and most tailgating are not detected by this detector.
- The P95 transfers above its design rate (5.9 % on val_A, 6.8 % on test). P99 transfers at 2.0 % / 1.8 %.
- **Platform differences:** re-running on another platform changes codes by up to about 3e-5. The nearest vehicle to a threshold is 3.9e-5 above P95
  (val_B, 016/normal_heavy_traffic_sparse veh013). No decision changed between the cloud and Mac runs. On a very different library stack, that one P95 decision could flip.
- The controlled-batch channels are read from `experiments_followup/amplitude_only_20260927_203146/data/r1_runs/`. They were built with the unchanged
  channel pipeline from the pinned exports and are hash-pinned; their codes match the independently saved `r1/R1_window_codes.parquet`.
