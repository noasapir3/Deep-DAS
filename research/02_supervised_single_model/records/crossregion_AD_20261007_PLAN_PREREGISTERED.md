# Experiments A–D on the cross-region supervised model — pre-registered plan

Written before any training. The frozen baseline is `DL_supervised_crossregion_P99` (checkpoint sha256 aa517c02…2083). It is never modified.
Each experiment is delivered into its own copy: `DL_supervised_crossregion_P99__exp{A,B,C,D}_…`.

## What changes (one thing at a time)

| Run | Input cleaning | Extra head features | Everything else |
|---|---|---|---|
| R0 (control) | no | none | the original p12 recipe, re-run on this platform, to measure platform/run noise |
| A | yes | none | identical recipe |
| B | yes | `osc` (lateral periodicity) | identical recipe; head input 16 → 17 |
| C | yes | `tg` (plausible short-headway share) | identical recipe; head input 16 → 17 |
| D | yes | `osc` + `tg` | identical recipe; head input 16 → 18 |

**Cleaning rule (fixed):** a bin with a leader present (m_lead > 0.5), bumper gap g_lead < 1.0 m, v_lon > 5 m/s and m_valid > 0 is physically impossible
(a duplicate-track leader). It is re-encoded exactly as channels.py encodes "no leader": m_lead = 0, g_lead = 60, th_lead = 10, dv_lead = 0.
It is applied to every input of the run (training, selection, calibration and evaluation), because it is part of the preprocessing. No vehicle is removed
and no label changes. The normalization (amplitude-only `norm.json`) is unchanged.

**Feature `osc`:** the share of the spectral power of d_lat in 0.25–0.6 Hz (Hann window, power above 0.02 Hz), over the vehicle's valid bins of its usable tracks
(concatenated), computed from the 13 input channels only. It is identical to the diagnostic definition.
**Feature `tg`:** the share of valid bins with a leader present, th_lead < 0.8 s and v_lon > 5 m/s, computed after cleaning (so the gap is ≥ 1 m).
**Feature scaling:** standardized by the mean and std over the training normal vehicles, fixed before training, and appended to the 16 standardized pooled codes.

**Recipe (unchanged from p12):** encoder initialized from the amplitude-only checkpoint, new head (Linear(16+k → 16), GELU, Linear(16 → 1)), seed 0,
AdamW (encoder lr 1e-4, head 1e-3, wd 1e-4), batches of 48 normal + 16 anomalous vehicles, ≤ 32 random windows per vehicle, 120 steps per epoch,
≤ 25 epochs, patience 6, checkpoint chosen by validation AP (val_A normals plus batch backgrounds vs the 17 val-region anomalies).
Training and selection data: the same cross-region split and labels (34 train / 17 val anomalies, the same normals).

## Thresholds (fixed rule, label-free)
P99 (primary) and P95 (alternative) of the run's own logits on the 1,242 sufficient-evidence normal val_B vehicles. They are computed and written
before any evaluation data are loaded. No threshold is chosen from C3, weaving or any test result.

## Evaluation (identical for baseline and every run)
- **The 9 valid anomalies:** speeding (orig 003), C1 real braking, stalled (orig 020), C4 contact-collision striker, reckless (orig 120), weaving (orig 037),
  overweight (orig 140), C2 corrected straddling, C3 realized tailgating.
- **False alarms:** normal test (513 sufficient evidence) and normal val_A (1,313).
- **Precision / recall / F1:** positives = the 9 valid anomalies, negatives = the 513 test normals. Also, separately, on the new-scene test batch
  (verified injected vehicles in unseen scenes vs the same 513 test normals).
- Vehicles with fewer than 3 windows get no decision (unchanged quality rule).

## Caveats known in advance
- C3 and the original weaving case were inspected before this plan. The features were motivated by them, so A–D on the 9 valid cases is not
  independent evidence. The new-scene batch is the independent check.
- One seed per run; R0 measures how much a re-run alone changes.
