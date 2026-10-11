# Stall diversity vs stall quantity — pre-registered plan
Written 2026-10-08, before the survey, the run plan, any simulation and any training of this experiment.

**Read-only throughout:**
- `DL_supervised_crossregion_P99`;
- the frozen candidate C_new_ENSEMBLE: `FROZEN_FINAL_CHOICE.json`, the 5 C_new checkpoints, their thresholds, plan and selection;
- the initial AE checkpoint.

Fingerprints are checked before (`data/FROZEN_INPUTS_sha256_BEFORE.txt`, plus the device-side lists) and after.
All writes go to `stall_diversity_20261008/` and to the new device copies `DL_supervised_crossregion_P99__E_stall_quantity` and `__E_stall_diversity`.
**The final holdout is not generated or opened.** Florentin is not touched.

## Hypothesis under test (not a conclusion)
The model misses the original stalled vehicle because training lacks two kinds of stops:
- vehicles standing from the start;
- vehicles that stop and never drive again;

and in particular these stops as seen through the noisy tracker. EXP1 raised exposure to the existing stall examples but added no new stop modes.

## Arms (the same amount of added data; the only difference is the stall mode)
Both arms keep **all** of C_new's training data (NEWDATA = 1: tg3 train, v4 train and its matched controls) and add the same 24 stall *slots*.
A slot is one (region, vehicle, start lane, planned route, stop location, sensing condition, speed).

| slot group (8 slots each) | E_stall_quantity | E_stall_diversity |
|---|---|---|
| A | move → stop 30 s → move (MSM) | MSM (**the same simulation runs**) |
| B | MSM | stands still from t = 0 to the end (START) |
| C | MSM | moves, then stops until the end of the recording (STOP_END) |

- **MSM** reproduces the existing training pattern: speed profile 0.9 × limit, instant freeze with the existing `frozen` mechanism, held 30 s, then released.
- **START** is `frozen` from t = 0, with v = 0, at the stop location.
- **STOP_END** uses the same instant freeze at the stop location, never released.
- **The stop location is identical across the three modes of a slot.**
  - MSM and STOP_END reach it along the same planned route (`sim.set_vehicle_route`).
  - The freeze fires when the vehicle reaches the stop location on that route. This is a position trigger instead of the earlier fixed 6 s trigger, so that the stop lands on the chosen sensing condition.
- **Recording:** 120 s for every run (the runner's cap), simulation seed 1 (as in all earlier batches), the same simulator and tracker. No tracker change, no ground-truth speed substitution.

## Sensing conditions (balanced, so the sensor mix is not a label cue)
Every stop location is on a DAS-covered segment, because otherwise a standing vehicle is not observed at all.
- **DAS_ONLY:** no camera within range + 10 m and FOV + 10°, and no GPS beacon within radius + 10 m of the stop point.
- **COMBINED:** a camera covers the stop point with margin (≤ 0.85 × range, ≤ 0.85 × half-FOV). GPS may or may not cover it.
- **Balance:** each training region has one DAS_ONLY slot and one COMBINED slot, so each mode gets 4 DAS_ONLY + 4 COMBINED slots in training.

## Normal examples (identical in both arms)
For every training slot, 2 extra runs:
1. **CONTROL:** the same vehicle, route and speed, with no stop (drives through the stop location).
   It counts as a clean normal only if its longest truth stop is < 2 s and it has no collision.
2. **LEGIT_STOP — a legitimate stop by the project's definitions.** The verification rules count a stall as a stop of ≥ 10 s, and hard braking as a deceleration of ≥ 4 m/s².
   - **Behaviour:** the vehicle brakes at 2.0 m/s² from the same route position so that it stops at the same location, stands 3 s (even slots) or 5 s (odd slots), and drives on.
   - **Verified** only if the truth shows a stop of 1–10 s, peak deceleration < 4 m/s², movement after the stop, and no collision. Otherwise the run is excluded and reported.
   - **Disclosed:**
     - The simulator has no traffic lights or stop signs, so this is the only legitimate-stop mechanism available.
     - The normal corpus contains almost no stopped vehicles (val_B and train normals: share of |v| < 0.5 m/s ≈ 0).
     - The severity study once used a 4 s freeze as a "mild" stall. That set is not used in training here.

**Background vehicles** use the p10 rule: usable as normal only if the truth longest stop is < 2 s and the minimum gap is ≥ 5 m.
Vehicles queued behind a stalled or stopping vehicle fail this rule and are **excluded and counted**, never auto-labelled normal.
veh002 is an ordinary background vehicle in these runs.

## Realization, tracking and windows (reported separately; nothing is filtered because a model finds it hard)
Ground-truth rules (truth v < 0.1 m/s = stopped):

| mode | verified if |
|---|---|
| START | stopped from the first truth sample to the last, ≥ 100 s |
| STOP_END | max v before the stop ≥ 3 m/s; the stop lasts until the last truth sample; ≥ 10 s |
| MSM | max v before ≥ 3 m/s; longest stop ≥ 10 s; v ≥ 2 m/s for ≥ 2 s after the stop |

- Collisions of veh001 exclude the run in every mode.
- **Tracking:** the import is unchanged (`read_export → build_channels → track_quality`, amplitude on the DL bins).
  Usable tracks need ≥ 32 valid bins and quality_ok, as before. A model decision requires ≥ 3 windows.
- **Reported per run:**
  - verified (truth);
  - track exists;
  - usable;
  - ≥ 3 windows;
  - observed sensor shares while stopped (m_das, m_cam, m_gps).
- **Information limit (reported, not assumed away):** a long STOP_END near the end of the recording and a LEGIT_STOP may look alike in the input. For the input statistics of every group, see `input_stats.csv`.

## Regions (drawn by a deterministic survey; no performance data used)
- **Training:**
  - Pool: `train`-split regions, excluding every region already used for added data — cross-region (131, 050, 090, 065), tailgating v1–v3 train, and v4 train.
  - Not drawable: the test regions (including 020, the original stalled scene) and the val_B regions.
  - **Eligible region:** a DAS_ONLY candidate and a COMBINED candidate, each with a planned route of ≥ 6 s and ≤ 14 s of driving at 0.9 × limit before the stop point, ≥ 60 m of route after it, and a start lane with no other vehicle.
  - **Draw:** 12 regions with numpy rng 20261009. Within a region, the candidate with the longest post-stop route is taken (ties: lane id).
- **New stall validation set (`stallval`):**
  - Pool: `val_A` regions not used before (excluding 002, 005 and the six tg3 val regions).
  - 6 regions drawn with rng 20261010: 3 DAS_ONLY and 3 COMBINED (one slot per region, the same eligibility rule).
  - Per region: START, STOP_END and MSM (18 stall vehicles, 6 per mode, 3+3 per sensing condition), plus one CONTROL and one LEGIT_STOP (5 s) (12 normal runs).
  - The training regions and the stallval regions are disjoint by construction, and disjoint from the original case and from the newscene regions.

## Training (unchanged C_new recipe)
- `x_train_eval_c.py` with NEWDATA = 1, `clean = 1`, features `tg`.
- The only change: the arm's added training vehicles (the slot anomalies of that arm, plus the shared normals).
- The same initial AE checkpoint and seeds 0–4.
- Unchanged: 48 + 16 per batch with the same uniform sampling over vehicles, 120 steps, 25 epochs, patience 6, the learning rates, weight decay and BCE loss.
- **Checkpoint selection:** validation AP on **the same validation set as C_new** (stallval is not added to it).
- **Exposure per type per epoch:** replayed exactly from the rng and reported.
  - Expected, from the vehicle counts: tailgating's share of anomaly draws falls from 60.5 % to about 47 %.
  - Each other non-stall type falls from about 94 to about 73 draws per epoch.
  - This is the same in both arms and is reported, not corrected (correcting it would be EXP1's balancing, which is excluded here).

## Evaluation
- Each seed and each ensemble gets its own P99 (primary) / P95 on the 1,242 sufficient-evidence val_B normals.
- **Ensemble:** the C_new method — the mean of the per-seed logits standardized with that seed's val_B normals.
- C_new (the 5 frozen checkpoints and their frozen thresholds) is **re-scored read-only** on stallval.
  The existing groups are compared to its saved `vehicle_scores.csv` as a determinism check.
- **Reported per seed, mean ± std and ensemble:**
  - stallval recall per mode and per sensing condition;
  - stallval alarms on CONTROL and LEGIT_STOP normals;
  - validation recall for each of the 9 types;
  - macro9;
  - val AP;
  - FA val_A / test;
  - decision coverage.
- **Diagnostic only, after checkpoint and threshold are fixed:** the original stalled vehicle, C3, the 9 valid cases, newscene.

## Replacement rule (fixed now)
An arm's ensemble replaces C_new_ENSEMBLE as the development candidate only if **all** of the following hold:
1. **Stall-mode improvement:** stallval macro recall over the 3 modes at its own P99 is strictly higher than C_new_ENSEMBLE's on the same set.
2. **Macro9 improvement:** macro9 is strictly higher than C_new_ENSEMBLE's.
   - **macro9** = the mean over the 9 types of the validation recall at P99.
   - The stall entry pools the 2 original validation stalls with the 18 stallval stalls.
   - The other 8 entries come from the original validation set.
3. **Tailgating:** validation tailgating ≥ 12/13.
4. **False alarms:** FA on val_A ≤ 1.5 % at its own P99.

- **Ties and both arms passing:** if both arms pass, the higher macro9 wins; ties go to the higher stallval macro, then to val AP.
- **Otherwise** C_new_ENSEMBLE stays.
- **Reported, not gating:** alarms on stallval LEGIT_STOP and CONTROL normals.
- **Not combined and not extended:** no balancing, no encoder freezing and no extra stop feature are combined into either arm, and no further experiment follows automatically.
