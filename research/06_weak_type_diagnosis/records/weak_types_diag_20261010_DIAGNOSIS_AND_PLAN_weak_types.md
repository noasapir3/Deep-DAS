# Weak anomaly types of E_stall_diversity_ENSEMBLE: diagnosis and proposed plan
2026-10-10. Read-only diagnosis. Nothing was trained, simulated, re-thresholded or changed.

## How the diagnosis was done
- **Folder:** `experiments_followup/weak_types_diag_20261010/`.
  - Scripts: `d1_diagnose.py`, `d2_dilution.py`, `d3_braking_window.py`.
  - Tables: `results/*.csv`.
- **Data:** the final-test North-TLV runs.
  - These runs now count as **exposed development data**. They are used for diagnosis only, never as evidence for a new model.
  - Analysis units: realised core anomalies (veh001, ≥ 3 windows) vs 1,018 normal-driving tracks, plus the E_DIV training set (exact composition rebuilt from `x_train_eval_s.py`).
- **Frozen models:** loaded read-only through `hs_scorer.py` (5 seeds, frozen normalisation and val_B constants).

**Five measurements per type, in the order the information flows:**
1. **Truth:** realised strength, from `realization.json`.
2. **Input:** AUC of simple statistics computed from the DL input channels themselves (what the network receives).
   - "13-stat probe" = a cross-validated logistic regression on 13 such statistics, grouped by run.
3. **Encoder:** a cross-validated linear probe (grouped by run) on the pooled 16-d code of each frozen seed, averaged.
   - Also on the initial autoencoder, before supervised training.
4. **Head and threshold:** AUC of the frozen ensemble score; recall at the frozen P99.
5. **Training data:** count, parameters and ranges of the training examples vs the test anomalies and the normals.

Extra checks:
- **Dilution check (`d2`):** the same frozen models on an 8-s crop around the event, compared with normals cropped around *their* own worst moment.
- **Collision check:** the score with the post-impact part cut away.

## Summary table (final-test tracks, exposed)

| Type | Truth realised | Input stat AUC | Input 13-stat probe | Encoder probe (E_DIV) | Initial AE probe | E_DIV score AUC | Recall @P99 |
|---|---|---|---|---|---|---|---|
| Hard braking | 11/11, 6 m/s² | 0.90 (a_lon p1), 0.975 (speed drop in event window) | 0.997 | 0.91 | 0.89 | **0.80** | 1/11 |
| Speeding | 12/12, ×1.11–1.44 | 0.87–0.90 | 0.86 | **0.81** | 0.70 | **0.78** | 4/12 |
| Overweight | 12/12, 30 t | 0.945 (amplitude) | 0.87 | 0.96 | 0.96 | **0.83** | 4/12 |
| Straddling | 12/12, 1.8 m | 0.998 (\|d_lat\| median) | 0.999 | 0.95 | 0.83 | 0.93 | 5/12 |
| Collision | 10/10, contact at ~8 m/s | 1.0 | 1.0 | 1.0 | 1.0 | 0.985 | 6/10 |
| Weaving | 11/11, std 1.05 m | 1.0 (0.25–0.6 Hz power) | 0.999 | 0.94 | 0.79 | 0.90 | 7/11 |
| *Tailgating (ref)* | | | 0.91 | 1.0 | 1.0 | 1.0 | 11/11 |
| *Stalled (ref)* | | | 0.92 | 0.93 | 0.97 | 0.97 | 12/14 tracks |

**Main conclusion.** For all six types, the distinguishing physics is in the input, and it reaches the encoder's pooled representation.
- The encoder probe AUC is ≥ 0.81, and ≥ 0.91 for five of the six.
- So this is **not** a case where the information never reaches the internal representation.
- The loss happens mainly **after** the encoder. The head's score AUC is lower than the encoder probe for:
  - braking: 0.80 vs 0.91;
  - overweight: 0.83 vs 0.96;
  - speeding: 0.78 vs 0.81;
  - weaving: 0.90 vs 0.94.
- For collision, straddling and weaving the ranking is good, but the scores sit just below P99, so the threshold margin is the problem.

**Why the head under-uses the information.** The training set gives each weak type **4 vehicles** (out of 105 positives).
- Each weak type: 3.8 % of positive draws.
- Tailgating: 49 vehicles, 47 %.
- Stalled: 28 vehicles, 27 %.
- All 4 of each weak type come from a single fixed design.
- The validation set used for checkpoint selection has only 2 per weak type, so the AP selection was practically blind to them.

This is the same situation tailgating was in before its expansion.

**Caveat.** The probes are trained and tested by cross-validation on the same exposed data, with 10–12 positives per type. Treat them as an indication that the information is present, not as an achievable score.

## Per-type diagnosis

### Hard braking (1/11): the hardest case
1. **Truth realised:** 11/11.
   - Peak 6 m/s², designed at t = 6 s.
   - Truth stop is only 0.26–0.28 s, i.e. a brake to almost zero and then driving on.
2. **Input:**
   - In the event window, the speed drop is 9–15 m/s (AUC 0.975 vs normals' same window) and a_lon p1 has AUC 0.90.
   - **But a_lon is noisy:** the minimum a_lon of normal tracks has a median of −7.5 m/s², i.e. *stronger than the designed brake*.
   - 40 % of training normal tracks have a_lon p1 < −5.5. The track-level a_lon minimum alone has AUC 0.70.
   - The tracker-derived acceleration has a noise floor comparable to the event.
3. **Encoder:** probe 0.91. The information is there, mostly through the speed profile.
4. **Head:** score AUC 0.80.
   - Scoring only the 8-s crop around the brake does **not** help: AUC 0.71–0.81, 0/11 above the normal-crop P99.
   - So this is **not** mainly a mean/std pooling dilution problem. The model simply does not treat this pattern as anomalous.
5. **Training:**
   - 4 examples, all identical: 6 m/s² at t = 6 s, at the very start of the track.
   - Training normals include many brake-to-stop events: stop-and-go scenes, and the 23 legitimate stops labelled normal on purpose.
6. **Look-alike normals:** many.
   - A legitimate stop differs from this hard brake only in deceleration (< 4 vs 6 m/s²) and dwell time (1–10 s vs 0.3 s).
   - Earlier finding (legit-stop diagnosis): the model's stop response is driven by stop **dwell** share. That is why it flags legitimate stops (20/28) but not hard brakes with a 0.3 s dwell.
7. **Cause:** too few, homogeneous examples, plus a low-SNR channel (tracking noise on a_lon), plus strong normal look-alikes. The threshold is not the cause.
8. **Shortcut risk:** every training and test brake is at t = 6 s near the track start, so "brake at track start" is a possible shortcut. New data must vary the timing.

**Expectation:** improvement is possible but uncertain, and it may cost legitimate-stop false alarms. This is the type least likely to improve from data alone.

### Speeding (4/12)
1. **Truth realised:** 12/12, p90 speed ratio 1.11–1.44 (median 1.40).
2. **Input:** r_lim AUC 0.87–0.90, limited by two things.
   - (a) **The normal generator itself exceeds the limit.**
     - `populate_vehicles` draws the mean speed as 1.0 × limit with Gaussian noise, and speed_max up to mean + 3–8 m/s.
     - So 26 % of training normal tracks have r_lim p90 > 1.15.
     - Speeding at ×1.1–1.3 therefore overlaps the normal tail *by design of the data*.
   - (b) **Map-matching limit errors:** a truth ×1.11+ speeder shows r_lim p90 of 0.88 in one track. Normals reach r_lim max ×2.5.
3. **Encoder:** probe 0.81 (initial AE 0.70). This is the weakest representation of the six; r_lim is a single channel.
4. **Head:** score AUC 0.78.
   - The score does **not** follow the speed ratio: ×1.50 → 0.33 and ×1.62 → −0.06, while some ×1.44 tracks score 3.2.
   - Detections cluster on short tracks (Spearman score vs n_windows = −0.53).
5. **Training:**
   - 4 examples with r_lim p90 of 2.40, 1.48, 1.54 and 2.36.
   - Two of them are extreme (×2.4, partly limit artefacts), unlike the test median ×1.45.
6. **Look-alikes:** many fast normal drivers, and the normal_fast_legal scenes.
7. **Cause:** few, unrepresentative examples, plus label-design overlap with the normal speed tail, plus a weak representation.
8. **Shortcut risk:** learning "extreme r_lim spikes", which normals also have from limit errors, instead of sustained excess speed.

### Overweight (4/12)
1. **Truth realised:** 12/12, all 30 t.
2. **Input:** DAS amplitude AUC 0.945, overlapping the heavy normal vehicles.
   - Normal amp p90 has a p99 of 4.15; the anomalies range 3.84–4.47.
   - One test track has **no DAS amplitude at all** (m_das = 0). It cannot be detected: a sensor-coverage limit.
3. **Encoder:** probe 0.96, and the initial AE already has 0.96. The representation is good.
4. **Head:** score AUC 0.83. **The largest encoder→head loss.**
   - The score does not follow amplitude: amp 4.43 → 1.51, while 3.84 → 2.60.
5. **Training:**
   - 4 examples, three with amp p90 of 4.44–4.61, above every test anomaly except one.
   - The head learned a boundary that is too high, from too narrow a range.
6. **Look-alikes:** legal heavy vehicles in the normal_heavy_vehicles scenes (4.6 % of training normal tracks have amp p90 > 3.8), but they are not near-boundary labelled examples.
7. **Cause:** too few examples and too narrow a severity range. The representation is fine, so **the best candidate for data expansion.**
8. **Shortcut risk:** amplitude also depends on DAS geometry and coverage. Weights, regions and sensing modes must vary.

### Lane straddling (5/12)
1. **Truth realised:** 12/12, all exactly 1.8 m offset (one fixed design).
2. **Input:** \|d_lat\| median AUC 0.998, very strong.
3. **Encoder:** probe 0.95 after supervision (initial AE 0.83).
4. **Head:** score AUC 0.93 and event-crop AUC 0.955. A margin problem.
   - The 2 test tracks with d_lat std > 1 m (lane assignment flipping while on the line) score 0.85, vs 2.03 for the others.
5. **Training:**
   - 4 examples, all at 1.8 m with d_lat std 0.18–0.28.
   - Test tracks reach d_lat std 2.5 m because map matching alternates lanes.
6. **Look-alikes:** few. Only 0.7 % of training normals have \|d_lat\| median > 1.2.
7. **Cause:** homogeneous examples, plus map-matching noise when the vehicle is exactly on the line.
8. **Shortcut risk:** low.

### Collision (6/10)
1. **Truth realised:** 10/10.
   - Contact at about 8 m/s relative speed, then the vehicle stays stopped for about 104 s.
2. **Input:** a sharp deceleration (−10 to −17 m/s²) followed by a long stationary period, AUC 1.0.
3. **Encoder:** probe 1.0.
4. **Head:** score AUC 0.985, but the scores sit at 1.5–4.7, around P99 = 1.97. **A threshold-margin problem.**
5. **Training:** 4 vehicles, one design (8 m/s into a frozen target).
   - One of the 6 positive tracks is a **purely stationary post-impact fragment** (speed 0, 55.8 s).
   - So part of the collision signal the model learned is "a long stop".
6. **Post-impact check:**
   - With the track cut 2 s after impact, 5/10 stay above threshold (6/10 on the full track).
   - Median score: 1.83 vs 2.11.
   - For some vehicles the score falls sharply (4.66 → 0.37); for others it rises.
   - **So the model detects collisions partly through the post-impact stop, but not only through it.**
   - Under the project's rules a stop ≥ 10 s after a collision is itself abnormal, so the binary label is correct. The concern is generalisation to collisions without a long stop.
7. **Cause:** few examples, one design, margin.
8. **Shortcut:** confirmed partially (post-impact stop). New data must include collisions where the impact phase is visible and the target is moving.

### Lane weaving (7/11)
1. **Truth realised:** 11/11.
   - Lateral std 1.05–1.06 m, just above the 1.0 criterion (amplitude 1.5 m, period 2.5 s, fixed).
2. **Input:** 0.25–0.6 Hz lateral power AUC 1.0; v_lat std 0.98.
3. **Encoder:** probe 0.94 (initial AE 0.79).
4. **Head:** score AUC 0.90. A margin problem.
5. **Training:** 4 examples with the identical design.
   - Test examples are in-distribution, so the failure is not a distribution shift. It is too few examples and margin.
6. **Look-alikes:** few (1.8 % of training normals have an oscillation ratio > 0.6).
7. **Cause:** too few examples; all of them minimal severity.
8. **Shortcut risk:** low.

### Cross-type checks
- **Pooling dilution:** not the main cause (event crops do not raise the scores). Short tracks help speeding detection, which points to a whole-track mean effect, not a missed local event.
- **Tailgating controls** (7/11 alarms in the final test): the known shortcut (slow pinned leader / close following).
  - It is part of the no-regression conditions below.
- **Sensor coverage:** one overweight track had no DAS amplitude. Undecided tracks (< 3 windows) are a coverage limit of the decision rule, not of the types.

## What is missing, per type

| Type | What is missing | Data alone enough? |
|---|---|---|
| Overweight | Range of weights and DAS conditions; near-boundary legal heavies | Likely |
| Straddling | Range of offsets, durations, routes; on-the-line cases | Likely |
| Weaving | Range of amplitude, frequency, duration | Likely |
| Collision | More examples; varied impact speed and target state; visible impact phase | Likely (margin) |
| Speeding | Realistic severities (×1.3–1.8) on many routes; fast-legal controls. Note: ×1.1–1.25 overlaps the normal generator by design | Partly |
| Hard braking | Varied decel, speed, timing and shape (partial and to-stop); firm-legal braking controls. Limited by the a_lon noise floor | Uncertain |

## Proposed data expansion (batch "WX"; nothing generated yet)

**Tooling:** the existing p10 `build()` and realisation logic with new parameter grids, a planned route via `sim.set_vehicle_route`, and backgrounds from `populate_vehicles` with new seeds. New code goes in a new folder; existing scripts are imported read-only.

**Anomalies, per weak type: 24 train + 8 validation vehicles**, one injected vehicle per run.

| Type | Grid (balanced over the 24 / 8) | Realisation (unchanged p10 rule unless noted) |
|---|---|---|
| Hard braking | decel 5 / 6 / 7 / 8 m/s²; start speed 8–15 m/s; shape: partial (40–70 % speed drop, then continue) or to near-stop and go; event time 6–40 s into the track | peak ≥ 4 m/s² |
| Speeding | ×1.30 / 1.45 / 1.60 / 1.80 of the limit; whole route or ≥ 15 s segment; light → heavy traffic | p90 ratio ≥ 1.10 (over the event segment for segment cases*) |
| Overweight | 20 / 25 / 30 / 40 t; DAS-only, DAS+camera, sparse; speeds 0.7–1.0 × limit | ≥ 20 t |
| Straddling | offset 1.7 / 1.8 / 1.9 / 2.1 m; whole route or ≥ 20 s segment | median \|d_lat\| ≥ 1.7 m (over the event segment*) |
| Collision | impact speed 5 / 8 / 12 m/s; target frozen or moving 2–4 m/s; target 60–130 m ahead | ≥ 1 contact |
| Weaving | amplitude 1.5 / 1.8 / 2.2 m × period 2 / 3 / 4.5 s; whole route or ≥ 20 s segment | lateral std ≥ 1.0 m (over the event segment*) |

\* **Segment events need the realisation statistic measured over the designed event window** (same thresholds). This is an extension of the realisation rule and **needs your approval**. Without it, use whole-route events only.

**Normal controls (labelled normal, matched and placed in the same runs/regions):**
- Firm legal braking, 2.5–3.5 m/s², partial slowdown and continue: 24 train + 8 val. Against hard braking.
- Fast legal driving at 0.95–1.05 × the limit: 16 + 6. Against speeding.
- Legal heavy vehicles of 8–15 t across sensing modes: 16 + 6. Against overweight.
- Normal lateral drift (0.5–0.9 m) and normal lane changes: 16 + 6. Against straddling and weaving.

**Unaffected background vehicles** of all these runs are added as normals, as before.

**Kept unchanged:**
- all E_DIV training data: tailgating v3 / v4 with controls, stall_div stalls and legitimate stops, the p10 batch and the normal corpus;
- the original 4 + 2 per type.

**Size:** about 144 + 48 anomaly runs and 72 + 26 control runs, about 290 runs in all, plus a planning-only feasibility step (start lanes, routes, collision geometry) before freezing.

## Split and independence
- **By region, following the existing cross-region split:**
  - new **train** runs only in train regions;
  - new **validation** runs only in validation regions.
- **The final-test (test) regions are kept out of train and validation entirely.** They are reserved for the new fresh test.
- **New randomness throughout:**
  - new sim_seed and bg_seed ranges, disjoint from all earlier ones;
  - injection start lanes never used before in that region.
- **Independence check before freezing:** the same check as the final test (scenario md5, start segments, identical background vehicles).

## Architecture
- **No change:** the same encoder, pooling, head, tg feature, cleaning, normalisation, loss, learning rates, epochs and early stopping.
- **The diagnosis supports this:** the information reaches the encoder for all six types.
- **If hard braking still fails after the expansion,** the next candidate would be a pooling change (e.g. max or attention over windows) in a separate round. It is not proposed now.

## Training arms (5 seeds each, seeds 0–4; nothing else changes)
- **W1 (data only):** the new examples join the positive pool. Positives are still sampled uniformly per vehicle.
  - Tailgating's share of positive draws falls from 47 % to about 20 %.
- **W2 (data + exposure-preserving sampler):** each batch keeps the 16 E_DIV positive draws over the original 105 positives (identical exposure for tailgating and stalled), plus 16 draws over the new weak-type positives.
  - The same 48 normals per batch, with new controls in the normal pool.
  - This changes only the sampler. The pos/neg ratio changes, but the threshold is re-derived on val_B anyway.
- **Checkpoint selection:** best AP per seed on the **expanded validation** (old val + 48 new anomalies + new val controls). This is the same AP rule as E_DIV.
- **Ensemble and threshold:** per-seed z with that arm's own val_B constants; ensemble mean; P99 on val_B normals. **No threshold copied from E_DIV.**

## Evaluation and selection rule (to be frozen before training)

**All metrics on validation data only**, for E_DIV (frozen, scored read-only), W1 and W2 alike:
- the expanded val;
- stallval (18 stalls + legitimate stops);
- stopval2;
- tailgating val v3 (12) and its controls;
- the 8/9 development cases.

**Reported:** macro recall over 9 types, per-type recall, single-seed recall spread, precision / F1, FA on val normal driving, FA on legitimate stops, FA on tailgating controls, decision coverage, ROC-AUC and AUPRC.

**A candidate passes only if all of these hold:**
1. **Weak types:** mean recall over the six weak types (8 val each) ≥ E_DIV + 0.15, **and** macro recall over the 9 types ≥ E_DIV + 0.08.
2. **No type loses more than 1 vehicle** vs E_DIV on its validation set.
   - Tailgating val ≥ E_DIV − 1.
   - Stallval stalls ≥ 17/18.
   - The original stalled case is detected.
3. **Development cases:** the 8/9 is kept (≥ 8/9).
4. **False alarms:**
   - val normal driving ≤ 1.5 % (E_DIV 1.07 %);
   - legitimate stops ≤ E_DIV + 5 percentage points;
   - tailgating-control alarms ≤ E_DIV + 1.
5. **Stability:** the weak-type gain is present in ≥ 3 of the 5 single seeds.

**Choice:**
- If both arms pass, choose the higher val macro recall; on a tie, W1 (simpler).
- **If neither passes, the round ends and E_DIV stays the final model.** No further arm is opened automatically.
- The exposed final test is reported for the candidate only as a secondary "exposed development" table, never used for selection.

## Fresh test (only if a candidate passes)
- **Design:** the same structure as the last final test (228 runs: 9 types, stop study, tailgating with controls, normal scenes), in the test regions.
- **New randomness:** new seeds, start lanes never used before, new backgrounds.
- **Frozen before simulation:** the plan, the evaluation code and both models' hashes.
- **Run once, after selection, never re-run.**
- **Reported:** the candidate vs the frozen E_DIV, with the previous test shown separately.

## Integrity
- **Hash checks after the diagnosis:**
  - 8/9 manifest 41/41 OK.
  - Final-test freeze manifest 57/58 OK. The one mismatch is the plan file with the disclosed erratum, as before.
  - Evaluation-code manifest 3/3 OK.
  - E_DIV checkpoints and FROZEN_CANDIDATE unchanged.
- **No file outside `weak_types_diag_20261010/` was created or modified.**

## Decisions needed from you
1. Approve the expansion sizes (24 train + 8 val per weak type, plus the controls).
2. Approve segment events with an event-window realisation rule, or keep whole-route events only.
3. Approve the two arms (W1 data-only, W2 exposure-preserving sampler), or only W1.
4. Approve the selection rule and its thresholds.
