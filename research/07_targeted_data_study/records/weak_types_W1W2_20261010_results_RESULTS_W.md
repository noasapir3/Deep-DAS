# W1 / W2: last improvement round. Results on validation (2026-10-10)

Frozen plan: `plan/PLAN_W1_W2_FROZEN.md`. Evaluation: `scripts/w07_evaluate.py` (frozen before scoring). Decision: `results/SELECTION_W.json`.
All numbers are on validation data (val_A regions + stallval / stopval2). The 9 development cases and the previous final test are not used.

## 1. Recall per anomaly type (ensemble at its own P99; seed range in brackets)

| Type | n | E_DIV | W1 | W2 |
|---|---|---|---|---|
| Hard braking (weak) | 9 | 2 = 0.22 [0.11–0.33] | 6 = 0.67 [0.33–0.56] | 7 = 0.78 [0.22–0.67] |
| Speeding (weak) | 9 | 2 = 0.22 [0.11–0.33] | 7 = 0.78 [0.67–0.78] | 7 = 0.78 [0.56–0.89] |
| Overweight (weak) | 10 | 2 = 0.20 [0.10–0.20] | 2 = 0.20 [0.10–0.20] | 3 = 0.30 [0.10–0.30] |
| Lane straddling (weak) | 9 | 3 = 0.33 [0.22–0.33] | 8 = 0.89 [0.67–0.89] | 8 = 0.89 [0.56–0.89] |
| Collision (weak) | 10 | 9 = 0.90 [0.60–1.00] | 10 = 1.00 [1.00–1.00] | 10 = 1.00 [1.00–1.00] |
| Lane weaving (weak) | 10 | 2 = 0.20 [0.10–0.40] | 7 = 0.70 [0.50–0.70] | 7 = 0.70 [0.40–0.70] |
| Reckless | 10 | 8 = 0.80 [0.80–0.80] | 10 = 1.00 [0.90–1.00] | 10 = 1.00 [0.90–1.00] |
| Tailgating | 21 | 20 = 0.95 [0.90–1.00] | 20 = 0.95 [0.81–0.95] | 21 = 1.00 [0.90–1.00] |
| Stalled | 20 | 20 = 1.00 [1.00–1.00] | 20 = 1.00 [0.95–1.00] | 20 = 1.00 [1.00–1.00] |
| **Weak6** (mean of 6 weak types) | | **0.346** [0.28–0.35] | **0.706** [0.58–0.65] | **0.741** [0.54–0.74] |
| **Macro9** (all 9 types) | | **0.537** [0.49–0.54] | **0.798** [0.68–0.76] | **0.827** [0.68–0.83] |

## 2. Performance and false alarms (ensemble)

| Metric | E_DIV | W1 | W2 |
|---|---|---|---|
| Precision | 0.716 | 0.811 | 0.788 |
| F1 | 0.670 | 0.822 | 0.823 |
| ROC-AUC (anomalies vs normal driving) | 0.868 | 0.962 | 0.962 |
| AUPRC | 0.697 | 0.864 | 0.877 |
| FA normal driving | 27/2382 = 1.13 % | 21/2382 = 0.88 % | 25/2382 = 1.05 % |
| FA legitimate stops (1–10 s) | 40/54 | 45/54 | 52/54 |
| FA tailgating matched controls | 6/7 | 0/7 | 2/7 |
| Stallval stalls detected | 18/18 | 18/18 | 18/18 |
| stopval2 abnormal stops (reported only) | 47/48 | 48/48 | 48/48 |
| Decision coverage A | 108/110 | 108/110 | 108/110 |
| Decision coverage N_drive | 2382/3105 | 2382/3105 | 2382/3105 |
| Decision coverage L | 54/54 | 54/54 | 54/54 |
| Decision coverage TC | 7/8 | 7/8 | 7/8 |

## 3. Stability across seeds (each seed at its own val_B P99)

| Model | Seed | Weak6 | Macro9 | Tailgating | Stalled | FA normal driving | FA legit stops | FA tg controls | AUPRC |
|---|---|---|---|---|---|---|---|---|---|
| E_DIV | 0 | 0.344 | 0.541 | 1.00 | 1.00 | 1.09 % | 38/54 | 6/7 | 0.666 |
| E_DIV | 1 | 0.344 | 0.535 | 0.95 | 1.00 | 1.39 % | 38/54 | 5/7 | 0.677 |
| E_DIV | 2 | 0.280 | 0.492 | 0.95 | 1.00 | 1.22 % | 43/54 | 4/7 | 0.666 |
| E_DIV | 3 | 0.348 | 0.533 | 0.90 | 1.00 | 0.59 % | 36/54 | 3/7 | 0.692 |
| E_DIV | 4 | 0.293 | 0.501 | 0.95 | 1.00 | 1.05 % | 38/54 | 7/7 | 0.669 |
| W1 | 0 | 0.652 | 0.763 | 0.95 | 1.00 | 1.13 % | 34/54 | 1/7 | 0.849 |
| W1 | 1 | 0.631 | 0.738 | 0.90 | 0.95 | 0.84 % | 38/54 | 0/7 | 0.832 |
| W1 | 2 | 0.581 | 0.683 | 0.81 | 0.95 | 0.97 % | 31/54 | 0/7 | 0.816 |
| W1 | 3 | 0.650 | 0.756 | 0.90 | 1.00 | 0.92 % | 35/54 | 0/7 | 0.835 |
| W1 | 4 | 0.650 | 0.756 | 0.90 | 1.00 | 0.50 % | 40/54 | 1/7 | 0.843 |
| W2 | 0 | 0.541 | 0.683 | 1.00 | 1.00 | 0.76 % | 36/54 | 0/7 | 0.842 |
| W2 | 1 | 0.741 | 0.827 | 1.00 | 1.00 | 1.64 % | 50/54 | 1/7 | 0.863 |
| W2 | 2 | 0.546 | 0.687 | 0.90 | 1.00 | 0.38 % | 39/54 | 0/7 | 0.861 |
| W2 | 3 | 0.650 | 0.750 | 0.95 | 1.00 | 1.05 % | 44/54 | 0/7 | 0.843 |
| W2 | 4 | 0.667 | 0.772 | 0.95 | 1.00 | 0.97 % | 52/54 | 1/7 | 0.862 |

## 4. Pre-registered selection rule C1–C9 (vs frozen E_DIV on the same data)

### W1: does NOT pass

| Condition | Value | Pass |
|---|---|---|
| C1_weak6_gain_ge_0.15 | `"+0.359"` | ✔ |
| C2_macro9_gain_ge_0.08 | `"+0.262"` | ✔ |
| C3_no_type_loses_more_than_1 | `{"det_diff": {"hard_braking": 4, "speeding": 5, "overweight_vehicle": 0, "lane_straddling": 5, "collision": 1, "lane_weaving": 5, "reckless_driver": 2, "tailgating": 0, "stalled_vehicle": 0}, "stallval_stalls": "18/18"}` | ✔ |
| C4_FA_drive | `{"cand": 0.008816120906801008, "E_DIV": 0.011335012594458438}` | ✔ |
| C5_FA_legit_stops_le_plus3 | `{"cand": 45, "E_DIV": 40, "n": 54}` | ✘ |
| C6_FA_tg_controls_le_plus1 | `{"cand": 0, "E_DIV": 6, "n": 7}` | ✔ |
| C7_seed_stability | `{"weak6_gain_per_seed": [0.3074074074074075, 0.2870370370370369, 0.30185185185185187, 0.3018518518518519, 0.35740740740740745], "tg_stalled_ok_per_seed": [true, true, false, true, true], "FA_drive_per_seed": [0.011335012594458438, 0.008396305625524769, 0.009655751469353484, 0.009235936188077247, 0.005037783375314861]}` | ✔ |
| C8_ranking | `{"AUPRC_diff": 0.1667933002099209, "ROC_AUC_diff": 0.094730540784277}` | ✔ |
| C9_coverage_equal | `{"coverage_A": ["108/110", "108/110"], "coverage_N_drive": ["2382/3105", "2382/3105"], "coverage_L": ["54/54", "54/54"], "coverage_TC": ["7/8", "7/8"], "coverage_val_B": ["1242/1537", "1242/1537"]}` | ✔ |

### W2: does NOT pass

| Condition | Value | Pass |
|---|---|---|
| C1_weak6_gain_ge_0.15 | `"+0.394"` | ✔ |
| C2_macro9_gain_ge_0.08 | `"+0.290"` | ✔ |
| C3_no_type_loses_more_than_1 | `{"det_diff": {"hard_braking": 5, "speeding": 5, "overweight_vehicle": 1, "lane_straddling": 5, "collision": 1, "lane_weaving": 5, "reckless_driver": 2, "tailgating": 1, "stalled_vehicle": 0}, "stallval_stalls": "18/18"}` | ✔ |
| C4_FA_drive | `{"cand": 0.010495382031905962, "E_DIV": 0.011335012594458438}` | ✔ |
| C5_FA_legit_stops_le_plus3 | `{"cand": 52, "E_DIV": 40, "n": 54}` | ✘ |
| C6_FA_tg_controls_le_plus1 | `{"cand": 2, "E_DIV": 6, "n": 7}` | ✔ |
| C7_seed_stability | `{"weak6_gain_per_seed": [0.19629629629629636, 0.3962962962962963, 0.26666666666666666, 0.3018518518518518, 0.37407407407407406], "tg_stalled_ok_per_seed": [true, true, true, true, true], "FA_drive_per_seed": [0.007556675062972292, 0.0163727959697733, 0.003778337531486146, 0.010495382031905962, 0.009655751469353484]}` | ✔ |
| C8_ranking | `{"AUPRC_diff": 0.1800091188227988, "ROC_AUC_diff": 0.09381316665111783}` | ✔ |
| C9_coverage_equal | `{"coverage_A": ["108/110", "108/110"], "coverage_N_drive": ["2382/3105", "2382/3105"], "coverage_L": ["54/54", "54/54"], "coverage_TC": ["7/8", "7/8"], "coverage_val_B": ["1242/1537", "1242/1537"]}` | ✔ |

**Passing candidates:** none  →  **Choice: none -> E_stall_diversity_ENSEMBLE stays the final model**

Bootstrap 95 % CI over validation runs of the gain vs E_DIV (reported only, not part of the rule): W1: ΔWeak6 [+0.18, +0.52], ΔMacro9 [+0.14, +0.37]; W2: ΔWeak6 [+0.23, +0.54], ΔMacro9 [+0.18, +0.39]

## 5. New validation runs: whole-route vs partial events (detected / n; reported only)

| Type | Extent | n | E_DIV | W1 | W2 |
|---|---|---|---|---|---|
| collision | event | 8 | 7 | 8 | 8 |
| hard_braking | event | 7 | 1 | 5 | 5 |
| lane_straddling | segment | 4 | 0 | 4 | 4 |
| lane_straddling | whole | 3 | 1 | 3 | 3 |
| lane_weaving | segment | 4 | 0 | 3 | 3 |
| lane_weaving | whole | 4 | 0 | 3 | 3 |
| overweight_vehicle | whole | 8 | 1 | 2 | 2 |
| reckless_driver | whole | 8 | 7 | 8 | 8 |
| speeding | segment | 3 | 0 | 1 | 1 |
| speeding | whole | 4 | 2 | 4 | 4 |
| tailgating | whole | 8 | 8 | 8 | 8 |

## 6. Conclusion (by the frozen rule)

**Neither W1 nor W2 meets the pre-registered rule, so no candidate is promoted. `E_stall_diversity_ENSEMBLE` stays the final model, and model development closes as agreed.** No new independent test is run.

**Both arms fail one and the same condition: C5, false alarms on legitimate stops.**
- 40/54 for E_DIV.
- 45/54 for W1, +5 where at most +3 was allowed.
- 52/54 for W2, +12.

**Every other condition passes in both arms:**
- large gains on the weak types: Weak6 +0.36 / +0.39 and Macro9 +0.26 / +0.29, with bootstrap 95 % CIs well above 0;
- no type loses a vehicle, and tailgating and stalled are kept;
- fewer false alarms on normal driving and on tailgating controls;
- higher AUPRC / ROC-AUC;
- seed stability;
- identical coverage.

The rule is applied as frozen and is not changed after the fact.

**What the round shows (for the report):**
- The six weak types *can* be learned from targeted, diverse data. The diagnosis said the information is present in the input and the encoder, and that is confirmed:
  - hard braking 0.22 → 0.67 / 0.78;
  - speeding 0.22 → 0.78;
  - straddling 0.33 → 0.89;
  - weaving 0.20 → 0.70;
  - collision 0.90 → 1.00.
- **Overweight barely moved (0.20 → 0.20 / 0.30)**, despite a good representation in the diagnosis. Amplitude overlap with legal heavy vehicles remains the limit.
- **The cost is exactly the trade-off the diagnosis predicted.** Learning hard braking (brake to near-stop or sharp slowdown) and the shorter stops in it pushes the model further towards alarming on short legitimate stops.
  - W2 gave the targeted controls 10× more exposure, but those controls are firm *non-stopping* braking, not 1–10 s legitimate stops. The legitimate-stop examples kept their original low exposure.
  - The trade-off between hard-braking recall and legitimate-stop false alarms is not solved by data volume alone.

## 7. Integrity
- **Hash guard of the frozen model and its backup:** 5,414 originals and 5,422 backup files, 0 differences at every stage (`guard/GUARD_LOG.txt`). Stages: start, before simulation, before training, before evaluation, after evaluation.
- **Data realised:**
  - anomalies 206/208: 2 speeding runs not realised, excluded;
  - controls 91/98: 7 firm-braking controls reached 4.0 m/s² through normal car-following and were excluded by the frozen clean rule;
  - tailgating pairs 8/8.
- **Disclosed incidents (nothing affected any result):**
  - Erratum 1: a missing `numpy` import in `w05_import.py`, fixed before any import output; the code manifest was re-frozen.
  - First training launch without the E_DIV env, stopped by the script's own assertion.
  - W1 seed 0 killed by the OS while loading data (out of memory) and run again unchanged (`LAUNCH_NOTE.txt`).
- **Plan manifest:** `PLAN_W1_W2_FROZEN.md` differs from its manifest hash only because erratum 1 was appended. All other plan and data files match.

## 8. Realised exposure (draws per batch of 64, averaged over seeds and epochs; rng replay logs `runs/*/model/exposure_per_epoch.csv`)

| | Tailgating | Stalled | Reckless | Each weak type (mean) | New targeted controls | Legitimate stops (existing) | Original normals |
|---|---|---|---|---|---|---|---|
| Planned E_DIV | 7.47 | 4.27 | 0.61 | 0.61 | – | ≈ 0.15 | 48 |
| W1 realised | 3.37 | 1.96 | 0.38 | 1.86 | 0.41 | 0.15 | 40.9 |
| W2 realised | 7.54 | 4.31 | 0.69 | 2.02 | 4.01 | 0.11 | 30.9 |
