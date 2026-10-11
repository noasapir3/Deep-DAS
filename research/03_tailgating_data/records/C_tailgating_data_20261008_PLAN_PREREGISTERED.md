# Controlled experiment: C without vs C with the new tailgating training data — pre-registered
Written 2026-10-08, before any training of this experiment. The frozen reference `DL_supervised_crossregion_P99` is never modified.
Every run is delivered into new copies, with the original's file hashes checked before and after.

## Question
Do the new tailgating training examples (tg3 train + v4 speed-varied tailgaters + v4 matched normal controls) improve version C,
everything else being equal?

## Arms (the only difference is the training data)
| | C_old | C_new |
|---|---|---|
| Model / preprocessing | version C: artifact cleaning + `tg` feature (definitions unchanged) | identical |
| Recipe | p12 recipe (seed-dependent head init and batch order, encoder from the amplitude-only checkpoint, 25 epochs, patience 6, selection by validation AP) | identical |
| **Training anomalies** | the 34 cross-region anomalies | the same 34 + tg3 train tailgaters + v4 realized tailgaters |
| **Training normals** | train corpus + cross-region train backgrounds | the same + unaffected tg3/v4 train backgrounds + v4 clean matched controls |
| **Validation (selection)** | val_A corpus normals + cross-region val backgrounds + 17 cross-region val anomalies + **tg3 val tailgaters (12) and their unaffected backgrounds** | identical |
| Calibration | val_B sufficient normals | identical |
| Seeds | 0, 1, 2, 3, 4 | 0, 1, 2, 3, 4 |

**Validation change, applied to both arms:** the tg3 tailgaters are added to validation in both arms, so selection can see tailgating.
For that reason C_old is retrained and the earlier C seeds are not reused.
**No validation, val_B, test, newscene or diagnostic vehicle is ever used for training.**

## Ensemble (fixed now)
- Per seed s: z_s = (logit_s − μ_s) / σ_s, where μ_s and σ_s are the mean and std of that seed's logits on the 1,242 sufficient-evidence val_B normals.
- Ensemble score = mean of z_s over the 5 seeds.
- **Thresholds of the ensemble** = P99 (primary) and P95 (alternative) of the ensemble score itself on the same val_B normals. No single-model threshold is copied.
- Quality rule unchanged: fewer than 3 windows means no decision.

## Candidate selection rule and false-alarm limit (validation data only; fixed now)
- **FA limit:** at its P99 threshold, an ensemble may flag at most **1.5 %** of the sufficient-evidence val_A corpus normals (≤ 19 of 1,313).
- **Candidate:** among the two ensembles that meet the FA limit, the one with the higher validation recall at P99 (17 cross-region + 12 tg3 val
  anomalies). Ties go to the higher validation AP. If neither meets the limit, there is no candidate and the frozen reference stays.
- **Disclosed:** the validation anomalies are also used for epoch selection inside each run, so validation recall is somewhat optimistic for both arms alike.
- The 9 valid cases, newscene and the test normals are reported for every seed and for the ensemble as **diagnostics only**. They take no part in the choice.

## Reported (per seed and ensemble, both arms)
- Detections of the 9 valid cases, C3, weaving.
- False alarms: test (513) and val_A (1,313).
- Precision / recall / F1 (9 valid vs 513 test normals).
- Validation AP and recall, overall and per type (tailgating separately).
- Newscene per type (diagnostic).

## End point (fixed now)
1. This experiment ends with one frozen candidate, or none.
2. The candidate (if any), the frozen reference `DL_supervised_crossregion_P99` and the shipped Z-score are evaluated **once** on a sealed final holdout,
   built only after the candidate is frozen, from geography never used for training, selection, calibration, normalization statistics or diagnosis.
3. After that single evaluation **the model work stops.** No model, threshold or preprocessing change follows from the holdout result.
   Every result is reported, including worse ones.
