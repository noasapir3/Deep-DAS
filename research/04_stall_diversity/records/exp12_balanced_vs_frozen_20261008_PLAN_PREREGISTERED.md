# Two development experiments after the stalled-vehicle regression — pre-registered (updated work plan)
Written 2026-10-08, after the read-only diagnosis (`stalled_regression_diag_20261008/`) and before any training of these experiments.

**Unchanged and read-only:** `DL_supervised_crossregion_P99` (frozen reference) and the frozen candidate C_new_ENSEMBLE with its freeze files
(`FROZEN_FINAL_CHOICE.json`, its 5 checkpoints, thresholds, plan and selection). Fingerprints are checked before and after.
Every write, training and result goes into the new copies `DL_supervised_crossregion_P99__exp1_balanced_type_sampling` and
`DL_supervised_crossregion_P99__exp2_frozen_encoder_head_only`.
**The final holdout is not generated or opened in this step.**

## What the diagnosis established (motivation only, not tuned on)
- The original stalled vehicle never moves (truth). The tracker gives it a noisy, non-zero speed, so its input never looks like a clean stop.
  The training stalls move, stop for 30 s and move again.
- It was already fragile before the new data: C_old 2/5 seeds, C_new 1/5.
- In C_new, the other eight types were drawn about 2.4× less often per epoch: 94 instead of 226 draws per type, while tailgating took 60.5 % of the anomaly draws.
- Before supervision the initial AE encoder placed this vehicle at the 97th percentile of the training-stall direction. After supervised fine-tuning it falls to 20–97 depending on the seed.

## Reference
The 5 saved C_new runs, seeds 0–4. They are not retrained. They were produced by `x_train_eval_c.py` with NEWDATA = 1 on the same data files,
and the experiment scripts differ from it only by the lines in `DIFF_exp1_vs_Cnew.txt` and `DIFF_exp2_vs_Cnew.txt`.

## Experiment 1 — type-balanced anomaly sampling (only change)
- Each anomalous slot of a batch first draws an anomaly type uniformly over the 9 training types, then a vehicle uniformly within that type.
  This gives an expected 1,920 / 9 ≈ 213 draws per type per epoch.
- **Unchanged:** data, features, 48 normals + 16 anomalies per batch, 120 steps, learning rates, weight decay, loss, epochs, patience,
  checkpoint selection by validation AP, the initial AE checkpoint and the window subsampling. No loss weights, no new features.
- Type labels are used for sampling and analysis only. They are never a model input.

## Experiment 2 — frozen encoder, head-only training (only change)
- Start from the same initial AE checkpoint as C_new (`amplitude_only_20260927_203146/model/checkpoint_best.pt`), not from a trained C_new model.
- **Frozen:** the encoder `enc` and the latent map `to_z` (requires_grad False, excluded from the optimizer, kept in eval mode).
  The encoder has no BatchNorm or Dropout, so there are no running statistics to change.
  After training, the encoder in the saved checkpoint is verified to be bit-identical to the initial AE checkpoint (recorded in `done.json`).
- **Head:** same architecture, learning rate 1e-3 and weight decay 1e-4. The feature standardization (`fmu`, `fsd`) comes from the initial encoder, as in C_new.
- C_new's original sampling (not the Experiment 1 balancing), same data, features, pooling and checkpoint rule.

## Common
- Seeds 0–4 and the same splits as C_new.
- **Ensemble** (same rule as C_new): mean of the per-seed logits standardized with each seed's val_B normals.
- **Thresholds:** each new model and each ensemble gets its own P99 (primary) / P95 on the 1,242 sufficient-evidence val_B normals.
  No numeric threshold is copied, and none is chosen from the stalled vehicle, C3 or weaving.
- **Reported per seed, as mean ± std and for the ensemble:**
  - validation recall per type (9 types);
  - validation macro recall (equal weight per type) and validation AP;
  - false alarms on val_A (1,313) and test (513);
  - decision coverage (share with ≥ 3 windows);
  - diagnostic only: the 9 valid cases per type and newscene per type.

## Candidate rule (validation only; fixed now)
A new ensemble (EXP1 or EXP2) replaces C_new_ENSEMBLE as the development candidate only if **all** of:
1. FA on val_A at its own P99 ≤ 1.5 % (the existing limit);
2. validation macro recall at P99 strictly higher than C_new_ENSEMBLE's;
3. validation tailgating recall at P99 no lower than C_new_ENSEMBLE's minus 1 vehicle (≥ 11/13).

If both qualify, the higher macro recall wins; ties go to the higher validation AP. Otherwise C_new_ENSEMBLE stays.
- **Disclosed:** validation has only 2 vehicles per non-tailgating type, so per-type validation recall moves in steps of 0.5, and the validation
  anomalies also choose the epoch.
- The 9 valid cases and newscene stay diagnostic. The two changes are **not** combined automatically, and **no third experiment** follows from these results.
