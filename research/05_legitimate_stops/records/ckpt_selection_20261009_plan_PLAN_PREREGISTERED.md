# Checkpoint selection: AP vs a task rule — pre-registered
Written 2026-10-09, before any rerun or per-epoch score of this check.

## Fixed and read-only
- **Not modified:** every existing model, threshold, result and freeze file (`DL_supervised_crossregion_P99`, C_new_ENSEMBLE, E_stall_diversity, F1, F2).
- **Fingerprints:** checked before and after.
- **Candidate:** E_stall_diversity stays the candidate unless the rule below says otherwise.
- **Holdout:** closed.

## What was saved
In every earlier run, only `checkpoint_best.pt` (best validation AP) exists, overwritten during training, plus `history.csv` (AP per epoch).
The behaviour of the other epochs on stops cannot be reconstructed from the log. The three relevant recipes are therefore **replayed**:
- REF = E_stall_diversity;
- F1;
- F2.

## Replay (identical recipes; only additions)
- **Script:** `x_train_eval_ck.py` = `x_train_eval_f.py` plus:
  - a checkpoint after every epoch;
  - the per-epoch logits of the original validation set and of the stallval + stopval2 stop vehicles (no-grad, eval mode, no random numbers).
- **Unchanged:** data, exposure/sampling, architecture, learning rates, epochs, patience, NTHREADS = 1 and seeds 0–4.
  The training budget, including the existing early stopping, is unchanged, so only the epochs that the original runs trained are candidates.
- **Reproduction check:** each replay's AP-selected checkpoint must be bit-identical to the saved original (`checkpoint_best.pt`). Any mismatch is reported, and that run is not interpreted.

## Data roles (development data only; no val_B, the 9 cases, newscene or holdout)
- **Temporary calibration normals (CAL):**
  - the normal-corpus vehicles of the original validation set from 11 of the 21 val_A regions (numpy rng 20261012 permutation, the first 11);
  - per checkpoint, the temporary threshold is the P99 of their logits (≥ 3 windows).
- **Selection set (SEL):**
  - the 29 original validation anomalies (9 types);
  - the normal-corpus vehicles of the other 10 val_A regions (regular-normal FA);
  - the 54 legitimate stops (3 / 5 / 8 s) and 30 controls;
  - the 66 abnormal stops of stallval + stopval2.
- **val_B:** used only after selection, for the final per-model P99 and the ensemble standardisation (the existing method).
- **Disclosed:** stallval / stopval2 now take part in selection, so they are no longer independent evidence of generalisation.

## Task-based selection rule (per run; all quantities at the CAL P99 threshold; ≥ 3 windows)
1. **Feasibility (all must hold):**
   - regular-normal FA on SEL ≤ 1.5 %;
   - tailgating ≥ 12/13;
   - abnormal-stop recall ≥ 0.95 (≥ 63/66).
2. **Objective:** J = (9 × macro9 + (1 − legitimate-stop alarm rate)) / 10.
   - macro9 is the equal-weight mean of the 9 type recalls; the stall entry pools the 2 original validation stalls with the 66 abnormal stops.
   - Correct handling of legitimate stops counts as a tenth term of equal weight.
3. **Tie-breaks, within 0.005 of the best J:** the lower legitimate-stop alarm rate, then the lower regular-normal FA, then the earlier epoch.
4. If no checkpoint is feasible, the AP checkpoint is kept, and this is reported.

## Comparison and replacement
- **Per seed, side by side:** the AP checkpoint vs the J checkpoint — the epoch, the recall per type, the legitimate-stop alarms (3 / 5 / 8 s), the regular-normal FA and the abnormal-stop recall.
- **Final evaluation:**
  - each selected checkpoint is fully re-scored (EVAL_ONLY);
  - its own val_B P99 is used;
  - the ensemble combines **all 5 seeds** of a recipe with the existing standardised-mean rule and its own val_B P99.
- **Replacement:** the J-selected ensembles (REF-J, F1-J, F2-J) are judged with **the unchanged rule of the F round**, against the frozen E_stall_diversity ensemble:
  - legitimate-stop alarms ≤ half of REF's and ≤ 20 %;
  - abnormal stops ≥ REF − 2;
  - GENTLE ≥ REF − 2;
  - non-stall validation detections ≥ REF − 1;
  - tailgating ≥ 12/13;
  - macro9 ≥ REF − 0.03;
  - FA val_A ≤ 1.5 %.
- **Outcome:** if none passes, the report says so, and E_stall_diversity stays.
- **No follow-up:** no further experiment is opened from the answer.

## Launch note
- The first launch failed at start-up (NameError: the stop-vehicle preload was placed before the window index existed). No epoch ran, and the partial folders were removed. The block was moved after the window index; nothing else changed.
- F2 seed 4 was started in parallel on the CPU freed after F2 seeds 0-2 finished (identical command, log logs_F2_s4_parallel.log); the chain refuses to retrain it later (done.json).
