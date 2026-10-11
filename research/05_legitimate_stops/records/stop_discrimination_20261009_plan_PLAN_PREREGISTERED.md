# Legitimate stop vs abnormal stop — F1 (controlled exposure) and F2 (temporal branch). Pre-registered.
Written 2026-10-09, before the stopval2 survey, before any simulation and before any training of this round.

## Fixed and read-only
These are not modified, and their fingerprints are checked before (done) and after:
- `DL_supervised_crossregion_P99`;
- C_new_ENSEMBLE and its freeze files;
- the selected candidate **E_stall_diversity** (`FROZEN_CANDIDATE_E_diversity.json`, its 5 checkpoints and thresholds).

New work goes only to `stop_discrimination_20261009/` and to the device copies `DL_supervised_crossregion_P99__F1_legit_stop_exposure` and `__F2_temporal_branch`.
**The final holdout is not generated or opened.**
Definitions are unchanged:
- **Legitimate stop:** a truth stop of < 10 s with peak deceleration < 4 m/s².
- **Abnormal stall:** a truth stop of ≥ 10 s.
An alarm on a legitimate stop is a false alarm.

## Audit of the existing development stops (`data/audit_existing_stops.csv`, done before this plan)
- **Two duration levels only.** Legitimate stops are 3 or 5 s; abnormal stops are ≥ 30 s (MSM) or the whole run (START / STOP_END).
- **The stop shares barely overlap.** Legitimate stops cover 0.08–0.38 of the windows, MSM covers 0.31–0.73.
- **The approach differs by class.** Every abnormal stop in the training data uses the instant freeze, and every legitimate stop uses 2 m/s² braking.
- **Consequence:** with these data, duration, stop share and the approach signature are confounded. They cannot show what a model uses.
- **Response:** the training data stay as they are (F1/F2 change only sampling and the head). The **validation** set is expanded so that it separates these factors (next section).
- **Observation limit:** for START / STOP_END the observation never covers the whole truth stop (burn-in at the start, recording end). These cases are reported as observation-limited and never relabelled.

## Expanded independent validation set "stopval2"
- **Regions:**
  - pool: val_A regions, excluding 002, 005 and the tg3 val regions (never training regions);
  - eligibility: a SHORT and a LONG observation slot;
  - selection: 12 regions drawn with numpy rng 20261011.
- **Slots per region:**
  - **SHORT:** the planned route ends at a boundary exit 60–150 m after the stop, so the vehicle leaves the map soon after.
  - **LONG:** ≥ 350 m of route after the stop.
  - **Slot choice:** the shortest post-stop route for SHORT, the longest for LONG (ties by lane id and s).
  - **Stop location:** on a DAS segment; 6–14 s of driving before the stop. The sensing condition is recorded (COMBINED / DAS_ONLY / PARTIAL).
- **Runs per slot (5), simulation seed 1, the same simulator and tracker:**

| mode | behaviour | label (project definition) |
|---|---|---|
| CONTROL | drives through | normal (verified only if longest truth stop < 2 s) |
| LEGIT_STOP 3 s | 2 m/s² braking to a stop, hold 3 s, resume | normal (verified: stop 1–10 s, decel < 4, moves after) |
| LEGIT_STOP 8 s | the same, hold 8 s | normal (same rule) |
| GENTLE_STALL 15 s | **the same 2 m/s² braking**, hold 15 s, resume | abnormal stall (verified: stop ≥ 10 s, decel < 4, moves after) |
| MSM 30 s | instant freeze, 30 s, resume (the existing pattern) | abnormal stall (the ST01 rule) |

- **What this gives:** 24 slots → 48 legitimate stops, 48 abnormal stops and 24 controls.
  - The same duration is observed in SHORT and LONG tracks.
  - Different durations end up with similar stop shares, e.g. 8 s in SHORT vs 15 s in LONG.
  - LEGIT 8 s vs GENTLE 15 s differ only in hold duration: same braking, same location, same scene.
- **Not filtered.** Unverified runs are excluded and reported, never relabelled. Tracking, ≥ 3 windows, observed span (from timestamps) and whether the observation covers the whole truth stop are reported per vehicle.
- **The model input is the tracker output only.** Ground truth is used only for labels and verification.
- **Evaluation pool:** the 6 stallval legitimate stops and 18 stallval abnormal stops are pooled with stopval2.
  - Legitimate stops: 54.
  - Abnormal stops: 66 (stallval START / STOP_END / MSM plus stopval2 GENTLE / MSM).

## Common to both arms (unchanged from E_stall_diversity)
- **Data and recipe:** E_stall_diversity's training data (NEWDATA = 1 + STALLARM = diversity), the initial AE checkpoint, seeds 0–4.
- **Training setup:** clean = 1, feature tg, 48 normals + 16 anomalies per batch, the same uniform anomaly draw, 120 steps, 25 epochs, patience 6, the learning rates (encoder 1e-4, head 1e-3), weight decay, BCE.
- **Checkpoint selection:** validation AP on the same original validation set.
- **Pooled path:** 32 random windows per vehicle in training, mean/std pooling, the same standardization.
- No loss weights, no new hand-made features, no tracker change.

## F1 — controlled exposure to legitimate stops (the only change: normal sampling)
- **Per batch:** 4 of the 48 normal slots are drawn uniformly (with replacement) from the 23 training legitimate stops. The other 44 are drawn uniformly from all other training normals (corpus, tailgating controls, stall-batch controls and backgrounds).
- **Why 4:** the stall anomalies currently get ≈ 16 × 28/105 = 4.3 slots per batch, so this matches the per-vehicle exposure of the two stop classes.
  This is a design choice made before any result, not a tuned value.
- **Expected effect on exposure:**
  - Legitimate stops: ≈ 480 draws per epoch, instead of ≈ 20.
  - Each legitimate-stop vehicle: ≈ 21 draws per epoch.
  - **Disclosed risk:** memorization of 23 vehicles (compare EXP1).
- **Reported:** actual draws and actual stop-overlapping windows drawn, by group (exact rng replay), with no window-level 1:1 balancing.

## F2 — the F1 sampling plus a small temporal branch in the head
- **Same data and the same F1 vehicle sampling.** The pooled path is unchanged.
- **Temporal branch:**
  - **Input:** the window codes of **one real contiguous run of windows of one track**, in time order. Per window step it receives:
    - the 8 codes;
    - the share of valid bins in the window (a missing-data mask);
    - the time since the previous window start, from the bin timestamps (normally 0.8 s, 0 for the first). Nothing is compressed.
  - **Layers:** Conv1d(10→16, k 5, dilation 1) → GELU → Conv1d(16→16, k 5, dilation 2) → GELU → Conv1d(16→16, k 5, dilation 4) → GELU → max over time.
  - **Receptive field:** 29 consecutive windows = 28 × 0.8 s + 6.4 s ≈ **28.8 s** of track time. This is enough to tell a < 10 s stop from a ≥ 10 s and a 30 s stop.
  - **No rule on duration.** The branch receives only codes, the mask and Δt.
- **Head:** Linear(16 pooled + 1 tg + 16 temporal → 16) → GELU → Linear(16 → 1). The new parameters are initialised from the seed.
- **Training:**
  - Pooled path: as before (32 random windows).
  - Temporal path: one random contiguous crop of up to 48 windows (≈ 44 s) from one of the vehicle's tracks, picked with probability proportional to its windows.
  - Random windows are never concatenated, and different tracks are never joined.
- **Evaluation:** the branch runs over all windows of every track; the maximum over time and over tracks is used.
- **Disclosed:**
  - The train crop (≤ 48) differs from the evaluation length (all windows). Max-pooling makes the feature monotone in the length.
  - A track shorter than the stop cannot show the duration; this is reported as an observation limit.

## Thresholds, ensemble, reference
- Each seed and each ensemble gets its own P99 / P95 on the 1,242 sufficient-evidence val_B normals.
- **Ensemble:** the C_new method — the mean of the per-seed logits standardized with that seed's val_B statistics.
- **Reference:** the frozen E_stall_diversity seeds, re-scored read-only on stopval2. The existing groups are checked against their saved scores.

## Reported separately (per seed, mean ± std, ensemble)
- **Legitimate stops:** the 3 s / 5 s / 8 s alarm rates, SHORT vs LONG, by stop-share bin.
- **The three abnormal stop modes** (START, STOP_END, MSM) plus GENTLE_STALL 15 s.
- **Tailgating**, and the other anomaly types on the original validation set.
- **FA** on val_A and test.
- **Decision coverage.**
- **macro9:** the mean of 9 equal-weighted type recalls. The stall entry pools the original validation stalls with all abnormal stops of stallval + stopval2.
- **Stop-share control (reported, not gating):** the alarm rate of legitimate vs abnormal stops within matched stop-share bins (0–0.2, 0.2–0.4, 0.4–0.7).
- **Diagnostics only:** the 9 valid cases and newscene, scored after the checkpoint and thresholds are fixed.

## Replacement rule (fixed now; reference = the E_stall_diversity ensemble on the same sets)
An arm's ensemble replaces E_stall_diversity only if **all** hold, at its own ensemble P99:
1. **Legitimate-stop false alarms:**
   - the alarm count on the 54 pooled legitimate stops is ≤ half of E_div's count;
   - and the alarm rate is ≤ 20 %.
2. **Abnormal stops kept:** recall on the 66 pooled abnormal stops ≥ E_div's minus 2 vehicles, and GENTLE_STALL recall ≥ E_div's minus 2 vehicles.
3. **Other anomalies kept:**
   - detected original-validation anomalies of the 8 non-stall types ≥ E_div's minus 1 vehicle;
   - tailgating ≥ 12/13;
   - macro9 ≥ E_div's − 0.03.
4. **Normal corpus:** FA on val_A ≤ 1.5 %.

- **If both arms pass:** the fewer legitimate-stop alarms wins; ties go to higher abnormal-stop recall, then to val AP.
- **Otherwise** E_stall_diversity stays.

**End point:**
- After F1 and F2 are reported, the work stops.
- No third experiment, no combination with other changes, and no threshold change.
- The 6 stops analysed yesterday are only part of the pool. A result is not reported as a success on them alone.
