# Last improvement round: W1 / W2 for the six weak anomaly types. Plan for approval (draft, 2026-10-10)

**Status:** a draft for Noa's approval. Nothing has been simulated or trained.
- The approved final model stays **E_stall_diversity_ENSEMBLE**, frozen and backed up.
  - Backup: `DL/FROZEN_FINAL_MODEL__E_stall_diversity_ENSEMBLE__20261010/`. Restore test: ALL_REPRODUCED.
- This round can replace it **only** if a candidate passes the rule in §6 on validation and then the new independent test in §8.
- **This is the last improvement round.** It has exactly two arms, W1 and W2. No third arm and no architecture change without Noa's explicit decision.

## 0. Separation from the frozen model
- **Work folders (new, separate):**
  - container: `experiments_followup/weak_types_W1W2_20261010/` with `plan/`, `scripts/`, `sims/`, `data/`, `runs/W1_s*`, `runs/W2_s*` and `results/`;
  - computer: `DL/diagnostics/weak_types_W1W2_20261010/`.
- **Frozen components are only read.**
  - The new training script is a copy of `x_train_eval_s.py`, changed only as stated in §5.
  - The frozen E_DIV is scored read-only through `hs_scorer.py`.
- **Hash guard:** every stage (generation, import, training, evaluation) starts by re-hashing all 5,414 source files listed in `SHA256_SOURCE_vs_BACKUP.tsv`.
  - It aborts on any change.
  - The guard is re-run at the end of the round, and the result is reported.
- **Results of the round** go only into this round's folders. `MODEL_RECORD.md` of the frozen model is never edited.

## 1. What is expanded and why (from the diagnosis, `weak_types_diag_20261010`)
The information is in the inputs and in the encoder (probe AUC 0.81–1.0). The losses come from a head trained on **4 single-design examples per type**, plus small margins.

| Type | Main gap found | Expansion focus |
|---|---|---|
| Overweight | head boundary learned from 3 high-amplitude examples | weights and DAS/sensing conditions; legal heavies as controls |
| Straddling | one offset (1.8 m); on-the-line map-matching flips | offsets, partial events, routes |
| Weaving | one design at minimum severity | amplitude × period, partial events |
| Collision | one design; part of the signal is the post-impact stop | impact speed, target moving vs stopped |
| Speeding | 2 of 4 examples extreme (×2.4); normal generator tail overlaps ×1.1–1.25 | ×1.30–1.80, partial events; fast-legal controls |
| Hard braking | 4 identical brakes at t = 6 s; a_lon noise; looks like a legitimate stop | decel × shape × timing; firm-but-legal braking controls |

**Kept unchanged in both arms:**
- all E_DIV training data: normal corpus, cross-region batch, tailgating v3/v4 with controls, stall_div stalls, legitimate stops and controls;
- the original validation sets.

## 2. New data: runs and split

**Regions (existing cross-region split, unchanged):**
- **train** runs only in the 98 *train* regions;
- **validation** runs only in the 21 *val_A* regions;
- *val_B* (21 regions) stays normals-only, for the thresholds, and gets no new runs;
- the 8 *test* regions are **not used** in this round. They are reserved for the new test.

**Independence (checked before freezing, like the last test):**
- new `sim_seed` and `bg_seed` ranges, disjoint from all earlier ones;
- background traffic regenerated per run (`populate_vehicles`, new seed);
- injection start lanes never used before in that region;
- md5 of each scenario differs from every earlier scenario.

### 2a. Anomaly runs (one injected vehicle per run, plus backgrounds)

| Type | Train | Val | Grid (balanced) | Whole route / partial (train; val) |
|---|---|---|---|---|
| Hard braking | 24 | 8 | decel 5 / 6 / 7 / 8 m/s² × shape {partial: 40–70 % speed drop then continue; to near-stop then go} × event time {6, 15, 30 s} | event by nature; event window = the planned brake ± 2 s |
| Speeding | 24 | 8 | ×1.30 / 1.45 / 1.60 / 1.80 limit; light → heavy traffic | 12 / 12; 4 / 4 (segment ≥ 15 s) |
| Overweight | 24 | 8 | 20 / 25 / 30 / 40 t × sensing {DAS-only, DAS+camera, sparse} × speed {0.7, 1.0}× limit | whole route |
| Straddling | 24 | 8 | offset 1.7 / 1.8 / 1.9 / 2.1 m | 12 / 12; 4 / 4 (segment ≥ 20 s) |
| Collision | 24 | 8 | impact rel. speed 5 / 8 / 12 m/s × target {stopped, moving 2–4 m/s}; target 60–130 m ahead | event by nature |
| Weaving | 24 | 8 | amplitude 1.5 / 1.8 / 2.2 m × period 2 / 3 / 4.5 s | 12 / 12; 4 / 4 (segment ≥ 20 s) |

**Measurement-only validation (no training examples):**
- **Reckless:** 8 val runs. Today val has only 2, too few to detect a regression.
- **Tailgating:** 8 val pairs (tailgater + matched control, tg14 tooling). Today val has *no* tailgating controls, so the FA on tailgating controls could not be measured on validation.

**Realisation (truth used for labels only; never as model input):**
- **The p10 thresholds are unchanged:**
  - speed ratio p90 ≥ 1.10;
  - peak decel ≥ 4 m/s²;
  - lateral std ≥ 1.0 m;
  - median \|d_lat\| ≥ 1.7 m;
  - ≥ 1 contact;
  - ≥ 20 t.
- **Partial events (approved in principle):** the event window `[t_start, t_end]` is written in `RUN_PLAN.csv` **before** simulation, from the injection plan.
  - The realisation statistic is computed over that window only, with the same thresholds.
  - No window is searched or chosen afterwards.
  - Whole-route events use the whole route, as before.
- **Failures:** a run that does not realise is excluded and reported, never relabelled or re-run.
- **Reporting:** partial and whole-route events are reported separately.

### 2b. Normal controls (labelled normal; in train and val regions)

| Control | Against | Train | Val |
|---|---|---|---|
| Firm legal braking 2.5–3.5 m/s², partial slowdown and continue | hard braking | 24 | 8 |
| Fast legal driving at 0.95–1.05 × limit | speeding | 16 | 6 |
| Legal heavy vehicles 8–15 t, all sensing modes | overweight | 16 | 6 |
| Normal lateral drift 0.5–0.9 m and normal lane changes | straddling, weaving | 16 | 6 |

- **Realisation check for controls (truth):**
  - braking peak < 4 m/s²;
  - speed ratio p90 < 1.10;
  - weight < 20 t;
  - median \|d_lat\| < 1.7 m and lateral std < 1.0 m.
- A control that crosses its anomaly threshold is excluded and reported.
- Unaffected background vehicles of all new runs are added as normals (unaffected = stop < 2 s and gap ≥ 5 m, as before).

### 2c. Run counts

| | Train | Val | Total |
|---|---|---|---|
| Weak-type anomalies | 144 | 48 | 192 |
| Reckless (measurement) | – | 8 | 8 |
| Tailgating pairs (measurement) | – | 16 | 16 |
| Normal controls | 72 | 26 | 98 |
| **Runs** | **216** | **98** | **314** |

Expected realised: about 95 % (last test 105/108).

**Feasibility step (after approval, before freezing):**
- a planning-only check of start lanes, route lengths and collision geometry;
- one dry-run per new mechanism (segment speed change, segment lateral offset, moving collision target), **discarded and not used** anywhere.
- If a mechanism is not feasible with the simulator, it is dropped and reported *before* the plan is frozen. Example: if a moving target cannot be scripted, collision keeps only the impact-speed grid.

## 3. Validation set used for selection: VAL_W
Validation data only. The 9 development cases and the exposed final test are **never** used for selection.

| Part | Content | Used for |
|---|---|---|
| Anomalies, 9 types | weak six: 2 old + 8 new each (10); reckless 2 + 8 (10); tailgating 13 + 8 (21); stalled: stallval 18 + 2 (20) | recall per type, macro recalls, checkpoint AP |
| Normal driving | val_A normal corpus + new val unaffected backgrounds + new val normal controls | FA normal driving, precision, AUC |
| Legitimate stops | stallval 6 + stopval2 54 = 60 | FA legitimate stops |
| Tailgating controls | 8 new matched controls | FA tailgating controls |
| val_B normals | unchanged | thresholds only |

## 4. Architecture and training recipe (identical in W1 and W2)
- **Same as E_DIV:**
  - architecture (ConvAE encoder → 8-d codes → mean/std pooling + tg feature → MLP head);
  - cleaning = 1, the 14 input channels and the normalisation (`norm.json`);
  - initialisation from the same autoencoder;
  - loss (BCE with logits), optimiser and learning rates (encoder 1e-4, head 1e-3, wd 1e-4);
  - 25 epochs × 120 steps, patience 6;
  - seeds 0–4.
- **Checkpoint per seed:** best AP on VAL_W (the same AP rule as E_DIV, on the expanded validation).
- **Threshold (per arm):**
  - per seed, z = (logit − val_B mean) / val_B std;
  - ensemble = mean of the 5 z;
  - **P99 on the val_B normals.**
- **No threshold is copied from E_DIV.**

## 5. W1 and W2: the only difference is sampling

**W1, data only.** The E_DIV sampler is unchanged: each batch draws 48 normals uniformly from the normal pool and 16 positives uniformly from the positive pool, both pools now including the new data.

**W2, data + exposure preservation.** Each batch = **the E_DIV batch exactly as it was** (48 normals from the *original* normal pool + 16 positives from the *original* 105 positives), **plus** an added sub-batch.
- **16 positives** uniformly from the 144 new weak-type positives.
- **8 normals:** 4 from the 72 new targeted controls and 4 from the new runs' unaffected backgrounds.
- So every original example keeps exactly its E_DIV exposure. The new data are added on top.
- The batch grows from 64 to 88 and the positive share from 25 % to 36 %. Steps, epochs, loss and learning rates are unchanged.
- **The 8-normal part needs your explicit approval.** It is there because the diagnosis showed that controls placed only in the folder are not seen enough (legitimate stops got about 1:105 exposure and were never learned). Without it, each new control is drawn about 20 times during training, against about 330 times for each new positive.

**Expected draws per batch** (positives; normals only for the new controls):

| | E_DIV | W1 | W2 |
|---|---|---|---|
| Tailgating (49 train) | 7.47 | 3.15 | **7.47** |
| Stalled (28) | 4.27 | 1.80 | **4.27** |
| Reckless (4, not expanded) | 0.61 | 0.26 | 0.61 |
| Each weak type (4 old + 24 new) | 0.61 | 1.80 | 3.28 |
| New targeted controls (72) | – | ≈ 0.4 | 4.0 |

Both arms report the realised exposure per type (rng replay), as in the stall round.

## 6. Selection rule (to be frozen with this plan)

**How everything is measured:**
- **Ensemble:** the arm's 5-seed ensemble at its own P99.
- **Seed level:** each single seed at its own frozen per-seed P99 (val_B), compared with the E_DIV seed of the same index at its own frozen per-seed P99.
- **Baseline:** E_DIV (frozen) is scored read-only on exactly the same VAL_W, and that is the reference.
- **Recall per type:** detected / decided realised vehicles.
- **Macro9:** the equal-weight mean over the 9 types.
- **Weak6:** the mean over hard braking, speeding, overweight, straddling, collision and weaving.

**A candidate (W1 or W2) passes only if all of the following hold.**

| # | Condition | Level |
|---|---|---|
| C1 | Weak6(cand) − Weak6(E_DIV) ≥ **0.15** | ensemble |
| C2 | Macro9(cand) − Macro9(E_DIV) ≥ **0.08** | ensemble |
| C3 | For every one of the 9 types: detected(cand) ≥ detected(E_DIV) − 1. In addition: tailgating ≥ E_DIV − 1 of 21; stallval stalls ≥ 17/18 | ensemble |
| C4 | FA on val normal driving ≤ **1.5 %** (E_DIV on val_A: 1.07 %) | ensemble |
| C5 | FA on the 60 legitimate stops ≤ FA(E_DIV) + **3** stops (≈ +5 pp) | ensemble |
| C6 | FA on the 8 tailgating controls ≤ FA(E_DIV) + **1** | ensemble |
| C7 | Weak6(seed k) − Weak6(E_DIV seed k) ≥ **0.10** in **≥ 3 of 5** seeds, and no seed has FA on val normal driving > 3 % | seeds |
| C8 | AUPRC ≥ AUPRC(E_DIV) − 0.02 and ROC-AUC ≥ ROC-AUC(E_DIV) − 0.01 (realised VAL_W anomalies vs val normal driving) | ensemble |
| C9 | Decision coverage equal to E_DIV on the same sets (same window rule; checked, not assumed) | – |

- **Reported, not gating:** precision, F1, recall by partial vs whole-route events, by severity and by sensing; per-seed spreads; the 8/9 cases and the exposed final test, each labelled as development/exposed.
- **Choice:**
  - if both pass, the higher Macro9; if within 0.02, the higher AUPRC; if still tied, W1 (simpler);
  - **if neither passes, E_stall_diversity_ENSEMBLE stays the final model** and development closes.

## 7. Order of work after approval
1. Feasibility check (§2c). Report anything dropped.
2. Freeze: this plan, `RUN_PLAN.csv`, the event windows, the selection code and the training-script diff (sha256 manifest).
3. Simulate, import (unchanged pipeline), realise (truth labels only), run the independence check. Report realisation per type.
4. Score frozen E_DIV read-only on VAL_W (baseline).
5. Train W1 and W2 (5 seeds each).
6. Evaluate, apply §6 once, and report.
7. Run the hash guard at the end.

## 8. New independent test (only if a candidate passes)
- **Plan:** presented for approval and frozen before simulating.
- **Design:** the 8 test regions and the same 228-run design as the last test (for comparability), plus a separately reported block with partial events and the new severity ranges.
- **New randomness:** new seeds, start lanes never used, new backgrounds. No run is used for training, checkpoints or thresholds.
- **Run once:** the candidate vs the frozen E_DIV on that test, never re-run. The previous test is shown separately and is not used for the decision.
- **Closing:** after this test (or after §6 if nothing passes), model development closes.

## Decisions needed
1. Approve the data plan: 314 runs, the grids, and the controls in §2.
2. Approve W2 as defined in §5, including the 8-normal sub-batch (or W2 with positives only).
3. Approve the selection rule C1–C9 and its numbers.
4. Approve adding the measurement-only validation runs: 8 reckless and 8 tailgating pairs.
