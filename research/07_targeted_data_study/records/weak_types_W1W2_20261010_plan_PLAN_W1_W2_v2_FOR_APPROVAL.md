# Last improvement round, W1 / W2: plan v2 for approval (2026-10-10)

Supersedes `PLAN_W1_W2_DRAFT_FOR_APPROVAL.md` in §5 (sampling) and §6 (selection rule).
- Everything else in the draft stands as approved in principle: the 314 runs, the grids, the controls, the 8 reckless val runs and the 8 tailgating val pairs, the partial-event rule, the region split and the feasibility step.
- **Nothing is simulated or trained before final approval.**
- `E_stall_diversity_ENSEMBLE` (E_DIV), its backup, the original models and the previous final-test results are read-only.
- New work lives only in `experiments_followup/weak_types_W1W2_20261010/` (on the computer: `DL/diagnostics/weak_types_W1W2_20261010/`).

## §5 (revised): sampling at the same batch size

### Why the first draft's W2 (88 per batch) is withdrawn
- A batch of 88 against 64 means W2 sees 37.5 % more vehicle draws in the same 3,000 steps.
- So W1 vs W2 would mix two changes: composition and training budget.

### Can both arms use 64?
Yes. The constraint is arithmetic:
- E_DIV uses 16 positive slots per batch, and tailgating + stalled already take 11.74 of them.
- Keeping those 11.74 exactly *and* giving the weak types more draws therefore needs more positive slots.
- At a fixed 64, those slots can only come from the normal slots.
- That is still a change of **composition only**, which is the question W2 should answer.

### Final definition (both arms: batch 64, 120 steps × 25 epochs = 192,000 vehicle draws, the same loss, learning rates, early stopping and seeds)

| Slots per batch | E_DIV (reference) | **W1**, data only, recipe unchanged | **W2**, exposure-preserving composition |
|---|---|---|---|
| Normals | 48 from original normals | 48 uniform from all normals (original + new backgrounds + new controls) | 36 uniform from original normals + new backgrounds; **4** from the new targeted controls |
| Positives | 16 uniform from 105 original | 16 uniform from 249 (105 original + 144 new) | **16 exactly as E_DIV** (uniform from the 105 original) + **8** uniform from the 144 new |
| Positive share | 25 % | 25 % | 37.5 % |

**Expected draws per batch** (new backgrounds ≈ 1,580, estimated from 7.3 unaffected backgrounds per run in the last test; final counts come from the data):

| | E_DIV | W1 | W2 |
|---|---|---|---|
| Tailgating (49) | 7.47 | 3.15 | **7.47** |
| Stalled (28) | 4.27 | 1.80 | **4.27** |
| Reckless (4) | 0.61 | 0.26 | **0.61** |
| Each weak type (4 old + 24 new) | 0.61 | 1.80 | 1.94 |
| Original normals (6,761) | 48 | 38.6 | 29.2 |
| New targeted controls (72) | – | 0.41 | 4.0 |

**Draws per vehicle over training:**
- original positive: 457 (E_DIV) / 193 (W1) / 457 (W2);
- new positive: – / 193 / 167;
- new control: – / 17 / 167;
- original normal: 21 / 17 / 13.

**What each comparison tests**
- **W1 vs E_DIV:** does the targeted data alone, with the training recipe unchanged, improve the weak types? The known cost is that it dilutes existing exposure: tailgating −58 %, stalled −58 %, reckless −58 %.
- **W2 vs E_DIV:** can the weak types gain while every original positive keeps exactly its E_DIV exposure and the new controls are seen as often as the new positives?
- **W2 vs W1:** the same data, the same batch and the same budget. The **only** difference is the per-batch composition.

**Disclosed cost of W2:** it sees each original normal about 40 % less than E_DIV (36 normal slots instead of 48, shared with the new backgrounds), and its positive share is 37.5 %.
- The threshold is re-calibrated at P99 on val_B either way.
- Any rise in false alarms is caught by C4–C6 below.

Both arms log the realised exposure per type (rng replay).

## §6 (revised): selection rule, fully specified (frozen before training)

### Sets (validation only; vehicle units as in every E_DIV validation; ≥ 3 windows = decided)

| Name | Content | n (planned) |
|---|---|---|
| A_t (anomalies of type t) | realised, decided validation vehicles of type t. Weak six: 2 old val_A + 8 new each. Reckless: 2 + 8. Tailgating: 13 old + 8 new. Stalled: 2 val_A + 18 stallval | 10 each for weak six and reckless; 21 tailgating; 20 stalled; 111 in all |
| N_drive (normal driving) | val_A normal corpus + unaffected backgrounds of all validation runs (old and new) + the 26 new val targeted controls | about 2,300 |
| L (legitimate stops, 1–10 s) | stallval 6 + stopval2 54 | 60 |
| TC (tailgating controls) | the 8 new matched controls | 8 |
| val_B | unchanged normals | thresholds only |

**Never used for selection:** the 9 development cases (8/9) and the previous final test. They are reported only, labelled "development / exposed".

### Scores and thresholds
- **Per seed:** z_k = (logit_k − μ_k) / σ_k, with μ_k and σ_k the mean and SD (ddof 0) of seed k's logits on val_B.
- **Ensemble:** z = mean_k z_k. Threshold T = P99 of the ensemble z over the decided val_B normals.
- **Single seed:** threshold T_k = P99 of seed k's logits on val_B.
- **E_DIV reference:** its frozen μ_k, σ_k, T = 1.9726181803233118 and per-seed T_k, scored read-only on exactly the same sets.

### Metrics (alarm = score > threshold)
- recall_t = alarms in A_t / |A_t|
- **Weak6** = mean of recall_t over {hard braking, speeding, overweight, straddling, collision, weaving}
- **Macro9** = mean of recall_t over all 9 types (equal weight)
- FA_drive = alarms in N_drive / |N_drive|; FA_L = number of alarms in L; FA_TC = number of alarms in TC
- Precision = TP / (TP + FP) and F1, on ∪A_t vs N_drive
- ROC-AUC and AUPRC of the ensemble z, on ∪A_t vs N_drive
- Coverage = decided / all, per set

### The conditions (a candidate must meet all nine)

| # | Condition | Set | Required value | Level |
|---|---|---|---|---|
| **C1** Weak-type gain | Weak6(cand) − Weak6(E_DIV) | A_t, weak six | **≥ +0.15** | ensemble |
| **C2** Overall gain | Macro9(cand) − Macro9(E_DIV) | A_t, all 9 | **≥ +0.08** | ensemble |
| **C3** No type loses | for each of the 9 types: alarms(cand) ≥ alarms(E_DIV) − 1. In addition, stallval stalls ≥ 17 / 18 | A_t; stallval | **every type**: at most −1 vehicle (tailgating at most −1 of 21, stalled at most −1 of 20) | ensemble |
| **C4** Normal driving | FA_drive(cand) | N_drive | **≤ 1.5 % and ≤ FA_drive(E_DIV) + 0.5 pp** | ensemble |
| **C5** Legitimate stops | FA_L(cand) − FA_L(E_DIV) | L (60) | **≤ +3** (≈ +5 pp) | ensemble |
| **C6** Tailgating controls | FA_TC(cand) − FA_TC(E_DIV) | TC (8) | **≤ +1** | ensemble |
| **C7** Seed stability | (a) Weak6_k(cand) − Weak6_k(E_DIV) ≥ +0.10 in **≥ 3 of 5** seeds (k = same seed index). (b) In ≥ 3 of 5 seeds, the tailgating and stalled recall_k(cand) ≥ recall_k(E_DIV) − 0.10. (c) FA_drive_k(cand) ≤ 3 % in **all 5** seeds | as above | as stated | seeds (each at its own T_k) |
| **C8** Ranking quality | AUPRC(cand) − AUPRC(E_DIV) and ROC-AUC(cand) − ROC-AUC(E_DIV) | ∪A_t vs N_drive | **≥ −0.02** and **≥ −0.01** | ensemble |
| **C9** Coverage | coverage(cand) = coverage(E_DIV) in every set | all | **equal** (same window rule; verified, not assumed) | – |

**Outcome**
- **Neither arm passes:** E_stall_diversity_ENSEMBLE stays the final model, and model development closes.
- **Exactly one passes:** that arm is frozen as the new candidate and goes to the new independent test.
- **Both pass:**
  1. the higher Macro9 is chosen;
  2. if the difference is < 0.02, the higher AUPRC;
  3. if the AUPRC difference is < 0.01, **W1** (no sampling change).
- **Reported for information only:** bootstrap 95 % CIs over validation runs for ΔWeak6 and ΔMacro9, and recall by partial vs whole-route event, severity and sensing.

**Checkpoint selection (before this rule, the same in both arms):** per seed, the epoch with the best AP on ∪A_t vs val_A normals. This is the same AP rule as E_DIV, on the expanded validation.

## End point
- **The candidate passes:** it is frozen (checkpoints, thresholds, hashes), and a new independent test plan in the 8 test regions (new seeds, start lanes, backgrounds) is presented for approval. It is frozen before simulation and run once. The candidate is compared with the frozen E_DIV, and the previous final test is shown separately and never used for the decision.
- **Then model development closes, whatever the outcome.**
