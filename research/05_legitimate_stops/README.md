# Study 05: False alarms on legitimate stops

> **Research question Q3** ([research index](../README.md)): why do more detections not necessarily justify replacing the model?
>
> **Status:** completed; no candidate replaced DeepDAS Ensemble v1, and this is its main documented limitation. **Data role:** validation. From the checkpoint-selection step on, these sets also took part in selection.

## Summary

**What we wanted to find out.** The final model flags most *legitimate* stops: gentle braking (< 4 m/s²), standing 1–10 s, driving on. By definition these are normal. Why does it flag them, and can that be fixed without breaking the other conditions?

**What was compared.**

1. **A read-only diagnosis** of the frozen model.
2. **A new validation set** that separates stop duration, stop share and approach.
3. **Two pre-registered remedies** on the same data, recipe and seeds: controlled exposure to legitimate stops, and exposure plus a temporal branch over the window sequence.
4. **Checkpoint choice by a task rule,** for the frozen recipe and for both remedies.

**Held constant:** the data, the recipe, the seeds and the threshold rule (each model at its own P99 on the threshold-region normals). The replacement rule was fixed before training.

**Data and its role.**

- **The pool:** the legitimate-stop validation set together with the stall validation set, 54 legitimate stops and 66 abnormal stops.
- **Role:** validation only.

<p align="center"><img src="../../figures/validation/png/S05_legitimate_stop_remedies.png" width="60%"></p>

*Figure.* Legitimate stops flagged (false alarms, of 54) by the frozen model and by each remedy, each at its own threshold, with the abnormal stops still detected; dashed: the pre-registered replacement rule.

**Result.**

- **Diagnosis, what was measured:**
  - **Stop windows:** with the windows that overlap the truth stop removed, no stop vehicle alarms.
  - **The tracker shortens stops:** a 30 s truth stop shows about 9 s.
  - **Training exposure:** training saw about 105× fewer legitimate-stop windows than stall windows.
- **The explanation these measurements support** (not proven as the only cause): the stop windows drive the score, and with mean/std pooling their *share* of the track matters more than the stop's duration. Legitimate stops were rarely seen in training, and their duration is weak in the tracked input.
- **Remedies:**
  - Alarms fell from 40 to 37 (exposure), 27 (temporal branch) and as low as 21 of 54 (temporal branch with a task-rule epoch).
  - None met the rule (at most 20 % of the stops and at most half of the reference) while keeping the other conditions. The temporal branch lost detections of other types.
  - The 8 s stops stayed almost always flagged in every variant (19–22 of 24).

**Decision.**

- DeepDAS Ensemble v1 stays the model, and the false alarms are documented as its main limitation.
- The final test showed the same problem: 20 of 28 legitimate stops flagged.

**What this does not show.** The following were not tried, because development was closed:

- a duration-aware feature computed from the track;
- a stop-specific post-filter;
- a tracker that preserves stop duration.

**Code, data and reproduction.**

- **Data:** `data/processed/anomaly_sets/legitimate_stop_validation/` (data package).
- **Records:** [`records/legit_stop_diag_20261009_README.md`](records/legit_stop_diag_20261009_README.md), [`records/stop_discrimination_20261009_RESULTS_F.md`](records/stop_discrimination_20261009_RESULTS_F.md), [`records/ckpt_selection_20261009_RESULTS_CKPT.md`](records/ckpt_selection_20261009_RESULTS_CKPT.md).
- **Reproduction:**
  - `scripts/reproduce_validation.py` regenerates the final model's scores on these sets (`results/validation/ensemble_scores_legitimate_stop_set.csv`).
  - The remedy runs are archived, not regenerated.

More figures: [figure gallery](../../figures/README.md) · All studies: [research index](../README.md) · Report map: [docs/QUESTION_MAP.md](../../docs/QUESTION_MAP.md)

---

## Full record

> *Documentation round v3 (2026-10-11): the statements marked † were reworded to separate what was measured from the explanation it supports. The text as first released is in `DeepDAS_Release_v1.0` and in the git history.*

**Status:** completed. **No candidate replaced DeepDAS Ensemble v1.** The false alarms on legitimate stops are documented as the model's main limitation.

**Data role:** validation only. The legitimate-stop validation set (`stopval2`) and `stallval` were used for every comparison in this study. From the checkpoint-selection step on they also took part in selection, so they are **not** independent evidence for any model.

## Problem
DeepDAS Ensemble v1 flags many **legitimate stops**: a vehicle brakes gently (< 4 m/s²), stands for 1–10 s, and drives on. By the project's definitions these are normal.
- In study 04 it flagged 4 of 6.
- On the pooled validation pool it flags **40 of 54** legitimate stops (stallval + stopval2).

## Step 1: read-only diagnosis (`records/legit_stop_diag_*`)
No model, threshold or label was changed. The frozen checkpoints were only loaded for scoring.

| Question | Measurement | Finding |
|---|---|---|
| Is the alarm caused by the stop itself? | Score each stallval vehicle with and without the windows that overlap the truth stop | With all windows, 4/6 legitimate stops and 18/18 abnormal stops alarm. Without the stop windows **none** alarms (a vehicle standing from the start has no windows left). † The alarms depend on the stop windows. |
| What does the input show of the stop? | Truth stop duration vs the stop visible in the tracked speed | The tracker shortens stops strongly: a 30 s truth stop shows about 9 s; a 120 s standstill shows about 15 s. Duration is a weak signal in the input. |
| How much did training see legitimate stops? | Exact replay of the training draws (matches the run logs in 107/107 epochs) | Each training legitimate stop (23) was drawn about 21× less often than each training stall (24), and the model saw about **105× fewer stop windows** from legitimate stops than from stalls (seed 0: 1,183 vs 125,095). |
| Does the model even fit them? | In-sample alarms on the 23 training legitimate stops | 7/23 flagged in-sample. |

**Proposed mechanism (†).** With mean/std pooling over all windows, the **share** of stop windows can drive the score more than the stop's duration. A short track with a 5 s stop has a stop share similar to a long track with a 30 s stall. The counterfactual above is consistent with this, but does not prove it to be the only cause.

## Step 2: an expanded validation set that separates the factors (`stopval2`)
**Why it was needed.** An audit of the existing stops showed that duration, stop share and approach were confounded:
- every training stall used an instant freeze;
- every legitimate stop used 2 m/s² braking.

**Design.** 12 val_A regions × 2 observation slots:
- **SHORT:** the vehicle leaves the map 60–150 m after the stop;
- **LONG:** ≥ 350 m of route after the stop.

**Five runs per slot:**

| Run | Behaviour | Label |
|---|---|---|
| Control | drives through | normal |
| Legitimate stop 3 s | 2 m/s² braking, stop 3 s, drive on | normal |
| Legitimate stop 8 s | 2 m/s² braking, stop 8 s, drive on | normal |
| Gentle stall 15 s | **the same 2 m/s² braking**, stop 15 s, drive on | stall |
| MSM 30 s | the existing instant freeze, 30 s | stall |

This gives 48 legitimate stops, 48 abnormal stops and 24 controls (`data/processed/anomaly_sets/legitimate_stop_validation/`).

**Evaluation pool.** Pooled with stallval:
- 54 legitimate stops (3 / 5 / 8 s: 24 / 6 / 24);
- 66 abnormal stops.

## Step 3: two pre-registered arms (`records/stop_discrimination_*`)
Both use the same data, recipe and seeds as DeepDAS Ensemble v1.
- **F1, controlled exposure.** 4 of the 48 normal slots of each batch are drawn from the 23 training legitimate stops. This matches the per-vehicle exposure of the stalls. It is the only change.
- **F2, a temporal branch.** F1 plus a small dilated 1-D convolution over the time-ordered window codes of one contiguous track segment (receptive field about 29 s), max-pooled over time, added to the head input. No duration rule is given.

**Replacement rule (fixed before training).** All of these must hold:
- legitimate-stop alarms ≤ half of the reference's, and ≤ 20 %;
- abnormal stops ≥ the reference's − 2, and gentle stalls ≥ the reference's − 2;
- non-stall detections ≥ the reference's − 1;
- tailgating ≥ 12/13;
- macro9 ≥ the reference's − 0.03;
- false alarms on val_A ≤ 1.5 %.

## Step 4: checkpoint selection by a task rule (`records/ckpt_selection_*`)
**Question.** Is the problem only *which epoch* is kept?

**How it was tested.**
- The three recipes (reference, F1, F2) were **replayed** with a checkpoint saved every epoch. The replays were bit-identical to the originals at the AP-selected epoch, for all 15 runs.
- A task-based objective J was used to choose the epoch. J weights 9 × macro9 against the legitimate-stop correctness, with feasibility limits on false alarms, tailgating and abnormal stops.
- **Disclosed:** stallval and stopval2 took part in this selection.

## Results (validation pool; each ensemble at its own val_B P99)

| | Legitimate stops flagged (of 54) | 3 s / 5 s / 8 s | Abnormal stops (of 66) | macro9 | False alarms, val_A | Passes rule |
|---|---|---|---|---|---|---|
| **DeepDAS Ensemble v1** (reference) | **40** (74 %) | 14/24 · 4/6 · 22/24 | 65 | 0.712 | 1.07 % | — |
| F1 | 37 (69 %) | 14/24 · 3/6 · 20/24 | 65 | 0.712 | 1.45 % | no |
| F2 | 27 (50 %) | 4/24 · 2/6 · 21/24 | 66 | 0.500 | 1.68 % | no |
| Reference, epoch by J | 41 | 15/24 · 4/6 · 22/24 | 66 | — | > 1.5 % | no |
| F1, epoch by J | 35 | 12/24 · 3/6 · 20/24 | 65 | lower | ≤ 1.5 % | no |
| F2, epoch by J | 21 (39 %) | 2/24 · 0/6 · 19/24 | 66 | lower | ≤ 1.5 % | no |

- **No variant met the legitimate-stop condition** (≤ 20 alarms and ≤ 20 %).
- The variants that reduced alarms the most (F2) lost other types: non-stall detections and macro9.
- **The 8 s legitimate stops stayed almost always flagged** in every variant: 19–22 of 24.

## Conclusions
1. **† The measurements support one explanation.** It combines the pooled stop-window share with very low training exposure to legitimate stops. Stop duration, the physical difference, is strongly shortened in the tracked input. It was not isolated as the single cause.
2. **Neither exposure (F1), a temporal branch (F2) nor epoch choice removes the problem** without breaking another condition. DeepDAS Ensemble v1 stays the model.
3. **The final test confirmed the limitation:** 20 of 28 legitimate stops flagged; 3 s 3/8, 5 s 8/11, 8 s 9/9. It is the first item of `docs/LIMITATIONS.md`.
4. **Not tried, by decision to close development:**
   - a duration-aware feature computed from the track;
   - a stop-specific post-filter;
   - a tracker change.

## Contents
| Path | Content |
|---|---|
| `records/` | The diagnosis (README and tables: window counterfactual, stop visibility, training fit, exposure replay), the F1/F2 plan with its addendum, results, selection and per-seed table, and the checkpoint-selection plan, results and selection, all verbatim |
| `../../data/processed/anomaly_sets/legitimate_stop_validation/`, `../../data/simulation_plans/legitimate_stop_validation/` | The stopval2 set and its plan |

The validation evaluation of DeepDAS Ensemble v1 on these sets is reproduced by `scripts/reproduce_validation.py` (`results/validation/ensemble_scores_legitimate_stop_set.csv`).
