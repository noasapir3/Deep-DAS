# Study 07: Targeted training data for the weak types (not adopted)

> **Research question Q3** ([research index](../README.md)): why do more detections not necessarily justify replacing the model?
>
> **Status:** completed; **not adopted**; the last round of model development. **Data role:** validation only. These models were trained **after the final test had been exposed**.

## Summary

**What we wanted to find out.** Does diverse, realisation-checked training data for the six weak types raise their recall without losing any existing ability?

**What was compared.**

- **Three models:** the frozen DeepDAS Ensemble v1 against two new five-seed arms:
  - a **targeted training-data expansion** (code `W1`);
  - the **same data with controlled, exposure-preserving sampling** (code `W2`).
- **Held constant:** the architecture, the splits, the seeds, the batch size and the training budget. Each model is thresholded at its own P99 on the threshold-region normals.
- **The decision rule:** nine conditions, all fixed before training (codes `C1`–`C9`).

**Data and its role.**

- **The data:** 314 new runs in training and selection regions. For each weak type there are 24 training and 8 validation vehicles from wide parameter grids, with matched legal controls.
- **Role:** validation only. No new test was run.

<p align="center"><img src="../../figures/validation/png/V5_controlled_comparisons_3panel.png" width="95%"></p>

*Figure, panel (c).*

- **Axes:** the mean recall over the six weak types against false alarms on 54 legitimate stops.
- **Models:** the frozen model, the targeted data expansion, and the same data with controlled sampling.
- **Markers:** per seed (small) and ensemble (large).
- **Dashed line:** the pre-registered limit, at most 3 more legitimate-stop alarms than the frozen model.

**Result** (frozen → targeted data / with controlled sampling):

- **Gains:**
  - mean weak-type recall 0.35 → 0.71 / 0.74, macro recall 0.54 → 0.80 / 0.83;
  - ROC-AUC 0.87 → 0.96;
  - false alarms on normal driving 1.13 % → 0.88 % / 1.05 %;
  - alarms on tailgating controls 6/7 → 0/7 / 2/7.
- **Cost:** legitimate-stop alarms rose from 40 to 45 and 52 of 54, past the limit of 43 (condition `C5`). Both arms passed the other eight conditions.

**Decision.**

- Neither arm is adopted, and DeepDAS Ensemble v1 remains the final model.
- Development was closed after this round.

**What this does not show.**

- **Final-test performance.** These arms have no final-test number: the test had been exposed, and a later model needs a new test.
- **Overweight.** It did not respond to the extra data.
- **Precision of the gains.** With 9–10 validation vehicles per weak type, one detection moves recall by 0.1. The direction is still consistent across seeds.

**Code, data and reproduction.**

- **Evaluation and rule:** [`code/evaluate_selection_rule.py`](code/evaluate_selection_rule.py). With the data package, `python research/07_targeted_data_study/code/evaluate_selection_rule.py outputs/w_round` reproduces the seven result files of [`results/`](results/) byte for byte.
- **Models:** `models/W1/`, `models/W2/`.
- **Data:** `data/` (data package); the generator is `python -m deepdas.batches.targeted_data`.
- **Training scripts:** in `code/as_run/`, exactly as run. They are not runnable from the repository.
- **Records:** [`records/weak_types_W1W2_20261010_plan_PLAN_W1_W2_FROZEN.md`](records/weak_types_W1W2_20261010_plan_PLAN_W1_W2_FROZEN.md), [`records/weak_types_W1W2_20261010_results_RESULTS_W.md`](records/weak_types_W1W2_20261010_results_RESULTS_W.md).

More figures: [figure gallery](../../figures/README.md) · All studies: [research index](../README.md) · Report map: [docs/QUESTION_MAP.md](../../docs/QUESTION_MAP.md)

---

## Full record

> *Documentation round v3 (2026-10-11): the statements marked † were reworded to separate what was measured from the explanation it supports. The text as first released is in `DeepDAS_Release_v1.0` and in the git history.*

**Status:** completed. **Not adopted.** Neither arm met the pre-registered rule, so DeepDAS Ensemble v1 remains the final model. This was the last round of model development.

**Data role:** validation only. The selection used val_A regions, stallval and the legitimate-stop validation set (stopval2). The development cases and the earlier final test were not used for any decision. No new test was run.

## Hypothesis
Study 06 found that the six weak types lack varied training examples. Adding diverse, realisation-checked examples of these types, together with normal controls matched to each type, raises their recall. The pre-registered **condition** was that this must not lose any existing ability: tailgating, stalls, false alarms on normal driving and on legitimate stops.

## Method (plan frozen before any simulation; `records/*PLAN_W1_W2_*`)
**Data.** 314 new runs in train and val_A regions, with new scenes, seeds and start lanes and no overlap with earlier sets (`data/`).
- **Weak-type anomalies, per type:** 24 training and 8 validation vehicles.
  - Hard braking: deceleration 5–8 m/s², partial or to a near-stop, at varied times.
  - Speeding: ×1.30–1.80 the limit, over the whole route or a segment.
  - Overweight: 20–40 t under different sensing conditions.
  - Straddling: offset 1.8–2.1 m.
  - Collision: impact speed 5–12 m/s, with a frozen or a moving target.
  - Weaving: different amplitudes and periods.
- **Partial events** are labelled only if the realisation statistic over the planned event window meets the unchanged threshold. This extension was explicitly approved.
- **Matched normal controls:**
  - firm legal braking (< 4 m/s²);
  - fast legal driving (p90 speed ratio < 1.10);
  - legal heavy vehicles (< 20 t);
  - lateral drift (< 1.7 m).

  A control is clean only if its targeted property stays on the legal side over the whole truth track. Otherwise it is excluded and reported.
- **Validation only:** 8 reckless runs and 8 tailgating pairs.
- **Totals:** 206 verified anomaly vehicles, 91 clean targeted controls, 8 matched tailgating controls and 2,126 unaffected backgrounds. 2 unverified anomalies and 7 controls that were not clean were excluded.

**Arms.** Both use the same architecture, splits, 5 seeds, batch of 64 and training budget as DeepDAS Ensemble v1. Checkpoints are chosen by validation AP on the expanded validation set.

| Slots per batch | DeepDAS Ensemble v1 | **W1**: data only | **W2**: exposure-preserving composition |
|---|---|---|---|
| Normal | 48 | 48 from all normals (original + new) | 36 original/new backgrounds + **4** new targeted controls |
| Anomalous | 16 from 105 | 16 from 249 (105 original + 144 new) | **16 exactly as before** + **8** from the 144 new |
| Anomalous share | 25 % | 25 % | 37.5 % (disclosed) |

**Thresholds.** Each candidate is thresholded at its own val_B P99. DeepDAS Ensemble v1 is scored read-only with its frozen constants on exactly the same sets.

**Selection rule C1–C9 (frozen before training; all must hold):**

| # | Condition |
|---|---|
| C1 | Weak6 recall (mean of the six weak types) gain ≥ +0.15 |
| C2 | Macro9 recall gain ≥ +0.08 |
| C3 | No type loses more than 1 detection; stallval stalls ≥ 17/18 |
| C4 | False alarms on normal driving ≤ 1.5 % and ≤ the reference's + 0.5 pp |
| C5 | False alarms on legitimate stops ≤ the reference's + 3 |
| C6 | False alarms on tailgating controls ≤ the reference's + 1 |
| C7 | Seed stability: Weak6 gain ≥ 0.10 in ≥ 3/5 seeds; tailgating and stalls hold in ≥ 3/5 seeds; normal false alarms ≤ 3 % in all seeds |
| C8 | AUPRC ≥ the reference's − 0.02 and ROC-AUC ≥ the reference's − 0.01 |
| C9 | Equal decision coverage |

C3 counts detections per type on a few vehicles each. This makes it a conservative condition, disclosed in the plan.

**Disclosed incidents** (`records/*LAUNCH_NOTE*`, plan erratum):
- **Erratum 1:** a missing `import numpy` in the import script stopped the first import before it wrote anything. The fix was the single import line, and the code manifest was re-frozen.
- **Training launch:** a launcher missing two settings was stopped by the script's own assertion before training.
- **W1 seed 0** was killed by out-of-memory while loading data and was rerun unchanged.

## Results (validation; ensembles at their own val_B P99)

| | DeepDAS Ensemble v1 | W1 | W2 |
|---|---|---|---|
| Hard braking / Speeding / Overweight | 0.22 / 0.22 / 0.20 | 0.67 / 0.78 / 0.20 | 0.78 / 0.78 / 0.30 |
| Straddling / Collision / Weaving | 0.33 / 0.90 / 0.20 | 0.89 / 1.00 / 0.70 | 0.89 / 1.00 / 0.70 |
| Reckless / Tailgating / Stalled | 0.80 / 0.95 / 1.00 | 1.00 / 0.95 / 1.00 | 1.00 / 1.00 / 1.00 |
| **Weak6 / Macro9** | 0.35 / 0.54 | **0.71 / 0.80** | **0.74 / 0.83** |
| ROC-AUC / AUPRC | 0.87 / 0.70 | 0.96 / 0.86 | 0.96 / 0.88 |
| False alarms, normal driving (of 2,382) | 1.13 % | 0.88 % | 1.05 % |
| **False alarms, legitimate stops (of 54)** | **40** | **45** | **52** |
| False alarms, tailgating controls (of 7) | 6 | 0 | 2 |

- **Rule outcome:** both arms pass C1–C4 and C6–C9. **Both fail C5**: W1 has +5 legitimate-stop alarms and W2 has +12, against a limit of +3. **No candidate.**
- **Bootstrap 95 % CI of the Weak6 gain** (reported only): W1 [+0.18, +0.52], W2 [+0.23, +0.54].

## Conclusions
1. **The weak types are learnable from targeted data.** Weak-type recall about doubles on validation, consistent with the diagnosis that the information is in the input and the encoder.
2. **Overweight is the exception.** Its amplitude overlaps legal heavy vehicles, and more data did not move it.
3. **† The gain came with more false alarms on legitimate stops, as the plan anticipated.** A plausible explanation: learning sharp slow-downs and brakes to a near-stop moves the model toward short stops. This was not tested separately. More data alone did not resolve the trade-off documented in study 05.
4. **The pre-registered rule was applied as written.** DeepDAS Ensemble v1 remains the final model, and development closed.

## Contents
| Path | Content |
|---|---|
| `data/` | The 314 runs in processed form (channels, amplitude, track metadata), per-vehicle labels, run report and run plan. The raw simulations are in the raw-data package (`docs/RAW_DATA_PACKAGE.md`) |
| `models/W1/`, `models/W2/` | Per seed: checkpoint, thresholds, training history, val_B logits, per-epoch exposure log |
| `results/` | The seven result files of the round (selection, metrics, calibration, unit scores, bootstrap CIs, partial-event recall); byte-identical to the original outputs |
| `code/evaluate_selection_rule.py` | The common evaluation and rule C1–C9, ported to the release paths. It reproduces all 7 result files of the round byte for byte (`docs/REPRODUCIBILITY.md`) |
| `code/as_run/` | The training and helper scripts exactly as they were run (not runnable from the release; see `code/as_run/README.txt`) |
| `records/` | The README of the round, the draft, v2 and frozen plans, the diff of the training script against the DeepDAS Ensemble v1 recipe, the launch note, the reproduction check, and the results and selection, all verbatim |

**Reproduce the evaluation:**
```
python research/07_targeted_data_study/code/evaluate_selection_rule.py outputs/w_round
```
The generator of the round is `python -m deepdas.batches.targeted_data` (verified to regenerate identical scenarios).
