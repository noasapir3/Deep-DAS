# Study 02: Supervised single model with anomalies from other regions

> **Research questions Q1 and Q2** ([research index](../README.md)): does supervision work where reconstruction failed, and is the result stable across random seeds? It also holds the history of the tailgating summary φ<sub>u</sub> (Q4).
>
> **Status:** completed; the selected model is shipped as the supervised single-model baseline. **Data role:** development and validation (not independent).

## Summary

**What we wanted to find out.**

- Can a few labelled anomalies, simulated in regions other than the evaluation regions, teach a classifier head on top of the pre-trained encoder?
- Are the follow-up changes (artefact cleaning, two summary features) stable across random seeds?

**What was compared.**

1. **The supervised single model:** a head on the amplitude-aware encoder.
2. **One change at a time:**
   - artefact cleaning;
   - cleaning + a weaving feature;
   - cleaning + the tailgating fraction φ<sub>u</sub>;
   - cleaning + both features.
3. **Seeds:** the original recipe and three of the variants, each trained with seeds 0, 1 and 2.

**Held constant:** the training data, the encoder initialisation, the training budget and the threshold rule (each model's own P99 on the threshold-region normals).

**Data and its role.**

- **Cross-region anomaly set:** 54 runs in regions outside the evaluation regions, giving 34 verified training anomalies and 17 verified selection anomalies.
- **Also used:** the 9 development cases and the early validation split.
- **Role:** development and validation. These sets were inspected while the models were chosen, so no number here is independent.

<p align="center"><img src="../../figures/development/png/S02_seed_dependence.png" width="60%"></p>

*Figure.* Development cases detected (of 9) by each recipe, per seed (open circles) and as the mean of three seeds (bar). The arms differ in one change each; the data and the threshold rule are fixed.

**Result.**

- **The selected single model** detects 7 of the 9 development cases; it misses weaving and tailgating.
- **Seed dependence:** the original recipe detects 7, 2 or 3 cases depending on the seed (2–7). Across the four recipes tested with three seeds each, the range is 2–8.
- **Tailgating:** no arm detects the tailgating case, with or without φ<sub>u</sub> (0 of 12 seed-study runs).

**Decision.**

- Supervision is adopted.
- Artefact cleaning and φ<sub>u</sub> become part of the recipe.
- Every later model is a **five-seed ensemble** with per-seed standardisation.
- The selected model is frozen as the single-model baseline.

**What this does not show.**

- **Why tailgating stays undetected.** With varied, verified tailgating data added and φ<sub>u</sub> kept, tailgating became detectable (study 03). Whether φ<sub>u</sub> was also needed is not identified: no model without it was trained on the final training data.
- **The value of φ<sub>u</sub>.** This is the only archived comparison with and without φ<sub>u</sub>, and it uses the early data ([study 08](../08_tailgating_feature_intervention/)).

**Code, data and reproduction.**

- **Model:** [`models/baselines/supervised_single_p99/`](../../models/baselines/supervised_single_p99/), scored by `src/deepdas/ensemble/scoring.py` (`SingleModel`).
- **Data:** `data/processed/anomaly_sets/cross_region/` (data package).
- **Records:** [`records/crossregion_AD_20261007_SUMMARY.md`](records/crossregion_AD_20261007_SUMMARY.md), [`records/seedstudy_20261007_SEED_SUMMARY.md`](records/seedstudy_20261007_SEED_SUMMARY.md).
- **Reproduction:**
  - The single model is scored in every development and final-test reproduction (`scripts/reproduce_development_cases.py`, `scripts/reproduce_final_test.py`).
  - The variant and seed runs are archived, not regenerated.

More figures: [figure gallery](../../figures/README.md) · All studies: [research index](../README.md) · Report map: [docs/QUESTION_MAP.md](../../docs/QUESTION_MAP.md)

---

## Full record

> *Documentation round v3 (2026-10-11): the statements marked † were reworded to separate what was measured from the explanation it supports. The text as first released is in `DeepDAS_Release_v1.0` and in the git history.*

**Status:** completed. The selected model is shipped as the **supervised single-model baseline** (`models/baselines/supervised_single_p99/`, internal identifier `CR_P99`). The follow-up experiments on features and seeds (A–D and the seed study) led to the design of DeepDAS Ensemble v1.
**Data role:** development and validation. The development cases and the new-scene check had been inspected before these models were chosen, so none of the numbers below is independent.

## Hypothesis
A small number of labelled anomalies, simulated in **other regions** than the evaluation, is enough to teach a classifier head on top of the pre-trained encoder. The head would detect the types that reconstruction error misses (study 01), at a comparable false-alarm rate.

## Method
**Cross-region anomaly set.** 54 runs (9 types × 6 runs; `data/processed/anomaly_sets/cross_region/`).
- Training anomalies come from regions 131, 050, 090 and 065; selection anomalies from 002 and 005. None is an evaluation region.
- An anomaly counts only if a truth-based realisation rule confirms it. 51 of 54 runs were verified; tailgating failed in three regions and those runs were excluded.
- This gives 34 training and 17 selection anomalies.

**Model.**
- 14 input channels: the 13 of study 01 plus `das_amp_log10`.
- The encoder is initialised from the amplitude-aware autoencoder.
- Mean and std pooling of the window codes (16 features), then an MLP head 16 → 16 → 1.

**Training.**
- Batches of 48 normal + 16 anomalous vehicles, at most 32 windows per vehicle.
- AdamW (encoder 1e-4, head 1e-3).
- At most 25 epochs; the checkpoint is chosen by validation AP.

**Thresholds.** The P99 (primary) and P95 of the logits of the 1,242 sufficient-evidence val_B normals.

**Follow-up experiments A–D** (pre-registered; one change at a time; `records/crossregion_AD_*`):
- **A: lead-artefact cleaning.** A bin whose "leader" is closer than 1 m at more than 5 m/s is a duplicate track, physically impossible. It is re-encoded as "no leader".
- **B:** A + the lateral-periodicity feature `osc`.
- **C:** A + the tailgating fraction `tg`, the share of bins with headway < 0.8 s at more than 5 m/s.
- **D:** A + both features.
- **R0:** the original recipe re-run on this platform. It reproduced the original exactly.

**Seed study.** Baseline, A, B and C were each trained with seeds 0, 1 and 2, to test whether the gains were stable (`records/seedstudy_*`).

**New-scene check.** The `normal_medium_traffic` scene of each of the 8 test regions, with the same 9 injections (59 verified anomalies; `data/processed/anomaly_sets/new_scene_check/`).
- Planned as an evaluation-only set.
- It was looked at for every variant of this study, so it is reported as **secondary/diagnostic** only.

## Results
**The selected single model (internal `BASELINE_frozen` = `CR_P99`).**
- Validation AP 0.571.
- **7 of 9** development cases detected at P99; lane weaving and tailgating (C3) missed.
- False alarms at P99: 7/513 test-split normals and 17/1,313 val_A normals.
- New-scene check: 28 of 47 decided anomalies.

**Experiments A–D, seed 0, P99:**

| Variant | Val AP | Development cases | Missed | False alarms (test split) |
|---|---|---|---|---|
| Baseline (= R0) | 0.571 | 7/9 | weaving, tailgating | 7/513 |
| A (cleaning) | 0.687 | 6/9 | speeding, weaving, tailgating | 8/513 |
| B (cleaning + osc) | 0.746 | 8/9 | tailgating | 15/513 |
| C (cleaning + tg) | 0.718 | 8/9 | tailgating | 12/513 |
| D (both) | 0.404 | 5/9 | stalled, weaving, straddling, tailgating | 11/513 |

**Seed study (mean ± std over seeds 0–2, P99):**

| | Baseline | A | B | C |
|---|---|---|---|---|
| Development cases detected | 4.0 ± 2.6 | 4.7 ± 1.5 | 6.0 ± 2.0 | 6.3 ± 2.1 |

- The baseline seeds detected 7, 2 and 3 cases.
- No variant detected the realised tailgating case (C3) at P99 with any seed.

## Conclusions
1. **Supervision works.** A supervised head on the pre-trained encoder detects types that reconstruction error missed: braking, straddling, overweight and the contact collision.
2. **Cleaning and `tg` together help.** Lead-artefact cleaning plus the tailgating fraction gave the best and most stable variant (C). Both became part of DeepDAS Ensemble v1. `osc` and the combination D were not adopted.
3. **Single models are strongly seed-dependent.** The baseline ranged from 2/9 to 7/9 by seed alone. **A 5-seed ensemble with per-seed standardisation was therefore adopted** for every later model.
4. **† On the early data, adding `tg` was not enough for tailgating.** Even with `tg`, tailgating stayed undetected at P99. The cross-region set held only a handful of realised tailgaters, from one design, which pointed to the data. → Study 03, where varied data with `tg` kept made it detectable. Whether `tg` was also needed is not identified (study 08).

## Contents
| Path | Content |
|---|---|
| `records/` | Plans and summaries as written at the time, verbatim: experiments A–D, the seed study, the new-scene check plan and per-type results, and the release record of the single model with its limitations |
| `../../models/baselines/supervised_single_p99/` | Checkpoint, thresholds and model description of the selected single model (shipped baseline) |

The single model is scored by `src/deepdas/ensemble/scoring.py` (`SingleModel`) in the development and final-test reproductions.
