# Seed study — pre-registered (before any run)

**Question:** is the 8/9 of B and C stable across seeds, or seed-dependent?
**Versions:** Baseline recipe (raw input, no extra features), A (cleaning), B (cleaning + `osc`), C (cleaning + `tg`). The definitions are those of
`crossregion_AD_20261007/PLAN_PREREGISTERED.md`, unchanged.
**Seeds:** 0, 1, 2. Seed 0 of each version already exists: Baseline = the frozen original (reproduced exactly by R0), and A/B/C seed 0 = experiments A/B/C.
Seeds 1 and 2 are new runs. The seed sets `random`, `numpy`, `torch` and the batch sampler. The encoder always starts from the same amplitude-only
checkpoint. The head initialisation and batch order change with the seed.
**Unchanged:** split, labels, recipe, epochs, patience, selection by val AP, and the threshold rule (val_B normals P99 primary / P95 alternative, frozen before evaluation).
**Reported per version (mean ± std over 3 seeds):** detections of the 9 valid cases, FA test / val_A, precision / recall / F1 (9 valid vs 513 test normals),
val AP, the C3 and weaving logits, percentiles and decisions. New-scene numbers are reported as SECONDARY only, because that set has already been used for analysis.
**No decision rule is tuned on test, C3 or weaving.** The decision criterion is fixed now: a version counts as a stable improvement only if
(i) its mean detections exceed the baseline's by ≥ 1 with every seed ≥ the baseline's seed mean, and (ii) its mean test FA is not more than 2 above the baseline's.
