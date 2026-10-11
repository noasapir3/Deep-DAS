# Study 03: Tailgating data

## At a glance

| | |
|---|---|
| **Research question** | Is tailgating undetected because training lacks realised, varied examples, rather than information? |
| **What was compared** | The same recipe (cleaning + φ<sub>u</sub>, 5 seeds) without (<code>C_old</code>) and with (<code>C_new</code>) the new tailgating sets. <b>Both arms include φ<sub>u</sub>.</b> |
| **Data** | Geometry-verified tailgating sets: v3 on planned routes (36 realised) and v4 with matched controls (23 tailgaters + 24 controls); 59 verified training tailgaters in total. Validation regions only. |
| **Result** | Validation tailgating recall 0.385 → 0.923 (ensemble). The development tailgating case went from 0/5 to 5/5 seeds. |
| **Conclusion and limits** | Realised, varied examples made tailgating learnable. A model trained on this data <i>without</i> φ<sub>u</sub> was never trained, so the gain cannot be split between the data and the feature (see study 08). |
| **Reproduce** | Archived study runs (<code>records/</code>). The data is in the data package (<code>data/processed/anomaly_sets/tailgating_*</code>), and the recipe is part of <code>scripts/train_ensemble.py</code>. |

<p align="center"><img src="../../figures/validation/png/V5_controlled_comparisons_3panel.png" width="95%"></p>

Panel (a) belongs to this study: tailgating recall on validation without and with the varied tailgating data, per seed (open) and for the 5-seed ensemble (filled). Panels (b) and (c) belong to studies 04 and 07.

More figures: [figure gallery](../../figures/) · All studies: [research index](../)

---

## Full record

**Status:** completed and adopted. The tailgating sets and the training recipe of this study are part of DeepDAS Ensemble v1. The resulting ensemble (internal `C_new_ENSEMBLE`) was the frozen candidate until study 04 replaced it.

**Data role:** validation only. The selection used validation regions. The development cases and the new-scene check were reported as diagnostics only.

## Hypothesis
Tailgating was undetected because training had almost no **realised and varied** tailgating examples: one design, mostly not realised. The cause was not missing information, since the tailgating fraction `tg` measures headway directly. Geometry-verified tailgating runs with varied speeds, durations and headways, together with normal runs matched in scene and speed, would make it learnable without raising false alarms.

## Method
**1. Realising tailgating in the simulator.** The data plans went through versions (`records/tailgating_dataset_*`); each one is written before its runs.
- **Why earlier injections failed.** Only 1/8 (new-scene check) and 3/6 (cross-region set) tailgating injections were realised: leader and follower had no shared route and split at the first junction.
- **v1 (pre-registered).** Leader and follower share one planned route; the leader is pinned to v_L, the follower gets a short IDM headway T_f.
  - 10/24 training and 7/12 validation runs were realised, below the pre-registered 80 % bar, so it was reported as is.
  - Diagnosed from truth only: start gap too large, T_f = 0.40 s too close to the 0.8 s label threshold, duration not controlled, and the label lost the leader at lane-segment boundaries.
- **v2 (versioned revision).** Smaller start gap, T_f ∈ {0.20, 0.30} s, release after duration D, and the gap measured along the shared route.
  - 17/24 training and 10/12 validation runs were realised.
  - Two remaining failure causes were found from the simulator truth:
    - the IDM free-road term inflated the gap at a desired speed of 1.3 v_L;
    - routes left the map.
- **v3 = the tailgating set on planned routes** (`data/processed/anomaly_sets/tailgating_routes/`).
  - Changes from v2: the follower has a desired speed of 2.5 v_L, with the speed limit ignored only during the event; routes stop at boundary exits.
  - Released after the episode with ordinary driving, so the vehicle does not become a speeder.
  - **36 realised tailgaters:** 24 training (train regions) and 12 validation (val_A regions).
- **v4 = the tailgating set with matched controls** (`data/processed/anomaly_sets/tailgating_matched_controls/`), training only.
  - Segments ≥ 50 km/h, so the leader can drive at 9.0 or 11.8 m/s.
  - T_f ∈ {0.20, 0.30} s, D ∈ {8, 15, 25} s.
  - **Every tailgating run has a matched control:** the identical run with the default IDM headway.
  - 23 realised tailgaters and 24 clean controls.

**2. Labelling.** A vehicle is labelled tailgating only if the simulator truth shows the episode. The rule is `tailgating_v1`: truth headway < 0.8 s at a speed ≥ 5 m/s for ≥ 5 s continuously, bumper gap ≥ 1 m throughout (measured along the shared route), and no follower collision.
- Control and background vehicles count as normal only if they never meet the condition for ≥ 1 s and have no collision.
- Vehicles that do not meet either definition are **excluded and counted**, never auto-labelled.

**3. Controlled training experiment** (pre-registered; `records/C_tailgating_data_*`).

| | Without new data (`C_old`) | With new data (`C_new`) |
|---|---|---|
| **Model** | variant C of study 02 (cleaning + `tg`) | identical |
| **Recipe** | 5 seeds | identical |
| **Training anomalies** | the 34 cross-region anomalies | the same + v3 train + v4 realised tailgaters |
| **Training normals** | the corpus and the cross-region backgrounds | the same + unaffected backgrounds + v4 clean matched controls |

- **Validation, identical in both arms:** the 17 cross-region validation anomalies plus the 12 v3 validation tailgaters.
- **Ensemble:** z-score per seed with that seed's val_B mean/std, mean over the 5 seeds, then the ensemble's own P99 on val_B.
- **Selection rule, fixed before training:**
  1. at most 1.5 % false alarms on the val_A normals;
  2. then the higher validation recall at P99;
  3. ties go to the higher validation AP.

## Results (validation; ensembles at their own val_B P99)

| | Without new data | **With new data** |
|---|---|---|
| Validation AP | 0.449 | **0.766** |
| Validation recall (29 anomalies) | 0.586 | **0.793** |
| Tailgating recall (12 + 1 validation) | 0.385 | **0.923** |
| Recall, other 8 types | 0.750 | 0.688 |
| False alarms, val_A normals | 1.0 % | 1.1 % |
| Development cases (diagnostic) | 7/9 (missed: weaving, tailgating) | 7/9 (missed: stalled, weaving) |

- **Per seed with new data:** tailgating validation recall was 0.85–0.92 for every seed.
- **The development tailgating case C3 was detected by all 5 seeds.** Without the new data, no seed detected it.
- **Selected:** the ensemble with new data.

## Conclusions
1. **The hypothesis held.** With realised, varied tailgating examples and matched controls, tailgating became reliably detectable: about 0.92 validation recall, stable across seeds, without raising false alarms.
2. **Realisation must be verified.** Tailgating cannot be assumed from the injection parameters. Two plan revisions were needed before the simulator produced it, and each revision was written down before it ran.
3. **A new weakness appeared.** The stalled development case was lost: the ensemble missed it and only 1 of 5 seeds detected it. Diagnosis: tailgating took about 60 % of the anomaly draws, which cut the exposure of the other types by about 2.4×. In addition, the training stalls covered only one stop mode. → Study 04.
4. **Matched controls remain a known cost.** In the final test, 7 of 11 tailgating matched controls that follow a slow leader were flagged. Within each pair, the tailgater always scored higher than its control (see `docs/LIMITATIONS.md`).

## Contents
| Path | Content |
|---|---|
| `records/` | The tailgating data plans (pre-registered, v2, v3, v4), the plan and results of the training experiment, its selection record and the per-seed table, and the release record of the selected ensemble with its limitations, all verbatim |
| `../../data/processed/anomaly_sets/tailgating_routes/`, `tailgating_matched_controls/` | The two tailgating sets (processed runs, vehicle labels) |
| `../../data/simulation_plans/tailgating_*` | Their run plans and region surveys |
| `src/deepdas/batches/tailgating_routes.py`, `tailgating_controls.py` | The generators (verified: they regenerate identical scenarios and truth) |
