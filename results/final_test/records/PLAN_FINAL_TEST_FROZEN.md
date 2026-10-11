# Final generalisation test — North-TLV test regions — FROZEN plan
Frozen 2026-10-09, before any simulation of this test. Approved by Noa in principle (228 runs, E_stall_diversity_ENSEMBLE as the main model).
This plan supersedes `PLAN_NORTH_TLV_TEST_DRAFT.md`, and closes the three points she raised.

## Status of results
- The **8/9** on the valid development cases stays a reproduced **development** result (`audit/a03_valid9_reproduction.csv`, |Δ| = 0).
- This test is reported **separately**, as a generalisation check on new scenes in known (test-split) regions.
- No model, checkpoint, threshold, feature, normalisation or seed is changed because of it. No training or new experiment follows it.

## Point 1 — scene independence (checked before freezing: `data/plan_checks/independence.csv`)
- **New traffic realisations:**
  - **Background vehicles:** every run regenerates them with the corpus generator logic (`populate_vehicles.build_vehicle` / `SCENARIO_SPECS`, unchanged) and a new seed (`bg_seed`). This gives new count, lanes, positions, speeds, speed-change behaviour and weights.
  - **Simulation seed:** a new one per run (`sim_seed` = 20261010 + index). This drives new random junction choices of the background vehicles, i.e. new routes, and new stochastic speed profiles.
  - **Injected vehicles:** planned routes from start segments never used for an injection in that region (excluded per region: the newscene lanes, the original-case lanes and the C1–C4 lanes).
- **Check result:**
  - 228 / 228 scenarios differ from all 22–23 earlier scenario files of their region (md5);
  - 0 injection start segments were used before;
  - 0 of 1,708 background vehicles are identical to an earlier vehicle (lane, position, weight, target speed);
  - 131 coincide in lane and position (± 0.5 m) only — a chance overlap, with different speed and weight.
- **What is the same:** the maps, the sensors of each scenario type, the simulator and the tracker. These are not new, by design.

## Point 2 — tailgating geometry (verified before freezing: `data/plan_checks/tailgating_geometry.csv`)
- **Mechanism:** leader and follower start on the same start lane (leader at 40 m, follower at 25.5 m), with the same planned lane sequence.
  - The route prefix covers 40 + v_L·(D + 8 + 10) m.
  - No background vehicle is placed on the start lane.
- **All 12 runs pass:**
  - available route / needed route = 1.23–3.88 (minimum margin +23 %);
  - leader speed ≤ 0.85 × the lowest limit on the route;
  - every route ends at a boundary exit or within the map.
- **Combinations:**
  - v_L 6 m/s × {8, 15} s × T_f {0.20, 0.30}: 8 runs;
  - v_L 9 m/s × 8 s × T_f {0.20, 0.30}: 4 runs.
- **Matched controls:** 12, identical except the default headway.
- **Unrealised runs:** a run that does not realise is reported as unrealised, never re-run.

## Point 3 — stop definition (no overlap; the project rule)
- **Abnormal stop (stall):** longest truth stop **≥ 10.0 s** (truth v < 0.1 m/s). This is the p10 / ST01 rule.
  - START: stopped from the first to the last sample.
  - STOP_END: moved (≥ 3 m/s) before, and stopped until the last sample.
  - MSM / GENTLE: moved before, and moves ≥ 2 s after.
- **Legitimate stop:** longest truth stop **1.0 s ≤ stop < 10.0 s**, peak decel < 4 m/s², moves after.
- **Exactly 10.0 s is abnormal.** The designed holds (3 / 5 / 8 s vs 15 / 30 s) stay away from the boundary.
- **Truth used for labels only.** A run whose realised stop falls on the other side of its design is excluded and reported, not relabelled.

## Runs (`data/RUN_PLAN_final.csv`, 228)
- **7 p10 types × 12:** speeding, reckless, hard braking, weaving, straddling, collision, overweight.
  - p10 `build()` parameters, plus a planned route from a new start lane.
  - Collision target placed ≤ 130 m ahead on the route.
- **Stalled × 12:** START, STOP_END, MSM 30 s and GENTLE 15 s, 3 each, at ST01 DAS stop slots.
- **Tailgating:** 12, plus 12 controls.
- **Stop study:** 12 slots × {CONTROL, LEGIT 3 s, LEGIT 5 s, LEGIT 8 s, GENTLE 15 s} = 60.
- **Normal scenes:** 8 regions × {light, medium, heavy, heavy vehicles, stop-and-go, medium DAS-only} = 48.
- **Background types of the injection runs** cycle through medium, light, heavy, DAS-only, DAS+camera and sparse.

## Realisation and inclusion (truth only for labels)
- **p10 types:** `p10.realization` unchanged.
  - speeding / reckless: p90 speed ratio ≥ 1.10;
  - braking: peak ≥ 4 m/s²;
  - weaving: lateral std ≥ 1.0 m;
  - straddling: median |d_lat| ≥ 1.7 m;
  - collision: ≥ 1 contact;
  - overweight: ≥ 20 t.
- **Stall and stop modes:** as in Point 3.
- **Tailgating:** tailgating_v1 along the shared route ≥ 5 s, gap ≥ 1 m, no follower collision.
  - A control counts as clean if its longest tailgating condition is < 1 s and it has no collision.
- **Normals:**
  - all vehicles of the normal scenes;
  - unaffected backgrounds (stop < 2 s and gap ≥ 5 m);
  - clean controls (stop-study CONTROL and clean tailgating controls).
  - Affected backgrounds and leaders are excluded and counted.
- **Pipeline:** each run is imported with the unchanged pipeline (`read_export` → `build_channels` with the region geometry → `track_quality`, amplitude on the DL bins).
- **Usable track:** ≥ 32 valid bins and quality_ok (label-free).
- **DL decision:** a track with ≥ 3 windows. **Z-score decision:** every track it scores.
- **Vehicle outcome (evaluation only, via the truth id):** alarm if any decided track alarms; undecided if none was decided. Undecided vehicles are reported, not counted.

## Detectors (read-only, sha256-verified before scoring; no training, no calibration)
- **E_stall_diversity_ENSEMBLE (main):** 5 seeds, ensemble z with the frozen val_B constants, P99 = 1.9726181803233118.
- **DL_supervised_crossregion_P99:** P99 = −5.523461642265342.
- **Z-score (shipped, unchanged):** `model_live.pkl`, 15.392 σ.
  - Disclosed: 13 / 156 features use the truth position error.
  - A truth-free sensitivity (the same constants, those 13 features left out of max|z|) is reported as analysis only.
- **DL inference:** `code/hs_scorer.py`, verified to reproduce the stored decisions.
  - Primary unit: the track (no `vehicle_id_oracle` in inference).
  - Secondary: the original oracle-grouped vehicle.

## Reported (once)
- **Per type (9):** injected → realised → tracked → decided → detected; recall; equal-weight macro recall.
- **Stalled:** by mode. **Tailgating:** by speed and duration.
- **Stop study:** false alarms on LEGIT 3 / 5 / 8 s vs GENTLE 15 s recall.
- **False alarms:** on normal driving and on legitimate stops, separately.
- **Overall metrics:** precision, recall, F1, ROC-AUC and AUPRC on realised core anomalies vs normal driving.
- **Coverage:**
  - per-detector coverage;
  - every metric on the common decided set and on each detector's own set.
- **Uncertainty:** 95 % bootstrap over runs; paired comparison on the common set (bootstrap of the macro-recall difference, McNemar on the pooled anomalies).
- **Focus analysis:** tailgating, stalled, weaving and overweight.
- **Evaluation code:** written while the simulations run and frozen (sha256) **before any detector is run**.

## Erratum (2026-10-09, evaluation code, before any detector was run)
- **The bug.** `h03_import.py` read the plan row as a pandas Series, so `row.mode` returned the Series method `mode()` instead of the column value.
  As a result, every stop-study veh001 (CONTROL, LEGIT) fell through to the abnormal-stop label.
- **The fix.** `row["mode"]`, in the three places where it is used. Nothing else changed, and the import was re-run.
- **Disclosed:** the frozen hash of the evaluation code was updated (`FREEZE_MANIFEST_evaluation_code_sha256.txt`; the old hash is kept as `..._v1_superseded`).
