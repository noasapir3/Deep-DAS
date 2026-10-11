# W1 / W2: the last improvement round. FROZEN plan (2026-10-10)
Approved by Noa on 2026-10-10 (14:12). This file = `PLAN_W1_W2_v2_FOR_APPROVAL.md` (sampling and selection rule, unchanged) + `PLAN_W1_W2_DRAFT_FOR_APPROVAL.md` (data) + the approval notes + the feasibility outcome.
- Frozen by sha256 in `plan/FREEZE_MANIFEST_W_sha256.txt`, before any simulation of the batch.
- The training script and the evaluation / selection code are frozen in a second manifest (`plan/FREEZE_MANIFEST_W_code_sha256.txt`). That happens **before any model, including the frozen E_DIV, is scored on the new validation data**, and before any training.

## Approval notes (recorded at Noa's request)
1. **W2 changes the sampling composition, including the normal : anomaly ratio.**
   - In W2, the positive share of each batch rises from 25 % (E_DIV, W1) to **37.5 %**: 40 normals + 24 positives in a batch of 64.
   - So W2 vs W1 tests a change of composition, which includes that ratio, at an identical batch size and training budget. This is part of the experiment by design.
2. **C3 is a conservative preservation condition, not a statistical test.**
   - It requires that no type loses more than one detected vehicle relative to E_DIV on the same validation set.
   - With 8–21 vehicles per type it cannot establish equivalence, and it is not meant to. It is a guard against visible losses.
   - The uncertainty of the gains is reported separately (bootstrap over runs), for information only.
3. **Protection of the frozen model.**
   - E_stall_diversity_ENSEMBLE and its full backup (`DL/FROZEN_FINAL_MODEL__E_stall_diversity_ENSEMBLE__20261010/`) stay unchanged.
   - `scripts/w00_hash_guard.py` re-hashes all 5,414 original files and 5,422 backup files at the start and end of every stage, and aborts on any difference.
   - New work only in `experiments_followup/weak_types_W1W2_20261010/`.
4. **End point.**
   - If no candidate passes C1–C9, E_stall_diversity_ENSEMBLE stays final and development closes.
   - If one passes, it is frozen as a new candidate. The existing model is not replaced or deleted, and a new test plan is presented for approval before any test.
   - No further experiments and no architecture change without Noa's explicit approval.

## Feasibility (dry runs DISCARDED: `sims_feasibility_DISCARDED/`, never imported or used; results in `data/FEASIBILITY_RESULTS.txt`)
All six new mechanisms work. Each was checked against its realisation rule:
- **Partial and to-stop braking:** peak 6.0 and 8.0 m/s², at the planned times.
- **Segment speeding:** window p90 1.28 at ×1.30. The fixed-route version reaches 1.32.
- **Segment weaving:** window lateral std 1.06.
- **Collision with a moving target (4 m/s, Δv 5 m/s):** contact.
- **Overweight at 40 t:** realised.
- **Segment straddling:** realised.

Adjustments made **before freezing**:
- **Straddling offset 1.7 m removed.** A fixed 1.7 m offset realises a median \|d_lat\| of exactly 1.70, which is borderline for the unchanged ≥ 1.7 rule. The grid is now **1.8 / 1.9 / 2.0 / 2.1 m**, on alternating sides.
- **Control clean rule.**
  - Normal vehicles of the simulator (backgrounds) routinely reach a truth speed ratio of 1.10–1.12. The simulator caps at 1.12× the limit, and backgrounds are never checked for speed.
  - A control is therefore *clean* if its **targeted property is on the legal side** of the matching anomaly rule over the whole truth track, with no stop ≥ 1 s and no collision:
    - firm braking: peak < 4 m/s²;
    - fast legal: p90 speed ratio < 1.10;
    - heavy legal: < 20 t;
    - lateral: median \|d_lat\| < 1.7 m and lateral std < 1.0 m.
  - The fast-legal target is set relative to the **lowest limit on the route**: 0.90 / 0.95 / 1.00, so that it stays legal everywhere along the route.
  - A control that is not clean is excluded and reported.

## Data (`data/RUN_PLAN_W.csv`, 314 runs; `scripts/w03_batch.py`)

| | Train (98 train regions) | Val (21 val_A regions) |
|---|---|---|
| Hard braking: decel 5 / 6 / 7 / 8 × {partial 40 / 55 / 70 % drop, to near-stop} × t = 6 / 12 / 20 s | 24 | 8 |
| Speeding: ×1.30 / 1.45 / 1.60 / 1.80, whole route or segment [3, 18] / [3, 23] s | 24 (12 + 12) | 8 (4 + 4) |
| Overweight: 20 / 25 / 30 / 40 t × DAS-only / DAS+camera / sparse × 0.7 / 0.95 × limit | 24 | 8 |
| Straddling: ±1.8 / 1.9 / 2.0 / 2.1 m, whole route or segment [3, 23] s | 24 (12 + 12) | 8 (4 + 4) |
| Weaving: amplitude 1.5 / 1.8 / 2.2 m × period 2 / 3 / 4.5 s, whole route or segment [3, 23] s | 24 (12 + 12) | 8 (4 + 4) |
| Collision: Δv 5 / 8 / 12 m/s × target stopped / 2 / 3 / 4 m/s × 60–130 m ahead | 24 | 8 |
| Firm legal braking: 2.5 / 3.0 / 3.5 m/s², partial | 24 | 8 |
| Fast legal driving | 16 | 6 |
| Legal heavy vehicles, 8–15 t | 16 | 6 |
| Lateral normal: offset 0.5–0.9 m, or gentle weave 0.4–0.6 m / 6–10 s | 16 | 6 |
| Reckless (p10 design; measurement only) | – | 8 |
| Tailgating + matched control (final-test mechanism; measurement only) | – | 8 + 8 |

**Independence (`data/plan_checks/independence_W.csv`, checked before simulation):**
- train and val regions are disjoint, and only *train* / *val_A* regions are used (no test, no val_B);
- 0 start segments were used by any earlier injection in their region, and start lanes are unique;
- 0 / 314 scenarios have the md5 of an earlier scenario;
- 0 / 2,828 background vehicles are identical to an earlier vehicle (16 coincide in lane and position only);
- seeds: sim 31,000,000+i and background 32,000,000+i, disjoint from every earlier batch.

**Realisation (truth used for labels only):**
- The p10 thresholds are unchanged. Partial events are evaluated inside the planned window; hard braking after its planned time.
- Runs that do not realise are excluded and reported, never re-run or relabelled.
- **Import:** the unchanged pipeline (h03 logic: `read_export` → `build_channels` → `track_quality`, with amplitude on the DL bins).

## Training (W1, W2) and selection: exactly as `PLAN_W1_W2_v2_FOR_APPROVAL.md` §5 and §6
- **Both arms:** batch 64, 120 steps × 25 epochs, patience 6, the same loss, learning rates, initialisation, seeds 0–4, the checkpoint rule (best AP on the expanded validation) and the threshold rule (ensemble P99 on val_B).
- **W1:** uniform 48 / 16.
- **W2:** 36 normals from the original normals + new backgrounds, 4 from the new targeted controls, 16 positives exactly as E_DIV, 8 from the new positives.
- **Selection rule:** C1–C9 exactly as specified in v2 §6, unchanged.

## Erratum 1 (2026-10-10, import code, before any import output existed and before any training or scoring)
`scripts/w05_import.py` lacked `import numpy as np`, which the a02 amplitude function (`amp_on_bins`, executed from the a02 source exactly as in h03) needs. The first import call stopped at the first run with a NameError and wrote nothing.
- **The fix:** the one import line. Nothing else changed.
- **Code manifest re-frozen:** `FREEZE_MANIFEST_W_code_sha256.txt`. The previous one is kept as `_v1_superseded`.
