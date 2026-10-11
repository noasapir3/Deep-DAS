# New-scene test batch — pre-registered (written before any run)

**Purpose:** an evaluation set of anomalies in scenes that were never used for training, model selection, threshold calibration or diagnosis.

**Scenes:** the `normal_medium_traffic` scenario of each of the 8 TEST-split regions (003, 010, 020, 037, 062, 083, 120, 140).
- These regions were never used for training (normals or anomalies), checkpoint selection (val_A) or threshold calibration (val_B).
- These specific scenarios were never injected or diagnosed. The earlier diagnosed cases (the 9 originals, C1–C4) are different scenario files in the same maps.
- Disclosed overlap: the background vehicles of these scenes are also test-normal vehicles. Only the injected vehicle is scored as a positive here;
  false alarms are still measured on the existing 513 sufficient-evidence test normals.

**Injection:** byte-identical `build()` of the cross-region p10 (same 9 types, same parameters, seed 1, veh001 on the region's longest lane).
**Labels:** the same pre-registered truth realization rules (p10 `realization()`); only verified injected vehicles count as positives.
**Import:** the same p11 pipeline (channels, quality gate, amplitude on DL bins).
**Use:** evaluation ONLY, after every model (baseline, A, B, C, D) and its thresholds are frozen. Never used for any choice.
