# Limitations and caveats: DL_supervised_crossregion_P99

**Status:** simulation / development only. This is not independent confirmation.

**Evidence used here:**
- the original experiment (`evaluation/EXPERIMENT_REPORT.md`);
- two later read-only val_A checks that scored **this exact checkpoint with its frozen thresholds**:
  - `DL/diagnostics/mild_tailgating_valA_check_20260928_185343/`;
  - `DL/diagnostics/mild_tailgating_valA_batch2_20260928_190604/`.

  Only this model's rows from those checks are quoted. The later `tgv1` model is not part of this release.

## 1. Mild tailgating: unresolved weakness

| Case (truth under the frozen `tailgating_v1` definition) | Logit (P99 −5.52) | P99 / P95 |
|---|---|---|
| C3, realized mild tailgating (evaluation region 062) | – | no / no |
| v089, realized mild (val_A 089; headway median 0.70 s; 16.7 s) | −7.97 | no / **yes** |
| w047, realized mild, isolated matched pair (val_A 047; 0.60 s; 13.2 s; input-visible) | −16.52 | no / no |
| w102, realized mild (val_A 102; 0.60 s; confounded pair: lateral difference, route-end stop) | −18.76 | no / no |
| r002, realized close tailgating (val_A selection set) | – | no / yes |
| Matched normal controls (v089, v122, w047, w102) | −23.9 to −15.8 | 0 of 4 alarm |

- **Detection:** 0 of 3 realized mild tailgaters are caught at P99, and 1 of 3 at P95.
- **Scores do move:** the tailgater scores above its matched control in every pair (by +3.8 to +10.2 logits), but not enough to cross P99.
- **Failed injection v122** (no tailgating_v1 event; desired speed 1.12 × limit): it alarms at P99 (−3.59). This is neither a hit nor a clean false alarm.
- **Training coverage:** the training data contained only two tailgating positives (r131, r090), both close. There were no mild ones.

## 2. Weaving is not fully stable

- **Original weaving** is caught at P95 only (no / yes).
- **val_A weaving:** 1 of 2 at P99 and 2 of 2 at P95 (these vehicles were used for selection).
- **The earlier same-scene pilot** caught weaving at P99. That gain did not survive cross-region training.

## 3. Collision alarms are not real-time impact detection

- **Vehicle-level score:** a collision alarm is a score over the vehicle's whole usable track, computed after the fact.
- **No timing:** the model outputs no time of impact and has no causal or online component.
- **Unclear cause:** an alarm can be driven by the stop, deceleration or behaviour around the event rather than the contact itself.
- **Wording:** describe it as "vehicle flagged in a recording that contains a collision", not as impact detection.

## 4. Other weak or missing types

- **Speeding:** 0 of 2 at P99 and 1 of 2 at P95 in val_A, although the original speeding case is caught.
- **Overweight and reckless:** 1 of 2 each in val_A at P99.
- **Unrealized original events:** the original tailgating and straddling events were not physically realized, so no detector can be credited or blamed for them.

## 5. Operating point and false alarms

- **P99 is the primary operating point.**
  - Normal test: 7 of 513 sufficient-evidence vehicles alarm; 119 have insufficient evidence.
  - Normal val_A: 17 of 1,313 alarm; 273 insufficient.
  - Unaffected background vehicles in the later val_A checks: 0 of 32 decided alarm (6 insufficient).
- **P95 transfers badly:** 49 of 513 alarm on test and 112 of 1,313 on val_A, far above the 5 % design rate.
- **Insufficient evidence is frequent:** about 17–19 % of normal vehicles get no decision. They are never counted as normal.

## 6. Evidence quality

- **Thin training and selection data:**
  - one training run, seed 0;
  - at most 4 training anomalies per type, and 2 per type for selection;
  - the best epoch was the last allowed epoch (24 of 25).
- **Not independent:** the evaluation cases (originals, C1–C4, K1–K3) had been inspected before model and threshold choices, and the val_A anomalies and normals were used for checkpoint selection.
- **Shared mechanisms:** C1 and C2 were injected with the same mechanisms used in training, so their detection shows transfer across regions, not across mechanisms.
- **Imperfect matching:** simulation controls matched by settings are not automatically isolated. Lateral noise and route effects confounded one of the later pairs.
