# Numerical differences between CPU architectures

The release was built and its outputs were produced on **Linux x86-64** (Python 3.13.16 with the pinned libraries of `requirements.txt`, CPU torch). It was then installed from scratch and run on **Linux aarch64 (ARM)**: Python 3.13.15, the same pinned versions, CPU torch. This was the virtual machine of the user's computer.

The model is identical on both: the same files, verified by sha256. The only source of difference is float32 arithmetic, whose accumulation order depends on the CPU and its math library.

## 1. Development cases (E2)

| Quantity | x86-64 (build) | aarch64 | E2 v1.0 (logits, z, ensemble ≤ 1e-5) | E2 v1.1 (z, ensemble ≤ 1e-5; decisions and tables identical) |
|---|---|---|---|---|
| Raw seed logits, max \|Δ\| | 6.7e-6 | **1.72e-5** | x86 pass / **ARM fail** | reported only |
| Per-seed z, max \|Δ\| | 1.2e-6 | 3.4e-6 | pass / pass | pass / pass |
| Ensemble score, max \|Δ\| | 5.9e-7 | 1.6e-6 | pass / pass | pass / pass |
| Decisions (8/9 DeepDAS, 7/9 single model) | identical | identical | pass / pass | pass / pass |
| Development tables | byte-identical | byte-identical | pass / pass | pass / pass |

- **The reference** is the stored scores of the selection round.
- **Why x86 shows 6.7e-6.** That reference was produced by the training-time evaluation, which scores vehicles in other window batches. The per-unit scoring that produced the original development table is reproduced exactly.
- **Distance from the threshold.** The closest development case to the threshold is 0.029 away from it (overweight, score 1.865), about 18,000 times the largest ensemble deviation.

## 2. Validation and final test on aarch64 (E3)

Every file below was regenerated on ARM and compared with the shipped x86 results.

| File | Rows | Raw logits, max \|Δ\| (reported) | z, max \|Δ\| | Ensemble, max \|Δ\| | Decisions, windows, labels |
|---|---|---|---|---|---|
| seed 0–4 `vehicle_scores.csv` | 5,095 each | 1.6e-5 – 3.4e-5 | – | – | identical (P99 and P95) |
| seed 0–4 `vehicle_scores_legitimate_stop_set.csv` | 693 each | 1.4e-5 – 2.9e-5 | – | – | identical |
| `ensemble_scores.csv` | 5,095 | – | ≤ 6.7e-6 | 1.6e-6 | identical |
| `ensemble_scores_legitimate_stop_set.csv` | 693 | – | ≤ 5.0e-6 | 1.8e-6 | identical |
| `validation_metrics.csv` | 6 | – | – | recomputed diagnostic P99: 1.1e-9 | all counts and recalls identical |
| `scores_dl_tracks.csv` | 1,592 | seeds ≤ 3.1e-5; single model 3.2e-5 | ≤ 5.2e-6 (derived) | 1.8e-6 | identical |
| `scores_dl_oracle_vehicles.csv` | 1,572 | the same | ≤ 5.2e-6 (derived) | 1.8e-6 | identical |
| `vehicle_outcomes.csv` | 2,068 | single model ≤ 3.2e-5 | – | 1.8e-6 | identical; Z-score columns exactly equal |
| T1–T7, `FINAL_TEST_REPORT.md`, development tables | – | – | – | – | **byte-identical** |

**The relative size of the logit differences.** The raw logits are of order 1–20 and the val_B standard deviations are 2.8–6.7, so a logit difference of 3e-5 is a relative difference of about 1e-6 and a z difference below 1e-5.

## 2b. Import from raw data on aarch64 (E4)
Three final-test runs were re-imported on ARM from the raw-data package.
- **Channels and track metadata:** bit-identical.
- **DAS log-amplitude:** identical except for the last bit of some float64 values (≤ 8.9e-16). This comes from the `log10` of the system math library.
- **Scoring:** the re-imported runs, scored with `scripts/score_tracks.py`, give identical decisions and window counts, with ensemble scores within 6.1e-7.

## 3. Decisions near the thresholds

For each decision rule, the table gives the decided unit closest to its threshold on x86, compared with the largest deviation observed on ARM.

| Decision | Units | Closest unit to the threshold | Largest ARM deviation | Ratio | Decisions changed on ARM |
|---|---|---|---|---|---|
| **DeepDAS Ensemble v1**, final test (tracks) | 1,327 | 2.5e-3 | 1.8e-6 | 1,400× | 0 |
| **DeepDAS Ensemble v1**, final test (vehicles) | 1,312 | 2.5e-3 | 1.8e-6 | 1,400× | 0 |
| **DeepDAS Ensemble v1**, validation | 5,026 | 1.7e-3 | 1.6e-6 | 1,035× | 0 |
| **DeepDAS Ensemble v1**, legitimate-stop set | 563 | 3.3e-3 | 1.2e-6 | 2,850× | 0 |
| Supervised single model (logit), final test | 1,327 | 5.7e-3 | 1.9e-5 | 300× | 0 |
| Single seeds at their own P99 (reference only) | 5,026 each | ≥ 5.6e-4 | ≤ 3.4e-5 | ≥ 17× | 0 |
| Single seeds at their own P95 (reference only) | 5,026 each | seed 1: **2.1e-5**; others ≥ 4.3e-4 | ≤ 3.4e-5 | seed 1: 0.85× | 0 |

**Conclusion.** Every decision of DeepDAS Ensemble v1 and of the single model is at least 300 times farther from its threshold than the largest cross-architecture deviation. None changed.

**The one borderline case.** It is not a decision of the released model. Validation vehicle `018/normal_heavy_traffic|veh015` lies 2.1e-5 above **seed 1's own P95**, an alternative per-seed operating point stored for reference.
- Its logit moved by 8e-6 on ARM, and that per-seed decision stayed the same.
- On another platform such a per-seed P95 flag could change.
- Its ensemble score is 1.13, far below the ensemble threshold 1.97, so the decision of DeepDAS Ensemble v1 is unaffected.

## 4. What this means for users
- **Exact reproduction.** On Linux x86-64 with the pinned versions, every file is reproduced byte for byte, and retraining reproduces the checkpoints byte for byte (E5).
- **On another architecture** (tested: Linux aarch64):
  - all decisions and all result tables are identical;
  - z and ensemble scores agree within 1e-5;
  - raw logits agree within about 3e-5.
- **Retraining** (E5) was verified only on x86-64. On another architecture the selected epochs, and so the weights, could differ. The released model is the frozen one in any case.
