# Addendum — written after the stopval2 simulation and import, before any training or scoring of F1/F2

- **stopval2 realization.** All 120 runs were verified by the truth rules: 48 legitimate stops (3 s and 8 s), 24 GENTLE_STALL 15 s, 24 MSM 30 s and 24 controls.
  Every veh001 has a usable track. Excluded backgrounds (queued or affected): 114. Unaffected backgrounds: 573. They are scored and reported as a normal group, not gating.
- **Sensing.** The survey found no DAS_ONLY stop location among the eligible SHORT/LONG slots of these regions. The stop locations are COMBINED (14 slots) or PARTIAL (10 slots).
  The DAS_ONLY sensing condition is covered only by the 3 DAS_ONLY stallval slots (6 legitimate + 9 abnormal stops). This is reported as a limitation.
- **Stop-share design check** (`data/eval_stop_table.csv`, truth used for analysis only):
  - legitimate SHORT stops cover 0.29–0.69 of the windows;
  - GENTLE 15 s LONG covers 0.19–0.40;
  - MSM 30 s LONG covers 0.31–0.54.
  The planned overlap of stop shares across duration classes exists.
- **Smoke test.** F2 was run end to end once in the scratch area with 3 steps and 1 epoch, only to check the code path. Nothing from it is kept or used.
- **Script.** `x_train_eval_f.py`; its full diff to `x_train_eval_s.py` is in `data/DIFF_x_train_eval_f_vs_s.txt`.
- **REF re-score.** The E_stall_diversity seeds are re-scored with FARM = REF, EVAL_ONLY = 1, their own checkpoints and their own frozen thresholds.
  The existing groups are compared with the saved scores.

## Operational note (during training, no results used)
- 11:17 F2 seed 4 was started in parallel on the CPU freed by the finished F1 chain (identical command; log logs_F2_s4_parallel.log). When the sequential chain reaches seed 4 the script refuses to retrain (done.json exists) and only writes a short refusal to logs_F2_s4.log.
