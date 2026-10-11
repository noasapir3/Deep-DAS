# Diagnosis of the alarms on legitimate stops (read-only, 2026-10-09)
No model, threshold or label was changed. The frozen E_div and C_new checkpoints were only loaded for scoring. The shipped Z-score model was copied read-only
(sha256 0acc8803…, identical to anomaly_model/outputs and DL/data/fp/zscore), and it reproduces the published anomaly scores to 5e-4.
- `d01_zscore_stallval.py`: the shipped Z-score on the 30 stallval runs → `zscore_stallval_*.csv`
- `d02_exposure_and_windows.py`:
  - exact rng replay of the E_div training draws, which matches the runs' own exposure logs in 107 of 107 epochs → `exposure_replay_*.csv`;
  - window counterfactual (all windows / without the windows that overlap the truth stop) → `window_counterfactual*.csv`
- `d03_visibility_and_train_fit.py`:
  - the truth stop duration compared with what the tracked speed shows → `visibility.csv`;
  - in-sample alarms on the training legit stops → `train_fit.csv`
