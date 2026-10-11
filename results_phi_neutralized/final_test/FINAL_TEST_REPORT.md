> **phi_mode = neutralized** - inference-time intervention on the frozen hybrid model (standardised phi_u input held at 0). Post-hoc analysis; not a model trained without phi_u; not an independent evaluation. 'DeepDAS Ensemble v1' below means this neutralized configuration.

# Final test: North-TLV test regions (independent, run once)

Frozen detectors and thresholds. Deployment unit = one usable track (no ground truth in inference); a vehicle counts as detected if any of
its decided tracks alarms (the truth vehicle id is used only for this evaluation mapping).

## T1 Per anomaly type: injected, realised, tracked, decided, detected

| type               |   injected |   realized |   realized_tracked |   realized_usable_track |   DeepDAS Ensemble v1 decided |   DeepDAS Ensemble v1 detected |   DeepDAS Ensemble v1 recall |   Supervised single model decided |   Supervised single model detected |   Supervised single model recall |   SimStudio Z-score decided |   SimStudio Z-score detected |   SimStudio Z-score recall |   common_n |   DeepDAS Ensemble v1 recall common |   Supervised single model recall common |   SimStudio Z-score recall common |
|:-------------------|-----------:|-----------:|-------------------:|------------------------:|------------------------------:|-------------------------------:|-----------------------------:|----------------------------------:|-----------------------------------:|---------------------------------:|----------------------------:|-----------------------------:|---------------------------:|-----------:|------------------------------------:|----------------------------------------:|----------------------------------:|
| Speeding           |         12 |         12 |                 12 |                      12 |                            12 |                              3 |                        0.25  |                                12 |                                  1 |                            0.083 |                          12 |                            0 |                      0     |         12 |                               0.25  |                                   0.083 |                             0     |
| Reckless driving   |         12 |         11 |                 11 |                      11 |                            10 |                              7 |                        0.7   |                                10 |                                  1 |                            0.1   |                          11 |                            0 |                      0     |         10 |                               0.7   |                                   0.1   |                             0     |
| Hard braking       |         12 |         12 |                 12 |                      11 |                            11 |                              1 |                        0.091 |                                11 |                                  0 |                            0     |                          12 |                            0 |                      0     |         11 |                               0.091 |                                   0     |                             0     |
| Stalled vehicle    |         12 |         12 |                 12 |                      12 |                            12 |                             12 |                        1     |                                12 |                                 12 |                            1     |                          12 |                           12 |                      1     |         12 |                               1     |                                   1     |                             1     |
| Lane weaving       |         12 |         12 |                 12 |                      11 |                            11 |                              7 |                        0.636 |                                11 |                                  3 |                            0.273 |                          12 |                            0 |                      0     |         11 |                               0.636 |                                   0.273 |                             0     |
| Lane straddling    |         12 |         12 |                 12 |                      12 |                            12 |                              5 |                        0.417 |                                12 |                                  8 |                            0.667 |                          12 |                            0 |                      0     |         12 |                               0.417 |                                   0.667 |                             0     |
| Collision          |         12 |         10 |                 10 |                      10 |                            10 |                              7 |                        0.7   |                                10 |                                  9 |                            0.9   |                          10 |                           10 |                      1     |         10 |                               0.7   |                                   0.9   |                             1     |
| Overweight vehicle |         12 |         12 |                 12 |                      12 |                            12 |                              4 |                        0.333 |                                12 |                                  3 |                            0.25  |                          12 |                            0 |                      0     |         12 |                               0.333 |                                   0.25  |                             0     |
| Tailgating         |         12 |         12 |                 12 |                      11 |                            11 |                             10 |                        0.909 |                                11 |                                  0 |                            0     |                          12 |                            1 |                      0.083 |         11 |                               0.909 |                                   0     |                             0.091 |

## T2 Overall: realised anomalies vs normal driving

| detector                |   macro_recall_own |   macro_recall_common |   pooled_recall | FA_normal_driving   | FA_normal_common   | FA_legit_stops   | FA_legit_common   |   precision |    F1 |   ROC_AUC |   AUPRC | coverage_anomalies   | coverage_normals   | coverage_legit   |
|:------------------------|-------------------:|----------------------:|----------------:|:--------------------|:-------------------|:-----------------|:------------------|------------:|------:|----------:|--------:|:---------------------|:-------------------|:-----------------|
| DeepDAS Ensemble v1     |              0.56  |                 0.56  |           0.554 | 20/1008 = 0.0198    | 20/1008 = 0.0198   | 19/28            | 19/28             |       0.737 | 0.633 |     0.906 |   0.737 | 101/105              | 1008/1708          | 28/32            |
| Supervised single model |              0.364 |                 0.364 |           0.366 | 5/1008 = 0.0050     | 5/1008 = 0.0050    | 14/28            | 14/28             |       0.881 | 0.517 |     0.769 |   0.515 | 101/105              | 1008/1708          | 28/32            |
| SimStudio Z-score       |              0.231 |                 0.232 |           0.219 | 13/1574 = 0.0083    | 7/1008 = 0.0069    | 23/32            | 22/28             |       0.639 | 0.326 |     0.783 |   0.394 | 105/105              | 1574/1708          | 32/32            |

## T3 Stops and tailgating

| group                                            |   n | DeepDAS Ensemble v1   | Supervised single model   | SimStudio Z-score   |
|:-------------------------------------------------|----:|:----------------------|:--------------------------|:--------------------|
| Legitimate stop, 3 s hold (false alarm)          |   9 | 3/8                   | 3/8                       | 4/9                 |
| Legitimate stop, 5 s hold (false alarm)          |  12 | 7/11                  | 4/11                      | 9/12                |
| Legitimate stop, 8 s hold (false alarm)          |  11 | 9/9                   | 7/9                       | 10/11               |
| Gentle 15 s stall, stop study (detection)        |  11 | 9/10                  | 8/10                      | 10/11               |
| Stop-study control, no stop (false alarm)        |  12 | 0/12                  | 0/12                      | 0/12                |
| Stalled from the start (detection)               |   3 | 3/3                   | 3/3                       | 3/3                 |
| Stops and stays stopped (detection)              |   3 | 3/3                   | 3/3                       | 3/3                 |
| Stops 30 s mid-route, then drives on (detection) |   3 | 3/3                   | 3/3                       | 3/3                 |
| Gentle 15 s stall (detection)                    |   3 | 3/3                   | 3/3                       | 3/3                 |
| Tailgating, leader 6 m/s, 8 s (detection)        |   4 | 3/3                   | 0/3                       | 0/4                 |
| Tailgating, leader 6 m/s, 15 s (detection)       |   4 | 4/4                   | 0/4                       | 0/4                 |
| Tailgating, leader 9 m/s, 8 s (detection)        |   4 | 3/4                   | 0/4                       | 1/4                 |
| Tailgating matched controls (false alarm)        |  12 | 7/11                  | 0/11                      | 0/12                |

## T4 Secondary variants

| variant                                           |   Speeding |   Reckless driving |   Hard braking |   Stalled vehicle |   Lane weaving |   Lane straddling |   Collision |   Overweight vehicle |   Tailgating |   macro | FA_normal   | FA_legit   |
|:--------------------------------------------------|-----------:|-------------------:|---------------:|------------------:|---------------:|------------------:|------------:|---------------------:|-------------:|--------:|:------------|:-----------|
| DeepDAS Ensemble v1, vehicle units                |      0.25  |                0.7 |          0.091 |             1     |          0.636 |             0.417 |         0.7 |                0.333 |        0.909 |   0.56  | 20/1009     | 19/28      |
| Supervised single model, vehicle units            |      0.083 |                0.1 |          0     |             0.917 |          0.273 |             0.667 |         0.9 |                0.25  |        0     |   0.354 | 5/1009      | 14/28      |
| SimStudio Z-score without truth-position features |      0     |                0   |          0     |             1     |          0     |             0     |         1   |                0     |        0.083 |   0.231 | 13/1574     | 23/32      |

## T5 Uncertainty (bootstrap over runs, common decided set) and McNemar tests

```
{
 "bootstrap_95ci": {
  "macro recall DeepDAS Ensemble v1 (common)": "[0.475, 0.640]",
  "macro recall Supervised single model (common)": "[0.303, 0.426]",
  "macro recall SimStudio Z-score (common)": "[0.222, 0.256]",
  "FA normal DeepDAS Ensemble v1 (common)": "[0.011, 0.029]",
  "FA normal Supervised single model (common)": "[0.001, 0.010]",
  "FA normal SimStudio Z-score (common)": "[0.003, 0.012]",
  "macro diff DeepDAS Ensemble v1 - SimStudio Z-score": "[0.238, 0.409]",
  "macro diff DeepDAS Ensemble v1 - Supervised single model": "[0.109, 0.284]",
  "P(DeepDAS Ensemble v1 macro > SimStudio Z-score macro)": 1.0,
  "P(DeepDAS Ensemble v1 macro > CR macro)": 1.0
 },
 "mcnemar": {
  "DeepDAS Ensemble v1 vs SimStudio Z-score (pooled common anomalies)": {
   "only DeepDAS Ensemble v1": 36,
   "only SimStudio Z-score": 3,
   "exact McNemar p": 0.0
  },
  "DeepDAS Ensemble v1 vs SimStudio Z-score (common normals, false alarms)": {
   "only DeepDAS Ensemble v1": 20,
   "only SimStudio Z-score": 7
  },
  "DeepDAS Ensemble v1 vs Supervised single model (pooled common anomalies)": {
   "only DeepDAS Ensemble v1": 26,
   "only Supervised single model": 7,
   "exact McNemar p": 0.0013
  },
  "DeepDAS Ensemble v1 vs Supervised single model (common normals, false alarms)": {
   "only DeepDAS Ensemble v1": 19,
   "only Supervised single model": 4
  }
 }
}
```
