# Final generalisation test — North-TLV test regions (single evaluation)

Frozen detectors; thresholds unchanged; DL decisions per track (no oracle id in inference); the vehicle outcome = any decided track alarms (truth id used only for evaluation).

## T1 Per type: injected → realised → tracked → decided → detected

| type               |   injected |   realized |   realized_tracked |   realized_usable_track |   E_DIV_decided |   E_DIV_detected |   E_DIV_recall |   CR_P99_decided |   CR_P99_detected |   CR_P99_recall |   Z_decided |   Z_detected |   Z_recall |   common_n |   E_DIV_recall_common |   CR_P99_recall_common |   Z_recall_common |
|:-------------------|-----------:|-----------:|-------------------:|------------------------:|----------------:|-----------------:|---------------:|-----------------:|------------------:|----------------:|------------:|-------------:|-----------:|-----------:|----------------------:|-----------------------:|------------------:|
| speeding           |         12 |         12 |                 12 |                      12 |              12 |                4 |          0.333 |               12 |                 1 |           0.083 |          12 |            0 |      0     |         12 |                 0.333 |                  0.083 |             0     |
| reckless_driver    |         12 |         11 |                 11 |                      11 |              10 |                7 |          0.7   |               10 |                 1 |           0.1   |          11 |            0 |      0     |         10 |                 0.7   |                  0.1   |             0     |
| hard_braking       |         12 |         12 |                 12 |                      11 |              11 |                1 |          0.091 |               11 |                 0 |           0     |          12 |            0 |      0     |         11 |                 0.091 |                  0     |             0     |
| stalled_vehicle    |         12 |         12 |                 12 |                      12 |              12 |               12 |          1     |               12 |                12 |           1     |          12 |           12 |      1     |         12 |                 1     |                  1     |             1     |
| lane_weaving       |         12 |         12 |                 12 |                      11 |              11 |                7 |          0.636 |               11 |                 3 |           0.273 |          12 |            0 |      0     |         11 |                 0.636 |                  0.273 |             0     |
| lane_straddling    |         12 |         12 |                 12 |                      12 |              12 |                5 |          0.417 |               12 |                 8 |           0.667 |          12 |            0 |      0     |         12 |                 0.417 |                  0.667 |             0     |
| collision          |         12 |         10 |                 10 |                      10 |              10 |                6 |          0.6   |               10 |                 9 |           0.9   |          10 |           10 |      1     |         10 |                 0.6   |                  0.9   |             1     |
| overweight_vehicle |         12 |         12 |                 12 |                      12 |              12 |                4 |          0.333 |               12 |                 3 |           0.25  |          12 |            0 |      0     |         12 |                 0.333 |                  0.25  |             0     |
| tailgating         |         12 |         12 |                 12 |                      11 |              11 |               11 |          1     |               11 |                 0 |           0     |          12 |            1 |      0.083 |         11 |                 1     |                  0     |             0.091 |

## T2 Overall (realised core anomalies vs normal driving)

| detector   |   macro_recall_own |   macro_recall_common |   pooled_recall | FA_normal_driving   | FA_normal_common   | FA_legit_stops   | FA_legit_common   |   precision |    F1 |   ROC_AUC |   AUPRC | coverage_anomalies   | coverage_normals   | coverage_legit   |
|:-----------|-------------------:|----------------------:|----------------:|:--------------------|:-------------------|:-----------------|:------------------|------------:|------:|----------:|--------:|:---------------------|:-------------------|:-----------------|
| E_DIV      |              0.568 |                 0.568 |           0.564 | 20/1008 = 0.0198    | 20/1008 = 0.0198   | 20/28            | 20/28             |       0.74  | 0.64  |     0.908 |   0.742 | 101/105              | 1008/1708          | 28/32            |
| CR_P99     |              0.364 |                 0.364 |           0.366 | 5/1008 = 0.0050     | 5/1008 = 0.0050    | 14/28            | 14/28             |       0.881 | 0.517 |     0.769 |   0.515 | 101/105              | 1008/1708          | 28/32            |
| Z          |              0.231 |                 0.232 |           0.219 | 13/1574 = 0.0083    | 7/1008 = 0.0069    | 23/32            | 22/28             |       0.639 | 0.326 |     0.783 |   0.394 | 105/105              | 1574/1708          | 32/32            |

## T3 Stops and tailgating detail

| group                              |   n | E_DIV   | CR_P99   | Z     |
|:-----------------------------------|----:|:--------|:---------|:------|
| LEGIT 3 s (false alarm)            |   9 | 3/8     | 3/8      | 4/9   |
| LEGIT 5 s (false alarm)            |  12 | 8/11    | 4/11     | 9/12  |
| LEGIT 8 s (false alarm)            |  11 | 9/9     | 7/9      | 10/11 |
| GENTLE 15 s stop study (detection) |  11 | 9/10    | 8/10     | 10/11 |
| stop-study CONTROL (false alarm)   |  12 | 0/12    | 0/12     | 0/12  |
| stalled START (detection)          |   3 | 3/3     | 3/3      | 3/3   |
| stalled STOP_END (detection)       |   3 | 3/3     | 3/3      | 3/3   |
| stalled MSM (detection)            |   3 | 3/3     | 3/3      | 3/3   |
| stalled GENTLE_STALL (detection)   |   3 | 3/3     | 3/3      | 3/3   |
| tailgating v6 D8 (detection)       |   4 | 3/3     | 0/3      | 0/4   |
| tailgating v6 D15 (detection)      |   4 | 4/4     | 0/4      | 0/4   |
| tailgating v9 D8 (detection)       |   4 | 4/4     | 0/4      | 1/4   |
| tailgating controls (false alarm)  |  12 | 7/11    | 0/11     | 0/12  |

## T4 Secondary variants

| variant                        |   speeding |   reckless_driver |   hard_braking |   stalled_vehicle |   lane_weaving |   lane_straddling |   collision |   overweight_vehicle |   tailgating |   macro | FA_normal   | FA_legit   |
|:-------------------------------|-----------:|------------------:|---------------:|------------------:|---------------:|------------------:|------------:|---------------------:|-------------:|--------:|:------------|:-----------|
| E_DIV (oracle-grouped units)   |      0.333 |               0.7 |          0.091 |             1     |          0.636 |             0.417 |         0.6 |                0.333 |        1     |   0.568 | 20/1009     | 19/28      |
| CR_P99 (oracle-grouped units)  |      0.083 |               0.1 |          0     |             0.917 |          0.273 |             0.667 |         0.9 |                0.25  |        0     |   0.354 | 5/1009      | 14/28      |
| Z-score truth-free sensitivity |      0     |               0   |          0     |             1     |          0     |             0     |         1   |                0     |        0.083 |   0.231 | 13/1574     | 23/32      |

## T5 Uncertainty (bootstrap over runs, common decided set) and McNemar
```
{
 "bootstrap_95ci": {
  "macro recall E_DIV (common)": "[0.486, 0.646]",
  "macro recall CR_P99 (common)": "[0.303, 0.426]",
  "macro recall Z (common)": "[0.222, 0.256]",
  "FA normal E_DIV (common)": "[0.011, 0.030]",
  "FA normal CR_P99 (common)": "[0.001, 0.010]",
  "FA normal Z (common)": "[0.003, 0.012]",
  "macro diff E_DIV - Z": "[0.249, 0.417]",
  "macro diff E_DIV - CR_P99": "[0.119, 0.290]",
  "P(E_DIV macro > Z macro)": 1.0,
  "P(E_DIV macro > CR macro)": 1.0
 },
 "mcnemar": {
  "E_DIV vs Z (pooled common anomalies)": {
   "only E_DIV": 38,
   "only Z": 4,
   "exact McNemar p": 0.0
  },
  "E_DIV vs Z (common normals, false alarms)": {
   "only E_DIV": 19,
   "only Z": 6
  },
  "E_DIV vs CR_P99 (pooled common anomalies)": {
   "only E_DIV": 28,
   "only CR_P99": 8,
   "exact McNemar p": 0.0012
  },
  "E_DIV vs CR_P99 (common normals, false alarms)": {
   "only E_DIV": 19,
   "only CR_P99": 4
  }
 }
}
```

## T6 Focus runs

| vk                                    | run_id                         | type               | mode               |   region | bg_type                    | label            | usable   |   v_lead |   dur_target_s |   E_DIV_score | E_DIV_alarm   |   CR_P99_score | CR_P99_alarm   |   Z_score | Z_alarm   |
|:--------------------------------------|:-------------------------------|:-------------------|:-------------------|---------:|:---------------------------|:-----------------|:---------|---------:|---------------:|--------------:|:--------------|---------------:|:---------------|----------:|:----------|
| ft_lane_weaving_00_140|veh001         | ft_lane_weaving_00_140         | lane_weaving       | lane_weaving       |      140 | normal_medium_traffic      | anomaly_verified | True     |      nan |            nan |          3.86 | True          |           1.89 | True           |      4.75 | False     |
| ft_lane_weaving_01_083|veh001         | ft_lane_weaving_01_083         | lane_weaving       | lane_weaving       |      083 | normal_light_traffic       | anomaly_verified | True     |      nan |            nan |          1.64 | False         |         -13.36 | False          |      4.75 | False     |
| ft_lane_weaving_02_062|veh001         | ft_lane_weaving_02_062         | lane_weaving       | lane_weaving       |      062 | normal_heavy_traffic       | anomaly_verified | True     |      nan |            nan |          2.71 | True          |         -12.49 | False          |      4.75 | False     |
| ft_lane_weaving_03_020|veh001         | ft_lane_weaving_03_020         | lane_weaving       | lane_weaving       |      020 | normal_medium_das_only     | anomaly_verified | True     |      nan |            nan |          3.17 | True          |          -0.93 | True           |      4.75 | False     |
| ft_lane_weaving_04_037|veh001         | ft_lane_weaving_04_037         | lane_weaving       | lane_weaving       |      037 | normal_medium_das_cam_only | anomaly_verified | True     |      nan |            nan |          3.09 | True          |          -7.13 | False          |      4.75 | False     |
| ft_lane_weaving_05_003|veh001         | ft_lane_weaving_05_003         | lane_weaving       | lane_weaving       |      003 | normal_medium_sparse       | anomaly_verified | True     |      nan |            nan |          0.73 | False         |         -31.37 | False          |      4.75 | False     |
| ft_lane_weaving_06_120|veh001         | ft_lane_weaving_06_120         | lane_weaving       | lane_weaving       |      120 | normal_medium_traffic      | anomaly_verified | True     |      nan |            nan |          2.94 | True          |          -4.11 | True           |      4.75 | False     |
| ft_lane_weaving_07_010|veh001         | ft_lane_weaving_07_010         | lane_weaving       | lane_weaving       |      010 | normal_light_traffic       | anomaly_verified | True     |      nan |            nan |          1.67 | False         |         -15.59 | False          |      4.75 | False     |
| ft_lane_weaving_08_120|veh001         | ft_lane_weaving_08_120         | lane_weaving       | lane_weaving       |      120 | normal_heavy_traffic       | anomaly_verified | True     |      nan |            nan |          2.37 | True          |         -10.41 | False          |      4.75 | False     |
| ft_lane_weaving_09_062|veh001         | ft_lane_weaving_09_062         | lane_weaving       | lane_weaving       |      062 | normal_medium_das_only     | anomaly_verified | True     |      nan |            nan |         -0.84 | False         |         -33.57 | False          |      4.75 | False     |
| ft_lane_weaving_10_140|veh001         | ft_lane_weaving_10_140         | lane_weaving       | lane_weaving       |      140 | normal_medium_das_cam_only | anomaly_verified | True     |      nan |            nan |          2.17 | True          |         -10.8  | False          |      4.75 | False     |
| ft_lane_weaving_11_010|veh001         | ft_lane_weaving_11_010         | lane_weaving       | lane_weaving       |      010 | normal_medium_sparse       | anomaly_verified | False    |      nan |            nan |        nan    | False         |         nan    | False          |     12.62 | False     |
| ft_overweight_vehicle_00_037|veh001   | ft_overweight_vehicle_00_037   | overweight_vehicle | overweight_vehicle |      037 | normal_medium_traffic      | anomaly_verified | True     |      nan |            nan |          2.16 | True          |          -7.83 | False          |     11.56 | False     |
| ft_overweight_vehicle_01_140|veh001   | ft_overweight_vehicle_01_140   | overweight_vehicle | overweight_vehicle |      140 | normal_light_traffic       | anomaly_verified | True     |      nan |            nan |          2.6  | True          |         -14.39 | False          |      6.59 | False     |
| ft_overweight_vehicle_02_062|veh001   | ft_overweight_vehicle_02_062   | overweight_vehicle | overweight_vehicle |      062 | normal_heavy_traffic       | anomaly_verified | True     |      nan |            nan |          4.57 | True          |          10.14 | True           |     11.07 | False     |
| ft_overweight_vehicle_03_083|veh001   | ft_overweight_vehicle_03_083   | overweight_vehicle | overweight_vehicle |      083 | normal_medium_das_only     | anomaly_verified | True     |      nan |            nan |          1.51 | False         |          -7.09 | False          |     11.87 | False     |
| ft_overweight_vehicle_04_003|veh001   | ft_overweight_vehicle_04_003   | overweight_vehicle | overweight_vehicle |      003 | normal_medium_das_cam_only | anomaly_verified | True     |      nan |            nan |          1.48 | False         |          -4.52 | True           |     11.67 | False     |
| ft_overweight_vehicle_05_120|veh001   | ft_overweight_vehicle_05_120   | overweight_vehicle | overweight_vehicle |      120 | normal_medium_sparse       | anomaly_verified | True     |      nan |            nan |          0.4  | False         |         -16.55 | False          |      6.98 | False     |
| ft_overweight_vehicle_06_010|veh001   | ft_overweight_vehicle_06_010   | overweight_vehicle | overweight_vehicle |      010 | normal_medium_traffic      | anomaly_verified | True     |      nan |            nan |          0.68 | False         |         -11.02 | False          |      6.98 | False     |
| ft_overweight_vehicle_07_020|veh001   | ft_overweight_vehicle_07_020   | overweight_vehicle | overweight_vehicle |      020 | normal_light_traffic       | anomaly_verified | True     |      nan |            nan |          0.71 | False         |         -11.84 | False          |      9.4  | False     |
| ft_overweight_vehicle_08_020|veh001   | ft_overweight_vehicle_08_020   | overweight_vehicle | overweight_vehicle |      020 | normal_heavy_traffic       | anomaly_verified | True     |      nan |            nan |          1.4  | False         |         -15.98 | False          |      7    | False     |
| ft_overweight_vehicle_09_010|veh001   | ft_overweight_vehicle_09_010   | overweight_vehicle | overweight_vehicle |      010 | normal_medium_das_only     | anomaly_verified | True     |      nan |            nan |          0.4  | False         |         -15.41 | False          |      6.98 | False     |
| ft_overweight_vehicle_10_083|veh001   | ft_overweight_vehicle_10_083   | overweight_vehicle | overweight_vehicle |      083 | normal_medium_das_cam_only | anomaly_verified | True     |      nan |            nan |          3.94 | True          |          10.59 | True           |     12.22 | False     |
| ft_overweight_vehicle_11_120|veh001   | ft_overweight_vehicle_11_120   | overweight_vehicle | overweight_vehicle |      120 | normal_medium_sparse       | anomaly_verified | True     |      nan |            nan |         -0.89 | False         |         -14.68 | False          |      3.4  | False     |
| ft_stalled_00_010_START|veh001        | ft_stalled_00_010_START        | stalled_vehicle    | START              |      010 | normal_medium_traffic      | anomaly_verified | True     |      nan |            nan |          5.21 | True          |           3.73 | True           |    486.92 | True      |
| ft_stalled_01_083_STOP_END|veh001     | ft_stalled_01_083_STOP_END     | stalled_vehicle    | STOP_END           |      083 | normal_light_traffic       | anomaly_verified | True     |      nan |            nan |          5.71 | True          |          36.42 | True           |    435.22 | True      |
| ft_stalled_02_003_MSM|veh001          | ft_stalled_02_003_MSM          | stalled_vehicle    | MSM                |      003 | normal_heavy_traffic       | anomaly_verified | True     |      nan |            nan |          5.16 | True          |           7.44 | True           |     98.45 | True      |
| ft_stalled_03_037_GENTLE_STALL|veh001 | ft_stalled_03_037_GENTLE_STALL | stalled_vehicle    | GENTLE_STALL       |      037 | normal_medium_das_only     | anomaly_verified | True     |      nan |            nan |          4.88 | True          |          14.46 | True           |     45.27 | True      |
| ft_stalled_04_062_START|veh001        | ft_stalled_04_062_START        | stalled_vehicle    | START              |      062 | normal_medium_das_cam_only | anomaly_verified | True     |      nan |            nan |          4.3  | True          |           7.1  | True           |    445.56 | True      |
| ft_stalled_05_140_STOP_END|veh001     | ft_stalled_05_140_STOP_END     | stalled_vehicle    | STOP_END           |      140 | normal_medium_sparse       | anomaly_verified | True     |      nan |            nan |          5.49 | True          |          34.37 | True           |    371.71 | True      |
| ft_stalled_06_020_MSM|veh001          | ft_stalled_06_020_MSM          | stalled_vehicle    | MSM                |      020 | normal_medium_traffic      | anomaly_verified | True     |      nan |            nan |          3.91 | True          |          -4.86 | True           |    263.41 | True      |
| ft_stalled_07_120_GENTLE_STALL|veh001 | ft_stalled_07_120_GENTLE_STALL | stalled_vehicle    | GENTLE_STALL       |      120 | normal_light_traffic       | anomaly_verified | True     |      nan |            nan |          3.17 | True          |          -0.79 | True           |     65.95 | True      |
| ft_stalled_08_010_START|veh001        | ft_stalled_08_010_START        | stalled_vehicle    | START              |      010 | normal_heavy_traffic       | anomaly_verified | True     |      nan |            nan |          5.09 | True          |           1.91 | True           |    574.07 | True      |
| ft_stalled_09_083_STOP_END|veh001     | ft_stalled_09_083_STOP_END     | stalled_vehicle    | STOP_END           |      083 | normal_medium_das_only     | anomaly_verified | True     |      nan |            nan |          5.3  | True          |          24.3  | True           |    358.41 | True      |
| ft_stalled_10_003_MSM|veh001          | ft_stalled_10_003_MSM          | stalled_vehicle    | MSM                |      003 | normal_medium_das_cam_only | anomaly_verified | True     |      nan |            nan |          5.65 | True          |          11.87 | True           |     79.24 | True      |
| ft_stalled_11_037_GENTLE_STALL|veh001 | ft_stalled_11_037_GENTLE_STALL | stalled_vehicle    | GENTLE_STALL       |      037 | normal_medium_sparse       | anomaly_verified | True     |      nan |            nan |          4.62 | True          |          20.79 | True           |     51.18 | True      |
| ft_tailgating_00_003_tg|veh001        | ft_tailgating_00_003_tg        | tailgating         | tailgating         |      003 | normal_medium_traffic      | anomaly_verified | True     |        6 |              8 |          4.5  | True          |         -23.19 | False          |      2.24 | False     |
| ft_tailgating_01_010_tg|veh001        | ft_tailgating_01_010_tg        | tailgating         | tailgating         |      010 | normal_light_traffic       | anomaly_verified | True     |        6 |              8 |          5.06 | True          |         -10.15 | False          |      2.08 | False     |
| ft_tailgating_02_020_tg|veh001        | ft_tailgating_02_020_tg        | tailgating         | tailgating         |      020 | normal_heavy_traffic       | anomaly_verified | True     |        6 |             15 |          5.38 | True          |         -17.56 | False          |      1.88 | False     |
| ft_tailgating_03_037_tg|veh001        | ft_tailgating_03_037_tg        | tailgating         | tailgating         |      037 | normal_medium_das_only     | anomaly_verified | True     |        6 |             15 |          4.69 | True          |         -15.42 | False          |      2.53 | False     |
| ft_tailgating_04_062_tg|veh001        | ft_tailgating_04_062_tg        | tailgating         | tailgating         |      062 | normal_medium_das_cam_only | anomaly_verified | True     |        6 |              8 |          4.34 | True          |         -17.86 | False          |      2.71 | False     |
| ft_tailgating_05_083_tg|veh001        | ft_tailgating_05_083_tg        | tailgating         | tailgating         |      083 | normal_medium_sparse       | anomaly_verified | False    |        6 |              8 |        nan    | False         |         nan    | False          |      3.18 | False     |
| ft_tailgating_06_120_tg|veh001        | ft_tailgating_06_120_tg        | tailgating         | tailgating         |      120 | normal_medium_traffic      | anomaly_verified | True     |        6 |             15 |          3.96 | True          |         -12.76 | False          |      5.32 | False     |
| ft_tailgating_07_140_tg|veh001        | ft_tailgating_07_140_tg        | tailgating         | tailgating         |      140 | normal_light_traffic       | anomaly_verified | True     |        6 |             15 |          4.9  | True          |         -15.45 | False          |      1.65 | False     |
| ft_tailgating_08_010_tg|veh001        | ft_tailgating_08_010_tg        | tailgating         | tailgating         |      010 | normal_heavy_traffic       | anomaly_verified | True     |        9 |              8 |          5.24 | True          |         -20.79 | False          |      2.34 | False     |
| ft_tailgating_09_020_tg|veh001        | ft_tailgating_09_020_tg        | tailgating         | tailgating         |      020 | normal_medium_das_only     | anomaly_verified | True     |        9 |              8 |          5.25 | True          |         -24.02 | False          |      2.31 | False     |
| ft_tailgating_10_037_tg|veh001        | ft_tailgating_10_037_tg        | tailgating         | tailgating         |      037 | normal_medium_das_cam_only | anomaly_verified | True     |        9 |              8 |          5.28 | True          |         -18.33 | False          |     33.42 | True      |
| ft_tailgating_11_062_tg|veh001        | ft_tailgating_11_062_tg        | tailgating         | tailgating         |      062 | normal_medium_sparse       | anomaly_verified | True     |        9 |              8 |          3.66 | True          |         -20.34 | False          |      1.95 | False     |