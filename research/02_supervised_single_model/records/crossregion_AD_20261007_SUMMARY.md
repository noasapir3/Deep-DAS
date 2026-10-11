# Experiments A–D — summary (each run at its own frozen val_B thresholds)

## P99

| run                      | best_epoch   |   val_AP |   valid9_detected | C3   |   C3_logit |   C3_pct_valB | weaving   |   weaving_logit |   weaving_pct_valB | FA_test   | FA_valA   | FA_valB   |   precision_9vsTest |   recall_9 |   F1_9vsTest | missed                                         |   threshold | newscene_detected   |   newscene_insufficient |   precision_new |   recall_new |   F1_new |
|:-------------------------|:-------------|---------:|------------------:|:-----|-----------:|--------------:|:----------|----------------:|-------------------:|:----------|:----------|:----------|--------------------:|-----------:|-------------:|:-----------------------------------------------|------------:|:--------------------|------------------------:|----------------:|-------------:|---------:|
| BASELINE_frozen          | frozen 24    | 0.5707   |                 7 | no   |     -14.78 |          80.5 | no        |           -8.69 |               97.4 | 7/513     | 17/1313   | 13/1242   |               0.5   |      0.778 |        0.609 | weaving, C3 tailgating                         |      -5.523 | 28/47               |                      12 |           0.8   |        0.596 |    0.683 |
| R0_repro_original_recipe | 24           | 0.570685 |                 7 | no   |     -14.78 |          80.5 | no        |           -8.69 |               97.4 | 7/513     | 17/1313   | 13/1242   |               0.5   |      0.778 |        0.609 | weaving, C3 tailgating                         |      -5.523 | 28/47               |                      12 |           0.8   |        0.596 |    0.683 |
| A_clean_only             | 11           | 0.687401 |                 6 | no   |      -9.64 |          91.1 | no        |           -4.08 |               99   | 8/513     | 14/1313   | 13/1242   |               0.429 |      0.667 |        0.522 | speeding, weaving, C3 tailgating               |      -4.062 | 28/47               |                      12 |           0.778 |        0.596 |    0.675 |
| B_clean_osc              | 8            | 0.745869 |                 8 | no   |      -9.75 |          87.4 | ALARM     |           -0.69 |               99.7 | 15/513    | 20/1313   | 13/1242   |               0.348 |      0.889 |        0.5   | C3 tailgating                                  |      -4.486 | 32/47               |                      12 |           0.681 |        0.681 |    0.681 |
| C_clean_tg               | 8            | 0.718011 |                 8 | no   |      -6.36 |          97.5 | ALARM     |           -1.74 |               99.5 | 12/513    | 15/1313   | 13/1242   |               0.4   |      0.889 |        0.552 | C3 tailgating                                  |      -4.32  | 31/47               |                      12 |           0.721 |        0.66  |    0.689 |
| D_clean_osc_tg           | 3            | 0.404238 |                 5 | no   |      -7.4  |          82.4 | no        |           -7.09 |               85.5 | 11/513    | 16/1313   | 13/1242   |               0.312 |      0.556 |        0.4   | stalled, weaving, C2 straddling, C3 tailgating |      -2.685 | 29/47               |                      12 |           0.725 |        0.617 |    0.667 |

### P99: the 9 valid cases (Y = alarm)

|                      | BASELINE_frozen   | R0_repro_original_recipe   | A_clean_only   | B_clean_osc   | C_clean_tg   | D_clean_osc_tg   |
|:---------------------|:------------------|:---------------------------|:---------------|:--------------|:-------------|:-----------------|
| speeding             | Y                 | Y                          | -              | Y             | Y            | Y                |
| C1 real braking      | Y                 | Y                          | Y              | Y             | Y            | Y                |
| stalled              | Y                 | Y                          | Y              | Y             | Y            | -                |
| C4 collision striker | Y                 | Y                          | Y              | Y             | Y            | Y                |
| reckless             | Y                 | Y                          | Y              | Y             | Y            | Y                |
| weaving              | -                 | -                          | -              | Y             | Y            | -                |
| overweight           | Y                 | Y                          | Y              | Y             | Y            | Y                |
| C2 straddling        | Y                 | Y                          | Y              | Y             | Y            | -                |
| C3 tailgating        | -                 | -                          | -              | -             | -            | -                |

### P99: new-scene test, detected / verified per type

|                            | BASELINE_frozen   | R0_repro_original_recipe   | A_clean_only   | B_clean_osc   | C_clean_tg   | D_clean_osc_tg   |
|:---------------------------|:------------------|:---------------------------|:---------------|:--------------|:-------------|:-----------------|
| anomaly_collision          | 7/7               | 7/7                        | 7/7            | 7/7           | 7/7          | 7/7              |
| anomaly_hard_braking       | 2/6               | 2/6                        | 2/6            | 3/6           | 3/6          | 3/6              |
| anomaly_lane_straddling    | 3/5               | 3/5                        | 3/5            | 3/5           | 3/5          | 1/5              |
| anomaly_lane_weaving       | 1/5               | 1/5                        | 1/5            | 3/5           | 2/5          | 1/5              |
| anomaly_overweight_vehicle | 2/5               | 2/5                        | 2/5            | 2/5           | 2/5          | 3/5              |
| anomaly_reckless_driver    | 3/5               | 3/5                        | 3/5            | 3/5           | 3/5          | 3/5              |
| anomaly_speeding           | 2/5               | 2/5                        | 2/5            | 2/5           | 2/5          | 2/5              |
| anomaly_stalled_vehicle    | 8/8               | 8/8                        | 7/8            | 8/8           | 8/8          | 8/8              |
| anomaly_tailgating         | 0/1               | 0/1                        | 1/1            | 1/1           | 1/1          | 1/1              |

## P95

| run                      | best_epoch   |   val_AP |   valid9_detected | C3    |   C3_logit |   C3_pct_valB | weaving   |   weaving_logit |   weaving_pct_valB | FA_test   | FA_valA   | FA_valB   |   precision_9vsTest |   recall_9 |   F1_9vsTest | missed                                         |   threshold | newscene_detected   |   newscene_insufficient |   precision_new |   recall_new |   F1_new |
|:-------------------------|:-------------|---------:|------------------:|:------|-----------:|--------------:|:----------|----------------:|-------------------:|:----------|:----------|:----------|--------------------:|-----------:|-------------:|:-----------------------------------------------|------------:|:--------------------|------------------------:|----------------:|-------------:|---------:|
| BASELINE_frozen          | frozen 24    | 0.5707   |                 8 | no    |     -14.78 |          80.5 | ALARM     |           -8.69 |               97.4 | 49/513    | 112/1313  | 63/1242   |               0.14  |      0.889 |        0.242 | C3 tailgating                                  |     -10.894 | 34/47               |                      12 |           0.41  |        0.723 |    0.523 |
| R0_repro_original_recipe | 24           | 0.570685 |                 8 | no    |     -14.78 |          80.5 | ALARM     |           -8.69 |               97.4 | 49/513    | 112/1313  | 63/1242   |               0.14  |      0.889 |        0.242 | C3 tailgating                                  |     -10.894 | 34/47               |                      12 |           0.41  |        0.723 |    0.523 |
| A_clean_only             | 11           | 0.687401 |                 8 | no    |      -9.64 |          91.1 | ALARM     |           -4.08 |               99   | 56/513    | 113/1313  | 63/1242   |               0.125 |      0.889 |        0.219 | C3 tailgating                                  |      -8.419 | 34/47               |                      12 |           0.378 |        0.723 |    0.496 |
| B_clean_osc              | 8            | 0.745869 |                 8 | no    |      -9.75 |          87.4 | ALARM     |           -0.69 |               99.7 | 46/513    | 81/1313   | 63/1242   |               0.148 |      0.889 |        0.254 | C3 tailgating                                  |      -7.666 | 35/47               |                      12 |           0.432 |        0.745 |    0.547 |
| C_clean_tg               | 8            | 0.718011 |                 9 | ALARM |      -6.36 |          97.5 | ALARM     |           -1.74 |               99.5 | 45/513    | 82/1313   | 63/1242   |               0.167 |      1     |        0.286 |                                                |      -7.722 | 35/47               |                      12 |           0.438 |        0.745 |    0.551 |
| D_clean_osc_tg           | 3            | 0.404238 |                 5 | no    |      -7.4  |          82.4 | no        |           -7.09 |               85.5 | 42/513    | 73/1313   | 63/1242   |               0.106 |      0.556 |        0.179 | stalled, weaving, C2 straddling, C3 tailgating |      -4.965 | 34/47               |                      12 |           0.447 |        0.723 |    0.553 |

### P95: the 9 valid cases (Y = alarm)

|                      | BASELINE_frozen   | R0_repro_original_recipe   | A_clean_only   | B_clean_osc   | C_clean_tg   | D_clean_osc_tg   |
|:---------------------|:------------------|:---------------------------|:---------------|:--------------|:-------------|:-----------------|
| speeding             | Y                 | Y                          | Y              | Y             | Y            | Y                |
| C1 real braking      | Y                 | Y                          | Y              | Y             | Y            | Y                |
| stalled              | Y                 | Y                          | Y              | Y             | Y            | -                |
| C4 collision striker | Y                 | Y                          | Y              | Y             | Y            | Y                |
| reckless             | Y                 | Y                          | Y              | Y             | Y            | Y                |
| weaving              | Y                 | Y                          | Y              | Y             | Y            | -                |
| overweight           | Y                 | Y                          | Y              | Y             | Y            | Y                |
| C2 straddling        | Y                 | Y                          | Y              | Y             | Y            | -                |
| C3 tailgating        | -                 | -                          | -              | -             | Y            | -                |

### P95: new-scene test, detected / verified per type

|                            | BASELINE_frozen   | R0_repro_original_recipe   | A_clean_only   | B_clean_osc   | C_clean_tg   | D_clean_osc_tg   |
|:---------------------------|:------------------|:---------------------------|:---------------|:--------------|:-------------|:-----------------|
| anomaly_collision          | 7/7               | 7/7                        | 7/7            | 7/7           | 7/7          | 7/7              |
| anomaly_hard_braking       | 4/6               | 4/6                        | 3/6            | 5/6           | 5/6          | 5/6              |
| anomaly_lane_straddling    | 5/5               | 5/5                        | 5/5            | 3/5           | 3/5          | 1/5              |
| anomaly_lane_weaving       | 1/5               | 1/5                        | 2/5            | 3/5           | 3/5          | 1/5              |
| anomaly_overweight_vehicle | 3/5               | 3/5                        | 3/5            | 3/5           | 2/5          | 4/5              |
| anomaly_reckless_driver    | 3/5               | 3/5                        | 3/5            | 3/5           | 3/5          | 4/5              |
| anomaly_speeding           | 2/5               | 2/5                        | 2/5            | 2/5           | 3/5          | 3/5              |
| anomaly_stalled_vehicle    | 8/8               | 8/8                        | 8/8            | 8/8           | 8/8          | 8/8              |
| anomaly_tailgating         | 1/1               | 1/1                        | 1/1            | 1/1           | 1/1          | 1/1              |
