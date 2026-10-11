# The two inference options of DeepDAS Ensemble v1

## What is the same in both
**The trained model is the same.** DeepDAS Ensemble v1 consists of five frozen networks. Each was **trained** with a classifier head of 17 inputs:

- **16 learned features:** the mean and standard deviation of the 8-d window codes over the track;
- **1 explicit physical summary:** the tailgating fraction

  φ<sub>u</sub> = |{ b ∈ B<sub>u</sub> : leader present ∧ headway < 0.8 s ∧ v<sub>lon</sub> > 5 m/s }| / |B<sub>u</sub>|,

  where B<sub>u</sub> is the set of valid 0.1 s bins of the unit.

**The inputs of φ<sub>u</sub>.** It is computed from the tracked data, not from simulator truth. Headway and the leader mask are also 2 of the 14 encoder input channels, so φ<sub>u</sub> adds no new measurement. It is an explicit track-level summary of information that the encoder receives window by window.

**The head input.** The head receives x<sub>17</sub> = (φ<sub>u</sub> − μ<sub>φ</sub>)/σ<sub>φ</sub>, with μ<sub>φ</sub> = 0.005994 and σ<sub>φ</sub> = 0.031635 stored in every checkpoint.

## The two options

| | `--phi-mode neutralized` | `--phi-mode active` |
|---|---|---|
| 17th head input | x<sub>17</sub> = 0 (the value of an average normal training vehicle) | x<sub>17</sub> = (φ<sub>u</sub> − μ<sub>φ</sub>)/σ<sub>φ</sub> |
| Weights, μ<sub>k</sub>, σ<sub>k</sub>, τ | the frozen ones | the frozen ones |
| What it is | an inference-time intervention on the trained hybrid model | the model exactly as trained, selected, frozen and tested |
| When it was defined | after the final test had been exposed | before the final test |
| Published results | `results_phi_neutralized/` (every file stamped `phi_mode = neutralized`) | `results/` |

**Not a model trained without the feature.** The neutralized option is the same network with one input held at zero. A network trained without φ<sub>u</sub> could have learned to use the headway channels differently. No such network was trained on the final training data ([study 08, §6](../research/08_tailgating_feature_intervention/README.md)).

## Results per option (never mix them)

| Check | Neutralized | Active |
|---|---|---|
| Development cases (9; diagnostic, not independent) | 7 / 9 | 8 / 9 |
| Validation: tailgating (selection data) | 9 / 13 | 12 / 13 |
| Validation: macro recall / AP | 0.688 / 0.737 | 0.714 / 0.781 |
| Final test: tailgating | 10 / 11 (post hoc) | 11 / 11 |
| Final test: macro recall | 0.560 (post hoc) | 0.568 |
| Final test: ROC-AUC / AUPRC | 0.906 / 0.737 (post hoc) | 0.908 / 0.742 |
| Final test: false alarms, normal driving / legitimate stops | 20 / 1,008; 19 / 28 (post hoc) | 20 / 1,008; 20 / 28 |

**Where the options differ.**

- **Development.** Only one case changes decision: the tailgating case. Its φ<sub>u</sub> = 0.81 lies 25 standard deviations above the training-normal mean, and its score falls from 3.73 to −0.08.
- **Final test.** Speeding, collision and tailgating each change by one vehicle.

**What the comparison supports.**

- With the feature active, the frozen model detects one more development case.
- Most test tailgaters stay detected without it: the learned representation carries most of their signal.
- It does **not** show that the tailgating improvement of the project came from the feature alone, or from the data alone.

Every number: [research/08 tables](../research/08_tailgating_feature_intervention/tables/) and [REPORT_V2_NUMBER_SOURCES.md](../research/08_tailgating_feature_intervention/tables/REPORT_V2_NUMBER_SOURCES.md).

## How to select an option

- **Reproduction scripts:** `--phi-mode neutralized` or `--phi-mode active`.
- **Python code and `scripts/score_tracks.py`:** the environment variable `DEEPDAS_PHI_MODE=neutralized|active`, read when a detector is created.
- **If nothing is given, the code uses `active`.** That default was kept so that the published results reproduce unchanged. The documentation always names the option explicitly.

Implementation: `src/deepdas/ensemble/phi_mode.py`, `Detector.head_in` in `src/deepdas/ensemble/detector.py`, and `scripts/_phi_io.py`.
