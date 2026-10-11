# Klausner experiment: the final model on the one real recording

**Status:** a qualitative observation on one real, **unlabelled** track. It is not an evaluation.

- **Inference only.** Nothing was trained, re-calibrated or changed; the five frozen checkpoints, their statistics and the frozen threshold τ = 1.973 were used as they are.
- **Not used for any decision.** The recording was not part of any training, selection or test set, and it changes no reported metric.

## Research question
Does DeepDAS Ensemble v1, trained only on simulated tracks, raise an alarm on the real bus recording from Klausner Street, and how does that compare with the earlier detectors?

## What was compared

| Detector | Decision on the real bus | Score | Source |
|---|---|---|---|
| **DeepDAS Ensemble v1, φ<sub>u</sub> active** (as trained) | **ALARM** | s<sub>u</sub> = 4.53 > τ = 1.973 | `klausner_scores.csv` (this run) |
| **DeepDAS Ensemble v1, φ<sub>u</sub> neutralized** | **ALARM** | s<sub>u</sub> = 4.55 > τ = 1.973 | `klausner_scores.csv` (this run) |
| Supervised single model | ALARM | logit 18.8 > −5.52 | `klausner_scores.csv` (this run) |
| SimStudio Z-score (shipped model) | flagged | max \|z\| = 27.5 > 15.39 (top feature: mean absolute jerk) | `klausner_track_summary.json` (this run) |
| Stage-1 autoencoder (study 01) | not flagged | track score 2.15 < 5.20 | archived: `research/01_unsupervised_autoencoder/records/unsup_REPORT.md` §4.6 |

**Per-seed results.** All five seeds alarm individually:

- seed logits 6.6, 7.8, 7.7, 9.3 and 3.5;
- every per-seed z is above τ.

**The two options agree.** For this bus φ<sub>u</sub> = 0, because it never follows a leader closely. Neutralizing the 17th input therefore barely changes the score: 4.53 vs 4.55.

## Data
The real export of SimStudio (`data/simstudio_inputs/klausner/`, part of the data package):

- **Recording.** One real DAS channel on an existing telecom fibre and one camera on Klausner Street, next to the TAU campus.
- **Event.** A city bus slows down, stands at a bus stop and pulls away. The fused track lasts 41.8 s and has 175 valid 0.1 s bins and 15 windows.
- **Processing.** The track was imported with the unchanged pipeline used for every simulated run: map matching on the real street geometry, the 14 channels, the DAS amplitude from the Kalman audit sheet, and the quality gate.

## Figure

<p align="center"><img src="figures/klausner_experiment.png" alt="Klausner experiment" width="95%"></p>

**Left panel:**

- the tracked longitudinal speed and acceleration of the bus;
- the DAS-measurement indicator.

- **Standing time.** The tracked speed stays below 1 m/s for 17.5 s. The stop is fragmented by brief interruptions: the longest continuous stop below 0.5 m/s is 7.9 s, in 6 stop events (SimStudio Z-score features).
- **Noise.** Around arrival and departure the real tracked kinematics are much noisier than anything in the simulation, with |a| up to about 70 m/s².

**Right panel:**

- the ensemble score in both options against the frozen threshold.

## Conclusion and limits

- **Result.** The final model flags the real bus in both inference options. So do the supervised single model and the SimStudio Z-score; the stage-1 autoencoder does not.
- **What it does not show:**
  - **No label.** There is no ground truth for this vehicle. A bus stopping at a bus stop is ordinary traffic, so whether this alarm is a true or a false alarm cannot be decided from this recording.
  - **The cause is not identified.** Two explanations are both plausible, and this one track cannot separate them:
    - *Stopping pattern.* The long, repeated stopping resembles the stop patterns the model learned to flag; in the simulation's labels, a stall is a stop of 10 s or more (`docs/ANOMALY_TYPES.md`).
    - *Sensor noise.* The real DAS-derived kinematics are far noisier than the simulated ones (the same noise drives the Z-score's top feature, mean absolute jerk). An input outside the training distribution can also produce a high score.
  - **Not a validation.** One unlabelled track cannot validate any detector. It shows that the full pipeline runs end to end on real fused data.

## How to reproduce
From the repository root, with the data package unpacked (a few seconds on CPU):
```bash
python models/deepdas_ensemble_v1/experiment_klausner/run_klausner_experiment.py
```
The script scores the track in both options and writes the files next to itself.

| File | Content |
|---|---|
| `run_klausner_experiment.py` | the experiment (import, scoring in both options, Z-score, figure) |
| `klausner_scores.csv` | per option: windows, five seed logits, ensemble score, decision, φ<sub>u</sub>, τ; single-model logit and decision |
| `klausner_track_summary.json` | all detectors, track facts, the archived autoencoder result and its source |
| `figures/klausner_experiment.{png,pdf}` | the figure above |
