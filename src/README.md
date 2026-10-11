# Source code: the `deepdas` package

The package is imported by every script in [`../scripts/`](../scripts/). It finds `models/`, `data/` and `results/` relative to the repository root ([`deepdas/paths.py`](deepdas/paths.py)).

| Module | Role | Main entry points |
|---|---|---|
| `ensemble/` | DeepDAS Ensemble v1 at inference: the detector (encoder + pooling + head), φ<sub>u</sub> and artefact cleaning, ensemble scoring, the `--phi-mode` option | `scoring.Ensemble`, `scoring.score_units`, `detector.load_detector`, `phi_mode` |
| `model/` | the 1-D convolutional autoencoder whose encoder the ensemble uses | `conv_ae.ConvAE` |
| `train/` | supervised training of one ensemble seed and calibration on val_B | `supervised.train_seed` |
| `evaluation/` | development, validation and final-test evaluation, metrics, the Klausner check of stage 1 | `development.reproduce`, `validation.ensemble_tables`, `final_test.score` / `final_test.analyze` |
| `data/` | import of SimStudio exports, map matching, the 14 channels, windows, quality gate, splits, labels | `channels.build_channels`, `dataset` |
| `amplitude.py`, `importing.py`, `data_sources.py` | DAS amplitude on the 0.1 s bins; import of one run; readers for processed runs | `importing.import_run` |
| `baseline/` | wrapper of the shipped SimStudio Z-score model | `zscore.shipped_model` |
| `batches/`, `sim/` | generators of the simulated anomaly sets and the run driver | – |
| `fp/`, `fp_anomaly_model/` | **SimStudio** simulator, sensors, Kalman tracker and Z-score model, unchanged from the SimStudio project | – |

**Input and output of the model.**

- **Input:** the channels and DAS amplitude of one or more tracks, as produced by `importing.import_run`.
- **Output** (per decision unit): the number of windows, the five seed logits, the ensemble score, and the decision `ALARM` / `no alarm` / `no decision` (fewer than 3 windows).

```python
from deepdas.ensemble.scoring import score_units      # set DEEPDAS_PHI_MODE=neutralized|active before creating the Ensemble
```

Architecture and design: [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).
