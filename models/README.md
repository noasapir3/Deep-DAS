# Models

The frozen models of the project, each with its checksum. **Nothing in this folder is retrained or recalibrated.**

| Folder | Model | Head inputs | Role |
|---|---|---|---|
| [`deepdas_ensemble_v1/`](deepdas_ensemble_v1/) | **DeepDAS Ensemble v1** (internal id `E_stall_diversity_ENSEMBLE` / `E_DIV`): five networks, frozen 2026-10-09, threshold τ = 1.9726181803233118 | 16 learned + φ<sub>u</sub> = 17 | the final model; **[current model card](MODEL_CARD_DeepDAS_Ensemble_v1.md)**; [original model card](deepdas_ensemble_v1/MODEL_CARD.md) (frozen) |
| [`deepdas_ensemble_v1/experiment_klausner/`](deepdas_ensemble_v1/experiment_klausner/) | the final model on the real Klausner recording (inference only) | – | one real stalled bus (detected) |
| [`baselines/supervised_single_p99/`](baselines/supervised_single_p99/) | supervised single model (`CR_P99`): one seed, early data, no artefact cleaning, own val_B P99 = −5.523 | 16 | frozen baseline |
| [`baselines/zscore/`](baselines/zscore/) | SimStudio Z-score model (`model_live.pkl`): 156 features, threshold 15.39 σ, never refitted | – | frozen baseline |

The W1/W2 checkpoints of study 07 (not adopted) are in [`../research/07_targeted_data_study/models/`](../research/07_targeted_data_study/models/).

**Two model cards.**
- **The current card,** [`MODEL_CARD_DeepDAS_Ensemble_v1.md`](MODEL_CARD_DeepDAS_Ensemble_v1.md), is the one to read. It compares the detectors on the common decided subset, covers both inference options and reports the Klausner experiment.
- **The original card,** [`deepdas_ensemble_v1/MODEL_CARD.md`](deepdas_ensemble_v1/MODEL_CARD.md), belongs to the frozen model folder, whose files are hash-checked (`scripts/verify_phi_modes.py`, P1), so it is not edited. The current card supersedes it on these points:

- **What its comparison table mixes.** The Z-score row (macro recall 0.231, ROC-AUC 0.783, AUPRC 0.394) is on the Z-score's own decided set: 105 anomalies and 1,574 normals.
- **The values on the same vehicles.** On the common decided subset used for the ensemble (101 anomalies, 1,008 normals), the Z-score has 0.232 / 0.819 / 0.496. Those are the values to compare with the ensemble's 0.568 / 0.908 / 0.742 (`results/final_test/T2_overall.csv`).
- **Real data.** The model card also predates the [Klausner experiment](deepdas_ensemble_v1/experiment_klausner/).

**The W1/W2 checkpoints** are the targeted training-data expansion of study 07 and the same data with controlled sampling.

## Inference options of the final model
Both options use the same five files. They differ only in the 17th head input; see [docs/INFERENCE_MODES.md](../docs/INFERENCE_MODES.md).
```bash
python scripts/reproduce_development_cases.py outputs/dev_active      --phi-mode active        # 8 / 9
python scripts/reproduce_development_cases.py outputs/dev_neutralized --phi-mode neutralized   # 7 / 9
```

## Check the files
```bash
python scripts/check_models.py      # sha256 against ensemble.json, architecture, both options (no data needed)
```

## Files of `deepdas_ensemble_v1/`

| File | Content |
|---|---|
| `ensemble.json` | the seeds, checkpoint sha256, frozen val_B mean / std per seed, decision rule and threshold |
| `seed_<k>/checkpoint.pt` | encoder + head, with the pooled-feature and φ<sub>u</sub> standardisation constants |
| `seed_<k>/thresholds.json`, `seed_<k>/training_history.csv` | per-seed reference percentiles; loss and validation AP per epoch |
| `input_normalization.json` | the fixed normalisation of the 14 input channels |
| `autoencoder_init.pt`, `autoencoder_config.json`, `autoencoder_training_history.csv` | the encoder initialisation (pre-trained autoencoder) |
| `freeze_record.json` | the record written when the model was frozen |
