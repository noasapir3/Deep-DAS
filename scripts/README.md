# Scripts

Command-line entry points. Each script puts `src/` on the Python path itself, so no package installation is needed. Run them from the repository root.

**Choosing the inference option.** The model scripts take `--phi-mode neutralized|active`. If the flag is omitted they use `active`, the published configuration, so always name the option ([docs/INFERENCE_MODES.md](../docs/INFERENCE_MODES.md)).

**Where outputs go.** Pass an output folder under `outputs/` (ignored by git) so the published files are not overwritten.

| Script | What it does | Needs the data package | Run time (measured, 2 CPU cores) |
|---|---|---|---|
| `check_models.py` | checks the five checkpoint hashes, the architecture and the two inference options; scores nothing | no | seconds |
| `reproduce_development_cases.py [out] --phi-mode …` | the 9 development cases | yes | a few seconds |
| `reproduce_validation.py [out] --phi-mode … [--workers N]` | all validation scores and the selection metrics | yes | about 3 min per option (`--workers 1`) |
| `reproduce_final_test.py [out] --phi-mode …` | final-test scores and tables T1–T7 | yes | about 1.5 min |
| `verify_phi_modes.py [--full] [--original DIR]` | checks both options against `results/` and `results_phi_neutralized/`, the frozen model files and the report numbers | yes | about 3 min; longer with `--full` |
| `score_tracks.py <processed_runs_dir> <out.csv>` | scores your own imported runs (option from `DEEPDAS_PHI_MODE`) | your runs | depends on the input |
| `import_runs.py <sims_dir> <out_dir> <group>` | converts SimStudio run folders into the processed format | raw runs | depends on the input |
| `recompute_zscore_from_raw.py` | recomputes the Z-score from raw exports | raw data package | – |
| `pretrain_autoencoder.py`, `train_ensemble.py <out_dir>` | retrain the encoder or the five seeds into a separate folder (refuses to write into `models/`) | yes | about 75 min for five seeds (release estimate) |
| `verify_release.py` | the checker of the original release folder `DeepDAS_Release_v1.0`; its manifest step does not apply to this repository | – | – |
| `_setup.py`, `_phi_io.py` | helpers: path setup, `--phi-mode` handling and stamping of outputs | – | – |

**Example:**
```bash
python scripts/reproduce_final_test.py outputs/ft_neutralized --phi-mode neutralized
cat outputs/ft_neutralized/RUN_MODE.json      # every output folder records the option it was made with
```
