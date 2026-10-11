# Reproducibility

The release can reproduce its results on three levels. Only the first two are self-contained.

| Level | Starts from | Needs | Reproduces | Time (2 CPU cores) |
|---|---|---|---|---|
| **1. Scores and tables** | processed data + frozen models (in the release) | the release only | development 8/9, all validation scores, final-test scores and tables T1–T7, the study-07 evaluation | about 10 min |
| **2. Training** | processed data + autoencoder initialisation (in the release) | the release only | the five frozen checkpoints of DeepDAS Ensemble v1, exactly (same platform) | about 75 min (two seeds in parallel) |
| **3. Raw data** | raw simulator exports | the separate **raw-data package** (`RAW_DATA_PACKAGE.md`) | the processed data (import), the Z-score scores, and the simulations themselves (generators) | from minutes (samples) to many hours (everything) |

## Level 1: scores and tables
```bash
python scripts/reproduce_development_cases.py    # → results/development/        (8 of 9)
python scripts/reproduce_validation.py           # → results/validation/          (5 seeds, two at a time)
python scripts/reproduce_final_test.py           # → results/final_test/          (T1–T7, FINAL_TEST_REPORT.md)
python research/07_targeted_data_study/code/evaluate_selection_rule.py outputs/w_round
python scripts/verify_release.py                 # E1, E2, E3, E9: regenerates into outputs/verify/ and compares
```
- **Each script takes an optional output directory.** `verify_release.py` always writes to `outputs/verify/` and compares the regenerated files with the shipped `results/`.
- **The shipped `results/` were checked against the original outputs of the project** (`results/ORIGINAL_OUTPUTS_SHA256.txt`): byte-identical.
- **Z-score track scores.** The final-test reproduction reads them from `data/processed/final_test/zscore_scores_precomputed.csv`, because recomputing them needs the raw exports (level 3).

## Level 2: training
```bash
python scripts/train_ensemble.py outputs/retrain --seeds 0,1,2 --threads 1 &
python scripts/train_ensemble.py outputs/retrain --seeds 3,4   --threads 1
```
- **What it writes:** per seed, `checkpoint_best.pt`, `history.csv`, the val_B thresholds and the validation scores.
- **It refuses to write into `models/`.** This is a verification and study tool. The released model is the frozen one.
- **Determinism.** With `--threads 1` and the library versions of `requirements.txt` on Linux x86-64 CPU, the retrained checkpoints are identical to the frozen ones (check E5, `RELEASE_VERIFICATION.md`).
- **Other platforms.** On another platform, small numerical differences can change the selected epoch.
- **The ensemble constants are not retrained.** They are recomputed from the val_B logits of the retrained seeds.

**The encoder initialisation** (`models/deepdas_ensemble_v1/autoencoder_init.pt`) can itself be retrained:
```bash
python scripts/pretrain_autoencoder.py outputs/autoencoder
```
This step was **not** part of the release verification, because it takes several hours on CPU. The shipped initialisation is the original file. Its training history is in `models/deepdas_ensemble_v1/autoencoder_training_history.csv`.

## Level 3: from raw data (needs the raw-data package)
```bash
# re-import raw runs into the processed format
python scripts/import_runs.py <raw>/final_test outputs/reimport final_test ft_collision_00_003 ...
# recompute the SimStudio Z-score of the final test from raw exports
python scripts/recompute_zscore_from_raw.py <raw>/final_test outputs/zscore_recomputed.csv
# regenerate a simulation from its plan (writes scenario, export, truth, run.json, realization.json)
python -m deepdas.batches.final_test run outputs/generated/final_test ft_collision_00_003
python -m deepdas.batches.<set> run <out> <run_id> ...   # tailgating_routes | tailgating_controls | stalls | stop_study | targeted_data
python -m deepdas.batches.cross_region <out> <run_id> ...   # the cross-region set (no 'run' sub-command)
```
**Comparisons.**
- Compare a re-imported run with `data/processed/<set>/runs/<run_id>_*.parquet`.
- Compare a regenerated run with its raw copy.

  | File | Expected |
  |---|---|
  | scenario JSON, `truth_trail.parquet`, `realization.json` | identical |
  | `run.json` | identical except the wall-time fields, the timestamped export-folder name and the recorded path of the source scenario (release layout instead of the working-tree layout) |

**Data that cannot be reproduced from this release alone:**
- The processed **normal corpus** was imported from the SimStudio project's own normal-traffic exports. These are referenced, with hashes, in the raw-data package.
- **Re-simulating the normal corpus** would need the SimStudio project's generation run. That run is not part of this work.

## What is guaranteed, and what is not
- **Guaranteed on the build platform:** every score, decision and table of levels 1–2 is identical to the original.
- **Elsewhere:**
  - verified on Linux aarch64: every decision and every result table identical, z and ensemble scores within 1e-5 (`CROSS_PLATFORM.md`);
  - a difference in any decision or table would be a failure of E2 or E3 (`EVALUATION_PROTOCOL.md` §6–7).
- **Not guaranteed:** wall times and timestamps in run records and logs.
