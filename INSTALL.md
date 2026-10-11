# Installation, data and reproduction

Contents:

1. [Requirements](#1-requirements)
2. [Install](#2-install)
3. [Check the installation (no data)](#3-check-the-installation-no-data)
4. [Get the data package](#4-get-the-data-package)
5. [Run the model](#5-run-the-model)
6. [Reproduce the published results](#6-reproduce-the-published-results)
7. [Tests](#7-tests)
8. [Troubleshooting](#8-troubleshooting)

## 1. Requirements

| Item | Requirement |
|---|---|
| Operating system | Linux x86-64 (build and verification platform). aarch64 Linux was tested for scoring (`docs/CROSS_PLATFORM.md`). macOS and Windows were **not** tested. |
| Python | 3.11 or newer; verified with 3.13 |
| GPU | not needed; everything runs on CPU |
| Memory | about 2.5 GB per scoring process; the full validation reproduction needs about 6 GB (or run it with `--workers 1`) |
| Disk | about 35 MB for the repository, 271 MB for the data package archive, about 410 MB once unpacked |

## 2. Install
```bash
git clone https://github.com/noasapir3/Deep-DAS.git DeepDAS
cd DeepDAS
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```
- **PyTorch build.** `--extra-index-url` selects the CPU build of PyTorch.
- **Pinned versions.** `requirements.txt` pins the exact versions with which the results were reproduced byte for byte; other versions may change the last digits of scores.

**Running the code.** The scripts put `src/` on the Python path themselves, so installing the package is optional. To import it from your own code, install it in editable mode with `pip install -e .`. It locates `models/` and `data/` relative to the repository folder, so a non-editable install would not find them.

## 3. Check the installation (no data)
```bash
python scripts/check_models.py
```
Expected output: ten `PASS` lines and `ALL CHECKS PASSED`. The script checks:

- the sha256 of the five frozen checkpoints against `models/deepdas_ensemble_v1/ensemble.json`;
- that they load with the expected architecture (head 17 → 16 → 1, 67,225 parameters per seed);
- that the two inference options set the 17th head input as documented (`docs/INFERENCE_MODES.md`).

It scores no data and produces no result.

## 4. Get the data package
The processed data (simulated tracks, anomaly sets, final test, SimStudio inputs) is **not stored in git**. It is distributed as one archive, `DeepDAS_data_v1.0.zip` (271 MB), attached to the release [`data-v1.0`](https://github.com/noasapir3/Deep-DAS/releases/tag/data-v1.0). [docs/DATA_PACKAGE.md](docs/DATA_PACKAGE.md) describes what it contains.

Download it, unpack it into the repository root, then check every file:
```bash
curl -L -o DeepDAS_data_v1.0.zip https://github.com/noasapir3/Deep-DAS/releases/download/data-v1.0/DeepDAS_data_v1.0.zip
unzip DeepDAS_data_v1.0.zip -d .              # creates data/processed, data/scenarios, data/simstudio_inputs, research/07_targeted_data_study/data
sha256sum -c --quiet DATA_PACKAGE_SHA256.txt  # no output = every file matches
```

## 5. Run the model
Every command below names the inference option explicitly. If `--phi-mode` is omitted, the code uses `active`.

**Development cases (9 vehicles, a few seconds):**
```bash
python scripts/reproduce_development_cases.py outputs/dev_neutralized --phi-mode neutralized   # prints 7 / 9
python scripts/reproduce_development_cases.py outputs/dev_active      --phi-mode active        # prints 8 / 9 (published result)
```

**Your own imported runs** (`scripts/import_runs.py` converts SimStudio run folders into the processed format):
```bash
DEEPDAS_PHI_MODE=neutralized python scripts/score_tracks.py <processed_runs_dir> outputs/scores_neutralized.csv
DEEPDAS_PHI_MODE=active      python scripts/score_tracks.py <processed_runs_dir> outputs/scores_active.csv
```
`score_tracks.py` reads the option from the environment variable `DEEPDAS_PHI_MODE`; it does not take the flag.

**The Klausner real recording** (both options at once, a few seconds):
```bash
python models/deepdas_ensemble_v1/experiment_klausner/run_klausner_experiment.py
```

Output folders under `outputs/` are ignored by git.

## 6. Reproduce the published results
Always pass an output folder outside `results/` and `results_phi_neutralized/`, so the published files are not overwritten. Then compare.

**Measured run times.** The times below were measured in a fresh clone on Linux x86-64 with 2 CPU cores, Python 3.13 and the pinned requirements. All the outputs below were byte-identical to the published files.

```bash
# development (a few seconds)
python scripts/reproduce_development_cases.py outputs/dev_active --phi-mode active
cmp outputs/dev_active/development_cases.csv results/development/development_cases.csv && echo identical

# final test (about 1.5 min per option)
python scripts/reproduce_final_test.py outputs/ft_active --phi-mode active
python scripts/reproduce_final_test.py outputs/ft_neutralized --phi-mode neutralized   # post-hoc analysis

# validation (about 3 min per option with --workers 1)
python scripts/reproduce_validation.py outputs/val_active --phi-mode active --workers 1
```

**All checks in one command:**
```bash
python scripts/verify_phi_modes.py          # about 3 min; add --full to also reproduce both validation runs
```
It checks that:

- the frozen model files are unchanged;
- `active` reproduces `results/` byte for byte;
- `neutralized` reproduces `results_phi_neutralized/` byte for byte;
- the 17th input behaves as documented;
- every number of the report is found in its source file.

`scripts/verify_release.py` is the checker of the original release folder (`DeepDAS_Release_v1.0`). Its manifest step E1 expects the release's own files, so it reports differences in this repository; use `verify_phi_modes.py` here.

**Training (not needed to use the model).** `scripts/train_ensemble.py <out_dir>` retrains the five seeds into a separate folder. The release record estimates about 75 min on 2 cores; this was not re-measured here. It refuses to write into `models/`. See `docs/REPRODUCIBILITY.md`.

## 7. Tests
```bash
pip install pytest
python -m pytest -q
```
**Results in a fresh clone (measured):**

| Setting | Result |
|---|---|
| Without the data package | 33 passed, 8 skipped. 5 failed and 3 errors, all `FileNotFoundError` for files of the data package. |
| With the data package | 48 passed, **1 expected failure**: `test_provenance_and_isolation.py::test_manifest_hashes_match`. |

**Why the manifest test fails.** It checks the files against the manifest of the original release folder. This repository deliberately differs from that folder: the informational file `model_live.json` is left out, and the front-page documents are replaced ([docs/REPOSITORY_MAP.md](docs/REPOSITORY_MAP.md)).

The code, models and results are checked by `scripts/check_models.py` and `scripts/verify_phi_modes.py` instead.

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `FileNotFoundError` under `data/processed/...` | The data package is not unpacked in the repository root (§4). |
| Killed during `reproduce_validation.py` | Not enough memory: add `--workers 1`. |
| Scores differ in the last digits | Different library versions or CPU architecture: decisions and tables should still match (`docs/CROSS_PLATFORM.md`). |
| `ModuleNotFoundError: deepdas` in your own code | Run `pip install -e .`, or add `src/` to `PYTHONPATH`. |
