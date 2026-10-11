# DeepDAS Release v1.0

**DeepDAS Ensemble v1** is a supervised deep model that flags anomalous driving behaviour from vehicle tracks. The tracks fuse distributed acoustic sensing (DAS), cameras and GPS in the SimStudio North Tel Aviv simulation. The model uses no simulator ground truth at inference.

This folder is the complete, self-contained release of the Deep DAS course project: the final model, its data in processed form, the code, the research record, and the reproduction of every reported result.

## Results at a glance

| | DeepDAS Ensemble v1 | Supervised single model | SimStudio Z-score |
|---|---|---|---|
| **Independent final test** (228 new runs, run once): macro recall over 9 types | **0.568** [0.486–0.646] | 0.364 | 0.231 |
| ROC-AUC / AUPRC | **0.908 / 0.742** | 0.769 / 0.515 | 0.783 / 0.394 |
| False alarms, normal driving | 2.0 % | 0.5 % | 0.8 % |
| False alarms, legitimate 1–10 s stops | 20/28 | 14/28 | 23/32 |

**Development and validation** (these sets were used to select the model, so they are not independent):
- 8 of 9 development cases detected;
- validation macro recall 0.714;
- validation AP 0.781.

**Main limitations** (`docs/LIMITATIONS.md`):
- false alarms on legitimate short stops and on tailgating controls;
- low recall on hard braking, speeding, overweight and lane straddling;
- simulation only.

## Contents

| Folder / file | What it is |
|---|---|
| `RESEARCH_OVERVIEW.md` | **Start here.** Question, model, research path, results (development / validation / final test kept apart), limitations |
| `INSTALL.md` | Installation and a 5-minute check |
| `models/deepdas_ensemble_v1/` | The final model (5 seeds, frozen thresholds) and its `MODEL_CARD.md` |
| `models/baselines/` | The supervised single model and the SimStudio Z-score model, as frozen for the comparison |
| `data/` | All data needed to reproduce, in processed form (`data/README_DATA.md`); raw data: `docs/RAW_DATA_PACKAGE.md` |
| `src/deepdas/` | The package: data pipeline, simulator (SimStudio, unchanged), generators, model, training, evaluation |
| `scripts/` | Reproduction, training, import, scoring and verification entry points |
| `results/` | Reproduced results: `development/`, `validation/`, `final_test/` (`results/README.md`) |
| `research/` | Seven studies, each with hypothesis, method, results and conclusions, and the original records (`research/README.md`) |
| `docs/` | Architecture, data pipeline, anomaly types, evaluation protocol, limitations, reproducibility, cross-platform differences, identifiers, history, archive index |
| `tests/` | Unit and equivalence tests |
| `RELEASE_VERIFICATION.md` | Results of the release checks E1–E10, including a clean installation on a second machine |
| `MANIFEST_SHA256.txt` | sha256 of every file of the release |

## Quick start
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python scripts/reproduce_development_cases.py      # 8 of 9, about 1 min
python scripts/verify_release.py --quick           # manifest + development + hygiene
```
Full reproduction and training: `docs/REPRODUCIBILITY.md`.

## Naming
Documentation uses release names. Data files, score tables and archived records keep the study's identifiers (`E_DIV` = DeepDAS Ensemble v1, `CR_P99` = supervised single model, `Z` = SimStudio Z-score), so that every result can be reproduced byte for byte (on the build platform; see `docs/CROSS_PLATFORM.md`). The full mapping is in `docs/IDENTIFIERS.md`.

## Provenance
- **Source.** The release was built from the frozen backup of the final model and the study records of the project. The original project folders were not changed.
- **Traceability.** Every copied file is listed with its source and hash in `docs/BUILD_PROVENANCE.tsv`. What stayed in the archive is listed in `docs/ARCHIVE_INDEX.md`.
- **Third-party code.** The SimStudio simulator, tracker and Z-score model (`src/deepdas/fp/`, `src/deepdas/fp_anomaly_model/`, `src/deepdas/batches/populate_vehicles.py`) are included unchanged from the SimStudio Final Project.
