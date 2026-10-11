# From the project package to this repository

**Source.** The repository is built from the review copy of the release package, `DeepDAS_Release_v1.1_review`. That copy is identical to the frozen `DeepDAS_Release_v1.0` except for:

- the documented `--phi-mode` option;
- study 08;
- the Klausner experiment.

**Layout.** The folder layout is kept, so every path used by the code is unchanged.

## What is the same

- **Code.** `src/`, `scripts/` (except the one added file below) and `tests/` are unchanged from the review copy.
- **Models.** `models/deepdas_ensemble_v1/` and `models/baselines/` are byte-identical; `scripts/check_models.py` verifies the hashes.
- **Results.** `results/` and `results_phi_neutralized/` are byte-identical; `scripts/verify_phi_modes.py` checks that they reproduce.
- **Records.** `research/` keeps its study records verbatim.
- **Manifests.** `MANIFEST_SHA256.txt` (v1.0) and `MANIFEST_SHA256_v1.1_review.txt` are kept unchanged as provenance records. They also list files that are now in the data package.

## Moved, added or left out

| Change | Where | Why |
|---|---|---|
| **Moved to the data package** | `data/processed/`, `data/scenarios/`, `data/simstudio_inputs/`, `research/07_targeted_data_study/data/` | Size: about 5,800 files, 407 MB in total ([DATA_PACKAGE.md](DATA_PACKAGE.md)) |
| **Left out** | `models/baselines/zscore/model_live.json` and its copy in `data/simstudio_inputs/zscore/` | An informational export holding a private path on the authors' computer; never read by the code (the model file `model_live.pkl` is included) |
| **Moved** | release `README.md` → [docs/RELEASE_README_v1.0.md](RELEASE_README_v1.0.md); release `INSTALL.md` → [docs/RELEASE_INSTALL_v1.0.md](RELEASE_INSTALL_v1.0.md) | Replaced by a repository front page and an install guide that cover the data package and both inference options |
| **Added** | `README.md`, `INSTALL.md`, `CREDITS.md`, `.gitignore` | Repository front page, setup and credits |
| **Added** | `scripts/check_models.py` | Installation check without data: model hashes and the two inference options. It scores nothing. |
| **Added** | `models/deepdas_ensemble_v1/experiment_klausner/` | The final model on the real Klausner recording (inference only, qualitative) |
| **Added** | `figures/` | Figure gallery. Each figure is in PNG and PDF, with its data CSV and generating script. The scripts are repository copies of the report-figure scripts, with paths made relative. |
| **Added** | `report/` | The final 5-page report, v3 (PDF, LaTeX, figures); the previous version v2.2 in `report/archive/v2_2/` |
| **Added** | `docs/INFERENCE_MODES.md`, `docs/DATA_PACKAGE.md`, `docs/BACKGROUND_SIMSTUDIO.md`, `docs/README.md`, this file, `docs/assets/simstudio/` | Documentation and background images |
| **Added** | a `README.md` in every main folder; a self-contained summary at the top of every study page; `research/README.md` indexed by four research questions; [QUESTION_MAP.md](QUESTION_MAP.md) | Orientation |
| **Added** | `research/01_unsupervised_autoencoder/outputs/figures/` | The 13 archived stage-1 figures, restored so that the image links of the verbatim study report resolve |
| **Edited** | a short note at the top of `results/README.md` and `RESEARCH_OVERVIEW.md` | States the inference option of `results/`, and which Z-score subset the release overview quotes |
| **Edited** (documentation round v3) | `RESEARCH_OVERVIEW.md`, `docs/LIMITATIONS.md`, `docs/EVALUATION_PROTOCOL.md`, `docs/DEVELOPMENT_HISTORY.md`, `docs/IDENTIFIERS.md`, `docs/INFERENCE_MODES.md` | Comparison tables on one population, study 08 and the Klausner experiment added, display names mapped; listed in [CHANGES_DOCS_v3.md](CHANGES_DOCS_v3.md). The records (`research/*/records/`, `results/*/records/`) and the frozen model folder are not edited. |
| **Added** | `models/MODEL_CARD_DeepDAS_Ensemble_v1.md` | The current model card, outside the hash-checked model folder: comparison on the common decided subset, both inference options, the Klausner experiment. The original card is unchanged. |
| **Added** | `figures/scripts/make_main_performance_figures.py` → `figures/final_test/` V6, V7 | The main final-test figures with the three detectors frozen before the test (values asserted equal to the four-configuration figures G3/G4) |

**Not published at all:**

- the raw simulator output;
- the working project folders;
- internal backups and transfer folders;
- the training logs of individual runs;
- the course-submission working files.

The release documents what stayed in the archive in [ARCHIVE_INDEX.md](ARCHIVE_INDEX.md).
