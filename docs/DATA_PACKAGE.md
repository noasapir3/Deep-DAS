# Data: what is in the repository and what is in the data package

## In the repository (git)

| Path | Content |
|---|---|
| `models/` | the frozen ensemble (5 checkpoints), the supervised single model, the SimStudio Z-score model (2.3 MB) |
| `results/`, `results_phi_neutralized/` | all published scores and tables (development, validation, final test) |
| `research/*/records`, `research/07_targeted_data_study/{models,results}`, `research/08_*/tables` | the study records, W1/W2 checkpoints and result tables |
| `data/simulation_plans/` | run plans, surveys and pre-simulation checks of every generated set |
| `data/README_DATA.md` | the description of every data folder and its conventions |

**What runs without the data package.** You can install the code, check the models (`scripts/check_models.py`), read every result, and regenerate every figure that is drawn from `results/` and the study records.

## In the data package `DeepDAS_data_v1.0.zip`

| Path after unpacking | Size | Content |
|---|---|---|
| `data/processed/` | 248 MB | channels, DAS amplitude, labels and metadata of the normal corpus; the anomaly sets; the development cases; the final test (runs, realisation records, precomputed Z-score) |
| `data/scenarios/` | 73 MB | SimStudio scenario files of the 148 North-TLV regions (the generators start from these) |
| `data/simstudio_inputs/` | 28 MB | SimStudio reference inputs: region geometry, the nine original anomaly scenarios, the **Klausner** real recording, reference exports, Z-score feature tables |
| `research/07_targeted_data_study/data/` | 60 MB | the W1/W2 targeted training and validation sets |
| `DATA_PACKAGE_SHA256.txt` | – | sha256 of every one of the 5,772 files |

| Archive | Size | sha256 |
|---|---|---|
| `DeepDAS_data_v1.0.zip` | 270,604,098 bytes | `2894af8d90dee693ba4bb341984063bbe0e2ccb3c51b94137fdf2401444a7a08` |

**Integrity.** Every file in the package is byte-identical to the frozen release package. Each was checked against the release manifest (`MANIFEST_SHA256_v1.1_review.txt`).

**One file is left out on purpose:** `model_live.json`, an informational export of the SimStudio Z-score that recorded a path on the authors' computer. The code never reads it; the model itself (`model_live.pkl`) is included.

**Why the data is not stored in git.** It holds about 5,800 files, and one of them is 96 MB, close to GitHub's 100 MB per-file limit. A single versioned archive keeps the repository small and makes the exact data version explicit.

## Where to download it
> **Status: hosting not yet set up.** The archive has been built and checked.
>
> **Proposed hosting:** a GitHub Release asset of this repository. Assets of up to 2 GB per file are included in a normal GitHub account at no cost.
>
> The download link will be added here once the authors approve the upload.

**Use after downloading:**
```bash
unzip DeepDAS_data_v1.0.zip -d .              # run from the repository root
sha256sum -c --quiet DATA_PACKAGE_SHA256.txt
```

## Raw simulator output
The raw SimStudio exports (about 4.5 GB, plus 10.9 GB of normal exports referenced by hash) are a separate package, `DeepDAS_RawData_v1.0`. They are needed only to re-import runs, to recompute the Z-score from raw data, or to re-simulate. Description: [RAW_DATA_PACKAGE.md](RAW_DATA_PACKAGE.md). It is not published with this repository.

## Attribution
The road network of the simulation regions is derived from OpenStreetMap (© OpenStreetMap contributors, available under the Open Database License). See [CREDITS.md](../CREDITS.md).
