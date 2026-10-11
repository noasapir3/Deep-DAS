# Data

**This folder in git** holds `simulation_plans/` (the run plans, surveys and pre-simulation checks of every generated set) and the data description [README_DATA.md](README_DATA.md).

**The data package.** The processed data the code reads is distributed separately; see [docs/DATA_PACKAGE.md](../docs/DATA_PACKAGE.md). After you unpack `DeepDAS_data_v1.0.zip` in the repository root, this folder also contains:

| Folder | Content |
|---|---|
| `processed/normal_corpus/` | 14 channels and DAS amplitude of every normal track, labels, metadata, region split |
| `processed/development_cases/` | the 9 development cases |
| `processed/anomaly_sets/<set>/` | the simulated anomaly sets (cross-region, tailgating, stalls and stops, legitimate stops, new-scene check) |
| `processed/final_test/` | the final test: frozen plan, runs, realisation records, precomputed Z-score |
| `scenarios/north_tlv/` | SimStudio scenario files of the 148 regions |
| `simstudio_inputs/` | SimStudio reference inputs, including the Klausner real recording |

**Conventions** (track keys, labels, excluded vehicles): [README_DATA.md](README_DATA.md).

**How the data was produced:** [docs/DATA_PIPELINE.md](../docs/DATA_PIPELINE.md).

**Ground truth.** Simulator truth is never a model input; it only labels vehicles and verifies realisation.
