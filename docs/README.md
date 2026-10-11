# Documentation

| Page | Read it for |
|---|---|
| [BACKGROUND_SIMSTUDIO.md](BACKGROUND_SIMSTUDIO.md) | the earlier SimStudio fibre-sensing project and what DeepDAS adds |
| [INFERENCE_MODES.md](INFERENCE_MODES.md) | the two inference options (`neutralized` / `active`) and their separate results |
| [ARCHITECTURE.md](ARCHITECTURE.md) | the 14 channels, the encoder, pooling, head and ensemble rule |
| [ANOMALY_TYPES.md](ANOMALY_TYPES.md) | the nine anomaly types, their realisation rules, stops and legitimate stops |
| [DATA_PIPELINE.md](DATA_PIPELINE.md) | from SimStudio exports to model input: import, map matching, quality gate, labels |
| [DATA_PACKAGE.md](DATA_PACKAGE.md) | what is in git and what is in the separate data package; download and checks |
| [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) | the separation of development, validation and final test; decision units; metrics |
| [LIMITATIONS.md](LIMITATIONS.md) | the known limitations of the final model |
| [LEAKAGE_AUDIT.md](LEAKAGE_AUDIT.md) | how truth leakage and split leakage were ruled out |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md), [CROSS_PLATFORM.md](CROSS_PLATFORM.md) | exact reproduction, pinned versions, differences between CPU architectures |
| [IDENTIFIERS.md](IDENTIFIERS.md) | internal identifiers (`E_DIV`, `CR_P99`, `C_new`, …) and their release names |
| [DEVELOPMENT_HISTORY.md](DEVELOPMENT_HISTORY.md) | the chronological development record |
| [REPOSITORY_MAP.md](REPOSITORY_MAP.md) | how this repository maps to the frozen release package |
| [RAW_DATA_PACKAGE.md](RAW_DATA_PACKAGE.md), [ARCHIVE_INDEX.md](ARCHIVE_INDEX.md), `BUILD_PROVENANCE.tsv`, `SIMSTUDIO_PROVENANCE.json` | provenance of every file of the release |
| [RELEASE_README_v1.0.md](RELEASE_README_v1.0.md), [RELEASE_INSTALL_v1.0.md](RELEASE_INSTALL_v1.0.md) | the front page and install guide of the original release folder |

**A note on archived documents.** The release documents were written for the release folder `DeepDAS_Release_v1.0`. Two kinds of content in them are historical:

- **Paths and commands** that assume the processed data inside the folder. In this repository, unpack the [data package](DATA_PACKAGE.md) first.
- **Z-score numbers on its own decided set** (macro recall 0.231, ROC-AUC 0.783, AUPRC 0.394). The repository front page and the report compare the Z-score on the same vehicles as the deep models: 0.232 / 0.819 / 0.496.
