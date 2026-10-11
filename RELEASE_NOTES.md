# Release notes

## v1.0 (2026-10-10)
This is the first and final release of the Deep DAS project.

### Model
- **DeepDAS Ensemble v1** (development identifier `E_stall_diversity_ENSEMBLE`), frozen before the final test.
- Weights, normalisation, standardisation constants and the threshold 1.9726181803233118 are byte-identical to the frozen model. The sha256 of each checkpoint is recorded in `models/deepdas_ensemble_v1/ensemble.json`.
- Baselines frozen for the comparison: the supervised single model (P99 −5.523461642265342) and the SimStudio Z-score model (threshold 15.392482911229285).

### Results reproduced by this release
| Kind | Result |
|---|---|
| Development | 8 of 9 cases |
| Validation | macro recall 0.714, AP 0.781 |
| Independent final test | macro recall 0.568, ROC-AUC 0.908, AUPRC 0.742 (all tables T1–T7) |
| Study 07 | the evaluation of the targeted-data round |

### What changed relative to the working project
- **Code.** Refactored from the project's package and experiment scripts into one package (`src/deepdas/`) with release-relative paths.
  - The algorithm, channels, data order, normalisation, pooling, thresholds and decisions are unchanged.
  - Every refactored path was checked for equivalence against the original outputs (`RELEASE_VERIFICATION.md`).
- **Names.** Professional display names in documentation and display reports. Study identifiers are kept in data and score tables (`docs/IDENTIFIERS.md`).
- **Structure.**
  - The data are organised by role: normal corpus, development cases, anomaly sets, final test.
  - The research record is organised by study, with a README per study.
- **Raw data.** The raw simulator output moved to a separate raw-data package.

### Verification criterion
- The E2 criterion was revised on 2026-10-10 (v1.0 → v1.1) after the clean-install check on Linux aarch64.
  - The 1e-5 tolerance now applies to the per-seed z and ensemble scores.
  - Every decision and every result table must stay identical.
  - Raw logits are reported, not gated.
- Nothing in the model or its results changed. Details: `docs/EVALUATION_PROTOCOL.md` §6 and `docs/CROSS_PLATFORM.md`.

### Not included
- Raw simulations (separate package).
- Models of arms that were not adopted.
- Logs.
- Superseded backups (`docs/ARCHIVE_INDEX.md`).

### Known limitations
See `docs/LIMITATIONS.md`.
