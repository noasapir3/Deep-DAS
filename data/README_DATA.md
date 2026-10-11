# Data

Everything needed to reproduce the scores, tables and training of the release is here, in processed form. The raw simulator output is a separate package (`docs/RAW_DATA_PACKAGE.md`). How the data were produced is described in `docs/DATA_PIPELINE.md`.

| Folder | Content |
|---|---|
| `processed/normal_corpus/` | `channels.parquet` (14 channels per 0.1 s bin of every track), `amplitude.parquet` (DAS log-amplitude per bin), `labels.parquet` (role, scenario, region, split, quality), `meta.parquet` (export of origin, oracle id, timing), `splits.json` (region split) |
| `processed/development_cases/` | `cases.csv` (the 9 development cases DC-01…DC-09) and the processed controlled runs of four of them; the other five come from the normal-corpus files (SimStudio anomaly scenarios) |
| `processed/anomaly_sets/<set>/` | `vehicles.csv` (one row per simulated vehicle: label, type, region, split role, track keys) and `runs/<run>_{channels,amp,track_meta}.parquet` |
| `processed/final_test/` | the independent final test: `run_plan.csv` (frozen plan), `vehicles.csv`, `run_report.csv`, `runs/`, `realization/<run>.json` (truth realisation record per run), `zscore_scores_precomputed.csv` (SimStudio Z-score per track, computed from the raw exports) |
| `scenarios/north_tlv/` | SimStudio scenario files of the 148 North-TLV regions (sensor layout and normal scenario types); the generators start from these |
| `simulation_plans/<set>/` | the run plan, surveys and pre-simulation checks of each generated set |
| `simstudio_inputs/` | SimStudio reference inputs: region geometry, the nine original anomaly scenarios, the Klausner bus recording, reference exports, Z-score feature tables |

**Conventions.**
- **Track key:** `<group>/<scenario or run>|T<track id>`.
- **Vehicle key:** `<…>|veh<id>` (the oracle id, used only for labels and evaluation).
- **Run ids** keep their original prefixes (`docs/IDENTIFIERS.md`).
- **Labels** are in `vehicles.csv`: `anomaly_verified` is a positive; the `normal_*` labels are negatives; every `*excluded` label is out of every evaluation (`docs/DATA_PIPELINE.md`).
- **Truth** is never part of the model input. Only the labels and the realisation records derive from it.
