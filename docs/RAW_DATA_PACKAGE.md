# Raw-data package (separate from the release)

The release holds the **processed** form of every data set, and that is enough for levels 1–2 of `REPRODUCIBILITY.md`: scores, tables and training.

The **raw** simulator output is needed only for level 3:
- re-import;
- Z-score recomputation;
- comparison with regenerated simulations.

Because of its size it is a separate package.

**Name and location:** `DeepDAS_RawData_v1.0/`, next to the release in the project folder (`final proj/`). It is not shipped inside the release. Without it, level-3 reproduction is **not available**, and the release does not claim it.

## Layout

| Folder | Runs | Content of each run folder | Release counterpart |
|---|---|---|---|
| `final_test/` | 228 | `<run>.sim.json`, export folder (`*_tracking_audit/`, Excel), `truth_trail.parquet`, `run.json`, `realization.json` | `data/processed/final_test/` |
| `stalls_and_stops/` | 118 | as above | `data/processed/anomaly_sets/stalls_and_stops/` |
| `legitimate_stop_validation/` | 120 | as above | `…/legitimate_stop_validation/` |
| `tailgating_routes/` | 36 | as above | `…/tailgating_routes/` |
| `tailgating_matched_controls/` | 48 | as above | `…/tailgating_matched_controls/` |
| `new_scene_check/` | 72 | as above | `…/new_scene_check/` |
| `cross_region/` | 54 | as above | `…/cross_region/` |
| `development_cases_controlled/` | 4 | the four controlled development scenarios (scenario, export, truth, `run.json`; the realisation check of the batch is `REALIZATION_CHECK.json`) | `data/processed/development_cases/` |
| `targeted_data_study/` | 314 | as above | `research/07_targeted_data_study/data/` |
| `simstudio_normal_exports/` | — | **a reference, not a copy**: `MANIFEST.tsv` lists the 1,865 SimStudio export folders of the normal corpus with their location in the SimStudio project (`maps/north_TLV/…`) and the sha256 of every file | `data/processed/normal_corpus/` |

**Every folder has a `SHA256SUMS.txt` with one line per file.** The copied folders hold 994 runs, 7,044 files and 4.49 GB in total.

The package root also contains:
- `PACKAGE_MANIFEST.tsv`: folder, number of runs, number of files, bytes, and the sha256 of the folder's checksum file;
- `README.md`: how to verify and use the package.

## Obtaining
The package was built from the project's own copies of the runs; the source of every folder is recorded in `PACKAGE_MANIFEST.tsv`.
- Five folders were copied locally from the frozen archive and the SimStudio project.
- Four folders (legitimate-stop validation, both tailgating sets, new-scene check) exist only in the working copy of the project. They were transferred and verified file by file against the hashes of that copy.
- Three folders were additionally cross-checked against an independent copy: final test, stall set, study 07.
- Nothing was re-simulated. To hand the project over, copy `DeepDAS_RawData_v1.0/` together with the release.

**The SimStudio normal exports are not duplicated** (about 11 GB). They stay in the SimStudio project, which is their original location, and are verified by hash against `simstudio_normal_exports/MANIFEST.tsv`.

## Verifying
```bash
cd DeepDAS_RawData_v1.0
for d in */; do (cd "$d" && sha256sum -c --quiet SHA256SUMS.txt) && echo "OK $d"; done
```
**Normal exports:** `python verify_simstudio_exports.py <path to the SimStudio maps/north_TLV folder>` (in the package root) checks every listed file.

## Using it with the release
```bash
python scripts/import_runs.py <package>/final_test outputs/reimport final_test <run_id> ...
python scripts/recompute_zscore_from_raw.py <package>/final_test outputs/zscore_recomputed.csv
python -m deepdas.batches.final_test run outputs/generated/final_test <run_id>   # then compare with <package>/final_test/<run_id>
```
The expected equalities are listed in `REPRODUCIBILITY.md`, level 3. The results obtained during the release build are in `RELEASE_VERIFICATION.md` (checks E4, E6, E7).
