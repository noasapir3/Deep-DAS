# Release verification: DeepDAS Release v1.0 (closing check, 2026-10-10)

The checks are defined in `docs/EVALUATION_PROTOCOL.md` §5–7, with E2 criterion **version 1.1**.

**Release verified:**
- `MANIFEST_SHA256.txt`: 6,096 files. That is every file of the release except `MANIFEST_SHA256.txt` and this file.
- On the user's computer the release folder contains exactly the files of the manifest plus these two.

**Platforms:**
- **Build platform:** Linux x86-64, Python 3.13.16.
- **Clean installation:** a new environment on the user's computer, Linux aarch64 (ARM) in the virtual machine of the Claude desktop app, Python 3.13.15.
  - The packages were downloaded fresh with `pip install --no-cache-dir -r requirements.txt`.
  - The release was copied from `final proj/DeepDAS_Release_v1.0/`, without access to the original project folders.

## Summary

| Check | Build platform (x86-64) | Clean installation (aarch64) |
|---|---|---|
| **E1** Manifest, frozen checkpoint hashes | PASS (6,096 files) | PASS |
| **E2** Development cases (v1.1) | PASS: z ≤ 1.2e-6, ensemble ≤ 5.9e-7; 8/9 and 7/9; decisions identical; tables byte-identical | PASS: z ≤ 3.4e-6, ensemble ≤ 1.6e-6; 8/9 and 7/9; decisions identical; tables byte-identical |
| **E3a** Shipped results = original outputs (26 files) | PASS | PASS |
| **E3b** Validation regenerated (13 files) | PASS: 13 byte-identical | PASS: 13 equivalent (decisions, windows and keys identical; z ≤ 6.7e-6; ensemble ≤ 1.8e-6) |
| **E3c** Final test regenerated (11 files) | PASS: 11 byte-identical | PASS: 8 tables byte-identical (T1–T7, report), 3 score files equivalent (ensemble ≤ 1.8e-6, z ≤ 5.2e-6) |
| **E4** Import from raw data | PASS: 15 runs of 7 sets identical | 3 final-test runs: channels and track metadata identical; amplitude identical except the last bit of float64 log10 (≤ 8.9e-16); scores from the re-imported runs give identical decisions |
| **E5** Retraining (verification only) | PASS: 5 checkpoints byte-identical | not run (about 75 min) |
| **E6** Generators | PASS: 8 runs of 8 generators | not run |
| **E7** Z-score from raw data | PASS: 46 tracks identical | PASS: 46 tracks identical |
| **E8** Clean installation | – | PASS: install, `pip check`, editable install, tests 49/49 |
| **E9** Self-containment and hygiene | PASS (81 code files) | PASS |
| **E10** Originals untouched | PASS: originals 5,414 and backup 5,422, 0 differences (before the build and at closing) | PASS: frozen backup (5,422) and study-07 folder (5,616), 0 differences |
| Raw-data package | – | PASS: 9 folders, every file against its checksum; the 36,297 referenced SimStudio files verified |

**Release folder ready: yes.** All checks pass under criterion v1.1. The remaining limitations are listed at the end.

## E2: criterion history (transparency)
| Version | Tolerance 1e-5 applied to | x86-64 | aarch64 |
|---|---|---|---|
| **1.0** (approved before the build) | raw seed logits, z, ensemble | PASS (logits 6.7e-6) | **FAIL**: raw logits up to 1.72e-5 (z 3.4e-6, ensemble 1.6e-6, decisions and tables identical) |
| **1.1** (approved 2026-10-10 after that result) | per-seed z and ensemble; decisions and result tables must be identical; raw logits reported | PASS | PASS |

**What did not change:**
- the model, its weights, checkpoints, thresholds, normalisation and algorithm;
- every shipped result.

**What changed** (release copy only):
- the verification code (`scripts/verify_release.py`, `scripts/reproduce_validation.py` option `--workers 1`, tests);
- the documentation.

The full numbers are in `docs/CROSS_PLATFORM.md`.

## Decisions near the threshold
Every unit of DeepDAS Ensemble v1 is at least 1.7e-3 from the threshold, that is 1,000–2,850 times the largest ARM deviation. The same holds for every decided unit of the single model (5.7e-3, 300 times).

| Set | Closest unit (on x86) | Largest ARM deviation | Decisions changed |
|---|---|---|---|
| Final test, tracks / vehicles | 2.5e-3 | 1.8e-6 | 0 / 1,327 and 0 / 1,312 |
| Validation | 1.7e-3 | 1.6e-6 | 0 / 5,026 |
| Legitimate-stop set | 3.3e-3 | 1.2e-6 | 0 / 563 |
| Development cases | 2.9e-2 | 1.6e-6 | 0 / 9 |

**The only decision whose margin is smaller than the ARM deviation** is not a decision of the released model. It is the alternative per-seed P95 flag of seed 1, for one validation vehicle: margin 2.1e-5, ARM change 8e-6, unchanged. That vehicle's ensemble score is 1.13, so the decision of DeepDAS Ensemble v1 is unaffected.

The comparison used by E3 was itself tested against injected errors (`tests/test_verify_comparator.py`). Each of the following makes it fail: a flipped decision, a flipped vehicle outcome, a changed window count, a score off by 2e-5, and a changed table cell.

## Closing check: a new machine

| Question | Result |
|---|---|
| Self-contained? | Yes. The code reads and writes only inside the release folder (E9). It imports nothing from the original project, and the editable install resolves the data and models inside the release |
| Dependencies complete? | Yes. Every third-party import of the code paths that are used is pinned in `requirements.txt` (numpy, pandas, pyarrow, torch, scikit-learn, scipy, python-calamine, openpyxl, matplotlib, tabulate). `python-docx` and `lxml` appear only in the simulator's optional Word report, which is switched off. `anomaly_model` appears only in a vendored helper that is never called (tested) |
| Installable? | Yes, from a fresh virtual environment with internet access to PyPI and the PyTorch CPU index (about 90 s of downloads on the test machine). Python ≥ 3.11 is required |
| Runnable? | Yes. The reproduction scripts, `score_tracks.py` (deployment scoring), `import_runs.py` and the Z-score recomputation all ran on the new machine |
| Reproducible? | Yes. Development 8/9, validation and final test (macro recall 0.568, ROC-AUC 0.908, AUPRC 0.742) are reproduced with identical decisions and byte-identical tables. On x86-64 every file is byte-identical, and retraining reproduces the checkpoints |
| Resource needs | Up to 3 GB RAM per scoring process. With less than about 6 GB, use `--workers 1` (`INSTALL.md`) |

## Details of the build-platform checks

### E5 retraining
An isolated copy of the release was trained with `scripts/train_ensemble.py --threads 1`, two seeds in parallel, writing outside `models/`.

| Seed | Epochs | Checkpoint sha256 | History | Thresholds | Validation scores |
|---|---|---|---|---|---|
| 0 | 17 | equal (`c382ea35…`) | equal | equal | byte-identical |
| 1 | 25 | equal (`1c7d53d0…`) | equal | equal | byte-identical |
| 2 | 15 | equal (`1d85b5a0…`) | equal | equal | byte-identical |
| 3 | 25 | equal (`2d265051…`) | equal | equal | byte-identical |
| 4 | 25 | equal (`92318107…`) | equal | equal | byte-identical |

Histories are compared without the wall-time column. Since then the release code has changed only in files that training does not use.

### E4 and E6
- **E4.** 2–3 runs per set were re-imported: final test, stalls and stops, legitimate-stop validation, both tailgating sets, new-scene check, study 07. All were identical.
- **E6.** One or two runs per generator were regenerated: final test (2), stalls, stop study, tailgating routes, tailgating controls, targeted data, cross-region.
  - Scenario, truth trail and realisation record were identical.
  - In `run.json` only these fields differ: wall time, the timestamped export folder, the recorded path of the source scenario, and one NaN field.

### Study 07
The evaluation reproduced all 7 result files byte for byte.

## Limitations that remain
1. **Byte identity of score files holds only on x86-64.** On aarch64 every decision and every table is identical, z and ensemble scores agree within 1e-5, and raw logits within 3.4e-5.
2. **Untested runs:**
   - retraining (E5) and the generator checks (E6) were run only on x86-64;
   - a native macOS run was not tested; the aarch64 test ran in Linux.
3. **Raw-data dependencies:**
   - level-3 reproduction (re-import, Z-score from raw, re-simulation) needs the separate raw-data package `final proj/DeepDAS_RawData_v1.0/`;
   - the 10.9 GB of SimStudio normal exports are referenced in place by hash, not copied into it.
4. **Installation needs internet access** for pip, or a local package mirror.
5. **Model limitations** are unchanged (`docs/LIMITATIONS.md`): false alarms on legitimate stops and tailgating controls, the weak types, decision coverage, and simulation only.

## Not modified
The original project folders, the frozen backup and the transfer folders (`DL/_transfer_*`, `_release_build_staging/`) were not modified or deleted. The build wrote only:
- `DeepDAS_Release_v1.0/`;
- `DeepDAS_RawData_v1.0/`;
- new files under `_release_build_staging/`, including the ARM outputs used for `docs/CROSS_PLATFORM.md` (`arm_outputs_20261010/`).
