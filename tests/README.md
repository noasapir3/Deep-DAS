# Tests

```bash
pip install pytest
python -m pytest -q
```

| File | What it checks | Needs the data package |
|---|---|---|
| `test_metrics.py`, `test_verify_comparator.py` | metric functions and the result comparator | no |
| `test_model_windows_norm.py` | windowing, normalisation and model shapes | no |
| `test_release_equivalence.py` | the model description against the freeze record; the development cases against the stored scores | partly |
| `test_baseline_equivalence.py` | the shipped Z-score model reproduces its published scores | yes |
| `test_dataset_integrity.py`, `test_map_matching_and_channels.py` | the processed dataset and the channel pipeline | yes |
| `test_severity_and_runner.py` | anomaly severity and the simulation runner | needs vendored scenario files |
| `test_provenance_and_isolation.py` | no machine-specific paths in code; the code reads only release-relative paths | no |

**Expected results (measured in a fresh clone):**

| Setting | Result |
|---|---|
| Without the data package | 33 passed, 8 skipped. 5 failed and 3 errors, from tests that read data-package files without a skip guard (`FileNotFoundError`). |
| With the data package | 48 passed, 1 expected failure: `test_manifest_hashes_match`. It checks against the original release manifest, from which this repository deliberately differs (`docs/REPOSITORY_MAP.md`). |
