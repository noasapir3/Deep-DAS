# Installation

**Requirements:**
- Python 3.11 or newer (built and verified with 3.13);
- about 1 GB of disk for the release and 2.5 GB of RAM per scoring process;
- no GPU.

The release is verified on Linux x86-64. Exact reproduction of the scores and of retrained weights is guaranteed with the pinned versions of `requirements.txt` (`docs/REPRODUCIBILITY.md`).

## 1. Environment
```bash
cd DeepDAS_Release_v1.0
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```
`--extra-index-url` selects the CPU build of torch. A CUDA build also works for scoring; exact equality is verified only on CPU.

## 2. Two ways to use the code
**Scripts, no installation of the package.** Every script in `scripts/` puts `src/` on the Python path itself:
```bash
python scripts/reproduce_development_cases.py
```
**As a package.** Install it in editable mode, so that it keeps finding the release's data and models, which are located relative to the release folder:
```bash
pip install -e .
python -c "from deepdas.ensemble.scoring import Ensemble; print(Ensemble().threshold)"
```
**Do not install the package non-editably.** It resolves the data and models relative to the release folder, so a copy in `site-packages` would not find them.

## 3. Check the installation (a few minutes)
```bash
python scripts/verify_release.py --quick
```
This runs:
- **E1:** every file against `MANIFEST_SHA256.txt`;
- **E2:** the nine development cases re-scored and compared with the stored z and ensemble scores (within 1e-5) and decisions; expected 8/9;
- **E9:** self-containment and hygiene.

The full check (`python scripts/verify_release.py`, about 10 minutes on 2 cores) also regenerates the validation and final-test results and compares them with `results/` (E3).
- **Tables** must be byte-identical everywhere.
- **Score files** must be byte-identical on x86-64. On another architecture they must have identical decisions, with z and ensemble scores within 1e-5 (`docs/CROSS_PLATFORM.md`).

With little memory (less than about 6 GB), or when a command time limit applies, run the checks in parts, with the validation seeds one after the other:
```bash
python scripts/verify_release.py --steps E1,E2,E9
python scripts/verify_release.py --steps E3a,E3b --workers 1
python scripts/verify_release.py --steps E3c
```

## 4. Tests
```bash
pip install pytest
python -m pytest -q
```

## 5. Raw data (optional)
Only for re-import, Z-score recomputation and simulation checks: see `docs/RAW_DATA_PACKAGE.md`.
