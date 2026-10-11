# DeepDAS Release v1.1_review

**Status:** this is a review copy of `DeepDAS_Release_v1.0`. It adds one documented inference option and a post-hoc study. Nothing was trained, re-calibrated or re-standardised. `DeepDAS_Release_v1.0` itself is unchanged and remains the reference release.

## What is the same as v1.0
- **Model weights and constants.** The five checkpoints, the input normalisation, the standardisation constants μ_k and σ_k, the threshold τ = 1.9726181803233118 and `ensemble.json` are byte-identical (checked by P1 in `scripts/verify_phi_modes.py`).
- **Model provenance.** The weights come from the frozen DeepDAS Ensemble v1 (development id `E_stall_diversity_ENSEMBLE`): five seeds trained with the study-04 recipe (expanded tailgating data plus diverse stall modes; head 17 → 16 → 1 with φ_u) and frozen before the final test. Training conditions are documented in `research/04_stall_diversity/` and `models/deepdas_ensemble_v1/`.
- **Published results.** Everything in `results/` is the original v1.0 output. Each `results/*/` folder now also carries a `RUN_MODE.json` stating `phi_mode: active`.

## What is new
| Item | Where |
|---|---|
| Inference option `--phi-mode {active,neutralized}` (or env `DEEPDAS_PHI_MODE`); default `active` | `src/deepdas/ensemble/phi_mode.py`, `Detector.head_in` |
| Reproduction scripts accept the option. Neutralized runs write to `results_phi_neutralized/` and stamp every output with the mode. | `scripts/reproduce_*.py`, `scripts/_phi_io.py` |
| Validation in neutralized mode uses the frozen μ_k, σ_k and τ (`frozen=True`). Nothing is recomputed. | `src/deepdas/evaluation/validation.py` |
| Neutralized-mode outputs (development, validation, exposed final test) | `results_phi_neutralized/` |
| Study 08: definition, history, tables, figures G0–G4, report number sources | `research/08_tailgating_feature_intervention/` |
| Verification of all of the above | `scripts/verify_phi_modes.py` |
| sha256 of every file of this copy | `MANIFEST_SHA256_v1.1_review.txt` |

**What `neutralized` means.** The 17th head input, x₁₇ = (φ_u − μ_φ)/σ_φ, is set to 0, the value of an average training-normal vehicle, using the stored constants. Everything else is unchanged.

**What it does not mean.** It is an intervention on the frozen hybrid model, defined after the final test had been exposed. It is **not** a model trained without φ_u, and **not** an independent evaluation. The hybrid (`active`) remains the selected, frozen and tested model. See `research/08_tailgating_feature_intervention/README.md`.

## Usage
```
python scripts/reproduce_development_cases.py --phi-mode neutralized
python scripts/reproduce_final_test.py        --phi-mode neutralized
python scripts/reproduce_validation.py        --phi-mode neutralized
python scripts/verify_phi_modes.py --original ../DeepDAS_Release_v1.0 --full
```
`--full` adds both validation reproductions, which take about 40 min on CPU.

## Verification (run on 2026-10-11): all checks passed
| Check | What it confirms |
|---|---|
| P1 | The 22 frozen model files match the v1.0 hashes. |
| P2 | The original v1.0 folder still verifies against its own manifest. |
| P3 | `active` reproduces the published development, validation and final-test tables byte for byte. |
| P4 | `neutralized` reproduces `results_phi_neutralized/` byte for byte. |
| P5 | x₁₇ = 0 for every φ value, and μ_φ, σ_φ are unchanged. |
| P6 | The report's four-row table equals the stored tables. |
| P7 | All 35 numbers in report v2 are found in their source files. |

**Note on the v1.0 manifest.** `MANIFEST_SHA256.txt` is kept as the v1.0 reference. In this copy, six code files intentionally differ from it:

- `detector.py`, `development.py` and `validation.py`;
- the three `reproduce_*` scripts.

Their active-mode behaviour is verified by P3. Use `MANIFEST_SHA256_v1.1_review.txt` for this copy.

## Not part of this copy (see `MISSING_BEFORE_SUBMISSION.md`)
Team contributions and the public code link are still placeholders. The package is therefore **not** ready for submission.

*Update in this repository (2026-10-11):*

- the public code link was added in report v2.2;
- the team contributions remain open;
- report v3, a reorganised draft, is in [`report/`](report/).

The review-copy folder itself is unchanged.
