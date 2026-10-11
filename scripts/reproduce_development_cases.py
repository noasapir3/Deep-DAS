#!/usr/bin/env python3
"""Reproduce the development result (nine development cases, one per anomaly type) with DeepDAS Ensemble v1.
Writes <out>/development_reproduction.csv (study identifiers; compared byte for byte by verify_release.py) and <out>/development_cases.csv.
Usage: python scripts/reproduce_development_cases.py [out_dir] [--phi-mode active|neutralized]
(default results/development; neutralized: results_phi_neutralized/development)"""
import sys
from pathlib import Path
import _setup  # noqa: F401
import _phi_io  # noqa: E402  (--phi-mode active|neutralized; default active = original hybrid)
from deepdas import paths
from deepdas.evaluation.development import reproduce

out = Path(sys.argv[1]) if len(sys.argv) > 1 else _phi_io.default_out(paths, "development"); out.mkdir(parents=True, exist_ok=True)
A, D = reproduce(); A.to_csv(out / "development_reproduction.csv", index=False); D.to_csv(out / "development_cases.csv", index=False)
_phi_io.stamp(out)
print(D.to_string(index=False)); print(f"\n[phi_mode={_phi_io.MODE}] DeepDAS Ensemble v1 detected {int(D.DeepDAS_Ensemble_v1_detected.sum())} / {len(D)} development cases")
