#!/usr/bin/env python3
"""Reproduce the independent final test: DL scores on every usable track and simulated vehicle (frozen models), the pre-registered tables
T1–T7 and vehicle_outcomes.csv (study identifiers; compared byte for byte by verify_release.py) and FINAL_TEST_REPORT.md (display names).
The Z-score track scores are read from data/processed/final_test/zscore_scores_precomputed.csv; recomputing them needs the raw-data
package (scripts/recompute_zscore_from_raw.py).
Usage: python scripts/reproduce_final_test.py [out_dir] [--phi-mode active|neutralized]
(default results/final_test; neutralized: results_phi_neutralized/final_test, post-hoc analysis of the exposed test)"""
import sys
from pathlib import Path
import _setup  # noqa: F401
import _phi_io  # noqa: E402  (--phi-mode active|neutralized; default active = original hybrid)
from deepdas import paths
from deepdas.evaluation import final_test as F
from deepdas.evaluation.report import final_test_report

out = Path(sys.argv[1]) if len(sys.argv) > 1 else _phi_io.default_out(paths, "final_test"); out.mkdir(parents=True, exist_ok=True)
F.score(out); T = F.analyze(out, out); final_test_report(out, out / "FINAL_TEST_REPORT.md")
_phi_io.stamp(out)
print(f"[phi_mode={_phi_io.MODE}]"); print(T["T2"].to_string(index=False))
