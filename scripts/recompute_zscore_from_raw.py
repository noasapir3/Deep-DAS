#!/usr/bin/env python3
"""Recompute the SimStudio Z-score track scores of the final test from the raw simulation exports (requires the separate raw-data package,
see docs/RAW_DATA_PACKAGE.md). Usage: python scripts/recompute_zscore_from_raw.py <raw_final_test_simulations_dir> [out_csv]"""
import sys
from pathlib import Path
import _setup  # noqa: F401
from deepdas.evaluation.final_test import zscore_from_raw

if len(sys.argv) < 2:
    sys.exit(__doc__)
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/zscore_scores_recomputed.csv"); out.parent.mkdir(parents=True, exist_ok=True)
zscore_from_raw(Path(sys.argv[1]), out); print("written", out)
