#!/usr/bin/env python3
"""Score imported runs with DeepDAS Ensemble v1 (deployment use: one decision per usable track, no ground truth).
Usage: python scripts/score_tracks.py <processed_runs_dir> <out_csv>
Output columns: unit (track key), n_windows, the five seed logits, ensemble score E_DIV_ens, decision E_DIV_decision (ALARM / no alarm /
no decision), and the single supervised model for comparison (CR_P99_*)."""
import sys
from pathlib import Path
import pandas as pd
import _setup  # noqa: F401
from deepdas.data_sources import read_runs
from deepdas.ensemble.scoring import score_units

if len(sys.argv) < 3:
    sys.exit(__doc__)
d = Path(sys.argv[1]); ch, amp = read_runs(d)
meta = pd.concat([pd.read_parquet(f) for f in sorted(d.glob("*_track_meta.parquet"))]); keys = meta[meta.usable].track_key.astype(str).tolist()
R = score_units(ch, amp, pd.DataFrame({"unit": keys, "tracks": [[k] for k in keys]})); R.to_csv(sys.argv[2], index=False)
print(R.E_DIV_decision.value_counts().to_string())
