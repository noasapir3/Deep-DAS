"""
Label-free track-quality gate.

The Final-Project exports contain tracks whose fused trajectory is physically
impossible for a sustained period — the filter coasting or switching vehicles
under sparse sensor coverage: reversing at several m/s, speeds far above the
simulator's cap (42 m/s), or the estimate leaving the road.  Such tracks are
tracker failures, not driver behaviour.  They are flagged here from the model
channels only (no truth, no label), excluded from training, and reported
separately in the evaluation.  The same rule is applied to every track,
normal or anomalous.

A track fails the gate when any of the following holds for at least MIN_RUN
consecutive valid 10 Hz bins (1 s):
    v_lon < V_REVERSE_MPS      (reversing; the simulator never reverses)
    v_lon > V_MAX_MPS          (above the simulator's speed cap)
    |d_lat| > D_LAT_MAX_M      (more than 5 m from the matched lane centre-line)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

V_REVERSE_MPS = -3.0
V_MAX_MPS = 45.0
D_LAT_MAX_M = 5.0
MIN_RUN = 10


def _longest_run(flag: np.ndarray) -> int:
    if not flag.any():
        return 0
    m = np.r_[0, flag.astype(np.int8), 0]
    d = np.diff(m)
    return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())


def track_quality(ch: pd.DataFrame) -> pd.DataFrame:
    """Per-track longest implausible runs and the resulting ``quality_ok`` flag.

    ``ch`` needs track_key, bin, v_lon, d_lat, m_valid, burn_in (the consolidated channel table).
    """
    df = ch[["track_key", "bin", "v_lon", "d_lat", "m_valid", "burn_in"]].copy()
    df["track_key"] = df["track_key"].astype(str)
    df = df[(~df["burn_in"]) & (df["m_valid"] > 0)].sort_values(["track_key", "bin"])
    keys = df["track_key"].to_numpy()
    v = df["v_lon"].to_numpy(dtype=np.float32)
    d = np.abs(df["d_lat"].to_numpy(dtype=np.float32))
    rev, fast, lat = v < V_REVERSE_MPS, v > V_MAX_MPS, d > D_LAT_MAX_M
    rows = []
    if len(keys):
        bounds = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])
        for a, b in zip(bounds[:-1], bounds[1:]):
            rows.append({"track_key": keys[a], "run_reverse": _longest_run(rev[a:b]), "run_fast": _longest_run(fast[a:b]),
                         "run_offlane": _longest_run(lat[a:b])})
    q = pd.DataFrame(rows, columns=["track_key", "run_reverse", "run_fast", "run_offlane"])
    q["quality_ok"] = (q["run_reverse"] < MIN_RUN) & (q["run_fast"] < MIN_RUN) & (q["run_offlane"] < MIN_RUN)
    return q
