"""
Loading of the processed data shipped with the release.

Every simulated set is stored as per-run parquet files (``runs/<run_id>_{channels,amp,track_meta}.parquet``) plus a vehicle table with
the truth label of every vehicle (labels are used only to define training targets and to evaluate, never as model input).
Track keys and run identifiers are the original ones, so that every result can be reproduced exactly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from . import paths
from .data.channels import CHANNELS

CH_COLS = ["track_key", "bin"] + list(CHANNELS) + ["burn_in"]

# display names of the nine anomaly types (identifiers are the ones stored in the data)
ANOMALY_TYPES = {"speeding": "Speeding", "reckless_driver": "Reckless driving", "hard_braking": "Hard braking", "stalled_vehicle": "Stalled vehicle",
                 "lane_weaving": "Lane weaving", "lane_straddling": "Lane straddling", "tailgating": "Tailgating", "collision": "Collision",
                 "overweight_vehicle": "Overweight vehicle"}
# simulated sets (directory under data/processed/anomaly_sets) and their display names
SETS = {"cross_region": "Cross-region anomaly set (9 types)", "tailgating_routes": "Tailgating set on planned routes",
        "tailgating_matched_controls": "Tailgating set with matched controls", "stalls_and_stops": "Stall and stop set",
        "legitimate_stop_validation": "Legitimate-stop validation set"}


def _cat(files) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def read_runs(run_dir: Path, prefix: str = "") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Channels and amplitude of all runs in a directory, concatenated in file-name order (as in every training and evaluation run)."""
    ch = _cat(sorted(Path(run_dir).glob(f"{prefix}*_channels.parquet"))); amp = _cat(sorted(Path(run_dir).glob(f"{prefix}*_amp.parquet")))
    if len(ch):
        ch = ch[CH_COLS]; ch["track_key"] = ch.track_key.astype(str)
    if len(amp):
        amp["track_key"] = amp.track_key.astype(str)
    return ch, amp


def normal_corpus(columns: List[str] | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(labels, channels, amplitude) of the normal driving corpus."""
    lab = pd.read_parquet(paths.PROCESSED / "labels.parquet"); lab["track_key"] = lab.track_key.astype(str)
    ch = pd.read_parquet(paths.PROCESSED / "channels.parquet", columns=columns or CH_COLS); ch["track_key"] = ch.track_key.astype(str)
    amp = pd.read_parquet(paths.PROCESSED / "amplitude.parquet"); amp["track_key"] = amp.track_key.astype(str)
    return lab, ch, amp


def usable_tracks(lab: pd.DataFrame) -> pd.DataFrame:
    """Usable = at least 32 valid bins and passing the label-free quality gate."""
    return lab[(lab.n_valid_bins >= 32) & lab.quality_ok.astype(bool)]


def anomaly_set(name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(vehicles, channels, amplitude) of one simulated set."""
    d = paths.ANOMALY_SETS / name
    v = pd.read_csv(d / "vehicles.csv", dtype={"region": str}); ch, amp = read_runs(d / "runs")
    return v, ch, amp


def final_test() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(run plan, vehicles, channels, amplitude) of the independent final test."""
    d = paths.FINAL_TEST
    plan = pd.read_csv(d / "run_plan.csv", dtype={"region": str}); v = pd.read_csv(d / "vehicles.csv", dtype={"region": str})
    ch, amp = read_runs(d / "runs")
    return plan, v, ch, amp


def load_json(p: Path) -> dict:
    return json.loads(Path(p).read_text())
