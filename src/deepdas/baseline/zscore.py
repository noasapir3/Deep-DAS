"""
The shipped Z-score baseline, unchanged.

* model      : ``data/fp/zscore/model_live.pkl`` (156 features, threshold 15.39 σ) — never refitted;
* features   : for the existing corpus the Final Project's own ``features_normal.csv`` /
               ``features_anomaly.csv`` rows are scored as published (the equivalence test shows the
               copied extractor reproduces them exactly); for the severity-study runs the copied,
               unchanged extractor is run on the new export folders;
* scoring    : ``scenario_scorer.score_dataframe`` (max |z| over the 156 features).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .. import paths
from ..fp_anomaly_model.feature_extractor import ScenarioLoader
from ..fp_anomaly_model.scenario_scorer import load_model, score_dataframe


def shipped_model() -> dict:
    return load_model(paths.ZSCORE_MODEL)


def _tidy(scores: pd.DataFrame, feats: pd.DataFrame, model: dict) -> pd.DataFrame:
    out = pd.DataFrame({
        "scenario_id": feats["scenario_id"].astype(str).to_numpy(),
        "track_id": feats["global_track_id"].astype(str).to_numpy(),
        "z_score_max": scores["z_score_max"].to_numpy(dtype=float),
        "flagged_fp_threshold": scores["flagged"].to_numpy(dtype=bool),
        "top_feature": scores["top_feature"].to_numpy(),
    })
    out["track_key"] = out["scenario_id"] + "|" + out["track_id"]
    return out


def score_published_corpus(model: Optional[dict] = None) -> pd.DataFrame:
    """Scores of the published normal (14,424) and anomaly (36) feature rows with the shipped model."""
    model = model or shipped_model()
    normal = pd.read_csv(paths.FP_DATA / "zscore" / "features_normal.csv.gz", low_memory=False)
    anomaly = pd.read_csv(paths.FP_DATA / "zscore" / "features_anomaly.csv", low_memory=False)
    parts = [_tidy(score_dataframe(df, model), df, model) for df in (normal, anomaly)]
    return pd.concat(parts, ignore_index=True)


def score_export_folder(export_dir: Path, scenario_json: Path, scenario_id: str, model: Optional[dict] = None) -> pd.DataFrame:
    """Copied extractor + shipped model on one export folder (used for the severity runs)."""
    model = model or shipped_model()
    feats = ScenarioLoader(export_dir, scenario_id=scenario_id, scenario_json=str(scenario_json)).extract()
    if "global_track_id" not in feats.columns or feats["global_track_id"].isna().all():
        return pd.DataFrame(columns=["scenario_id", "track_id", "z_score_max", "flagged_fp_threshold", "top_feature", "track_key"])
    return _tidy(score_dataframe(feats, model), feats, model)


def score_severity_runs(generated_root: Path, model: Optional[dict] = None) -> pd.DataFrame:
    model = model or shipped_model()
    parts: List[pd.DataFrame] = []
    for run_dir in sorted(p for p in Path(generated_root).iterdir() if p.is_dir()):
        info_path = run_dir / "run.json"
        if not info_path.exists():
            continue
        info = json.loads(info_path.read_text())
        export_dir = run_dir / info.get("result", {}).get("export_dir", "")
        if not export_dir.is_dir():
            continue
        parts.append(score_export_folder(export_dir, run_dir / info["scenario_json"], f"severity/{run_dir.name}", model))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
