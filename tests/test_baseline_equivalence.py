"""
The copied feature extractor and Z-score scorer reproduce the Final Project's
published numbers from the original export folders.

* ``001/normal_medium_traffic``: every one of the 166 feature values of its 11
  tracks equals the corresponding row of the published ``features_normal.csv``.
* ``region_003/anomaly_speeding``: features equal ``features_anomaly.csv`` and
  the shipped ``model_live.pkl`` gives the published ``z_score_max`` / flags.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deepdas import paths
from deepdas.fp_anomaly_model.feature_extractor import ScenarioLoader
from deepdas.fp_anomaly_model.scenario_scorer import load_model, score_dataframe

REF = paths.FP_DATA / "reference_exports"
ID_COLS = ["scenario_id", "global_track_id", "vehicle_id_oracle"]


def _export_dir(folder: Path) -> Path:
    return next(p for p in folder.iterdir() if p.is_dir() and p.name.endswith("_tracking_audit"))


def _extract(folder: Path, scenario_id: str) -> pd.DataFrame:
    export_dir = _export_dir(folder)
    scenario_json = next(folder.glob("*.sim.json"))
    return ScenarioLoader(export_dir, scenario_id=scenario_id, scenario_json=scenario_json).extract()


def _assert_frames_equal(got: pd.DataFrame, ref: pd.DataFrame) -> int:
    common = [c for c in got.columns if c in ref.columns]
    assert len(common) >= 166 + 3, f"only {len(common)} common columns"
    got = got.sort_values("global_track_id").reset_index(drop=True)
    ref = ref.sort_values("global_track_id").reset_index(drop=True)
    assert list(got["global_track_id"]) == list(ref["global_track_id"])
    n_checked = 0
    for c in common:
        g, r = got[c], ref[c]
        if pd.api.types.is_numeric_dtype(r) or pd.api.types.is_numeric_dtype(g):
            g = pd.to_numeric(g, errors="coerce").to_numpy(dtype=float)
            r = pd.to_numeric(r, errors="coerce").to_numpy(dtype=float)
            both_nan = np.isnan(g) & np.isnan(r)
            ok = both_nan | np.isclose(g, r, rtol=1e-6, atol=1e-6)
            assert ok.all(), f"column {c}: max abs diff {np.nanmax(np.abs(g - r))}"
        else:
            assert (g.astype(str).values == r.astype(str).values).all(), f"column {c} differs"
        n_checked += 1
    return n_checked


def test_normal_export_reproduces_features_normal_csv():
    scenario_id = "001/normal_medium_traffic"
    got = _extract(REF / "001_normal_medium_traffic", scenario_id)
    ref = pd.read_csv(paths.FP_DATA / "zscore" / "features_normal.csv.gz", low_memory=False)
    ref = ref[ref["scenario_id"] == scenario_id]
    assert len(ref) == len(got) == 11
    n = _assert_frames_equal(got, ref)
    assert n >= 169


def test_anomaly_export_reproduces_features_and_published_scores():
    scenario_id = "region_003/anomaly_speeding"
    got = _extract(REF / "region_003_anomaly_speeding", scenario_id)
    ref = pd.read_csv(paths.FP_DATA / "zscore" / "features_anomaly.csv", low_memory=False)
    ref = ref[ref["scenario_id"] == scenario_id]
    assert len(ref) == len(got) == 4
    _assert_frames_equal(got, ref)

    model = load_model(paths.ZSCORE_MODEL)
    assert len(model["feature_cols"]) == 156 and abs(model["threshold"] - 15.392482911229285) < 1e-9
    scores = score_dataframe(got, model).set_index("track_id")
    pub = pd.read_csv(paths.FP_DATA / "zscore" / "scores_features_anomaly.csv", low_memory=False)
    pub = pub[pub["scenario_id"] == scenario_id].set_index("global_track_id")
    for tid, row in pub.iterrows():
        assert abs(scores.loc[tid, "z_score_max"] - row["z_score_max"]) < 2e-3, tid
        assert bool(scores.loc[tid, "flagged"]) == bool(row["anomaly_flag"]), tid
        assert scores.loc[tid, "top_feature"] == row["top_feature"], tid
    # the injected vehicle (veh001) is the only flagged track
    assert list(scores.index[scores["flagged"]]) == ["T000001"]
