"""The cross-platform comparison of scripts/verify_release.py (E3) accepts tiny score differences and rejects every change of a decision,
a window count, a key or a table (docs/EVALUATION_PROTOCOL.md §5–6)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def vr():
    argv = sys.argv; sys.argv = ["verify_release.py", "--steps", "none"]; sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("verify_release", ROOT / "scripts/verify_release.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); sys.argv = argv
    return m


def _cmp(vr, tmp_path, rel, frame_or_text):
    ref = ROOT / "results" / rel; new = tmp_path / rel; new.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(frame_or_text, str):
        new.write_text(frame_or_text)
    else:
        frame_or_text.to_csv(new, index=False)
    return vr.compare_dirs(tmp_path, ROOT / "results", [rel])[0][rel]


def test_tiny_score_difference_is_equivalent(vr, tmp_path):
    x = pd.read_csv(ROOT / "results/validation/ensemble_scores.csv"); x["ens"] = x["ens"] + 5e-7
    assert _cmp(vr, tmp_path, "validation/ensemble_scores.csv", x).startswith("equivalent")


def test_score_beyond_tolerance_fails(vr, tmp_path):
    x = pd.read_csv(ROOT / "results/validation/ensemble_scores.csv"); x.loc[0, "ens"] += 2e-5
    assert _cmp(vr, tmp_path, "validation/ensemble_scores.csv", x) == "FAIL"


def test_decision_flip_fails(vr, tmp_path):
    x = pd.read_csv(ROOT / "results/final_test/scores_dl_tracks.csv"); i = x.index[x.E_DIV_decision == "ALARM"][0]
    x.loc[i, "E_DIV_decision"] = "no alarm"
    assert _cmp(vr, tmp_path, "final_test/scores_dl_tracks.csv", x) == "FAIL"


def test_vehicle_outcome_flip_fails(vr, tmp_path):
    x = pd.read_csv(ROOT / "results/final_test/vehicle_outcomes.csv"); x.loc[0, "E_DIV_alarm"] = not x.loc[0, "E_DIV_alarm"]
    assert _cmp(vr, tmp_path, "final_test/vehicle_outcomes.csv", x) == "FAIL"


def test_window_count_change_fails(vr, tmp_path):
    x = pd.read_csv(ROOT / "results/final_test/scores_dl_tracks.csv"); x.loc[0, "n_windows"] += 1
    assert _cmp(vr, tmp_path, "final_test/scores_dl_tracks.csv", x) == "FAIL"


def test_table_must_be_byte_identical(vr, tmp_path):
    t = (ROOT / "results/final_test/T2_overall.csv").read_text().replace("0.568", "0.569")
    assert _cmp(vr, tmp_path, "final_test/T2_overall.csv", t).startswith("FAIL")
