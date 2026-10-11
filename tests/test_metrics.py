"""Evaluation helpers: thresholds, recall at a false-alarm rate, Wilson intervals, vehicle aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from deepdas.evaluation.metrics import (_auroc, bootstrap_metric, far_by_scenario_type, recall_at_far, recall_by_type_level,
                                        summarise_detector, threshold_at, wilson)


def _toy_vehicles():
    rng = np.random.default_rng(0)
    normal = pd.DataFrame({"dl_score": rng.normal(1, 0.2, 400), "role": "normal", "split": "test", "group": "fp_normal",
                           "scenario_type": np.where(np.arange(400) % 2 == 0, "normal_a", "normal_b"), "anomaly_type": "", "level": ""})
    valb = pd.DataFrame({"dl_score": rng.normal(1, 0.2, 400), "role": "normal", "split": "val_B", "group": "fp_normal",
                         "scenario_type": "normal_a", "anomaly_type": "", "level": ""})
    inj = pd.DataFrame({"dl_score": np.r_[rng.normal(3, 0.2, 6), rng.normal(1, 0.2, 6)], "role": "injected", "split": "test",
                        "group": "severity", "scenario_type": "anomaly_x", "anomaly_type": "anomaly_x",
                        "level": ["severe"] * 6 + ["mild"] * 6})
    bg = pd.DataFrame({"dl_score": rng.normal(1, 0.2, 20), "role": "background", "split": "test", "group": "severity",
                       "scenario_type": "anomaly_x", "anomaly_type": "anomaly_x", "level": "mild"})
    return pd.concat([normal, valb, inj, bg], ignore_index=True)


def test_threshold_and_recall_at_far():
    neg = np.arange(1000, dtype=float)
    assert abs(threshold_at(neg, 0.99) - 989.01) < 0.1
    pos = np.array([995.0, 5.0, 999.0])
    assert abs(recall_at_far(pos, neg, 0.01) - 2 / 3) < 1e-9


def test_wilson_interval_bounds():
    lo, hi = wilson(3, 3)
    assert 0.4 < lo < 0.5 and hi == 1.0
    lo, hi = wilson(0, 3)
    assert lo == 0.0 and 0.5 < hi < 0.6
    assert np.isnan(wilson(0, 0)[0])


def test_bootstrap_auroc_contains_point_estimate():
    rng = np.random.default_rng(1)
    pos, neg = rng.normal(2, 1, 50), rng.normal(0, 1, 500)
    point, lo, hi = bootstrap_metric(pos, neg, _auroc, n_boot=200)
    assert lo <= point <= hi and 0.85 < point < 0.99


def test_summarise_detector_uses_injected_vs_test_normals_only():
    veh = _toy_vehicles()
    thr = threshold_at(veh.loc[(veh["role"] == "normal") & (veh["split"] == "val_B"), "dl_score"].to_numpy(), 0.99)
    s = summarise_detector(veh, "dl_score", {"p99": thr}, "toy")
    assert s["n_pos"] == 12 and s["n_neg"] == 400 and s["n_background"] == 20
    assert 0.7 < s["auroc_all"] < 0.8            # half of the positives are indistinguishable from normals
    assert abs(s["recall_p99"] - 0.5) < 1e-9     # the 6 severe ones
    assert s["far_test_p99"] < 0.05
    rt = recall_by_type_level(veh, "dl_score", thr).set_index("level")
    assert rt.loc["severe", "recall"] == 1.0 and rt.loc["mild", "recall"] == 0.0
    far = far_by_scenario_type(veh, "dl_score", thr).set_index("scenario_type")
    assert set(far.index) == {"normal_a", "normal_b", "ALL"} and far.loc["ALL", "n"] == 400
