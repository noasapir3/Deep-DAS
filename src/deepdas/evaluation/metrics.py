"""
Vehicle-level evaluation shared by the autoencoder and the Z-score baseline.

Evaluation unit  = (scenario, oracle vehicle); score = max over the vehicle's tracks.
Positives        = injected vehicles (Final-Project anomaly scenarios + severity runs).
Negatives        = vehicles of the held-out test regions' normal scenarios.
Background       = the other vehicles of anomaly scenarios (reported, never counted).
Both detectors are evaluated on exactly the same vehicles (those with at least one
track eligible for the autoencoder), so the comparison is like for like.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve

from .. import paths

DAS_SIGMA_BINS = [0.0, 0.32, 0.40, 0.50, 0.65, 10.0]
DAS_SIGMA_LABELS = ["≈0.30 (floor)", "0.32–0.40", "0.40–0.50", "0.50–0.65", ">0.65"]


def vehicle_table(dl_tracks: pd.DataFrame, zs_tracks: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Merge autoencoder and Z-score track scores into one vehicle-level table."""
    dl = dl_tracks[["track_key", "score", "scenario_id", "vehicle_id_oracle", "role", "anomaly_type", "level", "injection",
                    "group", "region", "scenario_type", "split", "n_valid_bins"]].copy()
    attr_cols = [c for c in dl_tracks.columns if c.startswith("attr_")]
    for c in attr_cols:
        dl[c] = dl_tracks[c]
    zs = zs_tracks[["track_key", "z_score_max", "top_feature"]]
    t = dl.merge(zs, on="track_key", how="left")
    m = meta[["track_key", "das_sigma_mean", "das_snr_mean", "duration_s", "n_das_meas", "n_cam_meas", "n_gps_meas"]]
    t = t.merge(m, on="track_key", how="left")
    g = t.groupby(["scenario_id", "vehicle_id_oracle"], sort=True)
    veh = pd.DataFrame({
        "dl_score": g["score"].max(),
        "z_score_max": g["z_score_max"].max(),
        "n_tracks": g.size(),
        "n_valid_bins": g["n_valid_bins"].sum(),
        "das_sigma_mean": g["das_sigma_mean"].mean(),
        "das_snr_mean": g["das_snr_mean"].mean(),
        "duration_s": g["duration_s"].sum(),
        "n_das_meas": g["n_das_meas"].sum(),
    })
    first = g.first()
    for c in ["role", "anomaly_type", "level", "injection", "group", "region", "scenario_type", "split"] + attr_cols:
        veh[c] = first[c]
    # top feature of the track that carries the vehicle's max z
    t["_z"] = t["z_score_max"].fillna(-1.0)
    idx = t.loc[t.groupby(["scenario_id", "vehicle_id_oracle"], sort=True)["_z"].idxmax()]
    veh["top_feature"] = pd.Series(idx["top_feature"].to_numpy(), index=pd.MultiIndex.from_frame(idx[["scenario_id", "vehicle_id_oracle"]]))
    veh = veh.reset_index()
    veh["das_sigma_bin"] = pd.cut(veh["das_sigma_mean"], DAS_SIGMA_BINS, labels=DAS_SIGMA_LABELS, right=False)
    return veh


def threshold_at(scores: np.ndarray, q: float) -> float:
    return float(np.percentile(scores, 100.0 * q))


def bootstrap_metric(pos: np.ndarray, neg: np.ndarray, fn, n_boot: int = 1000, seed: int = 0) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    point = fn(pos, neg)
    vals = []
    for _ in range(n_boot):
        p = rng.choice(pos, len(pos), replace=True)
        n = rng.choice(neg, len(neg), replace=True)
        vals.append(fn(p, n))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def _auroc(pos, neg):
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return roc_auc_score(y, np.r_[pos, neg])


def _auprc(pos, neg):
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return average_precision_score(y, np.r_[pos, neg])


def recall_at_far(pos: np.ndarray, neg: np.ndarray, far: float) -> float:
    """Recall when the threshold is set on the negatives themselves at the given false-alarm rate."""
    thr = threshold_at(neg, 1.0 - far)
    return float((pos > thr).mean())


def curves(pos: np.ndarray, neg: np.ndarray) -> Dict[str, np.ndarray]:
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    return {"fpr": fpr, "tpr": tpr, "precision": prec, "recall": rec}


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def summarise_detector(veh: pd.DataFrame, score_col: str, thresholds: Dict[str, float], name: str) -> dict:
    """Global metrics of one detector on the vehicle table."""
    neg = veh[(veh["role"] == "normal") & (veh["split"] == "test")][score_col].to_numpy(dtype=float)
    pos = veh[veh["role"] == "injected"][score_col].to_numpy(dtype=float)
    pos_fp = veh[(veh["role"] == "injected") & (veh["group"] == "fp_anomaly")][score_col].to_numpy(dtype=float)
    pos_sev = veh[(veh["role"] == "injected") & (veh["group"] == "severity")][score_col].to_numpy(dtype=float)
    bg = veh[veh["role"] == "background"][score_col].to_numpy(dtype=float)
    out = {"detector": name, "n_pos": int(len(pos)), "n_neg": int(len(neg)), "n_background": int(len(bg))}
    for label, p in (("all", pos), ("fp", pos_fp), ("severity", pos_sev)):
        if len(p) >= 2 and len(neg) >= 2:
            a, lo, hi = bootstrap_metric(p, neg, _auroc)
            out[f"auroc_{label}"] = a; out[f"auroc_{label}_ci"] = [lo, hi]
            a, lo, hi = bootstrap_metric(p, neg, _auprc)
            out[f"auprc_{label}"] = a; out[f"auprc_{label}_ci"] = [lo, hi]
            out[f"recall_at_1pct_far_{label}"] = recall_at_far(p, neg, 0.01)
            out[f"recall_at_5pct_far_{label}"] = recall_at_far(p, neg, 0.05)
    for tname, thr in thresholds.items():
        out[f"threshold_{tname}"] = thr
        out[f"far_test_{tname}"] = float((neg > thr).mean())
        out[f"recall_{tname}"] = float((pos > thr).mean()) if len(pos) else float("nan")
        out[f"recall_fp_{tname}"] = float((pos_fp > thr).mean()) if len(pos_fp) else float("nan")
        out[f"recall_severity_{tname}"] = float((pos_sev > thr).mean()) if len(pos_sev) else float("nan")
        out[f"background_flag_rate_{tname}"] = float((bg > thr).mean()) if len(bg) else float("nan")
    return out


def recall_by_type_level(veh: pd.DataFrame, score_col: str, thr: float) -> pd.DataFrame:
    inj = veh[veh["role"] == "injected"].copy()
    inj["detected"] = inj[score_col] > thr
    rows = []
    for (atype, level), g in inj.groupby(["anomaly_type", "level"], sort=True):
        k, n = int(g["detected"].sum()), int(len(g))
        lo, hi = wilson(k, n)
        rows.append({"anomaly_type": atype, "level": level, "n": n, "detected": k, "recall": k / n if n else np.nan,
                     "ci_lo": lo, "ci_hi": hi, "score_median": float(g[score_col].median()), "score_min": float(g[score_col].min())})
    return pd.DataFrame(rows)


def far_by_scenario_type(veh: pd.DataFrame, score_col: str, thr: float, split: str = "test") -> pd.DataFrame:
    neg = veh[(veh["role"] == "normal") & (veh["split"] == split)].copy()
    neg["flag"] = neg[score_col] > thr
    rows = []
    for st, g in neg.groupby("scenario_type", sort=True):
        k, n = int(g["flag"].sum()), int(len(g))
        lo, hi = wilson(k, n)
        rows.append({"scenario_type": st, "n": n, "flagged": k, "far": k / n if n else np.nan, "ci_lo": lo, "ci_hi": hi})
    k, n = int(neg["flag"].sum()), int(len(neg))
    lo, hi = wilson(k, n)
    rows.append({"scenario_type": "ALL", "n": n, "flagged": k, "far": k / n if n else np.nan, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def das_sigma_strata(veh: pd.DataFrame, score_col: str, thr: float) -> pd.DataFrame:
    rows = []
    for b in DAS_SIGMA_LABELS:
        g = veh[veh["das_sigma_bin"] == b]
        neg = g[(g["role"] == "normal") & (g["split"] == "test")]
        neg_all = g[g["role"] == "normal"]
        pos = g[g["role"] == "injected"]
        rows.append({
            "das_sigma_bin": b, "n_normal_test": int(len(neg)), "n_normal_all": int(len(neg_all)), "n_injected": int(len(pos)),
            "far_test": float((neg[score_col] > thr).mean()) if len(neg) else np.nan,
            "normal_score_median": float(neg_all[score_col].median()) if len(neg_all) else np.nan,
            "normal_score_p95": float(neg_all[score_col].quantile(0.95)) if len(neg_all) else np.nan,
            "recall": float((pos[score_col] > thr).mean()) if len(pos) else np.nan,
            "injected_score_median": float(pos[score_col].median()) if len(pos) else np.nan,
        })
    return pd.DataFrame(rows)
