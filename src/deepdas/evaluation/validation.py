"""
Validation results of DeepDAS Ensemble v1 (development data: these sets were used to select the model).

From the per-seed validation scores (deepdas.train.supervised.evaluate_frozen_seed): per-seed z-scores with the val_B mean / std of that
seed, the ensemble score (mean of the five z), its P99 on val_B, and the selection metrics (per-type validation recall, stall validation,
nine development cases, false alarms on the val_A normals). Identifiers are those of the study.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .. import paths

TYPES = ["anomaly_collision", "anomaly_hard_braking", "anomaly_lane_straddling", "anomaly_lane_weaving", "anomaly_overweight_vehicle",
         "anomaly_reckless_driver", "anomaly_speeding", "anomaly_stalled_vehicle", "anomaly_tailgating"]
SHORT = {t: t.replace("anomaly_", "").replace("_vehicle", "").replace("lane_", "") for t in TYPES}
MODES = ["START", "STOP_END", "MSM"]
V9 = ["speeding", "C1 real braking", "stalled", "C4 collision striker", "reckless", "weaving", "overweight", "C2 straddling", "C3 tailgating"]


def metrics(S, col, t99):
    hit = lambda d: (d[col] > t99) & (d.nw >= 3)
    val = S[S.group == "validation"]; vp = val[val.y == 1]; sv = S[S.group == "stallval"]; sp = sv[(sv.y == 1) & (sv.nw >= 3)]
    m = {f"val_{SHORT[t]}": float(hit(vp[vp.anomaly_type == t]).mean()) for t in TYPES}
    m["val_macro_recall_orig"] = float(np.mean([m[f"val_{SHORT[t]}"] for t in TYPES]))
    m["val_tailgating_n"] = f"{int(hit(vp[vp.anomaly_type == 'anomaly_tailgating']).sum())}/{int((vp.anomaly_type == 'anomaly_tailgating').sum())}"
    m["val_AP"] = float(average_precision_score(val.y, val[col]))
    for md in MODES:
        d = sp[sp.stall_mode == md]; m[f"stallval_{md}"] = float(hit(d).mean()); m[f"stallval_{md}_n"] = f"{int(hit(d).sum())}/{len(d)}"
    for se in ("DAS_ONLY", "COMBINED"):
        d = sp[sp.sensing == se]; m[f"stallval_{se}_n"] = f"{int(hit(d).sum())}/{len(d)}"
    m["stallval_macro3"] = float(np.mean([m[f"stallval_{md}"] for md in MODES]))
    os_ = vp[vp.anomaly_type == "anomaly_stalled_vehicle"]; pooled = pd.concat([hit(os_), hit(sp)])
    m["stall_pooled"] = float(pooled.mean()); m["stall_pooled_n"] = f"{int(pooled.sum())}/{len(pooled)}"
    m["macro9"] = float(np.mean([m[f"val_{SHORT[t]}"] if t != "anomaly_stalled_vehicle" else m["stall_pooled"] for t in TYPES]))
    sn = sv[(sv.y == 0) & (sv.nw >= 3)]
    for lab, nm in (("normal_matched_control", "control"), ("normal_legit_stop", "legit_stop"), ("normal_unaffected_background", "background")):
        d = sn[sn.anomaly_type == lab]; m[f"stallval_alarm_{nm}_n"] = f"{int(hit(d).sum())}/{len(d)}"
    va = S[S.group == "normal_val"]; te = S[S.group == "normal_test"]
    m["FA_valA"] = int(hit(va).sum()); m["FA_valA_rate"] = float(hit(va).mean()); m["FA_test"] = int(hit(te).sum())
    v9 = S[S.group == "valid9"].set_index("case")
    for c in V9:
        m[f"diag9_{c}"] = int(hit(v9.loc[[c]]).iloc[0])
    m["diag9_detected"] = int(sum(m[f"diag9_{c}"] for c in V9))
    ns = S[S.group == "newscene_anomaly"]
    for t in TYPES:
        d = ns[ns.anomaly_type == t]; m[f"diagNS_{SHORT[t]}"] = f"{int(hit(d).sum())}/{int((d.nw >= 3).sum())}"
    m["diagNS_total"] = f"{int(hit(ns).sum())}/{int((ns.nw >= 3).sum())}"
    return m


def ensemble_tables(seed_dirs, model_dir: Path = paths.ENSEMBLE, frozen: bool = False):
    """-> (per-seed + ensemble metrics, ensemble score table, legitimate-stop ensemble table).
    frozen=False (original release behaviour, used for the published active-mode tables): each seed is standardised with the val_B
    statistics of the logits being evaluated and the ensemble threshold is the P99 of their mean (identical to the frozen values when the
    frozen model is evaluated unchanged). frozen=True (always used for phi_mode='neutralized'): the frozen mu_k, sigma_k and tau of
    ensemble.json are used; nothing is re-estimated."""
    spec = json.loads((Path(model_dir) / "ensemble.json").read_text())
    rows, Z, ZL = [], [], []
    for s, d in enumerate(seed_dirs):
        d = Path(d); d = d / "evaluation" if (d / "evaluation/vehicle_scores.csv").exists() else d
        S = pd.read_csv(d / "vehicle_scores.csv"); S["nw"] = S.n_windows.fillna(3); S["key"] = S.group + "::" + S.vkey.astype(str)
        t99 = json.loads((model_dir / f"seed_{s}/thresholds.json").read_text())["thresholds_logit"]["P99"]
        m = metrics(S, "logit", t99); m.update({"seed": s, "thr_P99": t99}); rows.append(m)
        vb = S[S.group == "normal_val_B"].logit
        mu, sd = (spec["seeds"][s]["valB_mean"], spec["seeds"][s]["valB_std_ddof0"]) if frozen else (vb.mean(), vb.std(ddof=0))
        S["z"] = (S.logit - mu) / sd; Z.append(S.set_index("key"))
        L = d / "vehicle_scores_legitimate_stop_set.csv"
        if L.exists():
            Ls = pd.read_csv(L); Ls["nw"] = Ls.n_windows.fillna(3); Ls["z"] = (Ls.logit - mu) / sd; ZL.append(Ls)
    E = Z[0][["group", "vkey", "case", "y", "anomaly_type", "nw", "stall_mode", "sensing"]].copy()
    for s, z in enumerate(Z):
        assert (z.index == E.index).all() and (z.nw == E.nw).all(); E[f"z{s}"] = z.z
    E["ens"] = E[[f"z{s}" for s in range(len(Z))]].mean(axis=1); E = E.reset_index()
    t99 = float(spec["decision"]["threshold"]) if frozen else float(np.percentile(E[E.group == "normal_val_B"].ens, 99))
    m = metrics(E, "ens", t99); m.update({"seed": "ENSEMBLE", "thr_P99": t99}); rows.append(m)
    LE = None
    if len(ZL) == len(Z):
        LE = ZL[0][["vkey", "y", "anomaly_type", "nw", "stall_mode", "sensing", "obs", "hold_s"]].copy()
        for s, z in enumerate(ZL):
            LE[f"z{s}"] = z.z.values
        LE["ens"] = LE[[f"z{s}" for s in range(len(ZL))]].mean(axis=1)
    return pd.DataFrame(rows), E, LE
