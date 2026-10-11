"""
The independent final test (North-TLV test regions; 228 runs; run once on 2026-10-09).

score(): frozen DL detectors on every usable track (deployment unit, no ground truth) and, secondarily, on every simulated vehicle.
analyze(): the pre-registered analysis (tables T1–T7). Vehicle outcome = ALARM if any decided track of the vehicle alarms; the truth
vehicle id is used only for this evaluation mapping. Column and row identifiers are those of the study (``E_DIV`` = DeepDAS Ensemble v1,
``CR_P99`` = single supervised model, ``Z`` = SimStudio Z-score) so that the published tables are reproduced exactly.
The Z-score track scores need the raw simulation exports (separate raw-data package); the release ships them precomputed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score

from .. import paths
from ..data_sources import final_test
from ..ensemble.scoring import score_units


def score(out_dir: Path):
    """-> writes scores_dl_tracks.csv and scores_dl_oracle_vehicles.csv"""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    _, V, ch, amp = final_test()
    us = V[V.tracks_usable.fillna("") != ""].copy(); us["tracks"] = us.tracks_usable.str.split(";")
    T = us.explode("tracks")[["run_id", "vehicle_id_oracle", "tracks"]].rename(columns={"tracks": "track_key"})
    RT = score_units(ch, amp, pd.DataFrame({"unit": T.track_key, "tracks": T.track_key.apply(lambda k: [k])}))
    RT = RT.merge(T, left_on="unit", right_on="track_key"); RT.to_csv(out_dir / "scores_dl_tracks.csv", index=False)
    us["unit"] = us.run_id + "|" + us.vehicle_id_oracle; RV = score_units(ch, amp, us[["unit", "tracks"]]); RV.to_csv(out_dir / "scores_dl_oracle_vehicles.csv", index=False)
    return RT, RV


def analyze(scores_dir: Path, out_dir: Path, zscore_csv: Path = paths.FINAL_TEST / "zscore_scores_precomputed.csv"):
    """Pre-registered analysis of the final test. Reads the DL scores (score()) and the Z-score track scores; writes T1–T7 and vehicle_outcomes.csv."""
    scores_dir, R = Path(scores_dir), Path(out_dir); R.mkdir(parents=True, exist_ok=True)
    V = pd.read_csv(paths.FINAL_TEST / "vehicles.csv", dtype={"region": str}); RR = pd.read_csv(paths.FINAL_TEST / "run_report.csv", dtype={"region": str})
    DT = pd.read_csv(scores_dir / "scores_dl_tracks.csv"); DV = pd.read_csv(scores_dir / "scores_dl_oracle_vehicles.csv"); ZT = pd.read_csv(zscore_csv)
    ZTHR = 15.392482911229285; ETHR = 1.9726181803233118; CTHR = -5.523461642265342
    TYPES = ["speeding", "reckless_driver", "hard_braking", "stalled_vehicle", "lane_weaving", "lane_straddling", "collision", "overweight_vehicle", "tailgating"]
    V["vk"] = V.run_id + "|" + V.vehicle_id_oracle
    # ---- vehicle outcomes (truth id used only to map track decisions to vehicles)
    DT["vk"] = DT.run_id + "|" + DT.vehicle_id_oracle; d3 = DT[DT.n_windows >= 3]
    out = pd.DataFrame(index=V.vk)
    for det, sc, thr in (("E_DIV", "E_DIV_ens", ETHR), ("CR_P99", "CR_P99_logit", CTHR)):
        g = d3.groupby("vk")[sc].max(); out[f"{det}_decided"] = out.index.isin(g.index); out[f"{det}_score"] = g.reindex(out.index); out[f"{det}_alarm"] = out[f"{det}_score"] > thr
    DV = DV.rename(columns={"unit": "vk"}).set_index("vk")
    out["E_DIV_orac_decided"] = (DV.n_windows.reindex(out.index) >= 3).fillna(False); out["E_DIV_orac_score"] = DV.E_DIV_ens.reindex(out.index); out["E_DIV_orac_alarm"] = out.E_DIV_orac_score > ETHR
    out["CR_orac_decided"] = out.E_DIV_orac_decided; out["CR_orac_score"] = DV.CR_P99_logit.reindex(out.index); out["CR_orac_alarm"] = out.CR_orac_score > CTHR
    ZT["vk"] = ZT.run_id + "|" + ZT.vehicle_id_oracle.astype(str)
    for det, sc, thr in (("Z", "z_max", ZTHR), ("Z_noGT", "z_max_noGT", ZTHR)):
        g = ZT.groupby("vk")[sc].max(); out[f"{det}_decided"] = out.index.isin(g.index); out[f"{det}_score"] = g.reindex(out.index); out[f"{det}_alarm"] = out[f"{det}_score"] > thr
    X = V.set_index("vk").join(out); X["tracked"] = X.tracks_all.fillna("") != ""; X["usable"] = X.tracks_usable.fillna("") != ""
    X["common"] = X.E_DIV_decided & X.CR_P99_decided & X.Z_decided; X.to_csv(R / "vehicle_outcomes.csv")
    CORE = X[(X.block == "core") & (X.label == "anomaly_verified")]; NORM = X[X.label.isin(["normal_scene", "normal_unaffected_background", "normal_matched_control"])]
    LEG = X[X.label == "normal_legit_stop"]; SSA = X[X.label == "stopstudy_abnormal_verified"]
    DETS = ["E_DIV", "CR_P99", "Z"]


    def rate(d, det, set_="own"):
        d = d[d[f"{det}_decided"]] if set_ == "own" else d[d.common]
        return (int(d[f"{det}_alarm"].sum()), len(d))


    # ---- T1 funnel per type
    rows = []
    for t in TYPES:
        runs = RR[(RR.block == "core") & (RR.type == t)]; inj = X[(X.block == "core") & (X.type == t) & (X.vehicle_id_oracle == "veh001")]; c = CORE[CORE.type == t]
        r = {"type": t, "injected": len(runs), "realized": int(runs.verified.astype(bool).sum()), "realized_tracked": int(c.tracked.sum()), "realized_usable_track": int(c.usable.sum())}
        for det in DETS:
            h, n = rate(c, det); r[f"{det}_decided"] = n; r[f"{det}_detected"] = h; r[f"{det}_recall"] = round(h / n, 3) if n else np.nan
        hc = {det: rate(c, det, "common") for det in DETS}; r["common_n"] = hc["E_DIV"][1]
        for det in DETS:
            r[f"{det}_recall_common"] = round(hc[det][0] / hc[det][1], 3) if hc[det][1] else np.nan
        rows.append(r)
    T1 = pd.DataFrame(rows); T1.to_csv(R / "T1_per_type.csv", index=False)
    macro = {det: float(T1[f"{det}_recall"].mean()) for det in DETS}; macro_c = {det: float(T1[f"{det}_recall_common"].mean()) for det in DETS}

    # ---- T2 false alarms, coverage, overall metrics
    rows = []
    for det in DETS:
        fa, nf = rate(NORM, det); fl, nl = rate(LEG, det); fac, nfc = rate(NORM, det, "common"); flc, nlc = rate(LEG, det, "common")
        tp, npos = rate(CORE, det); prec = tp / (tp + fa) if (tp + fa) else np.nan; rec = tp / npos if npos else np.nan
        dd = pd.concat([CORE[CORE[f"{det}_decided"]].assign(y=1), NORM[NORM[f"{det}_decided"]].assign(y=0)])
        rows.append({"detector": det, "macro_recall_own": round(macro[det], 3), "macro_recall_common": round(macro_c[det], 3), "pooled_recall": round(rec, 3),
                     "FA_normal_driving": f"{fa}/{nf} = {fa / nf:.4f}", "FA_normal_common": f"{fac}/{nfc} = {fac / nfc:.4f}" if nfc else "",
                     "FA_legit_stops": f"{fl}/{nl}", "FA_legit_common": f"{flc}/{nlc}", "precision": round(prec, 3), "F1": round(2 * prec * rec / (prec + rec), 3) if prec == prec and rec else np.nan,
                     "ROC_AUC": round(roc_auc_score(dd.y, dd[f"{det}_score"]), 3), "AUPRC": round(average_precision_score(dd.y, dd[f"{det}_score"]), 3),
                     "coverage_anomalies": f"{int(CORE[f'{det}_decided'].sum())}/{len(CORE)}", "coverage_normals": f"{int(NORM[f'{det}_decided'].sum())}/{len(NORM)}",
                     "coverage_legit": f"{int(LEG[f'{det}_decided'].sum())}/{len(LEG)}"})
    T2 = pd.DataFrame(rows); T2.to_csv(R / "T2_overall.csv", index=False)

    # ---- T3 stop study, stalled by mode, tailgating detail
    rows = []
    for nm, d in [("LEGIT 3 s (false alarm)", LEG[LEG.legit_hold_s == 3]), ("LEGIT 5 s (false alarm)", LEG[LEG.legit_hold_s == 5]), ("LEGIT 8 s (false alarm)", LEG[LEG.legit_hold_s == 8]),
                  ("GENTLE 15 s stop study (detection)", SSA), ("stop-study CONTROL (false alarm)", X[(X.block == "stop_study") & (X["mode"] == "CONTROL") & (X.label == "normal_matched_control")])] + \
                 [(f"stalled {m} (detection)", CORE[(CORE.type == "stalled_vehicle") & (CORE["mode"] == m)]) for m in ("START", "STOP_END", "MSM", "GENTLE_STALL")] + \
                 [(f"tailgating v{v:g} D{dd:g} (detection)", CORE[(CORE.type == "tailgating") & (CORE.v_lead == v) & (CORE.dur_target_s == dd)]) for v, dd in ((6, 8), (6, 15), (9, 8))] + \
                 [("tailgating controls (false alarm)", X[(X.block == "tg_control") & (X.label == "normal_matched_control")])]:
        r = {"group": nm, "n": len(d)}
        for det in DETS:
            h, n = rate(d, det); r[det] = f"{h}/{n}"
        rows.append(r)
    T3 = pd.DataFrame(rows); T3.to_csv(R / "T3_stops_tailgating.csv", index=False)

    # ---- secondary: oracle-grouped DL units and truth-free Z-score
    rows = []
    for det, base in (("E_DIV (oracle-grouped units)", "E_DIV_orac"), ("CR_P99 (oracle-grouped units)", "CR_orac"), ("Z-score truth-free sensitivity", "Z_noGT")):
        rec = [(lambda c: c[f"{base}_alarm"][c[f"{base}_decided"]].mean())(CORE[CORE.type == t]) for t in TYPES]
        fa = NORM[NORM[f"{base}_decided"]][f"{base}_alarm"]; fl = LEG[LEG[f"{base}_decided"]][f"{base}_alarm"]
        rows.append({"variant": det, **{t: round(x, 3) for t, x in zip(TYPES, rec)}, "macro": round(float(np.nanmean(rec)), 3), "FA_normal": f"{int(fa.sum())}/{len(fa)}", "FA_legit": f"{int(fl.sum())}/{len(fl)}"})
    T4 = pd.DataFrame(rows); T4.to_csv(R / "T4_secondary.csv", index=False)

    # ---- bootstrap over runs (common decided set) + McNemar
    rng = np.random.default_rng(20261012); runs = X.run_id.unique(); BS = []
    CC = CORE[CORE.common]; NC = NORM[NORM.common]
    by_run_c = {r: g for r, g in CC.groupby("run_id")}; by_run_n = {r: g for r, g in NC.groupby("run_id")}
    for b in range(2000):
        smp = rng.choice(runs, len(runs), replace=True)
        c = pd.concat([by_run_c[r] for r in smp if r in by_run_c]); n = pd.concat([by_run_n[r] for r in smp if r in by_run_n])
        row = {}
        for det in DETS:
            rec = c.groupby("type")[f"{det}_alarm"].mean().reindex(TYPES); row[f"macro_{det}"] = float(rec.mean()); row[f"fa_{det}"] = float(n[f"{det}_alarm"].mean())
        BS.append(row)
    BS = pd.DataFrame(BS); ci = lambda s: f"[{s.quantile(.025):.3f}, {s.quantile(.975):.3f}]"
    boot = {f"macro recall {d} (common)": ci(BS[f"macro_{d}"]) for d in DETS} | {f"FA normal {d} (common)": ci(BS[f"fa_{d}"]) for d in DETS} | \
           {"macro diff E_DIV - Z": ci(BS.macro_E_DIV - BS.macro_Z), "macro diff E_DIV - CR_P99": ci(BS.macro_E_DIV - BS.macro_CR_P99),
            "P(E_DIV macro > Z macro)": round(float((BS.macro_E_DIV > BS.macro_Z).mean()), 3), "P(E_DIV macro > CR macro)": round(float((BS.macro_E_DIV > BS.macro_CR_P99).mean()), 3)}
    mc = {}
    for a, b in (("E_DIV", "Z"), ("E_DIV", "CR_P99")):
        x, y = CC[f"{a}_alarm"], CC[f"{b}_alarm"]; n01, n10 = int((~x & y).sum()), int((x & ~y).sum())
        mc[f"{a} vs {b} (pooled common anomalies)"] = {f"only {a}": n10, f"only {b}": n01, "exact McNemar p": round(binomtest(n10, n10 + n01, 0.5).pvalue, 4) if n10 + n01 else None}
        x, y = NC[f"{a}_alarm"], NC[f"{b}_alarm"]; mc[f"{a} vs {b} (common normals, false alarms)"] = {f"only {a}": int((x & ~y).sum()), f"only {b}": int((~x & y).sum())}
    json.dump({"bootstrap_95ci": boot, "mcnemar": mc}, open(R / "T5_uncertainty.json", "w"), indent=1)

    # ---- focus: tailgating, stalled, weaving, overweight (per run)
    F = X[(X.block == "core") & X.type.isin(["tailgating", "stalled_vehicle", "lane_weaving", "overweight_vehicle"]) & (X.vehicle_id_oracle == "veh001")]
    F = F[["run_id", "type", "mode", "region", "bg_type", "label", "usable", "v_lead", "dur_target_s", "E_DIV_score", "E_DIV_alarm", "CR_P99_score", "CR_P99_alarm", "Z_score", "Z_alarm"]].round(2)
    F.to_csv(R / "T6_focus_runs.csv", index=True)
    ZT.loc[ZT.vk.isin(CORE.index)].merge(CORE[["type"]], left_on="vk", right_index=True).groupby("type").top_feature.agg(lambda s: s.value_counts().head(2).to_dict()).to_csv(R / "T7_zscore_top_features_on_anomalies.csv")

    return {"T1": T1, "T2": T2, "T3": T3, "T4": T4, "T5": {"bootstrap_95ci": boot, "mcnemar": mc}, "T6": F}


def zscore_from_raw(raw_sims_dir: Path, out_csv: Path):
    """Recompute the Z-score track scores from the raw simulation exports (raw-data package): the shipped extractor and model, plus the
    truth-free sensitivity (the same constants, the 13 truth-position features left out of max|z|)."""
    from ..baseline.zscore import shipped_model
    from ..fp_anomaly_model.feature_extractor import ScenarioLoader
    from ..fp_anomaly_model.scenario_scorer import score_dataframe
    raw_sims_dir = Path(raw_sims_dir); V = pd.read_csv(paths.FINAL_TEST / "vehicles.csv", dtype={"region": str})
    runs = [r for r in sorted(V.run_id.unique()) if (raw_sims_dir / r).is_dir()]          # all 228 runs, or the subset present in the directory
    m = shipped_model(); F = m["feature_cols"]; GT = [c for c in F if "pos_err" in c or (c.startswith("dis_") and c.endswith("_err_mean_m")) or c.startswith("cov_err_")]
    mu = pd.Series(m["mean"], index=F) if not isinstance(m["mean"], dict) else pd.Series(m["mean"]); sd = pd.Series(m["std"], index=F) if not isinstance(m["std"], dict) else pd.Series(m["std"])
    NG = [c for c in F if c not in GT]; Z = []
    for r in runs:
        d = raw_sims_dir / r; info = json.loads((d / "run.json").read_text())
        f = ScenarioLoader(d / info["result"]["export_dir"], scenario_id=r, scenario_json=str(d / f"{r}.sim.json")).extract()
        if not len(f) or "global_track_id" not in f:
            continue
        s = score_dataframe(f, m); X = f.reindex(columns=NG).astype(float); zng = ((X - mu[NG]) / sd[NG].replace(0, np.nan)).abs().max(axis=1, skipna=True)
        Z.append(pd.DataFrame({"run_id": r, "track_id": f.global_track_id.astype(str).values, "vehicle_id_oracle": f.vehicle_id_oracle.astype(str).values,
                               "z_max": s.z_score_max.values, "z_flag": s.flagged.values.astype(bool), "top_feature": s.top_feature.values,
                               "z_max_noGT": zng.values, "z_flag_noGT": (zng.values > m["threshold"])}))
    ZT = pd.concat(Z, ignore_index=True); ZT.to_csv(out_csv, index=False); return ZT
