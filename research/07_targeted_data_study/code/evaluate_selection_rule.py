#!/usr/bin/env python3
"""
Targeted-data study (W1 / W2): the common evaluation of DeepDAS Ensemble v1 (frozen), W1 and W2 on the same validation data, and the
pre-registered selection rule C1–C9 (see ../README.md). Validation data only.
  1. Units = simulated validation vehicles (the unit of every validation result), decided = >= 3 windows; the same scoring path
     (artefact cleaning, tailgating feature, identical windows) for every model.
  2. Sanity: the ensemble logits on val_B / stall validation must equal the stored validation scores of DeepDAS Ensemble v1 (|d| <= 1e-4).
  3. Per model set: per-seed z with the val_B mean / std (DeepDAS Ensemble v1: frozen constants), ensemble = mean z, threshold = P99 on val_B
     (DeepDAS Ensemble v1: frozen 1.97262); single seed: its own val_B P99 (DeepDAS Ensemble v1: frozen per-seed P99).
  4. Metrics, C1–C9 and the choice rule; reported-only extras (partial vs whole-route events, bootstrap CIs).
Identifiers in the outputs are those of the study (E_DIV = DeepDAS Ensemble v1).  Usage: python evaluate_selection_rule.py [out_dir]
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = Path(__file__).resolve().parents[1]; ROOT = HERE.parents[1]; sys.path.insert(0, str(ROOT / "src"))
from deepdas import paths  # noqa: E402
from deepdas.data.dataset import window_starts  # noqa: E402
from deepdas.amplitude import attach_amplitude, to_arrays  # noqa: E402
from deepdas.data.channels import CHANNELS  # noqa: E402
from deepdas.ensemble.detector import T, STRIDE, clean_lead_artifacts, tailgating_fraction, load_detector  # noqa: E402
from deepdas.ensemble.scoring import Ensemble, load_norm  # noqa: E402
R = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"; R.mkdir(parents=True, exist_ok=True)
CH = list(CHANNELS); NORM = load_norm(); ENS = Ensemble()
WEAK = ["hard_braking", "speeding", "overweight_vehicle", "lane_straddling", "collision", "lane_weaving"]
TYPES9 = WEAK + ["reckless_driver", "tailgating", "stalled_vehicle"]
SETS = paths.ANOMALY_SETS

# ------------------------------------------------------------------ 1. units
U = []
lab = pd.read_parquet(paths.PROCESSED / "labels.parquet"); lab["track_key"] = lab.track_key.astype(str)
us = lab[(lab.n_valid_bins >= 32) & lab.quality_ok.astype(bool) & (lab.group == "fp_normal") & (lab.role == "normal")]
for (sid, vid), g in us.groupby(["scenario_id", "vehicle_id_oracle"]):
    sp = g.split.iloc[0]
    if sp in ("val_A", "val_B"):
        U.append({"vkey": f"{sid}|{vid}", "tracks": sorted(g.track_key), "set": "N_drive" if sp == "val_A" else "val_B", "type": "normal", "run": sid, "source": "normal corpus"})
BV = pd.read_csv(SETS / "cross_region/vehicles.csv")
for r in BV[BV.split_role == "val_A"].itertuples():
    t = r.anomaly_type.replace("anomaly_", "")
    if r.label == "anomaly_verified":
        U.append({"vkey": f"crossregion/{r.run_id}|{r.vehicle_id_oracle}", "tracks": r.tracks.split(";"), "set": "A", "type": t, "run": r.run_id, "source": "cross-region val"})
    elif r.label == "normal_unaffected_background":
        U.append({"vkey": f"crossregion/{r.run_id}|{r.vehicle_id_oracle}", "tracks": r.tracks.split(";"), "set": "N_drive", "type": "normal", "run": r.run_id, "source": "cross-region val"})
for r in pd.read_csv(SETS / "tailgating_routes/vehicles.csv").itertuples():
    if r.split_role != "val_A":
        continue
    if r.label == "anomaly_verified":
        U.append({"vkey": f"tailgating_v3/{r.run_id}|{r.vehicle_id_oracle}", "tracks": r.tracks.split(";"), "set": "A", "type": "tailgating", "run": r.run_id, "source": "tg v3 val"})
    elif r.label == "normal_unaffected_background":
        U.append({"vkey": f"tailgating_v3/{r.run_id}|{r.vehicle_id_oracle}", "tracks": r.tracks.split(";"), "set": "N_drive", "type": "normal", "run": r.run_id, "source": "tg v3 val"})
SB = pd.read_csv(SETS / "stalls_and_stops/vehicles.csv", dtype={"region": str})
for r in SB[SB.split_role == "stallval"].itertuples():
    k = f"stallval/{r.run_id}|{r.vehicle_id_oracle}"; tr = r.tracks.split(";") if isinstance(r.tracks, str) else []
    st, ty = {"anomaly_verified": ("A", "stalled_vehicle"), "normal_legit_stop": ("L", "legit_stop"), "normal_matched_control": ("N_drive", "stop_control"),
              "normal_unaffected_background": ("N_drive", "normal")}.get(r.label, (None, None))
    if st:
        U.append({"vkey": k, "tracks": tr, "set": st, "type": ty, "run": r.run_id, "source": "stallval"})
SV = pd.read_csv(SETS / "legitimate_stop_validation/vehicles.csv", dtype={"region": str})
for r in SV[SV.split_role == "stopval2"].itertuples():
    k = f"stopval2/{r.run_id}|{r.vehicle_id_oracle}"; tr = r.tracks.split(";") if isinstance(r.tracks, str) else []
    st, ty = {"anomaly_verified": ("R_stopval2_abnormal", "stopval2_abnormal_stop"), "normal_legit_stop": ("L", "legit_stop"),
              "normal_matched_control": ("N_drive", "stop_control"), "normal_unaffected_background": ("N_drive", "normal")}.get(r.label, (None, None))
    if st:
        U.append({"vkey": k, "tracks": tr, "set": st, "type": ty, "run": r.run_id, "source": "stopval2"})
WV = pd.read_csv(HERE / "data/vehicles.csv", dtype={"region": str}); WP = pd.read_csv(HERE / "data/run_plan.csv", dtype={"region": str}).set_index("run_id")
for r in WV[WV.split == "val"].itertuples():
    tr = r.tracks_usable.split(";") if isinstance(r.tracks_usable, str) else []; k = f"W/{r.run_id}|{r.vehicle_id_oracle}"
    st, ty = {"anomaly_verified": ("A", r.type), "normal_matched_control": ("TC", "tailgating_control"), "normal_targeted_control": ("N_drive", r.type),
              "normal_unaffected_background": ("N_drive", "normal")}.get(r.label, (None, None))
    if st:
        U.append({"vkey": k, "tracks": tr, "set": st, "type": ty, "run": r.run_id, "source": "W val", "extent": WP.loc[r.run_id, "extent"] if st == "A" else None})
U = pd.DataFrame(U); U["extent"] = U["extent"] if "extent" in U else None; U["tracks"] = U.tracks.apply(lambda t: [x for x in t if x])

# ------------------------------------------------------------------ data
def cat_dir(d, pat):
    return [pd.read_parquet(f) for f in sorted(d.glob(pat))]
keys = set(k for t in U.tracks for k in t)
ch = pd.read_parquet(paths.PROCESSED / "channels.parquet", columns=["track_key", "bin"] + CH + ["burn_in"])
amp = [pd.read_parquet(paths.PROCESSED / "amplitude.parquet")]; chs = [ch]
for d in (SETS / "cross_region/runs", SETS / "tailgating_routes/runs", SETS / "stalls_and_stops/runs", SETS / "legitimate_stop_validation/runs", HERE / "data/runs"):
    chs += [x[["track_key", "bin"] + CH + ["burn_in"]] for x in cat_dir(d, "*_channels.parquet")]; amp += cat_dir(d, "*_amp.parquet")
ch = pd.concat(chs, ignore_index=True); amp = pd.concat(amp, ignore_index=True); ch["track_key"] = ch.track_key.astype(str); amp["track_key"] = amp.track_key.astype(str)
ch = ch[ch.track_key.isin(keys)]; amp = amp[amp.track_key.isin(keys)]
RAW = to_arrays(attach_amplitude(ch, amp)); del ch, amp
XC = {k: torch.from_numpy(NORM.apply(clean_lead_artifacts(a))[0].T.copy()) for k, a in RAW.items()}
WIN, EXT = {}, {}
for u in U.itertuples():
    ks = [k for k in u.tracks if k in XC]; WIN[u.vkey] = [(k, s) for k in ks for s in window_starts(XC[k].shape[1], T, STRIDE)]
    EXT[u.vkey] = tailgating_fraction([clean_lead_artifacts(RAW[k]) for k in ks])
U["n_windows"] = U.vkey.map(lambda k: len(WIN[k])); U["decided"] = U.n_windows >= 3
U.drop(columns="tracks").to_csv(R / "units.csv", index=False)


@torch.no_grad()
def logits(models, vkeys):
    out = np.full((len(models), len(vkeys)), np.nan)
    for j, vk in enumerate(vkeys):
        ws = WIN[vk]
        if not ws:
            continue
        x = torch.stack([(lambda w: torch.cat([w, torch.zeros(w.shape[0], T - w.shape[1])], 1) if w.shape[1] < T else w)(XC[k][:, s:s + T]) for k, s in ws])
        e = torch.tensor([[EXT[vk]]], dtype=torch.float32)
        for i, m in enumerate(models):
            out[i, j] = m.score(x, e)
    return out


# ------------------------------------------------------------------ 2. models + sanity
MODELS = {"E_DIV": ENS.models}
for arm in ("W1", "W2"):
    MODELS[arm] = [load_detector(HERE / f"models/{arm}/seed_{s}/checkpoint.pt", 1) for s in range(5)]
vk = U.vkey.tolist(); LG = {m: logits(ms, vk) for m, ms in MODELS.items()}
ref = pd.read_csv(paths.RESULTS / "validation/seed_0/vehicle_scores.csv")
for grp in ("normal_val_B", "stallval"):
    rr = ref[ref.group == grp].copy(); rr["k"] = rr.vkey; rr = rr[rr.k.isin(set(vk))]
    mine = pd.Series(LG["E_DIV"][0], index=vk).reindex(rr.k)
    d = np.nanmax(np.abs(mine.values - rr.logit.values)); print(f"sanity E_DIV seed0 {grp}: n={len(rr)} max|d|={d:.2e}")
    assert len(rr) > 50 and d <= 1e-4, f"scorer does not reproduce stored E_DIV {grp} logits"


def calib(m):
    vb = (U.set == "val_B") & U.decided; L_ = LG[m]
    if m == "E_DIV":
        mu = np.array(ENS.mean); sd = np.array(ENS.std); Tk = np.array(ENS.seed_thresholds)
        Z = (L_ - mu[:, None]) / sd[:, None]; ens = Z.mean(0); T = ENS.threshold
        mu_c = L_[:, vb.values].mean(1); assert np.allclose(mu_c, mu, atol=1e-4), (mu_c, mu)              # frozen constants re-derived here
    else:
        mu = L_[:, vb.values].mean(1); sd = L_[:, vb.values].std(1); Tk = np.percentile(L_[:, vb.values], 99, axis=1)
        Z = (L_ - mu[:, None]) / sd[:, None]; ens = Z.mean(0); T = float(np.percentile(ens[vb.values], 99))
    return {"mu": mu.tolist(), "sd": sd.tolist(), "Tk": Tk.tolist(), "T": float(T), "ens": ens, "L": L_}


C = {m: calib(m) for m in MODELS}


# ------------------------------------------------------------------ 3. metrics
def metrics(score, thr, D):
    alarm = (score > thr) & D.decided.values; out = {}
    A = D.set == "A"
    for t in TYPES9:
        m = A & (D.type == t) & D.decided; out[f"n_{t}"] = int(m.sum()); out[f"det_{t}"] = int(alarm[m.values].sum()); out[f"rec_{t}"] = out[f"det_{t}"] / max(out[f"n_{t}"], 1)
    out["Weak6"] = float(np.mean([out[f"rec_{t}"] for t in WEAK])); out["Macro9"] = float(np.mean([out[f"rec_{t}"] for t in TYPES9]))
    nd = ((D.set == "N_drive") & D.decided).values; out["FA_drive_n"] = int(alarm[nd].sum()); out["N_drive"] = int(nd.sum()); out["FA_drive"] = out["FA_drive_n"] / max(out["N_drive"], 1)
    l_ = ((D.set == "L") & D.decided).values; out["FA_L"] = int(alarm[l_].sum()); out["n_L"] = int(l_.sum())
    tc = ((D.set == "TC") & D.decided).values; out["FA_TC"] = int(alarm[tc].sum()); out["n_TC"] = int(tc.sum())
    sv = ((D.source == "stallval") & (D.set == "A") & D.decided).values; out["stallval_stalls"] = int(alarm[sv].sum()); out["n_stallval_stalls"] = int(sv.sum())
    rs = ((D.set == "R_stopval2_abnormal") & D.decided).values; out["stopval2_abnormal_det"] = int(alarm[rs].sum()); out["n_stopval2_abnormal"] = int(rs.sum())
    an = (A & D.decided).values; tp = int(alarm[an].sum()); fp = out["FA_drive_n"]; out["precision"] = tp / max(tp + fp, 1); rec = tp / max(int(an.sum()), 1)
    out["pooled_recall"] = rec; out["F1"] = 2 * out["precision"] * rec / max(out["precision"] + rec, 1e-12)
    y = np.r_[np.ones(an.sum()), np.zeros(nd.sum())]; s = np.r_[score[an], score[nd]]
    out["ROC_AUC"] = float(roc_auc_score(y, s)); out["AUPRC"] = float(average_precision_score(y, s))
    for st in ("A", "N_drive", "L", "TC", "val_B"):
        mm = (D.set == st).values; out[f"coverage_{st}"] = f"{int(D.decided.values[mm].sum())}/{int(mm.sum())}"
    return out


ENS = {m: metrics(C[m]["ens"], C[m]["T"], U) for m in MODELS}
SEED = {m: [metrics(C[m]["L"][s], C[m]["Tk"][s], U) for s in range(5)] for m in MODELS}
pd.DataFrame([{"model": m, "level": "ensemble", **ENS[m]} for m in MODELS] + [{"model": m, "level": f"seed{s}", **SEED[m][s]} for m in MODELS for s in range(5)]).to_csv(R / "metrics_all.csv", index=False)
sc = pd.DataFrame({"vkey": vk})
for m in MODELS:
    sc[f"{m}_ens"] = C[m]["ens"]
    for s in range(5):
        sc[f"{m}_s{s}_logit"] = C[m]["L"][s]
sc.merge(U.drop(columns="tracks"), on="vkey").to_csv(R / "unit_scores.csv", index=False)
json.dump({m: {k: v for k, v in C[m].items() if k not in ("ens", "L")} for m in MODELS}, open(R / "calibration.json", "w"), indent=1)


# ------------------------------------------------------------------ 4. C1–C9
def rule(c):
    e, b = ENS[c], ENS["E_DIV"]; r = {}
    r["C1_weak6_gain_ge_0.15"] = {"value": e["Weak6"] - b["Weak6"], "pass": e["Weak6"] - b["Weak6"] >= 0.15 - 1e-12}
    r["C2_macro9_gain_ge_0.08"] = {"value": e["Macro9"] - b["Macro9"], "pass": e["Macro9"] - b["Macro9"] >= 0.08 - 1e-12}
    loss = {t: e[f"det_{t}"] - b[f"det_{t}"] for t in TYPES9}
    r["C3_no_type_loses_more_than_1"] = {"value": {"det_diff": loss, "stallval_stalls": f"{e['stallval_stalls']}/{e['n_stallval_stalls']}"},
                                         "pass": all(v >= -1 for v in loss.values()) and e["stallval_stalls"] >= 17}
    r["C4_FA_drive"] = {"value": {"cand": e["FA_drive"], "E_DIV": b["FA_drive"]}, "pass": e["FA_drive"] <= 0.015 + 1e-12 and e["FA_drive"] <= b["FA_drive"] + 0.005 + 1e-12}
    r["C5_FA_legit_stops_le_plus3"] = {"value": {"cand": e["FA_L"], "E_DIV": b["FA_L"], "n": e["n_L"]}, "pass": e["FA_L"] - b["FA_L"] <= 3}
    r["C6_FA_tg_controls_le_plus1"] = {"value": {"cand": e["FA_TC"], "E_DIV": b["FA_TC"], "n": e["n_TC"]}, "pass": e["FA_TC"] - b["FA_TC"] <= 1}
    a = [SEED[c][s]["Weak6"] - SEED["E_DIV"][s]["Weak6"] for s in range(5)]
    bb = [(SEED[c][s]["rec_tailgating"] >= SEED["E_DIV"][s]["rec_tailgating"] - 0.10 - 1e-12) and (SEED[c][s]["rec_stalled_vehicle"] >= SEED["E_DIV"][s]["rec_stalled_vehicle"] - 0.10 - 1e-12) for s in range(5)]
    cc = [SEED[c][s]["FA_drive"] for s in range(5)]
    r["C7_seed_stability"] = {"value": {"weak6_gain_per_seed": a, "tg_stalled_ok_per_seed": bb, "FA_drive_per_seed": cc},
                              "pass": sum(x >= 0.10 - 1e-12 for x in a) >= 3 and sum(bb) >= 3 and all(x <= 0.03 + 1e-12 for x in cc)}
    r["C8_ranking"] = {"value": {"AUPRC_diff": e["AUPRC"] - b["AUPRC"], "ROC_AUC_diff": e["ROC_AUC"] - b["ROC_AUC"]},
                       "pass": e["AUPRC"] - b["AUPRC"] >= -0.02 - 1e-12 and e["ROC_AUC"] - b["ROC_AUC"] >= -0.01 - 1e-12}
    cov = {k: (e[k], b[k]) for k in e if k.startswith("coverage_")}
    r["C9_coverage_equal"] = {"value": cov, "pass": all(x == y for x, y in cov.values())}
    r["ALL_PASS"] = all(v["pass"] for k, v in r.items() if k.startswith("C"))
    return r


SEL = {c: rule(c) for c in MODELS if c != "E_DIV"}
passing = [c for c in SEL if SEL[c]["ALL_PASS"]]
if not passing:
    choice = "none -> E_stall_diversity_ENSEMBLE stays the final model"
elif len(passing) == 1:
    choice = passing[0]
else:
    a, b = passing; da = ENS[a]["Macro9"] - ENS[b]["Macro9"]
    if abs(da) >= 0.02:
        choice = a if da > 0 else b
    else:
        dp = ENS[a]["AUPRC"] - ENS[b]["AUPRC"]; choice = (a if dp > 0 else b) if abs(dp) >= 0.01 else "W1"
json.dump({"rule": "PLAN_W1_W2_v2_FOR_APPROVAL.md section 6 (frozen)", "conditions": SEL, "passing": passing, "choice": choice,
           "ensemble_thresholds": {m: C[m]["T"] for m in MODELS}}, open(R / "SELECTION_W.json", "w"), indent=1, default=float)
print(json.dumps({"passing": passing, "choice": choice, **{c: {k: v["pass"] for k, v in SEL[c].items() if k.startswith("C")} for c in SEL}}, indent=1))

# ------------------------------------------------------------------ 5. reported only (not part of the rule)
rows = []
Aw = U[(U.set == "A") & (U.source == "W val") & U.decided]
for m in MODELS:
    al = pd.Series(C[m]["ens"] > C[m]["T"], index=U.index)
    for (t, ex), g in Aw.groupby(["type", "extent"]):
        rows.append({"model": m, "type": t, "extent": ex, "n": len(g), "detected": int(al[g.index].sum())})
pd.DataFrame(rows).to_csv(R / "recall_partial_vs_whole_Wval.csv", index=False)
rng = np.random.default_rng(0); runs = U[(U.set == "A") & U.decided].run.unique(); boot = {c: {"Weak6": [], "Macro9": []} for c in SEL}
Ad = U[(U.set == "A") & U.decided]
for _ in range(2000):
    pick = rng.choice(runs, len(runs), replace=True); idx = np.concatenate([Ad.index[Ad.run == r_].values for r_ in pick])
    for c in SEL:
        rec = {}
        for m in (c, "E_DIV"):
            al = C[m]["ens"][idx] > C[m]["T"]; ty = U.type.values[idx]
            rec[m] = {t: al[ty == t].mean() if (ty == t).any() else np.nan for t in TYPES9}
        boot[c]["Weak6"].append(np.nanmean([rec[c][t] for t in WEAK]) - np.nanmean([rec["E_DIV"][t] for t in WEAK]))
        boot[c]["Macro9"].append(np.nanmean([rec[c][t] for t in TYPES9]) - np.nanmean([rec["E_DIV"][t] for t in TYPES9]))
json.dump({c: {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot[c].items()} for c in SEL}, open(R / "bootstrap_gain_CI.json", "w"), indent=1)
print("written: results/metrics_all.csv, unit_scores.csv, SELECTION_W.json, recall_partial_vs_whole_Wval.csv, bootstrap_gain_CI.json")
