#!/usr/bin/env python3
"""Comparison tables: DeepDAS Ensemble v1 with phi_u ACTIVE (original hybrid, published results/) vs phi_u NEUTRALIZED at inference
(results_phi_neutralized/, produced by the reproduction scripts with --phi-mode neutralized). Every table carries the population and the
data role. Final-test rows are POST-HOC analysis of an already exposed test. Writes ../tables/*.csv."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
OUT = Path(__file__).resolve().parents[1] / "tables"; OUT.mkdir(exist_ok=True)
A, N = ROOT / "results", ROOT / "results_phi_neutralized"
SPEC = json.loads((ROOT / "models/deepdas_ensemble_v1/ensemble.json").read_text()); TAU = SPEC["decision"]["threshold"]
for d in ("development", "validation", "final_test"):
    assert json.loads((N / d / "RUN_MODE.json").read_text())["phi_mode"] == "neutralized", d


def phi_of_development_cases():
    from deepdas.evaluation.development import development_units
    from deepdas.amplitude import attach_amplitude, to_arrays
    from deepdas.ensemble.detector import tailgating_fraction, clean_lead_artifacts
    cases, CHA, AMA, VEH = development_units(); emu, esd = 0.005994477309286594, 0.03163501247763634; out = {}
    for r in VEH.itertuples():
        ks = r.track_key if isinstance(r.track_key, list) else [r.track_key]
        raw = to_arrays(attach_amplitude(CHA[CHA.track_key.isin(ks)], AMA[AMA.track_key.isin(ks)]))
        phi = tailgating_fraction([clean_lead_artifacts(raw[k]) for k in ks if k in raw]); out[r.vkey] = (phi, (phi - emu) / esd)
    return cases.set_index("unit_key"), out


# ---------------------------------------------------------------- development cases (diagnostic population, not independent)
da = pd.read_csv(A / "development/development_cases.csv"); dn = pd.read_csv(N / "development/development_cases.csv")
cases, PHI = phi_of_development_cases(); cid2unit = {c.case_id: u for u, c in cases.iterrows()}
D = pd.DataFrame({"population": "development cases (9, diagnostic)", "case_id": da.case_id, "anomaly_type": da.anomaly_type,
                  "phi_u": [PHI[cid2unit[c]][0] for c in da.case_id], "phi_u_standardised": [PHI[cid2unit[c]][1] for c in da.case_id],
                  "score_phi_active": da.DeepDAS_Ensemble_v1_score, "detected_phi_active": da.DeepDAS_Ensemble_v1_detected,
                  "score_phi_neutralized": dn.DeepDAS_Ensemble_v1_score, "detected_phi_neutralized": dn.DeepDAS_Ensemble_v1_detected,
                  "frozen_tau": TAU})
D["delta"] = D.score_phi_neutralized - D.score_phi_active
D.to_csv(OUT / "development_cases_comparison.csv", index=False)

# ---------------------------------------------------------------- validation (frozen mu_k, sigma_k, tau in both columns)
from deepdas.evaluation.validation import ensemble_tables  # noqa: E402
rows = {}
for name, d in (("phi active (original hybrid)", A / "validation"), ("phi neutralized (inference-time intervention)", N / "validation")):
    M, E, LE = ensemble_tables([d / f"seed_{s}" for s in range(5)], frozen=True)
    m = M[M.seed == "ENSEMBLE"].iloc[0].to_dict(); ok = LE[LE.nw >= 3]
    leg = ok[ok.anomaly_type == "normal_legit_stop"]; con = ok[ok.anomaly_type == "normal_matched_control"]
    vb = E[E.group == "normal_val_B"].ens
    rows[name] = {"population": "validation regions (val_A anomalies and normals; val_B normals; stop sets)",
                  "tailgating (12 v3 + 1 cross-region)": m["val_tailgating_n"], "macro recall, 9 types": round(m["macro9"], 3),
                  "validation AP": round(m["val_AP"], 3), "stall validation (3 modes)": f"{int(round(m['stallval_macro3'] * 18))}/18",
                  "false alarms, val_A normals": f"{int(m['FA_valA'])}/1313", "legitimate stops (54)":
                  f"{int(str(m['stallval_alarm_legit_stop_n']).split('/')[0]) + int((leg.ens > TAU).sum())}/{6 + len(leg)}",
                  "stop-set matched controls": f"{int((con.ens > TAU).sum())}/{len(con)}", "development cases": f"{int(m['diag9_detected'])}/9",
                  "val_B alarm rate at frozen tau": round(float((vb > TAU).mean()), 4), "frozen tau": TAU}
V = pd.DataFrame(rows); V.index.name = "metric"; V.to_csv(OUT / "validation_comparison.csv")

# ---------------------------------------------------------------- final test, exposed: POST HOC
fa, fn = A / "final_test", N / "final_test"
T2a, T2n = pd.read_csv(fa / "T2_overall.csv").set_index("detector"), pd.read_csv(fn / "T2_overall.csv").set_index("detector")
T1a, T1n = pd.read_csv(fa / "T1_per_type.csv").set_index("type"), pd.read_csv(fn / "T1_per_type.csv").set_index("type")
T3a, T3n = pd.read_csv(fa / "T3_stops_tailgating.csv").set_index("group"), pd.read_csv(fn / "T3_stops_tailgating.csv").set_index("group")
F = {"population": "final test, common decided subset (101 anomalies, 1,008 normal vehicles); exposed before this analysis (post hoc)"}
rowsF = []
for lab, a, n in [("macro recall (common)", T2a.loc["E_DIV", "macro_recall_common"], T2n.loc["E_DIV", "macro_recall_common"]),
                  ("ROC-AUC", T2a.loc["E_DIV", "ROC_AUC"], T2n.loc["E_DIV", "ROC_AUC"]), ("AUPRC", T2a.loc["E_DIV", "AUPRC"], T2n.loc["E_DIV", "AUPRC"]),
                  ("precision", T2a.loc["E_DIV", "precision"], T2n.loc["E_DIV", "precision"]), ("F1", T2a.loc["E_DIV", "F1"], T2n.loc["E_DIV", "F1"]),
                  ("false alarms, normal driving", T2a.loc["E_DIV", "FA_normal_common"], T2n.loc["E_DIV", "FA_normal_common"]),
                  ("false alarms, legitimate stops", T2a.loc["E_DIV", "FA_legit_common"], T2n.loc["E_DIV", "FA_legit_common"]),
                  ("tailgating matched controls", T3a.loc["tailgating controls (false alarm)", "E_DIV"], T3n.loc["tailgating controls (false alarm)", "E_DIV"])]:
    rowsF.append({"metric": lab, "phi active (original hybrid)": a, "phi neutralized (inference-time intervention)": n})
for t in T1a.index:
    rowsF.append({"metric": f"detected: {t} (of {int(T1a.loc[t, 'common_n'])})",
                  "phi active (original hybrid)": f"{int(round(T1a.loc[t, 'E_DIV_recall_common'] * T1a.loc[t, 'common_n']))}/{int(T1a.loc[t, 'common_n'])}",
                  "phi neutralized (inference-time intervention)": f"{int(round(T1n.loc[t, 'E_DIV_recall_common'] * T1n.loc[t, 'common_n']))}/{int(T1n.loc[t, 'common_n'])}"})
FT = pd.DataFrame(rowsF); FT.insert(0, "population", F["population"]); FT.to_csv(OUT / "final_test_comparison_POST_HOC.csv", index=False)

# ---------------------------------------------------------------- the four-row summary used in the report
S = pd.DataFrame([
    {"check": "development cases (9)", "phi neutralized": f"{int(D.detected_phi_neutralized.sum())}/9", "phi active": f"{int(D.detected_phi_active.sum())}/9"},
    {"check": "tailgating, validation", "phi neutralized": V.loc["tailgating (12 v3 + 1 cross-region)", "phi neutralized (inference-time intervention)"],
     "phi active": V.loc["tailgating (12 v3 + 1 cross-region)", "phi active (original hybrid)"]},
    {"check": "tailgating, exposed final test (post hoc)", "phi neutralized": FT.set_index("metric").loc["detected: tailgating (of 11)", "phi neutralized (inference-time intervention)"],
     "phi active": FT.set_index("metric").loc["detected: tailgating (of 11)", "phi active (original hybrid)"]},
    {"check": "macro recall, exposed final test (post hoc)", "phi neutralized": f"{T2n.loc['E_DIV', 'macro_recall_common']:.3f}", "phi active": f"{T2a.loc['E_DIV', 'macro_recall_common']:.3f}"}])
S.to_csv(OUT / "report_summary_table.csv", index=False)
print(D.round(4).to_string(index=False)); print(V.to_string()); print(FT.to_string(index=False)); print(S.to_string(index=False))
