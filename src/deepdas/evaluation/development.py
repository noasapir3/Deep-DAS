"""
The nine development cases (one per anomaly type): reproduction of the reported 8/9 with the frozen models.

Two unit definitions are scored: (a) the simulated vehicle (all its usable tracks), the unit of every development and validation result,
and (b) one unit per usable track (the deployment unit, no ground truth), mapped back to the case vehicle for evaluation only.
The result is compared with the ensemble scores stored when the model was selected.
"""
from __future__ import annotations

import pandas as pd

from .. import paths
from ..data_sources import CH_COLS, normal_corpus, usable_tracks
from ..ensemble.scoring import Ensemble, SingleModel, load_norm, score_units


def development_units():
    cases = pd.read_csv(paths.DEVELOPMENT_CASES / "cases.csv", keep_default_na=False)
    lab, ch, amp = normal_corpus()
    u = usable_tracks(lab); u = u[u.group == "fp_anomaly"]
    orig = u.groupby(["scenario_id", "vehicle_id_oracle"]).track_key.apply(sorted).reset_index(); orig["vkey"] = orig.scenario_id + "|" + orig.vehicle_id_oracle
    ch = ch[ch.track_key.isin(set(u.track_key))]; amp = amp[amp.track_key.isin(set(u.track_key))]
    cc, ca, cv = [ch], [amp], [orig[["vkey", "track_key"]]]
    for r in cases[cases.controlled_file_stem != ""].sort_values("internal_run").itertuples():
        D = paths.DEVELOPMENT_CASES
        c = pd.read_parquet(D / f"{r.controlled_file_stem}_channels.parquet"); m = pd.read_parquet(D / f"{r.controlled_file_stem}_track_meta.parquet")
        a = pd.read_parquet(D / f"{r.controlled_file_stem}_amplitude.parquet")
        for d_ in (c, m, a):
            d_["track_key"] = r.internal_run + "::" + d_.track_key.astype(str)
        cc.append(c[CH_COLS]); ca.append(a)
        vv = m[m.usable].groupby("vehicle_id_oracle").track_key.apply(sorted).reset_index(); vv["vkey"] = r.internal_run + "|" + vv.vehicle_id_oracle; cv.append(vv[["vkey", "track_key"]])
    CHA, AMA, VEH = pd.concat(cc, ignore_index=True), pd.concat(ca, ignore_index=True), pd.concat(cv, ignore_index=True)
    return cases, CHA, AMA, VEH[VEH.vkey.isin(set(cases.unit_key))]


def reproduce():
    """-> (reproduction table with the study's identifiers, display table)."""
    cases, CHA, AMA, VEH = development_units(); ens, single, norm = Ensemble(), SingleModel(), load_norm()
    A = score_units(CHA, AMA, VEH.rename(columns={"vkey": "unit", "track_key": "tracks"})[["unit", "tracks"]], ens, single, norm)
    T = VEH.explode("track_key")
    TT = score_units(CHA, AMA, pd.DataFrame({"unit": T.track_key, "tracks": T.track_key.apply(lambda k: [k])}), ens, single, norm).merge(T, left_on="unit", right_on="track_key")
    E = pd.read_csv(paths.RESULTS / "validation/ensemble_scores_selection_round.csv"); E = E[E.group == "valid9"].set_index("case")
    thr = ens.threshold; rows, disp = [], []
    for c in cases.itertuples():
        a = A[A.unit == c.unit_key].iloc[0]; t = TT[TT.vkey == c.unit_key]
        r = {"case": c.internal_case_label, "n_tracks": len(t), "n_windows": int(a.n_windows),
             "E_DIV_ens_saved": round(float(E.loc[c.internal_case_label, "ens"]), 4), "E_DIV_ens_reproduced": round(float(a.E_DIV_ens), 4),
             "E_DIV_alarm_original_units": bool(a.E_DIV_ens > thr), "E_DIV_alarm_track_units": bool((t.E_DIV_decision == "ALARM").any()),
             "CR_P99_logit": round(float(a.CR_P99_logit), 3), "CR_P99_alarm_original_units": bool(a.CR_P99_logit > single.threshold),
             "CR_P99_alarm_track_units": bool((t.CR_P99_decision == "ALARM").any())}
        rows.append(r)
        disp.append({"case_id": c.case_id, "anomaly_type": c.display_name, "source": c.source, "tracks": len(t), "windows": int(a.n_windows),
                     "DeepDAS_Ensemble_v1_score": round(float(a.E_DIV_ens), 4), "DeepDAS_Ensemble_v1_detected": bool(a.E_DIV_ens > thr),
                     "DeepDAS_Ensemble_v1_detected_track_units": r["E_DIV_alarm_track_units"],
                     "Supervised_single_model_logit": r["CR_P99_logit"], "Supervised_single_model_detected": r["CR_P99_alarm_original_units"]})
    R = pd.DataFrame(rows)
    if getattr(ens.models[0], "phi_mode", "active") == "neutralized":     # the saved score is the published HYBRID (phi active) value
        R = R.rename(columns={"E_DIV_ens_saved": "E_DIV_ens_saved_active_hybrid", "E_DIV_ens_reproduced": "E_DIV_ens_phi_neutralized"})
    return R, pd.DataFrame(disp)
