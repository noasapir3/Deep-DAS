#!/usr/bin/env python3
"""W05 — import the W runs with the UNCHANGED development pipeline (same code path as final_holdout h03_import.py: read_export -> build_channels with
the region geometry -> track_quality; DAS amplitude on the DL bins with the a02 function). Labels from realization.json only (truth).
Writes data/W_processed/<run>_{channels,amp,track_meta}.parquet, data/W_vehicles.csv and data/W_run_report.csv."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from python_calamine import CalamineWorkbook
W = Path(__file__).resolve().parents[1]; FU = W.parent; DL = FU.parent; sys.path.insert(0, str(DL / "src"))
from deepdas.data.import_exports import ExportRef, read_export  # noqa: E402
from deepdas.data.channels import build_channels  # noqa: E402
from deepdas.data.map_matching import LaneIndex  # noqa: E402
from deepdas.data.quality import track_quality  # noqa: E402
src = (FU / "amplitude_only_20260927_203146/scripts/a02_r1_channels_and_amplitude.py").read_text()
exec(src[src.index("def amp_on_bins"):src.index("stats, chk = []")])
SRC = W / "data/src/tiles_north_tlv_south"; B = W / "sims"; O = W / "data/W_processed"; O.mkdir(parents=True, exist_ok=True)
P = pd.read_csv(W / "data/RUN_PLAN_W.csv", dtype={"region": str}).set_index("run_id"); veh, runs = [], []


def label(row, real, vid):
    if vid == "veh001":
        if row.block == "anomaly":
            return "anomaly_verified" if real.get("verified") else "anomaly_UNVERIFIED_excluded"
        if row.block == "control":
            return "normal_targeted_control" if real.get("verified") else "control_NOT_clean_excluded"
        if row.block == "tg_control":
            return "normal_matched_control" if real.get("control_clean") else "control_NOT_clean_excluded"
    if vid == "veh002" and row.type in ("tailgating", "collision"):
        return "leader_or_target_excluded"
    return "normal_unaffected_background" if real.get("background", {}).get(vid, {}).get("unaffected_normal") else "background_affected_excluded"


done = set(p.name for p in O.glob("*_track_meta.parquet"))
for d in sorted(p for p in B.iterdir() if p.is_dir()):
    info = json.loads((d / "run.json").read_text()); real = json.loads((d / "realization.json").read_text()); rid = info["run_id"]; row = P.loc[rid]; reg = row.region
    if f"{rid}_track_meta.parquet" not in done:
        exp = d / info["result"]["export_dir"]
        ref = ExportRef(group="W", scenario_id=f"W/{rid}", region=reg, scenario_type=rid, export_dir=exp, scenario_json=d / f"{rid}.sim.json")
        rows50, meta = read_export(ref); rows50["track_key"] = rows50.track_key.astype(str); meta["track_key"] = meta.track_key.astype(str)
        ch = build_channels(rows50, meta, {reg: LaneIndex.from_scenario_json(SRC / reg / f"region_{reg}_sensors.sim.json")}); ch["track_key"] = ch.track_key.astype(str)
        vb = ch[ch.m_valid > 0].groupby("track_key").size(); q = track_quality(ch).set_index("track_key")
        meta["n_valid_bins"] = meta.track_key.map(vb).fillna(0).astype(int); meta["quality_ok"] = meta.track_key.map(q["quality_ok"]).fillna(True).astype(bool)
        meta["usable"] = (meta.n_valid_bins >= 32) & meta.quality_ok
        rws = CalamineWorkbook.from_path(str(exp / "tracking_audit.xlsx")).get_sheet_by_name("Kalman Audit").to_python(skip_empty_area=False)
        col = {str(h): i for i, h in enumerate(rws[0])}
        aud = pd.DataFrame([(x[col["Time (s)"]], str(x[col["Track ID"]]), x[col["DAS Amplitude A (raw units)"]]) for x in rws[1:] if x[col["Sensor Type"]] == "DAS"], columns=["t", "tid", "amp"])
        aud["t"] = pd.to_numeric(aud.t); aud["amp"] = pd.to_numeric(aud.amp, errors="coerce")
        A = pd.concat([amp_on_bins(rows50, aud, tk, tid)[0] for tk, tid in zip(meta.track_key, meta.track_id)], ignore_index=True) if len(meta) else pd.DataFrame(columns=["track_key", "bin", "amp_log10"])
        ch.to_parquet(O / f"{rid}_channels.parquet", index=False); A.to_parquet(O / f"{rid}_amp.parquet", index=False); meta.to_parquet(O / f"{rid}_track_meta.parquet", index=False)
    meta = pd.read_parquet(O / f"{rid}_track_meta.parquet")
    sc = json.loads((d / f"{rid}.sim.json").read_text())
    for vid in sc["vehicles"]:
        g = meta[meta.vehicle_id_oracle == vid]
        veh.append({"run_id": rid, "split": row.split, "block": row.block, "type": row.type, "extent": row.extent, "region": reg, "bg_type": row.bg_type,
                    "vehicle_id_oracle": vid, "label": label(row, real, vid), "tracks_all": ";".join(sorted(g.track_key)), "tracks_usable": ";".join(sorted(g[g.usable].track_key))})
    m1 = meta[meta.vehicle_id_oracle == "veh001"]
    runs.append({"run_id": rid, "split": row.split, "block": row.block, "type": row.type, "extent": row.extent, "region": reg, "verified": real.get("verified"),
                 "control_clean": real.get("control_clean"), "veh001_tracks": len(m1), "veh001_usable_tracks": int(m1.usable.sum()), "stop_reason": info["result"]["stop_reason"]})
    print(rid, "tracks", len(meta), "usable", int(meta.usable.sum()), flush=True)
V = pd.DataFrame(veh); V.to_csv(W / "data/W_vehicles.csv", index=False); R = pd.DataFrame(runs); R.to_csv(W / "data/W_run_report.csv", index=False)
print(V.groupby(["split", "block", "label"]).size().to_string())
