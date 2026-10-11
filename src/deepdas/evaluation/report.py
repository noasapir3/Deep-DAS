"""Human-readable reports of the published tables: study identifiers are replaced by the display names (the canonical CSV tables keep the
identifiers so that they can be reproduced byte for byte)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..data_sources import ANOMALY_TYPES

DETECTORS = {"E_DIV": "DeepDAS Ensemble v1", "CR_P99": "Supervised single model", "Z": "SimStudio Z-score"}
ROWS_T3 = {"LEGIT 3 s (false alarm)": "Legitimate stop, 3 s hold (false alarm)", "LEGIT 5 s (false alarm)": "Legitimate stop, 5 s hold (false alarm)",
           "LEGIT 8 s (false alarm)": "Legitimate stop, 8 s hold (false alarm)", "GENTLE 15 s stop study (detection)": "Gentle 15 s stall, stop study (detection)",
           "stop-study CONTROL (false alarm)": "Stop-study control, no stop (false alarm)", "stalled START (detection)": "Stalled from the start (detection)",
           "stalled STOP_END (detection)": "Stops and stays stopped (detection)", "stalled MSM (detection)": "Stops 30 s mid-route, then drives on (detection)",
           "stalled GENTLE_STALL (detection)": "Gentle 15 s stall (detection)", "tailgating v6 D8 (detection)": "Tailgating, leader 6 m/s, 8 s (detection)",
           "tailgating v6 D15 (detection)": "Tailgating, leader 6 m/s, 15 s (detection)", "tailgating v9 D8 (detection)": "Tailgating, leader 9 m/s, 8 s (detection)",
           "tailgating controls (false alarm)": "Tailgating matched controls (false alarm)"}
ROWS_T4 = {"E_DIV (oracle-grouped units)": "DeepDAS Ensemble v1, vehicle units", "CR_P99 (oracle-grouped units)": "Supervised single model, vehicle units",
           "Z-score truth-free sensitivity": "SimStudio Z-score without truth-position features"}


def _rename(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); cols = {}
    for c in df.columns:
        n = c
        for k, v in DETECTORS.items():
            if c == k or c.startswith(k + "_"):
                n = v + c[len(k):].replace("_", " "); break
        cols[c] = ANOMALY_TYPES.get(n, n)
    df = df.rename(columns=cols)
    for c in ("type", "detector"):
        if c in df:
            df[c] = df[c].map(lambda x: ANOMALY_TYPES.get(x, DETECTORS.get(x, x)))
    return df


def final_test_report(tables_dir: Path, out_md: Path):
    d = Path(tables_dir)
    T1 = _rename(pd.read_csv(d / "T1_per_type.csv")); T2 = _rename(pd.read_csv(d / "T2_overall.csv"))
    T3 = pd.read_csv(d / "T3_stops_tailgating.csv"); T3["group"] = T3.group.map(ROWS_T3); T3 = _rename(T3)
    T4 = pd.read_csv(d / "T4_secondary.csv"); T4["variant"] = T4.variant.map(ROWS_T4); T4 = _rename(T4)
    T5 = json.loads((d / "T5_uncertainty.json").read_text())
    txt = json.dumps(T5, indent=1)
    for k, v in DETECTORS.items():
        txt = txt.replace(k + " ", v + " ").replace(k + ")", v + ")").replace(" " + k + "\"", " " + v + "\"").replace("only " + k, "only " + v)
    out = ["# Final test: North-TLV test regions (independent, run once)", "",
           "Frozen detectors and thresholds. Deployment unit = one usable track (no ground truth in inference); a vehicle counts as detected if any of",
           "its decided tracks alarms (the truth vehicle id is used only for this evaluation mapping).", "",
           "## T1 Per anomaly type: injected, realised, tracked, decided, detected", "", T1.to_markdown(index=False), "",
           "## T2 Overall: realised anomalies vs normal driving", "", T2.to_markdown(index=False), "",
           "## T3 Stops and tailgating", "", T3.to_markdown(index=False), "",
           "## T4 Secondary variants", "", T4.to_markdown(index=False), "",
           "## T5 Uncertainty (bootstrap over runs, common decided set) and McNemar tests", "", "```", txt, "```"]
    Path(out_md).write_text("\n".join(out) + "\n")
