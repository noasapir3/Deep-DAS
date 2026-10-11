"""W01 library — shared helpers for the W1/W2 data batch. Imports the verified final-test builder (h01) and the earlier batch modules READ-ONLY
(no file of theirs is written); only the map source is pointed at this round's copy of the North-TLV scenario files (data/src, identical files)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
W = Path(__file__).resolve().parents[1]; FU = W.parent; DL = FU.parent
sys.path.insert(0, str(FU / "final_holdout_20261009/code")); sys.path.insert(0, str(DL / "src"))
import h01_north_test_batch as h01  # noqa: E402
SRC = W / "data/src/tiles_north_tlv_south"; h01.SRC = SRC; h01.st.RAW = SRC; h01.H = W   # in-memory only: h01.build uses H only for a relative path in run.json
SPL = json.loads((DL / "data/processed/splits.json").read_text())
TRAIN, VALA, VALB, TEST = SPL["train"], SPL["val_A"], SPL["val_B"], SPL["test"]


def prior_injection_segments():
    """Segments of every earlier injection start / stop lane, per region (all earlier batch plans + the p10 longest lane)."""
    out = {}
    for f in [f for f in FU.glob("*/data/RUN_PLAN*.csv") if W.name not in str(f)]:          # earlier batches only (not this round)
        P = pd.read_csv(f, dtype=str)
        if "region" not in P:
            continue
        for c in ("start_lane", "stop_lane", "lane"):
            if c in P:
                for r, l in zip(P.region, P[c]):
                    if isinstance(l, str) and l:
                        out.setdefault(r.zfill(3), set()).add(l.split("_")[0])
    for f in FU.glob("supervised_crossregion_20260927_200203/sims*/*/run.json"):
        j = json.loads(f.read_text()); l = j.get("injection", {}).get("lane")
        if l:
            out.setdefault(j["region"], set()).add(l.split("_")[0])
    for f in [f for f in FU.glob("*/sims*/*/run.json") if W.name not in str(f)]:
        try:
            j = json.loads(f.read_text())
        except Exception:
            continue
        pr = j.get("plan_row") or {}
        for c in ("start_lane", "stop_lane"):
            if pr.get(c) and pr.get("region"):
                out.setdefault(str(pr["region"]).zfill(3), set()).add(str(pr[c]).split("_")[0])
    return out
