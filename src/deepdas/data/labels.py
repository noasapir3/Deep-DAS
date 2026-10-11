"""
Track roles and anomaly labels.

A track is
    ``normal``      when it comes from a normal scenario (group ``fp_normal``);
    ``injected``    when its oracle vehicle carries an anomaly override in the
                    scenario file (or is named in ``run.json`` for severity runs);
    ``background``  for the other vehicles of an anomaly scenario — normal
                    drivers that may still react to the injected one, so they
                    are reported separately and never counted as positives.

The oracle vehicle id is used here ONLY to attach the label; it is not an input.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from .. import paths

NORMAL_FLEET_MAX_KG = 15000.0   # heaviest weight class of populate_vehicles.py (trucks 4.5-15 t)


def injected_vehicles(scenario_json: Path) -> Dict[str, str]:
    """vehicle id -> short description of the override(s) that make it anomalous."""
    d = json.loads(Path(scenario_json).read_text())
    out: Dict[str, str] = {}
    for vid, v in d.get("vehicles", {}).items():
        flags = []
        if v.get("ignore_speed_limit"):
            flags.append("ignore_speed_limit")
        if v.get("frozen"):
            flags.append("frozen")
        if v.get("allow_collision"):
            flags.append("allow_collision")
        if v.get("disable_idm"):
            flags.append("disable_idm")
        if float(v.get("a_cmd_override_mps2", 0) or 0) != 0.0:
            flags.append(f"a_cmd_override={v['a_cmd_override_mps2']}")
        if float(v.get("min_gap_m", 0) or 0) != 0.0:
            flags.append(f"min_gap_m={v['min_gap_m']}")
        if float(v.get("time_headway_s", 0) or 0) != 0.0:
            flags.append(f"time_headway_s={v['time_headway_s']}")
        if str(v.get("lateral_mode", "ou") or "ou").lower() != "ou":
            flags.append(f"lateral_mode={v['lateral_mode']}")
        if float(v.get("weight_kg", 0) or 0) > NORMAL_FLEET_MAX_KG:
            flags.append(f"weight_kg={v['weight_kg']}")
        if flags:
            out[vid] = ",".join(flags)
    return out


def build_labels(meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cache: Dict[str, Dict[str, str]] = {}
    for _, m in meta.iterrows():
        role, atype, level, injected_desc = "normal", "", "", ""
        if m["group"] == "fp_anomaly":
            key = m["scenario_id"]
            if key not in cache:
                cache[key] = injected_vehicles(paths.ANOMALY_SCENARIOS / m["scenario_json"])
            inj = cache[key]
            role = "injected" if m["vehicle_id_oracle"] in inj else "background"
            atype = m["scenario_type"]
            level = "fp"
            injected_desc = inj.get(m["vehicle_id_oracle"], "")
        elif m["group"] == "severity":
            role = "injected" if m["vehicle_id_oracle"] == m.get("injected_vehicle", "") else "background"
            atype = m["scenario_type"]
            level = str(m.get("level", ""))
            injected_desc = str(m.get("injection", "")) if role == "injected" else ""
        rows.append({
            "track_key": m["track_key"], "role": role, "anomaly_type": atype, "level": level,
            "injection": injected_desc, "vehicle_id_oracle": m["vehicle_id_oracle"], "scenario_id": m["scenario_id"],
            "group": m["group"], "region": m["region"], "scenario_type": m["scenario_type"],
        })
    return pd.DataFrame(rows)
