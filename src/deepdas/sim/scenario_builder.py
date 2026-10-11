"""
Severity study: 9 anomaly types × 3 levels × 3 seeds, each a copy of the Final
Project's scenario of that type in which ONLY the injected vehicle's parameters
change.  The background vehicles, map and sensors are identical to the original.

Levels follow the approved proposal (severe / moderate / mild); the heavy-vehicle
weights are the ones agreed in the design review (beyond the normal fleet's
15 t).  Two types need a mid-route trigger (hard braking, stall) and use the
runtime director; everything else is expressed through the scenario JSON's
per-vehicle override fields that the simulator already honours.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Dict, List

from .. import paths

LEVELS = ["severe", "moderate", "mild"]
SEEDS = [1, 2, 3]
TRIGGER_T = 6.0          # s after start for braking / stall events (vehicle drives normally before)

# anomaly type -> (Final-Project base scenario, injected vehicle)
BASES = {
    "speeding":        ("region_003_anomaly_speeding.sim.json", "veh001"),
    "hard_braking":    ("region_010_anomaly_hard_braking.sim.json", "veh001"),
    "stalled_vehicle": ("region_020_anomaly_stalled_vehicle.sim.json", "veh001"),
    "lane_weaving":    ("region_037_anomaly_lane_weaving.sim.json", "veh001"),
    "tailgating":      ("region_062_anomaly_tailgating.sim.json", "veh001"),
    "lane_straddling": ("region_083_anomaly_lane_straddling.sim.json", "veh001"),
    "collision":       ("region_120_anomaly_collision.sim.json", "veh001"),
    "overweight_vehicle": ("region_140_anomaly_overweight_vehicle.sim.json", "veh001"),
    "reckless_driver": ("region_120_anomaly_reckless_driver.sim.json", "veh001"),
}

# nominal severity parameters per level
PARAMS: Dict[str, Dict[str, dict]] = {
    "speeding":        {"severe": {"ratio": 2.0}, "moderate": {"ratio": 1.4}, "mild": {"ratio": 1.15}},
    "hard_braking":    {"severe": {"decel_mps2": 6.0}, "moderate": {"decel_mps2": 4.0}, "mild": {"decel_mps2": 2.5}},
    "stalled_vehicle": {"severe": {"duration_s": 30.0}, "moderate": {"duration_s": 10.0}, "mild": {"duration_s": 4.0}},
    "lane_weaving":    {"severe": {"amplitude_m": 3.0}, "moderate": {"amplitude_m": 1.5}, "mild": {"amplitude_m": 0.7}},
    "tailgating":      {"severe": {"T_s": 0.3, "s0_m": 0.5}, "moderate": {"T_s": 0.6, "s0_m": 1.5}, "mild": {"T_s": 0.9, "s0_m": 2.5}},
    "lane_straddling": {"severe": {"offset_m": 1.8}, "moderate": {"offset_m": 1.0}, "mild": {"offset_m": 0.5}},
    "collision":       {"severe": {"dv_mps": 15.0}, "moderate": {"dv_mps": 8.0}, "mild": {"dv_mps": 3.0}},
    "overweight_vehicle": {"severe": {"weight_kg": 40000.0}, "moderate": {"weight_kg": 30000.0}, "mild": {"weight_kg": 20000.0}},
    "reckless_driver": {"severe": {"ratio": 2.0, "amplitude_m": 3.0, "T_s": 0.3, "s0_m": 0.5},
                        "moderate": {"ratio": 1.4, "amplitude_m": 1.5, "T_s": 0.6, "s0_m": 1.5},
                        "mild": {"ratio": 1.15, "amplitude_m": 0.7, "T_s": 0.9, "s0_m": 2.5}},
}
WEAVE_PERIOD_S = 2.5


def _speed_limit_of_lane(scenario: dict, lane_id: str) -> float:
    seg_id = scenario["lanes"][lane_id]["segment_id"] if lane_id in scenario.get("lanes", {}) else lane_id.split("_fwd")[0].split("_bwd")[0]
    return float(scenario["segments"][seg_id].get("speed_limit_mps", 13.9))


def _set_speed_profile(v: dict, target: float) -> None:
    """Pin the stochastic speed profile to a ±0.5 m/s band around *target* (the profile cap is min+1)."""
    v["v"] = target
    v["target_speed_mps"] = target
    v["speed_mean_mps"] = target
    v["speed_min_mps"] = max(1.5, target - 0.5)
    v["speed_max_mps"] = target + 0.5
    v["speed_std_mps"] = 0.5


def _clear_overrides(v: dict) -> None:
    for k, default in (("ignore_speed_limit", False), ("frozen", False), ("allow_collision", False), ("disable_idm", False),
                       ("a_cmd_override_mps2", 0.0), ("a_cmd_override_until_t", 0.0), ("min_gap_m", 0.0), ("time_headway_s", 0.0),
                       ("lateral_mode", "ou"), ("lateral_fixed_offset_m", 0.0), ("lateral_weave_amplitude_m", 0.0),
                       ("lateral_weave_period_s", 0.0), ("lateral_weave_phase_rad", 0.0), ("lateral_offset_m", 0.0)):
        v[k] = default


def build_variant(anomaly_type: str, level: str, seed: int) -> dict:
    base_name, vid = BASES[anomaly_type]
    base = json.loads((paths.ANOMALY_SCENARIOS / base_name).read_text())
    sc = copy.deepcopy(base)
    v = sc["vehicles"][vid]
    p = PARAMS[anomaly_type][level]
    limit = _speed_limit_of_lane(sc, v["lane_id"])
    schedule: List[dict] = []
    injection: Dict[str, object] = {"vehicle": vid, "speed_limit_start_mps": limit}
    _clear_overrides(v)

    if anomaly_type == "speeding":
        _set_speed_profile(v, p["ratio"] * limit)
        v["ignore_speed_limit"] = True
        injection.update(ratio=p["ratio"], target_mps=p["ratio"] * limit)
    elif anomaly_type == "hard_braking":
        schedule.append({"t": TRIGGER_T, "vehicle": vid, "action": "brake", "decel_mps2": p["decel_mps2"]})
        injection.update(decel_mps2=p["decel_mps2"], trigger_t=TRIGGER_T)
    elif anomaly_type == "stalled_vehicle":
        _set_speed_profile(v, 0.9 * limit)     # the base scenario had it frozen from t=0 with no speed profile
        schedule.append({"t": TRIGGER_T, "vehicle": vid, "action": "stall", "duration_s": p["duration_s"]})
        injection.update(duration_s=p["duration_s"], trigger_t=TRIGGER_T)
    elif anomaly_type == "lane_weaving":
        v["lateral_mode"] = "weave"
        v["lateral_weave_amplitude_m"] = p["amplitude_m"]
        v["lateral_weave_period_s"] = WEAVE_PERIOD_S
        v["lateral_weave_phase_rad"] = (seed * 2.0 * math.pi / 3.0) % (2 * math.pi)
        injection.update(amplitude_m=p["amplitude_m"], period_s=WEAVE_PERIOD_S)
    elif anomaly_type == "tailgating":
        v["time_headway_s"] = p["T_s"]
        v["min_gap_m"] = p["s0_m"]
        _set_speed_profile(v, 1.12 * limit)    # faster than the leader so that IDM, not free driving, sets the gap
        injection.update(T_s=p["T_s"], s0_m=p["s0_m"], leader=sc["vehicles"]["veh002"]["id"])
    elif anomaly_type == "lane_straddling":
        v["lateral_mode"] = "straddle"
        v["lateral_fixed_offset_m"] = p["offset_m"]
        v["lateral_offset_m"] = p["offset_m"]
        injection.update(offset_m=p["offset_m"])
    elif anomaly_type == "collision":
        _set_speed_profile(v, p["dv_mps"])
        v["allow_collision"] = True
        v["ignore_speed_limit"] = True
        v["disable_idm"] = True
        injection.update(dv_mps=p["dv_mps"], target=sc["vehicles"]["veh002"]["id"])
    elif anomaly_type == "overweight_vehicle":
        v["weight_kg"] = p["weight_kg"]
        injection.update(weight_kg=p["weight_kg"])
    elif anomaly_type == "reckless_driver":
        _set_speed_profile(v, p["ratio"] * limit)
        v["ignore_speed_limit"] = True
        v["allow_collision"] = True
        v["lateral_mode"] = "weave"
        v["lateral_weave_amplitude_m"] = p["amplitude_m"]
        v["lateral_weave_period_s"] = WEAVE_PERIOD_S
        v["lateral_weave_phase_rad"] = (seed * 2.0 * math.pi / 3.0) % (2 * math.pi)
        v["time_headway_s"] = p["T_s"]
        v["min_gap_m"] = p["s0_m"]
        injection.update(ratio=p["ratio"], amplitude_m=p["amplitude_m"], T_s=p["T_s"], s0_m=p["s0_m"])
    else:
        raise ValueError(anomaly_type)

    region = base_name.split("_")[1]
    run_id = f"{anomaly_type}__{level}__s{seed}"
    return {
        "run_id": run_id,
        "region": region,
        "anomaly_type": f"anomaly_{anomaly_type}",
        "level": level,
        "seed": seed,
        "base_scenario": base_name,
        "injected_vehicle": vid,
        "nominal": p,
        "injection": injection,
        "schedule": schedule,
        "scenario": sc,
    }


def all_variants() -> List[dict]:
    return [build_variant(a, l, s) for a in BASES for l in LEVELS for s in SEEDS]


def write_variant(var: dict, root: Path) -> Path:
    run_dir = Path(root) / var["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    scenario_name = f"{var['run_id']}.sim.json"
    (run_dir / scenario_name).write_text(json.dumps(var["scenario"], indent=1))
    info = {k: v for k, v in var.items() if k != "scenario"}
    info["scenario_json"] = scenario_name
    (run_dir / "run.json").write_text(json.dumps(info, indent=1))
    return run_dir
