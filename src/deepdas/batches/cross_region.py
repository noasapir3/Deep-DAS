"""
Cross-region anomaly set: one vehicle of each of the nine anomaly types injected in six North-TLV regions (4 training, 2 validation).
The vehicle drives on the longest lane of the region's medium-traffic scenario; injections use only vehicle fields the simulator already
reads (speed profile, ignore_speed_limit, lateral mode, weight, allow_collision / disable_idm) and the AnomalyDirector (braking at t = 6 s,
stall at t = 6 s for 30 s). Realisation is measured from the simulator truth (labels only).
Each run: <out_dir>/<run_id>/ with scenario json, export, truth trail, run.json (parameter diff, director log, result) and realization.json.
Refuses to rerun an existing folder.
Usage:  python -m deepdas.batches.cross_region <out_dir> [run_id ...]      (no ids = list the plan)
"""
import copy, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from .. import paths  # noqa: E402
PLANS = paths.DATA / "simulation_plans" / "cross_region"; WORK = paths.GENERATED / "cross_region"
from deepdas.sim.runner import run_scenario  # noqa: E402
from deepdas.sim.anomaly_director import AnomalyDirector  # noqa: E402
from deepdas.sim.scenario_builder import _set_speed_profile, _speed_limit_of_lane, _clear_overrides, WEAVE_PERIOD_S  # noqa: E402
from deepdas.sim.severity import _lane_polylines, _project, realised_severity  # noqa: E402
from deepdas.fp.constants import IDM_VEHICLE_LENGTH_M  # noqa: E402

REGIONS = {"131": "train", "050": "train", "090": "train", "065": "train", "002": "val_A", "005": "val_A"}
TYPES = ["speeding", "reckless_driver", "hard_braking", "stalled_vehicle", "lane_weaving", "lane_straddling", "tailgating", "collision", "overweight_vehicle"]
SEED = 1; TRIG = 6.0


def plan():
    return [(f"r{r}_{t}", r, t) for r in REGIONS for t in TYPES]


def longest_lane(sc):
    L = _lane_polylines(sc)
    ln = {k: float(np.hypot(*np.diff(np.array(p), axis=0).T).sum()) for k, p in L.items()}
    two = {sid for sid, s in sc["segments"].items() if not s.get("one_way")}
    cand = sorted(ln, key=lambda k: (k.split("_")[0] in two, ln[k]), reverse=True)
    return cand[0], ln[cand[0]], L[cand[0]]


def place(v, lane, s, poly):
    v["lane_id"] = lane; v["s"] = float(s)
    (x0, y0), (x1, y1) = poly[0], poly[1]
    v["heading_rad"] = math.atan2(y1 - y0, x1 - x0); v["ax_world_mps2"] = 0.0; v["ay_world_mps2"] = 0.0; v["a_long_mps2"] = 0.0


def diff(a, b, p=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += diff(a.get(k, "<absent>"), b.get(k, "<absent>"), f"{p}/{k}")
    elif a != b:
        out.append({"path": p, "original": a, "new": b})
    return out


def build(region, typ):
    src = paths.SCENARIOS / region / "simulations/normal_medium_traffic" / f"region_{region}_normal_medium_traffic.sim.json"
    base = json.loads(src.read_text()); sc = copy.deepcopy(base)
    lane, llen, poly = longest_lane(sc)
    v = sc["vehicles"]["veh001"]; _clear_overrides(v); place(v, lane, 20.0, poly)
    limit = _speed_limit_of_lane(sc, lane); sched = []; inj = {"lane": lane, "lane_length_m": llen, "speed_limit_mps": limit}
    if typ == "speeding":
        _set_speed_profile(v, 1.4 * limit); v["ignore_speed_limit"] = True
    elif typ == "reckless_driver":
        _set_speed_profile(v, 1.4 * limit); v.update(ignore_speed_limit=True, allow_collision=True, lateral_mode="weave", lateral_weave_amplitude_m=1.5,
                                                     lateral_weave_period_s=WEAVE_PERIOD_S, lateral_weave_phase_rad=0.0, time_headway_s=0.6, min_gap_m=1.5)
    elif typ == "hard_braking":
        sched = [{"t": TRIG, "vehicle": "veh001", "action": "brake", "decel_mps2": 6.0}]
    elif typ == "stalled_vehicle":
        _set_speed_profile(v, 0.9 * limit); sched = [{"t": TRIG, "vehicle": "veh001", "action": "stall", "duration_s": 30.0}]
    elif typ == "lane_weaving":
        v.update(lateral_mode="weave", lateral_weave_amplitude_m=1.5, lateral_weave_period_s=WEAVE_PERIOD_S, lateral_weave_phase_rad=0.0)
    elif typ == "lane_straddling":
        v.update(lateral_mode="straddle", lateral_fixed_offset_m=-1.8, lateral_offset_m=-1.8)
    elif typ == "tailgating":
        w = sc["vehicles"]["veh002"]; _clear_overrides(w); place(w, lane, 60.0, poly); _set_speed_profile(w, 0.8 * limit)
        v.update(time_headway_s=0.3, min_gap_m=0.5); _set_speed_profile(v, 1.12 * limit); inj["leader"] = "veh002"
    elif typ == "collision":
        w = sc["vehicles"]["veh002"]; _clear_overrides(w); place(w, lane, min(llen - 15.0, 150.0), poly)
        w.update(frozen=True, v=0.0, target_speed_mps=0.0)
        _set_speed_profile(v, 8.0); v.update(allow_collision=True, ignore_speed_limit=True, disable_idm=True); inj["target"] = "veh002"
    elif typ == "overweight_vehicle":
        v["weight_kg"] = 30000.0
    return src, base, sc, sched, inj


def realization(sc, truth, collisions, typ):
    trig = TRIG if typ in ("hard_braking", "stalled_vehicle") else None
    rs = realised_severity(sc, [dict(r) for r in truth], "veh001", [c for c in collisions if "veh001" in c.get("vehicle_ids", [])], trigger_t=trig)
    lanes = _lane_polylines(sc); rows = [r for r in truth if r["vehicle_id"] == "veh001"]
    dl = np.array([_project(lanes, r)[1] for r in rows], float)
    rs["d_lat_abs_median"] = float(np.nanmedian(np.abs(dl))) if np.isfinite(dl).any() else float("nan")
    rule = {"speeding": rs.get("speed_ratio_p90", 0) >= 1.10, "reckless_driver": rs.get("speed_ratio_p90", 0) >= 1.10,
            "hard_braking": rs.get("decel_peak_mps2", 0) >= 4.0, "stalled_vehicle": rs.get("stop_longest_s", 0) >= 10.0,
            "lane_weaving": rs.get("lateral_std_m", 0) >= 1.0, "lane_straddling": rs["d_lat_abs_median"] >= 1.7,
            "tailgating": (rs.get("gap_min_m", 99) < 5.0) and (rs.get("headway_p10_s", 99) < 0.5),
            "collision": rs.get("n_collisions", 0) >= 1, "overweight_vehicle": rs.get("weight_kg", 0) >= 20000}
    rs["verified"] = bool(rule[typ]); rs["rule"] = typ
    # background vehicles: unaffected -> usable as normal examples
    bg = {}
    for vid in sc["vehicles"]:
        if vid == "veh001":
            continue
        r = realised_severity(sc, [dict(x) for x in truth], vid, [], trigger_t=None)
        role = "leader/target" if vid == "veh002" and typ in ("tailgating", "collision") else "background"
        ok = r.get("n_rows", 0) > 0 and role == "background" and r.get("stop_longest_s", 0) < 2.0 and not (r.get("gap_min_m", 99) < 5.0)
        bg[vid] = {"role": role, "unaffected_normal": bool(ok), "stop_longest_s": r.get("stop_longest_s"), "gap_min_m": r.get("gap_min_m")}
    rs["background"] = bg
    return rs


if __name__ == "__main__":
    B = Path(sys.argv[1]); ids = sys.argv[2:]
    if not ids:
        for p in plan():
            print(*p)
        sys.exit(0)
    for rid, region, typ in plan():
        if rid not in ids:
            continue
        d = B / rid
        if d.exists():
            print(rid, "exists - not rerun", flush=True); continue
        d.mkdir(parents=True)
        src, base, sc, sched, inj = build(region, typ)
        sj = d / f"{rid}.sim.json"; sj.write_text(json.dumps(sc, indent=1))
        hook = AnomalyDirector(sched) if sched else None
        t0 = time.time()
        res = run_scenario(sj, d, seed=SEED, hook=hook, record_truth=True)
        truth = res.pop("truth"); pd.DataFrame(truth).to_parquet(d / "truth_trail.parquet", index=False)
        real = realization(sc, truth, res["collisions"], typ)
        info = {"run_id": rid, "region": region, "split_role": REGIONS[region], "anomaly_type": f"anomaly_{typ}", "seed": SEED,
                "source_scenario": str(src.relative_to(paths.ROOT)), "source_md5": hashlib.md5(src.read_bytes()).hexdigest(),
                "scenario_md5": hashlib.md5(sj.read_bytes()).hexdigest(), "injection": inj, "parameter_changes": diff(base, sc),
                "director_schedule": sched, "director_log": hook.log if hook else [], "runner": "deepdas.sim.runner.run_scenario (SIM_DT 0.02, cap 120 s)",
                "wall_s": round(time.time() - t0, 1), "result": res}
        (d / "run.json").write_text(json.dumps(info, indent=1, default=str))
        (d / "realization.json").write_text(json.dumps(real, indent=1, default=str))
        print(rid, res["stop_reason"], round(res["sim_time_s"], 1), "wall", info["wall_s"], "verified", real["verified"], flush=True)
