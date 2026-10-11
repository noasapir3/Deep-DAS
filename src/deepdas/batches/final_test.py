"""
Independent final test in the 8 North-TLV test regions (228 runs, planned and frozen before any simulation; run once on 2026-10-09).
  plan  -> outputs/generated/final_test/run_plan.csv + plan_checks/   (geometry only; deterministic)
  run <out_dir> <run_id ...>                                           (uses the shipped plan; one folder per run; refuses to rerun)
Unchanged simulator, runner and tracker. Injection = the mechanisms of the training sets, imported unchanged: cross-region parameters
(7 types), stall / legitimate-stop hooks (START / STOP_END / MSM / LEGIT_STOP), GENTLE_STALL, and the tailgating route hook and label.
Scene independence: background vehicles regenerated with the corpus generator logic (populate_vehicles.build_vehicle and SCENARIO_SPECS,
unchanged) with new seeds, no background vehicle on an injection start lane, new simulation seeds (random routing / speed profiles), and
injection start lanes never used for an injection in that region.
"""
import copy, hashlib, json, math, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd

from .. import paths  # noqa: E402
from . import stalls as st  # noqa: E402
from . import stop_study as sd  # noqa: E402
from . import tailgating_controls as tg  # noqa: E402
from . import cross_region  # noqa: E402
from . import populate_vehicles as pv  # noqa: E402
PLANS = paths.DATA / "simulation_plans" / "final_test"; WORK = paths.GENERATED / "final_test"
from deepdas.sim.runner import run_scenario  # noqa: E402
from deepdas.sim.anomaly_director import AnomalyDirector  # noqa: E402
from deepdas.sim.scenario_builder import _set_speed_profile, _clear_overrides, WEAVE_PERIOD_S  # noqa: E402
from deepdas.sim.severity import realised_severity  # noqa: E402

SRC = paths.SCENARIOS; st.RAW = SRC
TEST = ["003", "010", "020", "037", "062", "083", "120", "140"]
PRIOR_SEGS = {"003": {"seg59", "seg1603"}, "010": {"seg473", "seg84", "seg485"}, "020": {"seg388", "seg382"}, "037": {"seg842", "seg1352"},
              "062": {"seg919"}, "083": {"seg871", "seg868", "seg865"}, "120": {"seg2493", "seg2342"}, "140": {"seg1615", "seg281"}}
CROSS_REGION_TYPES = ["speeding", "reckless_driver", "hard_braking", "lane_weaving", "lane_straddling", "collision", "overweight_vehicle"]
BG_CYCLE = ["normal_medium_traffic", "normal_light_traffic", "normal_heavy_traffic", "normal_medium_das_only", "normal_medium_das_cam_only", "normal_medium_sparse"]
NORMAL_TYPES = ["normal_light_traffic", "normal_medium_traffic", "normal_heavy_traffic", "normal_heavy_vehicles", "normal_stop_and_go", "normal_medium_das_only"]
TG_COMBOS = [(6.0, 0.20, 8.0), (6.0, 0.30, 8.0), (6.0, 0.20, 15.0), (6.0, 0.30, 15.0)] * 2 + [(9.0, 0.20, 8.0), (9.0, 0.30, 8.0)] * 2
SEED0 = 20261010
STALL_ABNORMAL_MIN_S = 10.0           # project rule (cross-region and stall sets): a stop of >= 10.0 s is a stall; a legitimate stop is 1.0 <= stop < 10.0 s


def base_path(r, typ):
    return SRC / r / "simulations" / typ / f"region_{r}_{typ}.sim.json"


def seg_of_lane(l):
    return l.split("_")[0]


def regen_background(sc, typ, seed, exclude_lanes):
    """Corpus generator logic, unchanged per vehicle (pv.build_vehicle, pv.SCENARIO_SPECS); only the lane draw skips injection start lanes."""
    random.seed(seed); spec = pv.SCENARIO_SPECS[typ]; lanes, segs = sc["lanes"], sc["segments"]
    ok = [l for l in lanes if l not in exclude_lanes]; n = random.randint(spec["n_min"], spec["n_max"]); veh = {}
    for i in range(1, n + 1):
        lid = random.choice(ok); poly = lanes[lid].get("polyline", [[0, 0], [1, 0]])
        vlim = float(segs.get(lanes[lid].get("segment_id", ""), {}).get("speed_limit_mps") or pv.DEFAULT_SPEED_LIMIT_MPS)
        veh[f"veh{i:03d}"] = pv.build_vehicle(f"veh{i:03d}", lid, poly, vlim, spec)
    sc["vehicles"] = veh
    return sc


# ------------------------------------------------------------------ plan (geometry only)
def region_geometry(r):
    p = base_path(r, "normal_medium_traffic"); w, meta, L, seg_of, lim, occ = st.lane_graph(p); exits = set(w.boundary_exit_nodes or set())
    return w, meta, L, seg_of, lim, exits


def plan():
    rng = np.random.default_rng(20261011); rows = []; used = {r: set() for r in TEST}; chk = []
    G = {r: region_geometry(r) for r in TEST}

    def free_lanes(r, minroute, minlane=25.0, lim_min=0.0):
        w, meta, L, seg_of, lim, exits = G[r]; tg.EXITS.clear(); tg.EXITS.update(exits); out = []
        for s in sorted(L):
            if seg_of_lane(s) in PRIOR_SEGS[r] or L[s] < minlane:
                continue
            route, tot = tg.best_route(meta, L, seg_of, s, cap=900.0, max_depth=20, lim=lim if lim_min else None, min_lim=lim_min)
            if lim_min and lim[seg_of[s]] < lim_min:
                continue
            if tot >= minroute:
                out.append((s, route, tot, min(lim[seg_of[x]] for x in route)))
        return out

    def regions_for(n):
        base = list(TEST); extra = list(rng.choice(TEST, n - len(TEST), replace=False)) if n > len(TEST) else []
        return [base[i] for i in rng.permutation(len(base))][:n] + extra

    k = 0
    # --- 7 cross-region types
    for typ in CROSS_REGION_TYPES:
        for j, r in enumerate(regions_for(12)):
            c = [x for x in free_lanes(r, 150.0) if x[0] not in used[r]] or free_lanes(r, 150.0)
            s, route, tot, mlim = c[int(rng.integers(len(c)))]; used[r].add(s)
            rows.append({"run_id": f"ft_{typ}_{j:02d}_{r}", "block": "core", "type": typ, "mode": typ, "region": r, "bg_type": BG_CYCLE[j % 6],
                         "start_lane": s, "route": ";".join(route), "route_m": round(tot, 1), "route_min_limit": mlim}); k += 1
    # --- stop slots (ST01 candidate rule) for stalled (12) and the stop study (12)
    cand = []
    for r in TEST:
        C = pd.DataFrame(st.candidates(r)); C = C[~C.start_lane.map(seg_of_lane).isin(PRIOR_SEGS[r])]
        cand.append(C)
    C = pd.concat(cand, ignore_index=True)
    slots = []
    for r in [TEST[i] for i in rng.permutation(8)] * 6:          # stop slots are distinct among themselves (they may share a start lane with a cross-region-type run)
        c = C[(C.region == r) & ~C.start_lane.isin({s["start_lane"] for s in slots if s["region"] == r})]
        if len(c):
            c = c.sort_values(["post_m", "start_lane", "stop_lane", "stop_s"], ascending=[False, True, True, True]).iloc[0]; slots.append(c.to_dict())
        if len(slots) == 24:
            break
    assert len(slots) == 24, len(slots)
    modes = ["START", "STOP_END", "MSM", "GENTLE_STALL"] * 3
    for j, c in enumerate(slots[:12]):
        m = modes[j]; used[c["region"]].add(c["start_lane"])
        rows.append({"run_id": f"ft_stalled_{j:02d}_{c['region']}_{m}", "block": "core", "type": "stalled_vehicle", "mode": m, "region": c["region"], "bg_type": BG_CYCLE[j % 6],
                     **{x: c[x] for x in ("start_lane", "stop_lane", "stop_s", "stop_pos_route_m", "drive_m", "post_m", "v_drive", "sensing", "route")},
                     "legit_hold_s": 15.0 if m == "GENTLE_STALL" else np.nan})
    for j, c in enumerate(slots[12:]):
        used[c["region"]].add(c["start_lane"])
        for m, hold in (("CONTROL", np.nan), ("LEGIT_STOP", 3.0), ("LEGIT_STOP", 5.0), ("LEGIT_STOP", 8.0), ("GENTLE_STALL", 15.0)):
            rows.append({"run_id": f"ft_stopstudy_{j:02d}_{c['region']}_{m}{'' if hold != hold else int(hold)}", "block": "stop_study", "type": "stop_study", "mode": m,
                         "region": c["region"], "bg_type": BG_CYCLE[j % 6], **{x: c[x] for x in ("start_lane", "stop_lane", "stop_s", "stop_pos_route_m", "drive_m", "post_m", "v_drive", "sensing", "route")},
                         "legit_hold_s": hold, "slot": j})
    # --- tailgating (12) + matched controls: shared planned route, geometry verified now
    tg_used = {r: set() for r in TEST}
    for j, (vl, tf, d) in enumerate(TG_COMBOS):
        need = tg.LEAD_S + vl * (d + tg.APPROACH_S + tg.TAIL_S); lim_min = vl / 0.85 if vl > 6.0 else 0.0
        opts = []
        for r in TEST:
            for s, route, tot, mlim in free_lanes(r, 1.10 * need, minlane=tg.LEAD_S + 5.0, lim_min=lim_min):
                if s not in tg_used[r] and mlim * 0.85 >= vl - 1e-9:      # distinct among tailgating runs (may share a start lane with another block's run)
                    opts.append((r, s, route, tot, mlim))
        # spread over regions: prefer the region with the fewest tailgating runs so far
        cnt = pd.Series([x["region"] for x in rows if x["type"] == "tailgating"] + [None]).value_counts()
        opts.sort(key=lambda o: (cnt.get(o[0], 0), o[0], o[1])); r, s, route, tot, mlim = opts[0]; tg_used[r].add(s)
        # the shortest prefix covering the requirement is what the hook uses (tg14 build logic)
        for kind in ("tailgating", "control"):
            rows.append({"run_id": f"ft_tailgating_{j:02d}_{r}_{'tg' if kind == 'tailgating' else 'ctl'}", "block": "core" if kind == "tailgating" else "tg_control",
                         "type": "tailgating", "mode": kind, "region": r, "bg_type": BG_CYCLE[j % 6], "start_lane": s, "route": ";".join(route), "route_m": round(tot, 1),
                         "route_min_limit": mlim, "v_lead": vl, "T_f": tf if kind == "tailgating" else 0.0, "dur_target_s": d, "route_m_needed": round(need, 1)})
        chk.append({"tg_run": j, "region": r, "start_lane": s, "start_lane_m": round(G[r][2][s], 1), "v_lead": vl, "T_f": tf, "D_s": d, "route_needed_m": round(need, 1),
                    "route_available_m": round(tot, 1), "margin": round(tot / need, 2), "route_min_limit_mps": mlim, "leader_speed_ok": mlim * 0.85 >= vl,
                    "route_ends_at_exit": G[r][1][route[-1]].get("to") in G[r][5], "n_route_lanes": len(route), "start_lane_free_of_background": True})
    # --- normal scenes
    for r in TEST:
        for typ in NORMAL_TYPES:
            rows.append({"run_id": f"ft_normal_{r}_{typ.replace('normal_', '')}", "block": "normal", "type": "normal", "mode": "normal", "region": r, "bg_type": typ})
    P = pd.DataFrame(rows); P.insert(1, "sim_seed", SEED0 + np.arange(len(P))); P.insert(2, "bg_seed", SEED0 + 100000 + np.arange(len(P)))
    (WORK / "plan_checks").mkdir(parents=True, exist_ok=True); P.to_csv(WORK / "run_plan.csv", index=False)
    pd.DataFrame(chk).to_csv(WORK / "plan_checks/tailgating_geometry.csv", index=False)
    print(P.groupby(["block", "type"]).size().to_string()); print("total runs", len(P)); print(pd.DataFrame(chk).to_string(index=False))


# ------------------------------------------------------------------ build + hooks
class Composite:
    def __init__(self, *hooks):
        self.hooks = [h for h in hooks if h is not None]; self.log = []

    def __call__(self, sim, world, t):
        for h in self.hooks:
            h(sim, world, t)


class RouteOnly:
    def __init__(self, routes):
        self.routes, self.done, self.log = routes, False, []

    def __call__(self, sim, world, t):
        if not self.done:
            for vid, (lanes, dest) in self.routes.items():
                if vid in world.vehicles:
                    sim.set_vehicle_route(vid, lanes, dest); self.log.append({"t": t, "vehicle": vid, "n_lanes": len(lanes)})
            self.done = True


def lane_at(route, L, pos):
    acc = 0.0
    for lid in route:
        if pos <= acc + L[lid]:
            return lid, pos - acc
        acc += L[lid]
    return route[-1], L[route[-1]] - 1.0


def build(row):
    base = base_path(row.region, row.bg_type); sc = json.loads(base.read_text())
    excl = set() if row.block == "normal" else {row.start_lane}
    sc = regen_background(sc, row.bg_type, int(row.bg_seed), excl); meta = {"base_scenario": str(base.relative_to(paths.ROOT)), "base_md5": hashlib.md5(base.read_bytes()).hexdigest()}
    if row.block == "normal":
        return sc, None, {}, meta
    w, gmeta, L, seg_of, lim, exits = region_geometry(row.region); route = row.route.split(";"); dest = gmeta[route[-1]]["to"]
    place = lambda v, lane, s: v.update(lane_id=lane, s=float(s), heading_rad=math.atan2(*(lambda p: (p[1][1] - p[0][1], p[1][0] - p[0][0]))(w.lanes[lane].polyline)),
                                        ax_world_mps2=0.0, ay_world_mps2=0.0, a_long_mps2=0.0)
    v = sc["vehicles"]["veh001"]; _clear_overrides(v); limit = float(lim[seg_of[row.start_lane]]); inj = {"speed_limit_mps": limit}
    if row.type in CROSS_REGION_TYPES:
        s0 = min(20.0, 0.5 * L[row.start_lane]); place(v, row.start_lane, s0); sched = []; t = row.type
        if t == "speeding":
            _set_speed_profile(v, 1.4 * limit); v["ignore_speed_limit"] = True
        elif t == "reckless_driver":
            _set_speed_profile(v, 1.4 * limit); v.update(ignore_speed_limit=True, allow_collision=True, lateral_mode="weave", lateral_weave_amplitude_m=1.5,
                                                         lateral_weave_period_s=WEAVE_PERIOD_S, lateral_weave_phase_rad=0.0, time_headway_s=0.6, min_gap_m=1.5)
        elif t == "hard_braking":
            sched = [{"t": cross_region.TRIG, "vehicle": "veh001", "action": "brake", "decel_mps2": 6.0}]
        elif t == "lane_weaving":
            v.update(lateral_mode="weave", lateral_weave_amplitude_m=1.5, lateral_weave_period_s=WEAVE_PERIOD_S, lateral_weave_phase_rad=0.0)
        elif t == "lane_straddling":
            v.update(lateral_mode="straddle", lateral_fixed_offset_m=-1.8, lateral_offset_m=-1.8)
        elif t == "collision":
            tw = sc["vehicles"]["veh002"]; _clear_overrides(tw); tot = sum(L[x] for x in route); tpos = s0 + min(130.0, tot - s0 - 15.0)
            tl, ts = lane_at(route, L, tpos); place(tw, tl, ts); tw.update(frozen=True, v=0.0, target_speed_mps=0.0)
            _set_speed_profile(v, 8.0); v.update(allow_collision=True, ignore_speed_limit=True, disable_idm=True); inj.update(target="veh002", target_lane=tl, target_s=ts)
        elif t == "overweight_vehicle":
            v["weight_kg"] = 30000.0
        hook = Composite(RouteOnly({"veh001": (route, dest)}), AnomalyDirector(sched) if sched else None)
        return sc, hook, {"route": route, "schedule": sched, **inj}, meta
    if row.type in ("stalled_vehicle", "stop_study"):
        r2 = row._replace(mode="LEGIT_STOP") if row.mode == "GENTLE_STALL" else row
        _set_speed_profile(v, float(row.v_drive)); off, acc = {}, 0.0
        for lid in route:
            off[lid] = acc; acc += L[lid]
        if row.mode == "START":
            place(v, row.stop_lane, float(row.stop_s)); v.update(frozen=True, v=0.0)
        else:
            place(v, row.start_lane, st.S0)
        return sc, st.StallHook(r2, route, dest, off), {"route": route}, meta
    if row.type == "tailgating":
        need = tg.LEAD_S + row.v_lead * (row.dur_target_s + tg.APPROACH_S + tg.TAIL_S); rr, acc = [], 0.0
        for lid in route:
            rr.append(lid); acc += L[lid]
            if acc >= need:
                break
        d2 = gmeta[rr[-1]]["to"]
        for vid, s, speed in (("veh002", tg.LEAD_S, row.v_lead), ("veh001", tg.FOLL_S, row.v_lead)):
            x = sc["vehicles"][vid]; _clear_overrides(x); place(x, row.start_lane, s); _set_speed_profile(x, speed)
        f = sc["vehicles"]["veh001"]; _set_speed_profile(f, 2.5 * row.v_lead); f["v"] = row.v_lead; f["ignore_speed_limit"] = True
        if row.mode == "tailgating":
            f["time_headway_s"] = float(row.T_f); f["min_gap_m"] = tg.S0_F
        # the leader start lane is free of other background vehicles by construction (regen_background excludes it)
        for vid in list(sc["vehicles"]):
            if vid not in ("veh001", "veh002") and sc["vehicles"][vid]["lane_id"] == row.start_lane:
                raise RuntimeError("background on tailgating start lane")
        return sc, tg.RouteHook({"veh001": (rr, d2), "veh002": (rr, d2)}, tg.APPROACH_S + row.dur_target_s, row.v_lead), {"route": rr}, meta
    raise ValueError(row.type)


def bg_status(sc, truth, skip):
    out = {}
    for vid in sc["vehicles"]:
        if vid in skip:
            continue
        r = realised_severity(sc, [dict(x) for x in truth], vid, [], trigger_t=None)
        ok = r.get("n_rows", 0) > 0 and r.get("stop_longest_s", 0) < 2.0 and not (r.get("gap_min_m", 99) < 5.0)
        out[vid] = {"unaffected_normal": bool(ok), "stop_longest_s": r.get("stop_longest_s"), "gap_min_m": r.get("gap_min_m")}
    return out


def realization(row, sc, truth, coll, inj):
    if row.block == "normal":
        return {"verified": True, "rule": "normal scene: every vehicle is a normal vehicle (corpus definition)"}
    if row.type in CROSS_REGION_TYPES:
        real = cross_region.realization(sc, truth, coll, row.type); real["background"] = bg_status(sc, truth, {"veh001"} | ({"veh002"} if row.type == "collision" else set()))
        return real
    if row.type in ("stalled_vehicle", "stop_study"):
        if row.mode == "GENTLE_STALL":
            real = sd.realization(row, sc, truth, coll)
        else:
            real = st.realization(row, sc, truth, coll)
        real["stall_abnormal_min_s"] = STALL_ABNORMAL_MIN_S; real["background"] = bg_status(sc, truth, {"veh001"}); return real
    if row.type == "tailgating":
        real, R = tg.label(sc, truth, coll, inj["route"]); real["background"] = bg_status(sc, truth, {"veh001", "veh002"}); return real


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "plan":
        plan()
    elif cmd == "run":
        B = Path(sys.argv[2]); ids = set(sys.argv[3:]); P = pd.read_csv(PLANS / "run_plan.csv", dtype={"region": str})
        for row in P.itertuples():
            if row.run_id not in ids:
                continue
            d = B / row.run_id
            if d.exists():
                print(row.run_id, "exists - not rerun"); continue
            d.mkdir(parents=True); sc, hook, inj, meta = build(row)
            sj = d / f"{row.run_id}.sim.json"; sj.write_text(json.dumps(sc, indent=1)); t0 = time.time()
            res = run_scenario(sj, d, seed=int(row.sim_seed), hook=hook, record_truth=True)
            truth = res.pop("truth"); pd.DataFrame(truth).to_parquet(d / "truth_trail.parquet", index=False)
            real = realization(row, sc, truth, res["collisions"], inj)
            info = {"run_id": row.run_id, "plan_row": {k: (v if v == v else None) for k, v in row._asdict().items()}, "scenario_md5": hashlib.md5(sj.read_bytes()).hexdigest(),
                    **meta, "injection": {k: v for k, v in inj.items()}, "wall_s": round(time.time() - t0, 1), "result": res,
                    "hook_log": getattr(hook, "log", None)}
            (d / "run.json").write_text(json.dumps(info, indent=1, default=str)); (d / "realization.json").write_text(json.dumps(real, indent=1, default=str))
            print(row.run_id, res["stop_reason"], round(res["sim_time_s"], 1), "verified", real.get("verified"), real.get("control_clean"), flush=True)
