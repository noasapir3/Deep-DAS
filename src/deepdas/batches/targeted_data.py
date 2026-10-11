"""
Targeted-data study (research, not adopted): 314 runs in train / val_A regions with diverse severities of the six weak anomaly types
(hard braking, speeding, overweight, lane straddling, collision, lane weaving), partial (segment) events, normal controls (firm legal
braking, fast legal driving, legal heavy vehicles, normal lateral drift) and measurement-only validation runs (reckless driving,
tailgating with matched controls).
  plan                       -> outputs/generated/targeted_data_study/run_plan.csv (geometry only; deterministic; reads the route survey)
  run <out_dir> <run_id ...>  -> simulate shipped-plan runs (refuses to rerun an existing folder)
Unchanged simulator, runner, tracker and sensor scenarios. Injection = the mechanisms of the final-test builder (reckless driving and
tailgating) and, for the six weak types, the same vehicle override fields plus a per-step hook (WHook) that only sets fields the simulator
already consumes: a_cmd_override_mps2/until (scripted braking), the speed-profile fields + ignore_speed_limit re-applied with
sim.apply_vehicle_macro_profile (segment speeding), and the native lateral_window_start_t/until_t (segment straddling / weaving).
Realisation uses the simulator truth only (labels); the cross-region thresholds are unchanged; for partial events they are computed inside
the event window written in the plan before simulation.
"""
import copy, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from .. import paths  # noqa: E402
from . import final_test  # noqa: E402
tg = final_test.tg; cross_region = final_test.cross_region
PLANS = paths.DATA / "simulation_plans" / "targeted_data_study"; WORK = paths.GENERATED / "targeted_data_study"
from deepdas.sim.runner import run_scenario  # noqa: E402
from deepdas.sim.scenario_builder import _set_speed_profile, _clear_overrides  # noqa: E402
from deepdas.sim.severity import realised_severity, _lane_polylines, _project  # noqa: E402

BG_CYCLE = final_test.BG_CYCLE
SENS = ["normal_medium_das_only", "normal_medium_das_cam_only", "normal_medium_sparse"]
SEED_SIM, SEED_BG = 31000000, 32000000          # disjoint from every earlier batch (cross-region set: 1; final test: 20261010+i / 20361010+i)
S0 = 20.0


# ------------------------------------------------------------------ hook
class WHook:
    """Per-step hook. events: list of dicts sorted by t:  brake {decel, drop_frac or None}, speed_on {target}, speed_off."""
    def __init__(self, routes, events):
        self.routes, self.events, self.done_routes, self.log, self.saved = routes, sorted(events, key=lambda e: e["t"]), False, [], None

    def __call__(self, sim, world, t):
        if not self.done_routes:
            for vid, (lanes, dest) in self.routes.items():
                if vid in world.vehicles:
                    sim.set_vehicle_route(vid, lanes, dest); self.log.append({"t": t, "route": vid, "n_lanes": len(lanes)})
            self.done_routes = True
        while self.events and t >= self.events[0]["t"]:
            e = self.events.pop(0); v = world.vehicles.get("veh001")
            if v is None:
                self.log.append({"t": t, "action": e["action"], "status": "vehicle_absent"}); continue
            if e["action"] == "brake":
                v0 = max(0.0, float(v.v)); a = abs(float(e["decel"]))
                dv = v0 * float(e["drop_frac"]) if e.get("drop_frac") else v0
                v.a_cmd_override_mps2 = -a; v.a_cmd_override_until_t = t + dv / a + (0.0 if e.get("drop_frac") else 0.2)
                self.log.append({"t": t, "action": "brake", "v0": v0, "decel": a, "drop_frac": e.get("drop_frac"), "until_t": v.a_cmd_override_until_t})
            elif e["action"] == "speed_on":
                self.saved = {k: getattr(v, k) for k in ("speed_min_mps", "speed_max_mps", "speed_mean_mps", "speed_std_mps", "target_speed_mps", "ignore_speed_limit")}
                x = float(e["target"]); v.speed_min_mps = max(1.5, x - 0.5); v.speed_max_mps = x + 0.5; v.speed_mean_mps = x; v.speed_std_mps = 0.5
                v.target_speed_mps = x; v.ignore_speed_limit = True; sim.apply_vehicle_macro_profile("veh001")
                self.log.append({"t": t, "action": "speed_on", "target": x, "v_now": float(v.v)})
            elif e["action"] == "speed_off":
                for k, val in (self.saved or {}).items():
                    setattr(v, k, val)
                sim.apply_vehicle_macro_profile("veh001"); self.log.append({"t": t, "action": "speed_off", "v_now": float(v.v)})


# ------------------------------------------------------------------ build
def place(w, v, lane, s):
    p = w.lanes[lane].polyline
    v.update(lane_id=lane, s=float(s), heading_rad=math.atan2(p[1][1] - p[0][1], p[1][0] - p[0][0]), ax_world_mps2=0.0, ay_world_mps2=0.0, a_long_mps2=0.0)


def build(row):
    """-> scenario dict, hook, injection info, meta. Reckless / tailgating rows use the unchanged final-test build."""
    if row.type in ("reckless_driver", "tailgating"):
        return final_test.build(row)
    base = final_test.base_path(row.region, row.bg_type); sc = json.loads(base.read_text())
    sc = final_test.regen_background(sc, row.bg_type, int(row.bg_seed), {row.start_lane})
    meta = {"base_scenario": str(base.relative_to(paths.ROOT)), "base_md5": hashlib.md5(base.read_bytes()).hexdigest()}
    w, gmeta, L, seg_of, lim, exits = final_test.region_geometry(row.region); route = row.route.split(";"); dest = gmeta[route[-1]]["to"]
    v = sc["vehicles"]["veh001"]; _clear_overrides(v); limit = float(lim[seg_of[row.start_lane]]); s0 = min(S0, 0.5 * L[row.start_lane])
    place(w, v, row.start_lane, s0); ev = []; routes = {"veh001": (route, dest)}; inj = {"speed_limit_mps": limit, "route": route}
    t, x = row.type, row
    win = (float(x.win_start), float(x.win_end)) if x.extent == "segment" else None
    if t in ("hard_braking", "ctl_firm_braking"):
        ev = [{"t": float(x.t_event), "action": "brake", "decel": float(x.decel), "drop_frac": None if x.shape == "stop" else float(x.drop_frac)}]
    elif t in ("speeding", "ctl_fast_legal"):
        tgt = float(x.speed_ratio) * limit
        if t == "ctl_fast_legal":
            _set_speed_profile(v, float(x.speed_ratio) * float(x.route_min_limit))   # legal: relative to the LOWEST limit on the route; no ignore_speed_limit
        elif win is None:
            _set_speed_profile(v, tgt); v["ignore_speed_limit"] = True
        else:
            ev = [{"t": win[0], "action": "speed_on", "target": tgt}, {"t": win[1], "action": "speed_off"}]
    elif t in ("overweight_vehicle", "ctl_heavy_legal"):
        v["weight_kg"] = float(x.weight_kg); _set_speed_profile(v, float(x.speed_ratio) * limit)
    elif t in ("lane_straddling", "ctl_lateral"):
        if t == "ctl_lateral" and x.lat_mode == "weave":
            v.update(lateral_mode="weave", lateral_weave_amplitude_m=float(x.amplitude_m), lateral_weave_period_s=float(x.period_s), lateral_weave_phase_rad=0.0)
        else:
            v.update(lateral_mode="straddle", lateral_fixed_offset_m=float(x.offset_m), lateral_offset_m=float(x.offset_m) if win is None else 0.0)
        if win is not None:
            v.update(lateral_window_start_t=win[0], lateral_window_until_t=win[1])
    elif t == "lane_weaving":
        v.update(lateral_mode="weave", lateral_weave_amplitude_m=float(x.amplitude_m), lateral_weave_period_s=float(x.period_s), lateral_weave_phase_rad=0.0)
        if win is not None:
            v.update(lateral_window_start_t=win[0], lateral_window_until_t=win[1])
    elif t == "collision":
        tw = sc["vehicles"]["veh002"]; _clear_overrides(tw); tpos = s0 + float(x.target_dist_m); tl, ts = final_test.lane_at(route, L, tpos); place(w, tw, tl, ts)
        vt = float(x.target_speed)
        if vt <= 0:
            tw.update(frozen=True, v=0.0, target_speed_mps=0.0)
        else:
            _set_speed_profile(tw, vt); tw["speed_min_mps"] = max(0.5, vt - 0.3); routes["veh002"] = (route[route.index(tl):], dest)
        _set_speed_profile(v, vt + float(x.impact_dv)); v.update(allow_collision=True, ignore_speed_limit=True, disable_idm=True)
        inj.update(target="veh002", target_lane=tl, target_s=ts, target_speed=vt)
    else:
        raise ValueError(t)
    return sc, WHook(routes, ev), inj, meta


# ------------------------------------------------------------------ realisation (truth = labels only)
def stats(sc, truth, coll, win=None, trig=None):
    tr = [dict(r) for r in truth if win is None or win[0] <= r["t"] <= win[1]]
    rs = realised_severity(sc, tr, "veh001", [c for c in coll if "veh001" in c.get("vehicle_ids", [])], trigger_t=trig)
    lanes = _lane_polylines(sc); rows = [r for r in tr if r["vehicle_id"] == "veh001"]
    dl = np.array([_project(lanes, r)[1] for r in rows], float)
    rs["d_lat_abs_median"] = float(np.nanmedian(np.abs(dl))) if len(dl) and np.isfinite(dl).any() else float("nan")
    return rs


RULE = {"speeding": lambda s: s.get("speed_ratio_p90", 0) >= 1.10, "hard_braking": lambda s: s.get("decel_peak_mps2", 0) >= 4.0,
        "lane_weaving": lambda s: s.get("lateral_std_m", 0) >= 1.0, "lane_straddling": lambda s: s.get("d_lat_abs_median", 0) >= 1.7,
        "collision": lambda s: s.get("n_collisions", 0) >= 1, "overweight_vehicle": lambda s: s.get("weight_kg", 0) >= 20000}


CTL_PROP = {"ctl_firm_braking": lambda s: s.get("decel_peak_mps2", 9) < 4.0, "ctl_fast_legal": lambda s: s.get("speed_ratio_p90", 9) < 1.10,
            "ctl_heavy_legal": lambda s: s.get("weight_kg", 9e9) < 20000, "ctl_lateral": lambda s: s.get("d_lat_abs_median", 9) < 1.7 and s.get("lateral_std_m", 9) < 1.0}


def control_clean(typ, s):
    """A control is clean if its targeted property stays on the legal side of the matching anomaly rule over the whole truth track, and it has
    no stop >= 1 s and no collision. (Its other behaviour is normal driving, as for every background vehicle — e.g. normal vehicles reach a
    truth speed ratio of up to the simulator's 1.12x cap; that is not checked for non-speed controls, exactly as for backgrounds.)"""
    return bool(s.get("n_rows", 0) > 0 and CTL_PROP[typ](s) and s.get("stop_longest_s", 9) < 1.0 and s.get("n_collisions", 0) == 0)


def realization(row, sc, truth, coll, inj):
    if row.type in ("reckless_driver", "tailgating"):
        return final_test.realization(row, sc, truth, coll, inj)
    whole = stats(sc, truth, coll)
    out = {"whole": whole, "extent": row.extent}
    if row.type.startswith("ctl_"):
        out["verified"] = control_clean(row.type, whole); out["rule"] = "control: targeted property legal over the whole truth track (see control_clean); no stop >= 1 s; no collision"
    else:
        if row.extent == "segment":
            ws = stats(sc, truth, coll, win=(float(row.win_start), float(row.win_end))); out["window"] = ws; use = ws
            out["rule"] = f"p10 threshold inside the planned event window [{row.win_start}, {row.win_end}] s"
        elif row.type == "hard_braking":
            use = stats(sc, truth, coll, trig=float(row.t_event)); out["window"] = use; out["rule"] = "peak decel >= 4 after the planned brake time"
        else:
            use = whole; out["rule"] = "p10 threshold over the whole truth track"
        out["verified"] = bool(RULE[row.type](use))
    out["background"] = final_test.bg_status(sc, truth, {"veh001"} | ({"veh002"} if row.type == "collision" else set()))
    return out


# ------------------------------------------------------------------ plan
def plan():
    rng = np.random.default_rng(20261012); R = pd.read_csv(PLANS / "route_survey.csv", dtype={"region": str})
    used = {}; rows = []; cnt = {}

    def pick(split, need_m, lim_min=0.0):
        c = R[(R.split == split) & (R.route_m >= need_m) & (R.route_min_limit >= lim_min)]
        c = c[[s not in used.get(r, set()) for r, s in zip(c.region, c.start_lane)]]
        assert len(c), (split, need_m)
        c = c.assign(n=c.region.map(lambda r: cnt.get(r, 0)), u=rng.random(len(c))).sort_values(["n", "u"]).iloc[0]
        used.setdefault(c.region, set()).add(c.start_lane); cnt[c.region] = cnt.get(c.region, 0) + 1
        return c

    def add(split, typ, i, need_fn, **kw):
        lim0 = kw.pop("lim_min", 0.0); c = pick(split, need_fn(14.0), lim0)        # need computed with a 14 m/s reference (conservative: max limit 13.9)
        rows.append({"run_id": f"w_{'tr' if split == 'train' else 'va'}_{typ}_{i:02d}_{c.region}", "split": "train" if split == "train" else "val",
                     "block": "control" if typ.startswith("ctl_") else "anomaly", "type": typ, "region": c.region, "start_lane": c.start_lane,
                     "route": c.route, "route_m": c.route_m, "route_min_limit": c.route_min_limit, "extent": "whole", **kw})

    for split, n_an in (("train", 24), ("val_A", 8)):
        # hard braking: decel x shape x event time (partial drop fractions cycle)
        g = [(d, sh, te) for te in (6.0, 12.0, 20.0) for d in (5.0, 6.0, 7.0, 8.0) for sh in ("partial", "stop")]
        g = g if n_an == 24 else [g[i] for i in (0, 3, 5, 6, 9, 12, 15, 22)]
        for i, (d, sh, te) in enumerate(g):
            add(split, "hard_braking", i, lambda v, te=te: S0 + v * (te + 15.0), decel=d, shape=sh, drop_frac=(0.4, 0.55, 0.7)[i % 3] if sh == "partial" else np.nan,
                t_event=te, extent="event", win_start=te, win_end=te + 6.0, bg_type=BG_CYCLE[i % 6])
        # speeding: ratio x extent
        for i in range(n_an):
            r = (1.30, 1.45, 1.60, 1.80)[i % 4]; seg = (i // 4) % 2 == 1; D = (15.0, 20.0)[(i // 8) % 2]
            if seg:
                add(split, "speeding", i, lambda v, r=r, D=D: S0 + v * 15.0 + r * v * D + 30.0, speed_ratio=r, extent="segment", win_start=3.0, win_end=3.0 + D, bg_type=BG_CYCLE[i % 6])
            else:
                add(split, "speeding", i, lambda v: 150.0, speed_ratio=r, bg_type=BG_CYCLE[i % 6])
        # overweight: weight x sensing x speed
        for i in range(n_an):
            add(split, "overweight_vehicle", i, lambda v: 150.0, weight_kg=(20000.0, 25000.0, 30000.0, 40000.0)[i % 4], bg_type=SENS[(i // 4) % 3],
                speed_ratio=(0.7, 0.95)[(i // 12 + i) % 2])
        # straddling: offset (alternating side) x extent
        for i in range(n_an):
            o = (1.8, 1.9, 2.0, 2.1)[i % 4] * (1 if (i // 4) % 2 == 0 else -1); seg = (i // 2) % 2 == 1
            if seg:
                add(split, "lane_straddling", i, lambda v: S0 + v * 38.0, offset_m=o, extent="segment", win_start=3.0, win_end=23.0, bg_type=BG_CYCLE[i % 6])
            else:
                add(split, "lane_straddling", i, lambda v: 150.0, offset_m=o, bg_type=BG_CYCLE[i % 6])
        # weaving: amplitude x period x extent
        combos = [(a, p) for a in (1.5, 1.8, 2.2) for p in (2.0, 3.0, 4.5)]
        for i in range(n_an):
            a, p = combos[i % 9]; seg = i % 2 == 1
            if seg:
                add(split, "lane_weaving", i, lambda v: S0 + v * 38.0, amplitude_m=a, period_s=p, extent="segment", win_start=3.0, win_end=23.0, bg_type=BG_CYCLE[i % 6])
            else:
                add(split, "lane_weaving", i, lambda v: 150.0, amplitude_m=a, period_s=p, bg_type=BG_CYCLE[i % 6])
        # collision: impact speed x target state x distance
        for i in range(n_an):
            dv = (5.0, 8.0, 12.0)[i % 3]; vt = (0.0, 2.0, 3.0, 4.0)[(i // 3) % 4]; dist = (60.0, 85.0, 110.0, 130.0)[(i // 2) % 4]
            add(split, "collision", i, lambda v, dv=dv, vt=vt, dist=dist: S0 + dist + vt * dist / dv + 40.0, impact_dv=dv, target_speed=vt, target_dist_m=dist,
                extent="event", bg_type=BG_CYCLE[i % 6])
    # controls
    for split, n in (("train", (24, 16, 16, 16)), ("val_A", (8, 6, 6, 6))):
        for i in range(n[0]):
            te = (6.0, 12.0, 20.0)[i % 3]
            add(split, "ctl_firm_braking", i, lambda v, te=te: S0 + v * (te + 15.0), decel=(2.5, 3.0, 3.5)[(i // 3) % 3], shape="partial", drop_frac=(0.4, 0.55, 0.7)[i % 3],
                t_event=te, extent="event", win_start=te, win_end=te + 6.0, bg_type=BG_CYCLE[i % 6])
        for i in range(n[1]):
            add(split, "ctl_fast_legal", i, lambda v: 150.0, speed_ratio=(0.90, 0.95, 1.0)[i % 3], bg_type=BG_CYCLE[i % 6])
        for i in range(n[2]):
            add(split, "ctl_heavy_legal", i, lambda v: 150.0, weight_kg=(8000.0, 10000.0, 12000.0, 15000.0)[i % 4], bg_type=SENS[i % 3], speed_ratio=(0.7, 0.95)[(i // 4) % 2])
        for i in range(n[3]):
            if i % 2 == 0:
                add(split, "ctl_lateral", i, lambda v: 150.0, lat_mode="offset", offset_m=(0.5, 0.7, 0.9)[(i // 2) % 3] * (1 if i % 4 == 0 else -1), bg_type=BG_CYCLE[i % 6])
            else:
                add(split, "ctl_lateral", i, lambda v: 150.0, lat_mode="weave", amplitude_m=(0.4, 0.5, 0.6)[(i // 2) % 3], period_s=(6.0, 8.0, 10.0)[(i // 2) % 3], bg_type=BG_CYCLE[i % 6])
    # measurement-only validation: reckless (cross-region design) and tailgating pairs (final-test mechanism, geometry verified like the final test)
    for i in range(8):
        add("val_A", "reckless_driver", i, lambda v: 150.0, bg_type=BG_CYCLE[i % 6])
    for j, (vl, tf, d) in enumerate(final_test.TG_COMBOS[:8]):
        need = tg.LEAD_S + vl * (d + tg.APPROACH_S + tg.TAIL_S); lim_min = vl / 0.85 if vl > 6.0 else 0.0
        c = pick("val_A", 1.10 * need, lim_min)
        for kind in ("tailgating", "control"):
            rows.append({"run_id": f"w_va_tailgating_{j:02d}_{c.region}_{'tg' if kind == 'tailgating' else 'ctl'}", "split": "val", "block": "anomaly" if kind == "tailgating" else "tg_control",
                         "type": "tailgating", "mode": kind, "region": c.region, "start_lane": c.start_lane, "route": c.route, "route_m": c.route_m,
                         "route_min_limit": c.route_min_limit, "extent": "whole", "v_lead": vl, "T_f": tf if kind == "tailgating" else 0.0, "dur_target_s": d,
                         "route_m_needed": round(need, 1), "bg_type": BG_CYCLE[j % 6]})
    P = pd.DataFrame(rows); P.insert(1, "sim_seed", SEED_SIM + np.arange(len(P))); P.insert(2, "bg_seed", SEED_BG + np.arange(len(P)))
    P["mode"] = P["mode"].fillna(P.type) if "mode" in P else P.type
    assert P.run_id.is_unique and len(P) == 314, len(P)
    assert not set(P[P.split == "train"].region) & set(P[P.split == "val"].region)
    WORK.mkdir(parents=True, exist_ok=True); P.to_csv(WORK / "run_plan.csv", index=False)
    print(P.groupby(["split", "block", "type"]).size().to_string()); print("runs", len(P), "| train regions", P[P.split == "train"].region.nunique(), "| val regions", P[P.split == "val"].region.nunique())


def run_rows(P, out):
    for row in P.itertuples():
        d = out / row.run_id
        if d.exists():
            print(row.run_id, "exists - not rerun", flush=True); continue
        d.mkdir(parents=True); sc, hook, inj, meta = build(row)
        sj = d / f"{row.run_id}.sim.json"; sj.write_text(json.dumps(sc, indent=1)); t0 = time.time()
        res = run_scenario(sj, d, seed=int(row.sim_seed), hook=hook, record_truth=True)
        truth = res.pop("truth"); pd.DataFrame(truth).to_parquet(d / "truth_trail.parquet", index=False)
        real = realization(row, sc, truth, res["collisions"], inj)
        info = {"run_id": row.run_id, "plan_row": {k: (v if v == v else None) for k, v in row._asdict().items()}, "scenario_md5": hashlib.md5(sj.read_bytes()).hexdigest(),
                **meta, "injection": {k: v for k, v in inj.items()}, "wall_s": round(time.time() - t0, 1), "result": res, "hook_log": getattr(hook, "log", None)}
        (d / "run.json").write_text(json.dumps(info, indent=1, default=str)); (d / "realization.json").write_text(json.dumps(real, indent=1, default=str))
        print(row.run_id, res["stop_reason"], round(res["sim_time_s"], 1), "verified", real.get("verified"), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "plan":
        plan()
    elif cmd == "run":
        P = pd.read_csv(PLANS / "run_plan.csv", dtype={"region": str}); ids = set(sys.argv[3:])
        run_rows(P[P.run_id.isin(ids)], Path(sys.argv[2]))
