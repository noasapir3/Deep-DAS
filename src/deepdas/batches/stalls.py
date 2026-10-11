"""
Stall and stop set: stalled vehicles (START: stopped from the first sample; STOP_END: drives, then stops until the end; MSM: drives,
stops 30 s, drives on), legitimate stops (brake at 2 m/s², hold < 10 s, drive on) and matched controls, on DAS stop slots of planned
routes in training regions, plus a stall validation group in val_A regions. Unchanged simulator, runner and tracker; injection only through
vehicle fields the simulator already reads (frozen, a_cmd_override_*, speed profile) and its public routing API (sim.set_vehicle_route).
Rule: a stop of >= 10.0 s is a stall; a legitimate stop is 1.0 <= stop < 10.0 s.
Usage:
  python -m deepdas.batches.stalls survey                      -> outputs/generated/stalls_and_stops/survey_*.csv   (no simulation)
  python -m deepdas.batches.stalls plan                        -> outputs/generated/stalls_and_stops/run_plan.csv  (deterministic)
  python -m deepdas.batches.stalls run <out_dir> <run_id ...>  -> one folder per run, using the shipped plan (refuses to rerun)
"""
import copy, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd

from .. import paths  # noqa: E402
PLANS = paths.DATA / "simulation_plans" / "stalls_and_stops"; WORK = paths.GENERATED / "stalls_and_stops"
from deepdas.fp.project_io import load_world  # noqa: E402
from deepdas.fp.bus import EventBus  # noqa: E402
from deepdas.fp.sim_core import Simulation  # noqa: E402
from deepdas.sim.runner import run_scenario  # noqa: E402
from deepdas.sim.scenario_builder import _set_speed_profile, _clear_overrides  # noqa: E402
from deepdas.sim.severity import realised_severity  # noqa: E402

RAW = paths.SCENARIOS
SPLITS = json.loads((paths.PROCESSED / "splits.json").read_text())
USED_TRAIN = {"131", "050", "090", "065"}
for f in ("tailgating_routes/earlier_plans/pilot_plan.csv", "tailgating_routes/earlier_plans/second_pilot_plan.csv", "tailgating_routes/run_plan.csv", "tailgating_matched_controls/run_plan.csv"):
    USED_TRAIN |= set(pd.read_csv(paths.DATA / "simulation_plans" / f, dtype={"region": str}).region)
USED_VAL = {"002", "005"} | {r for r in USED_TRAIN if r in SPLITS["val_A"]}
SEED = 1; S0 = 5.0; DRIVE_MIN_S, DRIVE_MAX_S, POST_MIN_M = 6.0, 14.0, 60.0
MSM_HOLD_S, LEGIT_DECEL = 30.0, 2.0


def scen_path(r):
    return RAW / r / "simulations/normal_medium_traffic" / f"region_{r}_normal_medium_traffic.sim.json"


def lane_graph(path):
    w = load_world(path)
    for seg in w.segments.values():
        seg.lanes = 1
    sim = Simulation(EventBus(), w); sim.rebuild_lanes()
    L = {lid: float(np.hypot(*np.diff(np.array(l.polyline, float), axis=0).T).sum()) for lid, l in w.lanes.items()}
    meta = sim._lane_meta; seg_of = {lid: l.segment_id for lid, l in w.lanes.items()}
    lim = {sid: float(s.speed_limit_mps or 13.9) for sid, s in w.segments.items()}
    occ = {}
    for vid, v in w.vehicles.items():
        occ.setdefault(v.lane_id, []).append(vid)
    return w, meta, L, seg_of, lim, occ


def best_route(meta, L, seg_of, start, exits, cap=450.0, max_depth=14):
    """Longest simple route from `start` (no segment twice, no U-turn, stops at boundary exits) — tg12/tg14 search without a limit filter."""
    out = {}
    for lid, m in meta.items():
        out.setdefault(m.get("from"), []).append(lid)
    best = ([start], L[start])

    def dfs(path, segs, length):
        nonlocal best
        if length > best[1]:
            best = (list(path), length)
        if length >= cap or len(path) >= max_depth:
            return
        m = meta[path[-1]]
        if m.get("to") in exits:
            return
        for nxt in sorted(out.get(m.get("to"), [])):
            if meta[nxt].get("to") == m.get("from") or seg_of[nxt] in segs:
                continue
            path.append(nxt); segs.add(seg_of[nxt]); dfs(path, segs, length + L[nxt]); path.pop(); segs.discard(seg_of[nxt])
            if best[1] >= cap:
                return
    dfs([start], {seg_of[start]}, L[start])
    return best


def point_at(poly, s):
    p = np.array(poly, float); d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    i = min(max(int(np.searchsorted(d, s) - 1), 0), len(p) - 2); f = (s - d[i]) / max(d[i + 1] - d[i], 1e-9)
    return p[i] + f * (p[i + 1] - p[i])


def coverage(sc, xy):
    """(cam_inner, cam_outer, gps_inner, gps_outer) for a point, from the scenario's sensor definitions."""
    ci = co = gi = go = False
    for c in sc["cameras"].values():
        dx, dy = xy[0] - c["x"], xy[1] - c["y"]; r = math.hypot(dx, dy)
        ang = abs((math.atan2(dy, dx) - c["heading_rad"] + math.pi) % (2 * math.pi) - math.pi); hf = math.radians(c["fov_deg"]) / 2
        ci |= (r <= 0.85 * c["range_m"]) and (ang <= 0.85 * hf); co |= (r <= c["range_m"] + 10) and (ang <= hf + math.radians(10))
    for g in sc["gps"].values():
        r = math.hypot(xy[0] - g["x"], xy[1] - g["y"]); gi |= r <= g["radius_m"]; go |= r <= g["radius_m"] + 10
    return ci, co, gi, go


def candidates(r):
    p = scen_path(r); sc = json.loads(p.read_text()); w, meta, L, seg_of, lim, occ = lane_graph(p)
    exits = set(w.boundary_exit_nodes or set()); das = {d["segment_id"] for d in sc["das"].values()}
    rows = []
    for start in sorted(L):
        if L[start] < S0 + 5 or set(occ.get(start, [])) - {"veh001", "veh002"}:
            continue
        v = 0.9 * lim[seg_of[start]]; route, total = best_route(meta, L, seg_of, start, exits)
        off, acc = {}, 0.0
        for lid in route:
            off[lid] = acc; acc += L[lid]
        for lid in route:
            if seg_of[lid] not in das:
                continue
            for s in np.arange(10.0, L[lid] - 10.0 + 1e-9, 5.0):
                d = off[lid] + s - S0; post = total - (off[lid] + s)
                if not (DRIVE_MIN_S * v <= d <= DRIVE_MAX_S * v and post >= POST_MIN_M):
                    continue
                ci, co, gi, go = coverage(sc, point_at(w.lanes[lid].polyline, s))
                cls = "COMBINED" if ci else ("DAS_ONLY" if not co and not go else None)
                if cls:
                    rows.append({"region": r, "sensing": cls, "start_lane": start, "stop_lane": lid, "stop_s": float(s), "stop_pos_route_m": round(off[lid] + s, 2),
                                 "drive_m": round(d, 1), "post_m": round(post, 1), "v_drive": round(v, 3), "gps_at_stop": gi, "route": ";".join(route)})
    return rows


def pick(C, region, cls):
    c = C[(C.region == region) & (C.sensing == cls)]
    return c.sort_values(["post_m", "start_lane", "stop_lane", "stop_s"], ascending=[False, True, True, True]).iloc[0] if len(c) else None


def survey():
    for name, pool in (("train", [r for r in SPLITS["train"] if r not in USED_TRAIN]), ("stallval", [r for r in SPLITS["val_A"] if r not in USED_VAL])):
        rows = []
        for r in pool:
            if scen_path(r).exists():
                rows += candidates(r)
        C = pd.DataFrame(rows); WORK.mkdir(parents=True, exist_ok=True); C.to_csv(WORK / f"survey_{name}.csv", index=False)
        print(name, "pool", len(pool), "regions with DAS_ONLY", C[C.sensing == "DAS_ONLY"].region.nunique(), "with COMBINED", C[C.sensing == "COMBINED"].region.nunique())


def plan():
    rows = []
    C = pd.read_csv(PLANS / "survey_train.csv", dtype={"region": str})
    both = sorted(set(C[C.sensing == "DAS_ONLY"].region) & set(C[C.sensing == "COMBINED"].region))
    regs = [both[i] for i in np.random.default_rng(20261009).permutation(len(both))[:12]]
    slot = 0
    for i, r in enumerate(regs):
        for cls, grp in (("DAS_ONLY", "ABC"[i % 3]), ("COMBINED", "ABC"[(i + 1) % 3])):
            c = pick(C, r, cls); base = {"region": r, "split_role": "train", "slot": slot, "sensing": cls, "group": grp, **{k: c[k] for k in ("start_lane", "stop_lane", "stop_s", "stop_pos_route_m", "drive_m", "post_m", "v_drive", "gps_at_stop", "route")}}
            rows.append({**base, "mode": "MSM", "arms": "both" if grp == "A" else "quantity"})
            if grp != "A":
                rows.append({**base, "mode": "START" if grp == "B" else "STOP_END", "arms": "diversity"})
            rows.append({**base, "mode": "CONTROL", "arms": "both"})
            rows.append({**base, "mode": "LEGIT_STOP", "arms": "both", "legit_hold_s": 3.0 if slot % 2 == 0 else 5.0})
            slot += 1
    V = pd.read_csv(PLANS / "survey_stallval.csv", dtype={"region": str}); order = sorted(V.region.unique())
    order = [order[i] for i in np.random.default_rng(20261010).permutation(len(order))]; chosen = {}
    for cls in ("DAS_ONLY", "COMBINED"):
        for r in order:
            if len([k for k, v in chosen.items() if v == cls]) == 3:
                break
            if r not in chosen and pick(V, r, cls) is not None:
                chosen[r] = cls
    for j, (r, cls) in enumerate(chosen.items()):
        c = pick(V, r, cls); base = {"region": r, "split_role": "stallval", "slot": 100 + j, "sensing": cls, "group": "V", "arms": "eval", **{k: c[k] for k in ("start_lane", "stop_lane", "stop_s", "stop_pos_route_m", "drive_m", "post_m", "v_drive", "gps_at_stop", "route")}}
        for m in ("START", "STOP_END", "MSM", "CONTROL", "LEGIT_STOP"):
            rows.append({**base, "mode": m, **({"legit_hold_s": 5.0} if m == "LEGIT_STOP" else {})})
    P = pd.DataFrame(rows); P.insert(0, "run_id", [f"st_{r.split_role[:2]}_{r.region}_s{r.slot}_{r.mode}" for r in P.itertuples()])
    WORK.mkdir(parents=True, exist_ok=True); P.to_csv(WORK / "run_plan.csv", index=False)
    print(P.groupby(["split_role", "mode", "sensing"]).size().unstack(fill_value=0)); print(P.groupby(["split_role", "arms"]).size())


class StallHook:
    """Route at t = 0; position-triggered freeze (MSM / STOP_END) or gentle stop (LEGIT_STOP); START is frozen in the scenario itself."""
    def __init__(self, row, route, dest, off):
        self.row, self.route, self.dest, self.off = row, route, dest, off
        self.done = self.stopped = self.released = self.braking = False; self.t_stop = None; self.log = []

    def pos(self, v):
        return self.off[v.lane_id] + v.s if v.lane_id in self.off else None

    def __call__(self, sim, world, t):
        v = world.vehicles.get("veh001")
        if not self.done:
            if v is not None and self.row.mode != "START":
                sim.set_vehicle_route("veh001", self.route, self.dest); self.log.append({"t": t, "action": "route", "n_lanes": len(self.route)})
            self.done = True
        if v is None or self.row.mode in ("START", "CONTROL"):
            return
        p = self.pos(v); target = float(self.row.stop_pos_route_m)
        if self.row.mode in ("MSM", "STOP_END"):
            if not self.stopped and p is not None and p >= target:
                v.frozen = True; self.stopped = True; self.t_stop = t; self.log.append({"t": t, "action": "freeze", "pos": p, "v0": float(v.v)})
            if self.row.mode == "MSM" and self.stopped and not self.released and t >= self.t_stop + MSM_HOLD_S:
                v.frozen = False; self.released = True; self.log.append({"t": t, "action": "release"})
        elif self.row.mode == "LEGIT_STOP":
            if not self.braking and p is not None and p + v.v ** 2 / (2 * LEGIT_DECEL) >= target:
                v.a_cmd_override_mps2 = -LEGIT_DECEL; v.a_cmd_override_until_t = t + v.v / LEGIT_DECEL + 0.5; self.braking = True
                self.log.append({"t": t, "action": "brake", "pos": p, "v0": float(v.v)})
            if self.braking and not self.stopped and v.v < 0.1:
                v.frozen = True; self.stopped = True; self.t_stop = t; self.log.append({"t": t, "action": "hold", "pos": p})
            if self.stopped and not self.released and t >= self.t_stop + float(self.row.legit_hold_s):
                v.frozen = False; v.a_cmd_override_mps2 = 0.0; v.a_cmd_override_until_t = 0.0; self.released = True; self.log.append({"t": t, "action": "release"})


def build(row):
    src = scen_path(row.region); base = json.loads(src.read_text()); sc = copy.deepcopy(base)
    w, meta, L, seg_of, lim, occ = lane_graph(src); route = row.route.split(";"); off, acc = {}, 0.0
    for lid in route:
        off[lid] = acc; acc += L[lid]
    v = sc["vehicles"]["veh001"]; _clear_overrides(v); _set_speed_profile(v, float(row.v_drive))
    lane, s = (row.stop_lane, float(row.stop_s)) if row.mode == "START" else (row.start_lane, S0)
    poly = w.lanes[lane].polyline; (x0, y0), (x1, y1) = poly[0], poly[1]
    v.update(lane_id=lane, s=s, heading_rad=math.atan2(y1 - y0, x1 - x0), ax_world_mps2=0.0, ay_world_mps2=0.0, a_long_mps2=0.0)
    if row.mode == "START":
        v.update(frozen=True, v=0.0)
    return src, base, sc, route, meta[route[-1]]["to"], off


def stops(rows):
    t = np.array([r["t"] for r in rows]); v = np.array([r["v"] for r in rows]); st = v < 0.1; out, i = [], 0
    while i < len(st):
        if st[i]:
            j = i
            while j + 1 < len(st) and st[j + 1]:
                j += 1
            out.append((t[i], t[j], t[j] - t[i] + (t[1] - t[0] if len(t) > 1 else 0))); i = j + 1
        else:
            i += 1
    return t, v, out


def realization(row, sc, truth, collisions):
    rows = sorted([r for r in truth if r["vehicle_id"] == "veh001"], key=lambda r: r["t"])
    t, v, S = stops(rows); a = np.array([r["a_long_mps2"] or 0.0 for r in rows])
    coll = [c for c in collisions if "veh001" in c.get("vehicle_ids", [])]
    longest = max(S, key=lambda x: x[2]) if S else (None, None, 0.0)
    out = {"mode": row.mode, "n_rows": len(rows), "t_first": float(t[0]) if len(t) else None, "t_last": float(t[-1]) if len(t) else None,
           "stop_longest_s": float(longest[2]), "stop_t0": longest[0], "stop_t1": longest[1], "n_collisions_veh001": len(coll),
           "v_max_before_stop": float(v[t < longest[0]].max()) if longest[0] is not None and (t < longest[0]).any() else 0.0,
           "decel_peak_mps2": float(-a.min()) if len(a) else 0.0}
    after = (t > longest[1]) & (v >= 2.0) if longest[1] is not None else np.zeros(len(t), bool)
    out["moving_after_s"] = float(after.sum() * 0.02)
    ends = longest[1] is not None and abs(longest[1] - t[-1]) < 0.05
    m = row.mode
    if m == "START":
        ok = bool(S) and S[0][0] - t[0] < 0.05 and ends and longest[2] >= 100
    elif m == "STOP_END":
        ok = out["v_max_before_stop"] >= 3 and ends and longest[2] >= 10
    elif m == "MSM":
        ok = out["v_max_before_stop"] >= 3 and longest[2] >= 10 and out["moving_after_s"] >= 2
    elif m == "LEGIT_STOP":
        ok = 1 <= longest[2] < 10 and out["decel_peak_mps2"] < 4 and out["moving_after_s"] >= 2
    else:
        ok = longest[2] < 2
    out["verified"] = bool(ok and not coll)
    bg = {}
    for vid in sc["vehicles"]:
        if vid == "veh001":
            continue
        r = realised_severity(sc, [dict(x) for x in truth], vid, [], trigger_t=None)
        ok = r.get("n_rows", 0) > 0 and r.get("stop_longest_s", 0) < 2.0 and not (r.get("gap_min_m", 99) < 5.0)
        bg[vid] = {"role": "background", "unaffected_normal": bool(ok), "stop_longest_s": r.get("stop_longest_s"), "gap_min_m": r.get("gap_min_m")}
    out["background"] = bg
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "survey":
        survey()
    elif cmd == "plan":
        plan()
    elif cmd == "run":
        B = Path(sys.argv[2]); ids = set(sys.argv[3:]); P = pd.read_csv(PLANS / "run_plan.csv", dtype={"region": str})
        for row in P.itertuples():
            if row.run_id not in ids:
                continue
            d = B / row.run_id
            if d.exists():
                print(row.run_id, "exists - not rerun"); continue
            d.mkdir(parents=True); src, base, sc, route, dest, off = build(row)
            sj = d / f"{row.run_id}.sim.json"; sj.write_text(json.dumps(sc, indent=1))
            hook = StallHook(row, route, dest, off); t0 = time.time()
            res = run_scenario(sj, d, seed=SEED, hook=hook, record_truth=True)
            truth = res.pop("truth"); pd.DataFrame(truth).to_parquet(d / "truth_trail.parquet", index=False)
            real = realization(row, sc, truth, res["collisions"])
            kind = {"MSM": "anomaly_stalled_vehicle", "START": "anomaly_stalled_vehicle", "STOP_END": "anomaly_stalled_vehicle",
                    "CONTROL": "normal_matched_control", "LEGIT_STOP": "normal_legit_stop"}[row.mode]
            info = {"run_id": row.run_id, "region": row.region, "split_role": row.split_role, "mode": row.mode, "arms": row.arms, "anomaly_type": kind, "seed": SEED,
                    "plan_row": row._asdict(), "source_scenario": str(src.relative_to(paths.ROOT)), "source_md5": hashlib.md5(src.read_bytes()).hexdigest(),
                    "scenario_md5": hashlib.md5(sj.read_bytes()).hexdigest(), "hook_log": hook.log, "wall_s": round(time.time() - t0, 1), "result": res}
            (d / "run.json").write_text(json.dumps(info, indent=1, default=str)); (d / "realization.json").write_text(json.dumps(real, indent=1, default=str))
            print(row.run_id, res["stop_reason"], round(res["sim_time_s"], 1), "stop", round(real["stop_longest_s"], 1), "verified", real["verified"], flush=True)
