"""
Tailgating set on planned routes (training and validation). A leader (veh002) and a follower (veh001) start on the same lane and receive the same planned lane sequence
(sim.set_vehicle_route at t = 0), long enough for the target following duration; the planned route stops at boundary exits.
  * the leader cruises at a pinned speed v_L (±0.5 m/s band, <= 0.85 x the lowest limit on the route);
  * the follower wants to drive faster with IDM time headway T_f and minimum gap 2 m during the event, so its steady headway is
    about T_f + s0 / v_L < 0.8 s while the gap stays physical; after the event duration D the follower returns to the default headway;
  * only scenario fields the simulator already reads and its public routing API are used.
Label (truth only): truth headway < 0.8 s, own speed >= 5 m/s, continuously >= 5 s, bumper gap >= 1.0 m throughout the episode and no
collision of the follower.
Usage:
  python -m deepdas.batches.tailgating_routes survey                     -> outputs/generated/<set>/region_survey.csv   (no simulation)
  python -m deepdas.batches.tailgating_routes plan                       -> outputs/generated/<set>/run_plan.csv        (deterministic)
  python -m deepdas.batches.tailgating_routes run <out_dir> <run_id ...>  -> one folder per run, using the shipped plan (refuses to rerun)
"""
import copy, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd

from .. import paths  # noqa: E402
PLANS = paths.DATA / "simulation_plans" / "tailgating_routes"; WORK = paths.GENERATED / "tailgating_routes"
from deepdas.fp.project_io import load_world  # noqa: E402
from deepdas.fp.bus import EventBus  # noqa: E402
from deepdas.fp.sim_core import Simulation  # noqa: E402
from deepdas.fp.constants import IDM_VEHICLE_LENGTH_M  # noqa: E402
from deepdas.sim.runner import run_scenario  # noqa: E402
from deepdas.sim.scenario_builder import _set_speed_profile, _clear_overrides  # noqa: E402
from deepdas.sim.severity import _lane_polylines, _project  # noqa: E402

RAW = paths.SCENARIOS
SPLITS = json.loads((paths.PROCESSED / "splits.json").read_text())
USED_BEFORE = {"131", "050", "090", "065", "002", "005"}          # regions of the earlier cross-region batch: excluded for variety
V_LEAD = [6.0, 9.0, 12.0]; T_F = [0.20, 0.30]; DUR = [8.0, 15.0, 25.0]
COMBOS = [(v, t, d) for v in V_LEAD for t in T_F for d in DUR]   # 18 combinations
S0_F, LEAD_S, FOLL_S, APPROACH_S, TAIL_S = 2.0, 40.0, 25.5, 8.0, 10.0   # start bumper gap 10 m
H_MAX, V_MIN, DUR_MIN, GAP_MIN = 0.8, 5.0, 5.0, 1.0
N_REGIONS = {"train": 12, "val_A": 6}; RUNS_PER_REGION = 2; SEED = 1


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


EXITS = set()


def best_route(meta, L, seg_of, start, cap=700.0, max_depth=14):
    """Longest simple route (no segment twice, no U-turn) from `start`, stopping once `cap` metres are reached."""
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
        cur = path[-1]; m = meta[cur]
        if m.get("to") in EXITS:                           # a vehicle leaves the map at a boundary exit; the route cannot continue
            return
        for nxt in sorted(out.get(m.get("to"), [])):
            if meta[nxt].get("to") == m.get("from") or seg_of[nxt] in segs:
                continue
            path.append(nxt); segs.add(seg_of[nxt]); dfs(path, segs, length + L[nxt]); path.pop(); segs.discard(seg_of[nxt])
            if best[1] >= cap:
                return
    dfs([start], {seg_of[start]}, L[start])
    return best


def survey():
    rows = []
    for sp in ("train", "val_A", "val_B"):
        for r in SPLITS[sp]:
            p = scen_path(r)
            if not p.exists():
                continue
            w, meta, L, seg_of, lim, occ = lane_graph(p); EXITS.clear(); EXITS.update(w.boundary_exit_nodes or set())
            cands = []
            for lid in sorted(L):
                if L[lid] < LEAD_S + 5 or set(occ.get(lid, [])) - {"veh001", "veh002"}:
                    continue
                route, length = best_route(meta, L, seg_of, lid)
                cands.append((length, lid, route))
            if not cands:
                rows.append({"region": r, "split": sp, "route_m": 0}); continue
            length, lid, route = max(cands, key=lambda c: (c[0], c[1]))
            rows.append({"region": r, "split": sp, "start_lane": lid, "route_m": round(length, 1), "n_lanes": len(route),
                         "route_min_limit": min(lim[seg_of[x]] for x in route), "route": ";".join(route)})
        print(sp, "done", flush=True)
    WORK.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(WORK / "region_survey.csv", index=False)


def plan():
    PILOT = pd.read_csv(PLANS / "earlier_plans/pilot_plan.csv", dtype={"region": str})            # the same regions and split as the pilot plan
    S = pd.read_csv(PLANS / "region_survey.csv", dtype={"region": str})
    S = S[S.region.isin(set(PILOT.region))].set_index("region").loc[list(dict.fromkeys(PILOT.region))].reset_index()
    rng = np.random.default_rng(20261007); rows = []; k = 0
    for sp in ("train", "val_A"):
        c = S[(S.split == sp) & (~S.region.isin(USED_BEFORE)) & (S.route_m >= 6.0 * (DUR_MIN + APPROACH_S) + LEAD_S)
              & (S.route_min_limit * 0.85 >= 6.0)].copy()
        for r in c.itertuples():
            for j in range(RUNS_PER_REGION):
                v, t, d = COMBOS[k % len(COMBOS)]; k += 1
                vl = min(v, 0.85 * r.route_min_limit)
                rows.append({"run_id": f"tg3_{r.region}_{j}", "region": r.region, "split_role": sp, "v_lead": round(vl, 2), "v_lead_nominal": v,
                             "T_f": t, "dur_target_s": d, "steady_headway_s": round(t + S0_F / vl, 3),
                             "route_m_available": r.route_m, "route_m_needed": round(LEAD_S + vl * (d + APPROACH_S + TAIL_S), 1),
                             "expected_follow_s": round(min(d, max(0.0, (r.route_m - LEAD_S) / vl - APPROACH_S)), 1)})
    P = pd.DataFrame(rows); WORK.mkdir(parents=True, exist_ok=True); P.to_csv(WORK / "run_plan.csv", index=False); print(P.to_string(index=False))


class RouteHook:
    def __init__(self, routes, release_t, v_lead):
        self.routes, self.done, self.release_t, self.released, self.log, self.v_lead = routes, False, release_t, False, [], v_lead

    def __call__(self, sim, world, t):
        if not self.done:
            for vid, (lanes, dest) in self.routes.items():
                if vid in world.vehicles:
                    sim.set_vehicle_route(vid, lanes, dest); self.log.append({"t": t, "vehicle": vid, "n_lanes": len(lanes)})
            self.done = True
        if not self.released and t >= self.release_t:
            v = world.vehicles.get("veh001")
            if v is not None:
                v.time_headway_s = 0.0; v.min_gap_m = 0.0          # back to the simulator defaults (IDM T 1.2 s, s0 2 m)
                v.ignore_speed_limit = False; vt = float(self.v_lead)  # and back to an ordinary speed profile at the leader's speed
                v.target_speed_mps = v.speed_mean_mps = vt; v.speed_min_mps = max(1.5, vt - 0.5); v.speed_max_mps = vt + 0.5
            self.released = True; self.log.append({"t": t, "vehicle": "veh001", "action": "release to default IDM headway"})


def build(row):
    src = scen_path(row.region); base = json.loads(src.read_text()); sc = copy.deepcopy(base)
    S = pd.read_csv(PLANS / "region_survey.csv", dtype={"region": str}).set_index("region").loc[row.region]
    full = S.route.split(";"); w, meta, L, seg_of, lim, occ = lane_graph(src)
    need = LEAD_S + row.v_lead * (row.dur_target_s + APPROACH_S + TAIL_S); route, acc = [], 0.0
    for lid in full:                                   # shortest prefix of the shared route that covers the target duration
        route.append(lid); acc += L[lid]
        if acc >= need:
            break
    dest = meta[route[-1]]["to"]
    lane = route[0]; poly = w.lanes[lane].polyline; (x0, y0), (x1, y1) = poly[0], poly[1]
    for vid, s, speed in (("veh002", LEAD_S, row.v_lead), ("veh001", FOLL_S, row.v_lead)):
        v = sc["vehicles"][vid]; _clear_overrides(v); v["lane_id"] = lane; v["s"] = s; v["heading_rad"] = math.atan2(y1 - y0, x1 - x0)
        v["ax_world_mps2"] = v["ay_world_mps2"] = v["a_long_mps2"] = 0.0; _set_speed_profile(v, speed)
    f = sc["vehicles"]["veh001"]; _set_speed_profile(f, 2.5 * row.v_lead); f["v"] = row.v_lead; f["ignore_speed_limit"] = True
    f["time_headway_s"] = float(row.T_f); f["min_gap_m"] = S0_F
    return src, base, sc, {"veh001": (route, dest), "veh002": (route, dest)}


def label(sc, truth, collisions, route=None):
    lanes = _lane_polylines(sc)
    off, acc = {}, 0.0
    for lid in (route or []):
        off[lid] = acc; acc += float(np.hypot(*np.diff(np.array(lanes[lid], float), axis=0).T).sum()) if lid in lanes else 0.0
    for r in truth:
        r["s"], r["d_lat"] = _project(lanes, r)
    by_t = {}
    for r in truth:
        by_t.setdefault(round(r["t"], 3), []).append(r)
    rows = sorted([r for r in truth if r["vehicle_id"] == "veh001"], key=lambda r: r["t"])
    rec = []
    for r in rows:
        best, who = None, None
        on_route = r["lane_id"] in off
        for o in by_t.get(round(r["t"], 3), []):
            if o["vehicle_id"] == "veh001":
                continue
            if on_route and o["lane_id"] in off:              # both on the shared route: gap along the route (crosses lane boundaries)
                g = (off[o["lane_id"]] + o["s"]) - (off[r["lane_id"]] + r["s"]) - IDM_VEHICLE_LENGTH_M
            elif o["lane_id"] == r["lane_id"]:                 # otherwise the same-lane rule
                g = o["s"] - r["s"] - IDM_VEHICLE_LENGTH_M
            else:
                continue
            if g >= -IDM_VEHICLE_LENGTH_M and (best is None or g < best):
                best, who = g, o["vehicle_id"]
        h = best / max(r["v"], 0.5) if best is not None else np.inf
        rec.append((r["t"], r["v"], best if best is not None else np.inf, h, who))
    R = pd.DataFrame(rec, columns=["t", "v", "gap", "h", "leader"])
    ok = (R.h < H_MAX) & (R.v >= V_MIN) & (R.gap >= GAP_MIN)
    brk = (R.t.diff().fillna(0) > 0.1) | (~ok)
    grp = brk.cumsum(); runs = R[ok].groupby(grp[ok])
    best = max(((g.t.iloc[-1] - g.t.iloc[0], g) for _, g in runs), key=lambda x: x[0], default=(0.0, None))
    coll = [c for c in collisions if "veh001" in c.get("vehicle_ids", [])]
    out = {"episode_s": round(float(best[0]), 2), "n_collisions_follower": len(coll),
           "min_gap_m_overall": float(R.gap[np.isfinite(R.gap)].min()) if np.isfinite(R.gap).any() else None}
    if best[1] is not None:
        g = best[1]
        out.update({"episode_t0": float(g.t.iloc[0]), "episode_h_median": float(g.h.median()), "episode_gap_min": float(g.gap.min()),
                    "episode_v_median": float(g.v.median()), "episode_leader_share_veh002": float((g.leader == "veh002").mean())})
    out["verified"] = bool(out["episode_s"] >= DUR_MIN and out["n_collisions_follower"] == 0)
    out["rule"] = "tailgating_v1 (gap along the shared route) + gap>=1 m + no follower collision"
    # background vehicles, as in the cross-region set: unaffected ones are usable normals
    bg = {}
    for vid in sc["vehicles"]:
        if vid in ("veh001", "veh002"):
            continue
        rr = [x for x in truth if x["vehicle_id"] == vid]
        v = np.array([x["v"] for x in rr]) if rr else np.zeros(0)
        stop = 0.0
        if len(rr) > 1:
            cur = 0.0
            for i in range(1, len(rr)):
                cur = cur + rr[i]["t"] - rr[i - 1]["t"] if v[i] < 0.1 else 0.0; stop = max(stop, cur)
        bg[vid] = {"role": "background", "unaffected_normal": bool(len(rr) > 0 and stop < 2.0), "stop_longest_s": stop}
    bg["veh002"] = {"role": "leader", "unaffected_normal": False}
    out["background"] = bg
    return out, R


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
            d.mkdir(parents=True); src, base, sc, routes = build(row)
            sj = d / f"{row.run_id}.sim.json"; sj.write_text(json.dumps(sc, indent=1))
            hook = RouteHook(routes, APPROACH_S + row.dur_target_s, row.v_lead); t0 = time.time()
            res = run_scenario(sj, d, seed=SEED, hook=hook, record_truth=True)
            truth = res.pop("truth"); pd.DataFrame(truth).to_parquet(d / "truth_trail.parquet", index=False)
            real, R = label(sc, truth, res["collisions"], routes["veh001"][0]); R.to_parquet(d / "follower_headway_truth.parquet", index=False)
            info = {"run_id": row.run_id, "region": row.region, "split_role": row.split_role, "anomaly_type": "anomaly_tailgating", "seed": SEED,
                    "plan_row": row._asdict(), "source_scenario": str(src.relative_to(paths.ROOT)), "source_md5": hashlib.md5(src.read_bytes()).hexdigest(),
                    "scenario_md5": hashlib.md5(sj.read_bytes()).hexdigest(), "routes": {k: v[0] for k, v in routes.items()}, "hook_log": hook.log,
                    "wall_s": round(time.time() - t0, 1), "result": res}
            (d / "run.json").write_text(json.dumps(info, indent=1, default=str)); (d / "realization.json").write_text(json.dumps(real, indent=1, default=str))
            print(row.run_id, res["stop_reason"], round(res["sim_time_s"], 1), "episode", real["episode_s"], "verified", real["verified"], flush=True)
