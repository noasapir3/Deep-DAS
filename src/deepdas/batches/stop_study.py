"""
Legitimate-stop validation set (val_A regions, new stop slots): CONTROL, LEGIT_STOP with a hold of 3 s or 8 s, GENTLE_STALL (the same
2 m/s² braking as LEGIT_STOP, held 15 s, then resumes) and MSM 30 s. Reuses the injection, routing and realisation code of
deepdas.batches.stalls unchanged.
Usage: python -m deepdas.batches.stop_study survey | plan | run <out_dir> <run_id ...>
"""
import json, sys, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from .. import paths  # noqa: E402
from . import stalls as st  # noqa: E402
PLANS = paths.DATA / "simulation_plans" / "legitimate_stop_validation"; WORK = paths.GENERATED / "legitimate_stop_validation"

POOL = [r for r in st.SPLITS["val_A"] if r not in st.USED_VAL]          # val_A minus 002/005 and the validation regions of the tailgating-routes set (stall-validation regions allowed, other slots)
SHORT_POST, LONG_POST = (60.0, 150.0), 350.0


def candidates(r):
    p = st.scen_path(r); sc = json.loads(p.read_text()); w, meta, L, seg_of, lim, occ = st.lane_graph(p)
    exits = set(w.boundary_exit_nodes or set()); das = {d["segment_id"] for d in sc["das"].values()}; rows = []
    for start in sorted(L):
        if L[start] < st.S0 + 5 or set(occ.get(start, [])) - {"veh001", "veh002"}:
            continue
        v = 0.9 * lim[seg_of[start]]; route, total = st.best_route(meta, L, seg_of, start, exits); ends_exit = meta[route[-1]].get("to") in exits
        off, acc = {}, 0.0
        for lid in route:
            off[lid] = acc; acc += L[lid]
        for lid in route:
            if seg_of[lid] not in das:
                continue
            for s in np.arange(10.0, L[lid] - 10.0 + 1e-9, 5.0):
                d = off[lid] + s - st.S0; post = total - (off[lid] + s)
                if not (st.DRIVE_MIN_S * v <= d <= st.DRIVE_MAX_S * v):
                    continue
                obs = "SHORT" if (ends_exit and SHORT_POST[0] <= post <= SHORT_POST[1]) else ("LONG" if post >= LONG_POST else None)
                if obs is None:
                    continue
                ci, co, gi, go = st.coverage(sc, st.point_at(w.lanes[lid].polyline, s))
                rows.append({"region": r, "obs": obs, "sensing": "COMBINED" if ci else ("DAS_ONLY" if not co and not go else "PARTIAL"), "start_lane": start, "stop_lane": lid,
                             "stop_s": float(s), "stop_pos_route_m": round(off[lid] + s, 2), "drive_m": round(d, 1), "post_m": round(post, 1), "v_drive": round(v, 3),
                             "gps_at_stop": gi, "route": ";".join(route)})
    return rows


def survey():
    rows = []
    for r in POOL:
        if st.scen_path(r).exists():
            rows += candidates(r)
    C = pd.DataFrame(rows); WORK.mkdir(parents=True, exist_ok=True); C.to_csv(WORK / "survey.csv", index=False)
    g = C.groupby("region").obs.agg(lambda x: set(x)); print("pool", len(POOL), "regions with SHORT and LONG:", int(g.apply(lambda s: {"SHORT", "LONG"} <= s).sum()))


MODES = [("CONTROL", None), ("LEGIT_STOP", 3.0), ("LEGIT_STOP", 8.0), ("GENTLE_STALL", 15.0), ("MSM", 30.0)]


def plan():
    C = pd.read_csv(PLANS / "survey.csv", dtype={"region": str})
    ok = sorted(r for r, g in C.groupby("region") if {"SHORT", "LONG"} <= set(g.obs))
    regs = [ok[i] for i in np.random.default_rng(20261011).permutation(len(ok))[:12]]; rows = []; slot = 200
    for r in regs:
        for obs in ("SHORT", "LONG"):
            c = C[(C.region == r) & (C.obs == obs)]
            c = c.sort_values(["post_m", "start_lane", "stop_lane", "stop_s"], ascending=[obs == "SHORT", True, True, True]).iloc[0]
            base = {"region": r, "split_role": "stopval2", "slot": slot, "obs": obs, "sensing": c.sensing, "group": "V2", "arms": "eval",
                    **{k: c[k] for k in ("start_lane", "stop_lane", "stop_s", "stop_pos_route_m", "drive_m", "post_m", "v_drive", "gps_at_stop", "route")}}
            for m, hold in MODES:
                rows.append({**base, "mode": m, "legit_hold_s": hold if m in ("LEGIT_STOP", "GENTLE_STALL") else np.nan})
            slot += 1
    P = pd.DataFrame(rows); P.insert(0, "run_id", [f"sv2_{r.region}_s{r.slot}_{r.mode}{'' if r.mode not in ('LEGIT_STOP', 'GENTLE_STALL') else int(r.legit_hold_s)}" for r in P.itertuples()])
    WORK.mkdir(parents=True, exist_ok=True); P.to_csv(WORK / "run_plan.csv", index=False)
    print(P.groupby(["obs", "sensing"]).slot.nunique()); print(P.drop_duplicates("slot")[["slot", "region", "obs", "sensing", "drive_m", "post_m", "v_drive"]].to_string(index=False))


class Hook(st.StallHook):
    """GENTLE_STALL = LEGIT_STOP mechanics (same braking and hold logic) with a 15 s hold; everything else is the ST01 hook unchanged."""
    def __init__(self, row, *a):
        if row.mode == "GENTLE_STALL":
            row = row._replace(mode="LEGIT_STOP")
        super().__init__(row, *a)


def realization(row, sc, truth, coll):
    if row.mode == "GENTLE_STALL":
        r = st.realization(row._replace(mode="LEGIT_STOP"), sc, truth, coll)
        r["mode"] = "GENTLE_STALL"
        r["verified"] = bool(r["stop_longest_s"] >= 10 and r["decel_peak_mps2"] < 4 and r["moving_after_s"] >= 2 and r["n_collisions_veh001"] == 0)
        return r
    return st.realization(row, sc, truth, coll)


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
            d.mkdir(parents=True); src, base, sc, route, dest, off = st.build(row)
            sj = d / f"{row.run_id}.sim.json"; sj.write_text(json.dumps(sc, indent=1))
            hook = Hook(row, route, dest, off); t0 = time.time()
            res = st.run_scenario(sj, d, seed=st.SEED, hook=hook, record_truth=True)
            truth = res.pop("truth"); pd.DataFrame(truth).to_parquet(d / "truth_trail.parquet", index=False)
            real = realization(row, sc, truth, res["collisions"])
            kind = {"MSM": "anomaly_stalled_vehicle", "GENTLE_STALL": "anomaly_stalled_vehicle", "CONTROL": "normal_matched_control", "LEGIT_STOP": "normal_legit_stop"}[row.mode]
            info = {"run_id": row.run_id, "region": row.region, "split_role": row.split_role, "mode": row.mode, "arms": row.arms, "anomaly_type": kind, "seed": st.SEED,
                    "plan_row": row._asdict(), "source_scenario": str(src.relative_to(paths.ROOT)), "source_md5": hashlib.md5(src.read_bytes()).hexdigest(),
                    "scenario_md5": hashlib.md5(sj.read_bytes()).hexdigest(), "hook_log": hook.log, "wall_s": round(time.time() - t0, 1), "result": res}
            (d / "run.json").write_text(json.dumps(info, indent=1, default=str)); (d / "realization.json").write_text(json.dumps(real, indent=1, default=str))
            print(row.run_id, res["stop_reason"], round(res["sim_time_s"], 1), "stop", round(real["stop_longest_s"], 1), "verified", real["verified"], flush=True)
