"""
Realised severity of a run, measured from the simulator's own truth events.

This is a description of what was simulated (used for the x-axis of the
recall-versus-severity figures and for checking that an injection actually
happened).  It is never an input of either detector.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..fp.constants import IDM_VEHICLE_LENGTH_M
from ..fp.geometry import polyline_nearest_s
from ..fp.bus import EventBus
from ..fp.project_io import dict_to_world
from ..fp.sim_core import Simulation


def _lane_polylines(scenario: dict) -> Dict[str, list]:
    """Lane polylines exactly as the simulator rebuilds them (one lane per direction, two-way
    segments offset by half a lane width) — NOT the ``lanes`` block stored in the scenario file."""
    world = dict_to_world(scenario)
    for seg in world.segments.values():
        seg.lanes = 1
    Simulation(EventBus(), world).rebuild_lanes()
    return {lid: [tuple(map(float, p)) for p in l.polyline] for lid, l in world.lanes.items()}


def _project(lanes: Dict[str, list], r: dict):
    """(s, d_lat) of a truth row on its lane polyline, using the copied geometry helper."""
    poly = lanes.get(r["lane_id"])
    if not poly or len(poly) < 2:
        return float("nan"), float("nan")
    s, q, d = polyline_nearest_s(poly, (r["x"], r["y"]))
    acc, j = 0.0, 1
    for k in range(1, len(poly)):
        L = float(np.hypot(poly[k][0] - poly[k - 1][0], poly[k][1] - poly[k - 1][1]))
        j = k
        if acc + L >= s:
            break
        acc += L
    tx, ty = poly[j][0] - poly[j - 1][0], poly[j][1] - poly[j - 1][1]
    n = max(np.hypot(tx, ty), 1e-9)
    tx, ty = tx / n, ty / n
    return s, -(r["x"] - q[0]) * ty + (r["y"] - q[1]) * tx


def realised_severity(scenario: dict, truth_all: List[dict], injected: str, collisions: List[dict],
                      trigger_t: Optional[float] = None) -> dict:
    """Summarise the injected vehicle's true trajectory (and its gap to the vehicle ahead)."""
    rows = [r for r in truth_all if r["vehicle_id"] == injected]
    if not rows:
        return {"n_rows": 0}
    seg_limit = {sid: float(s.get("speed_limit_mps", 13.9)) for sid, s in scenario["segments"].items()}
    lanes = _lane_polylines(scenario)
    for r in truth_all:                       # arc position of every vehicle, needed for gaps
        if "s" not in r:
            r["s"], r["d_lat"] = _project(lanes, r)
    t = np.array([r["t"] for r in rows], dtype=float)
    v = np.array([r["v"] for r in rows], dtype=float)
    a = np.array([r["a_long_mps2"] or 0.0 for r in rows], dtype=float)
    lim = np.array([seg_limit.get(r["segment_id"], 13.9) for r in rows], dtype=float)
    d_lat = np.array([r["d_lat"] for r in rows], dtype=float)
    # gap to the vehicle ahead on the same lane (all vehicles' truth at the same t)
    by_t: Dict[float, List[dict]] = {}
    for r in truth_all:
        by_t.setdefault(round(r["t"], 3), []).append(r)
    gaps, headways = [], []
    for r in rows:
        best = None
        for o in by_t.get(round(r["t"], 3), []):
            if o["vehicle_id"] == injected or o["lane_id"] != r["lane_id"]:
                continue
            g = o["s"] - r["s"] - IDM_VEHICLE_LENGTH_M
            if g >= 0.0 and (best is None or g < best):
                best = g
        if best is not None:
            gaps.append(best)
            headways.append(best / max(r["v"], 0.5))
    stopped = v < 0.1
    # longest stop after the trigger (or overall)
    longest, cur = 0.0, 0.0
    for i in range(1, len(rows)):
        if stopped[i] and (trigger_t is None or t[i] >= trigger_t):
            cur += t[i] - t[i - 1]
            longest = max(longest, cur)
        else:
            cur = 0.0
    post = t >= (trigger_t or -1.0)
    out = {
        "n_rows": int(len(rows)),
        "duration_s": float(t[-1] - t[0]),
        "speed_max_mps": float(v.max()),
        "speed_ratio_max": float((v / lim).max()),
        "speed_ratio_p90": float(np.percentile(v / lim, 90)),
        "decel_peak_mps2": float(-a[post].min()) if post.any() else float("nan"),
        "lateral_abs_max_m": float(np.nanmax(np.abs(d_lat))) if np.isfinite(d_lat).any() else float("nan"),
        "lateral_std_m": float(np.nanstd(d_lat)) if np.isfinite(d_lat).any() else float("nan"),
        "gap_min_m": float(min(gaps)) if gaps else float("nan"),
        "headway_min_s": float(min(headways)) if headways else float("nan"),
        "headway_p10_s": float(np.percentile(headways, 10)) if headways else float("nan"),
        "stop_longest_s": float(longest),
        "stopped_frac": float(stopped.mean()),
        "weight_kg": float(scenario["vehicles"][injected].get("weight_kg", 0.0)),
        "n_collisions": int(len(collisions)),
        "collision_rel_speed_mps": float(max((c.get("rel_speed_mps", 0.0) for c in collisions), default=float("nan"))),
    }
    return out
