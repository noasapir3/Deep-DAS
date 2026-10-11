"""
Headless scenario runner — a seeded re-implementation of the Final Project's
``scripts/run_all_simulations.py::run_scenario`` on top of the *copied, unchanged*
simulator, tracker, Kalman filter and export code (``deepdas.fp``).

Differences from the original runner (all outside the simulator):
  * ``random.seed(seed)`` before the world is loaded, so a run is reproducible;
  * an optional per-step ``hook(sim, world, t)`` (the anomaly director) that
    mutates ``Vehicle`` override fields the simulator already honours;
  * the vehicles' simulator truth can be recorded from ``world.vehicle_state``
    events to describe the *realised* severity (never used as model input);
  * ``export_all`` is called with ``gids=[]`` (no PNGs) and the module flag
    ``EXPORT_WORD_REPORT`` switched off at runtime — the xlsx workbooks, which are
    the only files the pipeline reads, are produced by exactly the same code.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..fp import audit as audit_mod
from ..fp.bus import EventBus
from ..fp.project_io import load_world
from ..fp.sim_core import Simulation
from ..fp.tracking import TrackManager

SIM_DT = 0.02                      # identical to run_all_simulations.SIM_DT
DEFAULT_MAX_DURATION = 120.0       # identical safety cap
CAPTURE_TOPICS = ("world.vehicle_state", "world.vehicle_stuck", "world.collision", "world.route_complete",
                  "sensor.gps", "sensor.camera", "sensor.das")
TRACKER_TOPICS = ("world.vehicle_state", "world.vehicle_stuck", "sensor.gps", "sensor.camera", "sensor.das")


def _normalize_lanes(world) -> None:
    """Enforce one lane per direction — copied from run_all_simulations._normalize_lanes."""
    for seg in world.segments.values():
        try:
            seg.lanes = 1
        except Exception:
            pass


def run_scenario(sim_json: Path, out_dir: Path, seed: int, hook: Optional[Callable] = None,
                 record_truth: bool = False, max_duration_s: float = DEFAULT_MAX_DURATION,
                 export: bool = True) -> Dict:
    t_start = time.time()
    random.seed(seed)

    world = load_world(Path(sim_json))
    _normalize_lanes(world)
    bus = EventBus()
    sim = Simulation(bus, world)
    sim.rebuild_lanes()

    all_events: list = []
    bus_subscribe = bus.subscribe

    def _capture(ev, _lst=all_events):
        _lst.append(ev)

    for topic in CAPTURE_TOPICS:
        bus_subscribe(topic, _capture)

    tm = TrackManager(world, bus=bus)      # live tracker, as in the original runner (its output is not exported)

    def _tm_cb(ev, _m=tm):
        try:
            _m.on_event(ev)
        except Exception:
            pass

    for topic in TRACKER_TOPICS:
        bus_subscribe(topic, _tm_cb)

    truth: List[dict] = []
    collisions: List[dict] = []
    if record_truth:
        def _truth(ev, _lst=truth):
            p = ev.payload or {}
            _lst.append({k: p.get(k) for k in ("t", "vehicle_id", "lane_id", "segment_id", "x", "y", "v", "a_long_mps2", "heading_rad")})
        bus_subscribe("world.vehicle_state", _truth)

    def _coll(ev, _lst=collisions):
        _lst.append(dict(ev.payload or {}))
    bus_subscribe("world.collision", _coll)

    stuck_ids: set = set()

    def _track_stuck(ev, _s=stuck_ids):
        vid = (ev.payload or {}).get("vehicle_id")
        if vid:
            _s.add(str(vid))
    bus_subscribe("world.vehicle_stuck", _track_stuck)

    safety_steps = int(max_duration_s / SIM_DT)
    steps_run, stop_reason = 0, "safety_cap"
    for _ in range(safety_steps):
        if hook is not None:
            hook(sim, world, sim.t)
        sim.step(SIM_DT)
        steps_run += 1
        if not world.vehicles:
            stop_reason = "all_exited"
            break
        if world.vehicles and stuck_ids.issuperset(world.vehicles.keys()):
            stop_reason = "all_stuck"
            break

    result = {
        "events": len(all_events), "steps": steps_run, "sim_time_s": steps_run * SIM_DT, "stop_reason": stop_reason,
        "seed": seed, "collisions": collisions, "n_truth_rows": len(truth),
    }
    if export:
        ts = time.strftime("%Y%m%d_%H%M%S")
        export_dir = Path(out_dir) / f"{Path(sim_json).stem}_export_{ts}_tracking_audit"
        audit_mod.EXPORT_WORD_REPORT = False       # runtime flag; the workbooks are unaffected
        audit_mod.EXPORT_PDF_REPORT = False
        paths_out = audit_mod.export_all(all_events, export_dir, world=world, gids=[])
        result["export_dir"] = export_dir.name
        result["n_audit_rows"] = paths_out.get("n_audit_rows")
        result["n_traj_rows"] = paths_out.get("n_traj_rows")
    result["truth"] = truth
    result["elapsed_s"] = time.time() - t_start
    return result
