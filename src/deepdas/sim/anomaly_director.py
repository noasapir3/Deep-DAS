"""
Runtime anomaly director — the DL-side hook that starts mid-route events.

It only sets ``Vehicle`` override fields that the copied simulator already
consumes in ``sim_core.Simulation.step`` (``a_cmd_override_mps2`` /
``a_cmd_override_until_t`` for scripted braking, ``frozen`` for stalls); the
simulator code is untouched.

Schedule format (JSON-serialisable, stored in ``run.json``)::

    [{"t": 6.0, "vehicle": "veh001", "action": "brake", "decel_mps2": 6.0},
     {"t": 6.0, "vehicle": "veh001", "action": "stall", "duration_s": 10.0}]

``brake`` decelerates at the given rate until the vehicle (nearly) stops — the
override window is v(t0)/decel + 0.2 s, computed when the event fires — after
which the vehicle resumes its normal target-speed tracking.
``stall`` freezes the vehicle for ``duration_s`` seconds and then releases it.
"""
from __future__ import annotations

from typing import Dict, List, Optional


class AnomalyDirector:
    def __init__(self, schedule: List[dict]):
        self.schedule = sorted(schedule, key=lambda a: a["t"])
        self._pending = list(self.schedule)
        self._release: List[dict] = []
        self.log: List[dict] = []

    def __call__(self, sim, world, t: float) -> None:
        while self._pending and t >= self._pending[0]["t"]:
            act = self._pending.pop(0)
            veh = world.vehicles.get(act["vehicle"])
            if veh is None:
                self.log.append({"t": t, "action": act["action"], "status": "vehicle_absent"})
                continue
            if act["action"] == "brake":
                v0 = max(0.0, float(veh.v))
                a = -abs(float(act["decel_mps2"]))
                until = t + v0 / abs(a) + 0.2
                veh.a_cmd_override_mps2 = a
                veh.a_cmd_override_until_t = until
                self.log.append({"t": t, "action": "brake", "v0": v0, "decel": abs(a), "until_t": until, "status": "applied"})
            elif act["action"] == "stall":
                veh.frozen = True
                self._release.append({"t": t + float(act["duration_s"]), "vehicle": act["vehicle"]})
                self._release.sort(key=lambda a: a["t"])
                self.log.append({"t": t, "action": "stall", "v0": float(veh.v), "duration_s": act["duration_s"], "status": "applied"})
            else:
                raise ValueError(f"unknown action {act['action']}")
        while self._release and t >= self._release[0]["t"]:
            rel = self._release.pop(0)
            veh = world.vehicles.get(rel["vehicle"])
            if veh is not None:
                veh.frozen = False
                self.log.append({"t": t, "action": "release", "status": "applied"})
