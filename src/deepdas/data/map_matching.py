"""
Map matching of Kalman positions onto the lane centre-lines of a region.

The lane geometry is built by the *copied, unchanged* simulator
(``deepdas.fp.sim_core.Simulation.rebuild_lanes``) from the region's scenario
file, so the DL pipeline sees exactly the polylines the vehicles drove on.
The projection itself is a vectorised re-implementation of
``deepdas.fp.geometry.polyline_nearest_s`` (tested against it in
``tests/test_map_matching.py``) because the pure-Python original is too slow for
millions of points.

Conventions (identical to the simulator's ``pose_with_lateral``):
    tangent (tx, ty) points along the direction of travel of the lane;
    the lateral offset is positive to the LEFT of travel, n = (-ty, tx).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..fp.bus import EventBus
from ..fp.project_io import dict_to_world
from ..fp.sim_core import Simulation

OFF_ROAD_M = 8.0          # a point farther than this from every lane is "off-road"
DIRECTION_PENALTY_M = 30.0  # added to the distance of lanes running against the velocity
HYSTERESIS_M = 1.0        # keep the previous lane while it is within this distance of the best


def load_world_with_lanes(scenario_json: Path):
    """World built the way the Final Project's headless runner builds it (one lane per direction)."""
    d = json.loads(Path(scenario_json).read_text())
    world = dict_to_world(d)
    for seg in world.segments.values():   # mirrors run_all_simulations._normalize_lanes
        seg.lanes = 1
    sim = Simulation(EventBus(), world)
    sim.rebuild_lanes()
    return world


@dataclass
class LaneIndex:
    lane_ids: List[str]
    segment_ids: List[str]
    speed_limit: np.ndarray        # (n_lanes,)
    lane_length: np.ndarray        # (n_lanes,)
    # flattened polyline segments, sorted by lane
    a: np.ndarray                  # (N, 2)
    b: np.ndarray                  # (N, 2)
    lane_of_seg: np.ndarray        # (N,)
    s0_of_seg: np.ndarray          # (N,) arc length at point a
    seg_len: np.ndarray            # (N,)
    tangent: np.ndarray            # (N, 2) unit
    lane_starts: np.ndarray        # (n_lanes,) first flattened segment of each lane

    @classmethod
    def from_world(cls, world) -> "LaneIndex":
        lane_ids, segment_ids, limits, lengths = [], [], [], []
        A, B, lane_of, s0, ln, tg, starts = [], [], [], [], [], [], []
        for li, (lid, lane) in enumerate(sorted(world.lanes.items())):
            poly = [tuple(map(float, p)) for p in lane.polyline]
            seg = world.segments.get(lane.segment_id)
            lane_ids.append(lid)
            segment_ids.append(lane.segment_id)
            limits.append(float(getattr(seg, "speed_limit_mps", 13.9)) if seg is not None else 13.9)
            starts.append(len(A))
            acc = 0.0
            for i in range(1, len(poly)):
                a, b = np.array(poly[i - 1]), np.array(poly[i])
                L = float(np.hypot(*(b - a)))
                if L < 1e-9:
                    continue
                A.append(a); B.append(b); lane_of.append(li); s0.append(acc); ln.append(L); tg.append((b - a) / L)
                acc += L
            if starts[-1] == len(A):   # degenerate lane: add a zero-length stub so reduceat works
                a = np.array(poly[0]) if poly else np.zeros(2)
                A.append(a); B.append(a + 1e-6); lane_of.append(li); s0.append(0.0); ln.append(1e-6); tg.append(np.array([1.0, 0.0]))
                acc = 0.0
            lengths.append(acc)
        return cls(lane_ids, segment_ids, np.array(limits), np.array(lengths), np.array(A), np.array(B),
                   np.array(lane_of), np.array(s0), np.array(ln), np.array(tg), np.array(starts))

    @classmethod
    def from_scenario_json(cls, scenario_json: Path) -> "LaneIndex":
        return cls.from_world(load_world_with_lanes(scenario_json))

    # ------------------------------------------------------------------ #
    def project_all(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Distance and parameter t of every point to every flattened segment. Shapes (M, N)."""
        ab = self.b - self.a                                   # (N, 2)
        ab2 = np.maximum((ab ** 2).sum(1), 1e-12)              # (N,)
        w = xy[:, None, :] - self.a[None, :, :]                # (M, N, 2)
        t = np.clip((w * ab[None]).sum(2) / ab2[None], 0.0, 1.0)
        q = self.a[None] + t[..., None] * ab[None]
        d = np.sqrt(((xy[:, None, :] - q) ** 2).sum(2))
        return d, t

    def match(self, xy: np.ndarray, vxy: Optional[np.ndarray] = None, chunk: int = 512,
              hysteresis_m: float = HYSTERESIS_M) -> Dict[str, np.ndarray]:
        """Match points (M, 2) of ONE track (in time order) to lanes.

        Returns lane index, s, signed lateral offset, tangent, distance.  ``hysteresis_m`` keeps the
        previous lane while it is within that distance of the best candidate (0 disables it)."""
        M = len(xy)
        lane = np.full(M, -1, dtype=np.int32)
        seg = np.full(M, -1, dtype=np.int64)
        tpar = np.zeros(M)
        dist = np.full(M, np.inf)
        prev_lane = -1
        for c0 in range(0, M, chunk):
            c1 = min(M, c0 + chunk)
            d, t = self.project_all(xy[c0:c1])                 # (m, N)
            cost = d.copy()
            if vxy is not None:
                v = vxy[c0:c1]
                speed = np.hypot(v[:, 0], v[:, 1])
                cos = (v @ self.tangent.T) / np.maximum(speed, 1e-9)[:, None]
                cost += np.where((speed[:, None] > 1.0) & (cos < 0.0), DIRECTION_PENALTY_M, 0.0)
            lane_cost = np.minimum.reduceat(cost, self.lane_starts, axis=1)     # (m, n_lanes)
            for i in range(c1 - c0):
                best = int(np.argmin(lane_cost[i]))
                if prev_lane >= 0 and lane_cost[i, prev_lane] <= lane_cost[i, best] + hysteresis_m:
                    best = prev_lane
                # best flattened segment within the chosen lane
                s_start = self.lane_starts[best]
                s_end = self.lane_starts[best + 1] if best + 1 < len(self.lane_starts) else len(self.a)
                j = s_start + int(np.argmin(cost[i, s_start:s_end]))
                lane[c0 + i] = best
                seg[c0 + i] = j
                tpar[c0 + i] = t[i, j]
                dist[c0 + i] = d[i, j]
                prev_lane = best
        s = self.s0_of_seg[seg] + tpar * self.seg_len[seg]
        tangent = self.tangent[seg]
        q = self.a[seg] + tpar[:, None] * (self.b[seg] - self.a[seg])
        delta = xy - q
        d_lat = -delta[:, 0] * tangent[:, 1] + delta[:, 1] * tangent[:, 0]   # left-positive
        return {
            "lane": lane,
            "s": s,
            "d_lat": d_lat,
            "tangent": tangent,
            "dist": dist,
            "speed_limit": self.speed_limit[lane],
            "off_road": dist > OFF_ROAD_M,
        }
