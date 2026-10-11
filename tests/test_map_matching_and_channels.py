"""
Map matching and channel construction.

* The vectorised projection equals the Final Project's ``polyline_nearest_s``
  (copied, unchanged) on random points of a real region.
* The lateral sign follows the simulator's ``pose_with_lateral`` (left > 0).
* A synthetic straight-lane track gives v_lon, d_lat, v_lat, r_lim and the
  leader gap / headway exactly.
* The channel table contains no simulator-truth column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deepdas import paths
from deepdas.data.channels import CHANNELS, MASK_CHANNELS, VALUE_CHANNELS, build_channels
from deepdas.data.map_matching import LaneIndex, load_world_with_lanes
from deepdas.fp.constants import IDM_VEHICLE_LENGTH_M
from deepdas.fp.geometry import point_at_s, polyline_nearest_s, pose_with_lateral, tangent_at_s


@pytest.fixture(scope="module")
def region001():
    world = load_world_with_lanes(paths.region_geometry_json("001"))
    return world, LaneIndex.from_world(world)


def test_vectorised_projection_matches_fp_geometry(region001):
    world, li = region001
    rng = np.random.default_rng(0)
    xs = np.array([n.x for n in world.nodes.values()])
    ys = np.array([n.y for n in world.nodes.values()])
    pts = np.column_stack([rng.uniform(xs.min(), xs.max(), 300), rng.uniform(ys.min(), ys.max(), 300)])
    m = li.match(pts, vxy=None, hysteresis_m=0.0)
    lanes = [world.lanes[lid] for lid in li.lane_ids]
    for k, p in enumerate(pts):
        best = min(((polyline_nearest_s(l.polyline, tuple(p))[2], j) for j, l in enumerate(lanes)))
        d_ref, j_ref = best
        assert abs(m["dist"][k] - d_ref) < 1e-6
        if j_ref == m["lane"][k]:
            s_ref = polyline_nearest_s(lanes[j_ref].polyline, tuple(p))[0]
            assert abs(m["s"][k] - s_ref) < 1e-6
        else:  # a tie (equidistant lanes) — distances must still agree
            assert abs(li.match(p[None, :], hysteresis_m=0.0)["dist"][0] - d_ref) < 1e-6


def test_lateral_sign_matches_pose_with_lateral(region001):
    world, li = region001
    lane = world.lanes[li.lane_ids[5]]
    L = li.lane_length[5]
    for s, off in ((0.3 * L, 0.8), (0.6 * L, -1.1)):
        p = pose_with_lateral(lane.polyline, s, off)
        tx, ty = tangent_at_s(lane.polyline, s)
        m = li.match(np.array([p]), vxy=np.array([[tx * 10, ty * 10]]))
        assert li.lane_ids[m["lane"][0]] == lane.id
        assert abs(m["d_lat"][0] - off) < 1e-6
        assert abs(m["s"][0] - s) < 1e-6


def _synthetic_world():
    """Two vehicles on one straight 400 m lane (speed limit 13.9 m/s)."""
    class Seg:  # minimal stand-ins for the dataclasses
        def __init__(self, sid, lim):
            self.id, self.speed_limit_mps = sid, lim

    class Lane:
        def __init__(self, lid, sid, poly):
            self.id, self.segment_id, self.polyline = lid, sid, poly

    class W:
        pass

    w = W()
    w.segments = {"segA": Seg("segA", 13.9)}
    w.lanes = {"segA_fwd_lane1": Lane("segA_fwd_lane1", "segA", [(0.0, 0.0), (400.0, 0.0)])}
    return w


def test_channels_on_synthetic_straight_lane():
    li = LaneIndex.from_world(_synthetic_world())
    t = np.arange(0.02, 20.0, 0.02)
    rows = []
    for key, x0, v, dlat in (("s/1|T1", 10.0, 10.0, 0.5), ("s/1|T2", 40.0, 8.0, -0.2)):
        rows.append(pd.DataFrame({
            "track_key": key, "t": t, "update_kind": 2,
            "x_hat": x0 + v * t, "y_hat": dlat, "vx_hat": v, "vy_hat": 0.0, "ax_hat": 0.0, "ay_hat": 0.0,
            "has_das": (np.arange(len(t)) % 2 == 0), "has_cam": False, "has_gps": False,
        }))
    rows50 = pd.concat(rows, ignore_index=True)
    meta = pd.DataFrame({"track_key": ["s/1|T1", "s/1|T2"], "region": ["synthetic"] * 2, "scenario_id": ["s/1"] * 2})
    ch = build_channels(rows50, meta, {"synthetic": li})
    assert set(CHANNELS) <= set(ch.columns)
    t1 = ch[(ch["track_key"] == "s/1|T1") & (ch["m_valid"] > 0)]
    assert np.allclose(t1["v_lon"], 10.0, atol=1e-5) and np.allclose(t1["d_lat"], 0.5, atol=1e-5)
    assert np.allclose(t1["v_lat"], 0.0, atol=1e-5) and np.allclose(t1["a_lon"], 0.0, atol=1e-5)
    assert np.allclose(t1["r_lim"], 10.0 / 13.9 - 1.0, atol=1e-5)
    assert set(np.round(t1["m_das"].unique().astype(float), 6)) <= {0.4, 0.5, 0.6} and abs(t1["m_das"].mean() - 0.5) < 0.02
    assert np.all(t1["m_cam"] == 0)
    # leader: T2 starts 30 m ahead and T1 closes at 2 m/s; gap = 30 - 2t - 4.5 until T1 passes at t = 12.75 s
    before = t1[t1["t_rel"] < 11.0]
    expected_gap = 30.0 - 2.0 * (before["t_rel"].to_numpy() + 0.06) - IDM_VEHICLE_LENGTH_M
    assert np.allclose(before["g_lead"], expected_gap, atol=0.05)
    assert np.allclose(before["th_lead"], expected_gap / 10.0, atol=0.01)
    assert np.allclose(before["dv_lead"], 2.0, atol=1e-5) and np.all(before["m_lead"] == 1)
    after = t1[t1["t_rel"] > 16.0]
    assert np.all(after["m_lead"] == 0) and np.allclose(after["g_lead"], 60.0) and np.allclose(after["th_lead"], 10.0)
    t2 = ch[ch["track_key"] == "s/1|T2"]
    assert np.all(t2[t2["t_rel"] < 11.0]["m_lead"] == 0)
    assert np.all(t2[t2["t_rel"] > 16.0]["m_lead"] == 1)
    # burn-in: the first second is flagged
    assert (ch[ch["track_key"] == "s/1|T1"]["burn_in"].to_numpy()[:10]).all()
    assert not ch[ch["track_key"] == "s/1|T1"]["burn_in"].to_numpy()[10:].any()


def test_channel_table_has_no_truth_columns():
    forbidden = {"true_x", "true_y", "true_s", "pos_err_m", "segment_id", "hat_s", "innovation_x", "NIS_x",
                 "vehicle_id_oracle", "das_amplitude", "fiber_distance_m", "lateral_offset_m", "snr", "sigma_m"}
    p = paths.PROCESSED / "channels.parquet"
    if not p.exists():
        pytest.skip("dataset not built")
    cols = set(pd.read_parquet(p).columns)
    assert not (cols & forbidden), cols & forbidden
    assert set(VALUE_CHANNELS + MASK_CHANNELS) <= cols
