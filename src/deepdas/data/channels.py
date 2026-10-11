"""
From 50 Hz Kalman rows to the 10 Hz road-frame channels the autoencoder sees.

Value channels (reconstructed by the model)
    v_lon   longitudinal speed along the matched lane tangent          [m/s]
    a_lon   longitudinal acceleration                                   [m/s²]
    d_lat   signed offset from the matched lane centre-line (left > 0)  [m]
    v_lat   lateral velocity                                            [m/s]
    r_lim   v_lon / speed limit of the matched segment − 1              [-]
    g_lead  bumper-to-bumper gap to the nearest track ahead on the same lane, capped [m]
    th_lead time headway g_lead / max(v_lon, 0.5), capped               [s]
    dv_lead closing speed v_lon − v_lon(leader)                         [m/s]
Mask channels (inputs only)
    m_das, m_cam, m_gps  fraction of the bin's rows carrying that sensor's measurement
    m_lead               1 when a leader within the cap exists
    m_valid              1 for real, on-road, post-burn-in bins (0 for padding / off-road / burn-in)

Everything here is computed from the exported Kalman state and the copied map
geometry.  No simulator truth is read (see docs/LEAKAGE_AUDIT.md).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..fp.constants import IDM_VEHICLE_LENGTH_M
from .map_matching import LaneIndex

BIN_S = 0.1
BURN_IN_S = 1.0
LEAD_GAP_CAP_M = 60.0
LEAD_TH_CAP_S = 10.0
VALUE_CHANNELS = ["v_lon", "a_lon", "d_lat", "v_lat", "r_lim", "g_lead", "th_lead", "dv_lead"]
MASK_CHANNELS = ["m_das", "m_cam", "m_gps", "m_lead", "m_valid"]
CHANNELS = VALUE_CHANNELS + MASK_CHANNELS


def bin_rows(rows50: pd.DataFrame) -> pd.DataFrame:
    """Average the 50 Hz rows of every track into 0.1 s bins (time-based, so variable-rate exports work too)."""
    df = rows50[["track_key", "t", "x_hat", "y_hat", "vx_hat", "vy_hat", "ax_hat", "ay_hat",
                 "has_das", "has_cam", "has_gps"]].copy()
    df["track_key"] = df["track_key"].astype(str)
    t0 = df.groupby("track_key")["t"].transform("min")
    df["bin"] = np.floor((df["t"].to_numpy(dtype=np.float64) - t0.to_numpy(dtype=np.float64)) / BIN_S + 1e-6).astype(np.int32)
    df["t_rel"] = df["t"].astype(np.float64) - t0.astype(np.float64)
    for c in ("has_das", "has_cam", "has_gps"):
        df[c] = df[c].astype(np.float32)
    agg = df.groupby(["track_key", "bin"], sort=True).agg(
        t_rel=("t_rel", "min"), n_rows=("t", "size"),
        x=("x_hat", "mean"), y=("y_hat", "mean"), vx=("vx_hat", "mean"), vy=("vy_hat", "mean"),
        ax=("ax_hat", "mean"), ay=("ay_hat", "mean"),
        m_das=("has_das", "mean"), m_cam=("has_cam", "mean"), m_gps=("has_gps", "mean"),
    ).reset_index()
    return agg


def road_frame(binned: pd.DataFrame, lane_index: LaneIndex) -> pd.DataFrame:
    """Map-match the binned positions and express the kinematics in the lane frame."""
    out = binned.copy()
    xy = out[["x", "y"]].to_numpy(dtype=np.float64)
    vxy = out[["vx", "vy"]].to_numpy(dtype=np.float64)
    parts = []
    for key, idx in out.groupby("track_key", sort=False).indices.items():   # hysteresis is per track
        m = lane_index.match(xy[idx], vxy[idx])
        parts.append(pd.DataFrame({
            "_i": idx, "lane": m["lane"], "s": m["s"], "d_lat": m["d_lat"], "tx": m["tangent"][:, 0],
            "ty": m["tangent"][:, 1], "match_dist": m["dist"], "speed_limit": m["speed_limit"], "off_road": m["off_road"],
        }))
    mm = pd.concat(parts).sort_values("_i")
    for c in ("lane", "s", "d_lat", "tx", "ty", "match_dist", "speed_limit", "off_road"):
        out[c] = mm[c].to_numpy()
    tx, ty = out["tx"].to_numpy(), out["ty"].to_numpy()
    vx, vy, ax, ay = (out[c].to_numpy(dtype=np.float64) for c in ("vx", "vy", "ax", "ay"))
    out["v_lon"] = vx * tx + vy * ty
    out["v_lat"] = -vx * ty + vy * tx
    out["a_lon"] = ax * tx + ay * ty
    out["r_lim"] = out["v_lon"] / np.maximum(out["speed_limit"].to_numpy(), 1.0) - 1.0
    return out


def leader_features(rf: pd.DataFrame, scenario_of: Dict[str, str]) -> pd.DataFrame:
    """Nearest track ahead on the same lane at the same time bin, within the same scenario."""
    out = rf.copy()
    out["scenario"] = out["track_key"].map(scenario_of)
    out["_i"] = np.arange(len(out))
    # absolute bin index inside the scenario (all tracks of an export share the simulation clock)
    out["abs_bin"] = np.floor((out["t_abs"].to_numpy(dtype=np.float64)) / BIN_S + 1e-6).astype(np.int64)
    srt = out.sort_values(["scenario", "abs_bin", "lane", "s"]).reset_index(drop=True)
    same = (srt["scenario"].shift(-1) == srt["scenario"]) & (srt["abs_bin"].shift(-1) == srt["abs_bin"]) & (srt["lane"].shift(-1) == srt["lane"])
    gap = srt["s"].shift(-1) - srt["s"] - IDM_VEHICLE_LENGTH_M
    v_lead = srt["v_lon"].shift(-1)
    has = same & (gap < LEAD_GAP_CAP_M)
    g = np.where(has, np.clip(gap, 0.0, LEAD_GAP_CAP_M), LEAD_GAP_CAP_M)
    th = np.where(has, np.minimum(g / np.maximum(srt["v_lon"].to_numpy(), 0.5), LEAD_TH_CAP_S), LEAD_TH_CAP_S)
    dv = np.where(has, srt["v_lon"] - v_lead, 0.0)
    res = pd.DataFrame({"_i": srt["_i"], "g_lead": g, "th_lead": th, "dv_lead": dv, "m_lead": has.astype(np.float32)}).sort_values("_i")
    for c in ("g_lead", "th_lead", "dv_lead", "m_lead"):
        out[c] = res[c].to_numpy()
    return out.drop(columns=["_i", "scenario"])


def build_channels(rows50: pd.DataFrame, meta: pd.DataFrame, lane_index_of_region: Dict[str, LaneIndex]) -> pd.DataFrame:
    """Full 50 Hz → 10 Hz channel table for one shard (one region or one scenario)."""
    rows50 = rows50.copy()
    rows50["track_key"] = rows50["track_key"].astype(str)
    binned = bin_rows(rows50)
    t0_abs = rows50.groupby("track_key")["t"].min().astype(np.float64)
    binned["t_abs"] = binned["track_key"].map(t0_abs).to_numpy() + binned["t_rel"].to_numpy()
    region_of = dict(zip(meta["track_key"], meta["region"]))
    scenario_of = dict(zip(meta["track_key"], meta["scenario_id"]))
    parts: List[pd.DataFrame] = []
    for region, keys in pd.Series(region_of).groupby(pd.Series(region_of)).groups.items():
        sub = binned[binned["track_key"].isin(set(keys))]
        if len(sub):
            parts.append(road_frame(sub, lane_index_of_region[region]))
    rf = pd.concat(parts, ignore_index=True)
    rf = leader_features(rf, scenario_of)
    rf["burn_in"] = rf["t_rel"] < BURN_IN_S - 1e-9
    rf["m_valid"] = ((~rf["burn_in"]) & (~rf["off_road"]) & (rf["n_rows"] > 0)).astype(np.float32)
    for c in VALUE_CHANNELS:
        rf.loc[rf["off_road"], c] = np.nan
    cols = ["track_key", "bin", "t_rel", "t_abs", "n_rows"] + CHANNELS + ["burn_in", "off_road", "lane", "s", "match_dist", "speed_limit"]
    rf = rf[cols].sort_values(["track_key", "bin"]).reset_index(drop=True)
    for c in VALUE_CHANNELS + MASK_CHANNELS + ["t_rel", "t_abs", "s", "match_dist", "speed_limit"]:
        rf[c] = rf[c].astype(np.float32)
    return rf
