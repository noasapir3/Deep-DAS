"""
Reconstruction-error scoring: window → track → vehicle.

window error  masked MSE over the six target channels (identical to the training loss)
track score   95th percentile of its window errors (scoring stride 8 bins = 0.8 s)
vehicle score max over the tracks of the same oracle vehicle in the same scenario
Per-channel errors are kept for attribution (share of the window error carried by each channel).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .. import paths
from ..data.dataset import TARGET_CHANNELS, RobustNorm, WindowDataset, eligible_tracks, load_labels, load_track_arrays, masked_mse
from ..model.conv_ae import ConvAE

TRACK_PERCENTILE = 95.0


def load_run(run_id: str):
    run_dir = paths.EXPERIMENTS / run_id
    cfg = json.loads((run_dir / "config.json").read_text())
    norm = RobustNorm.from_json(json.loads((run_dir / "norm.json").read_text()))
    st = torch.load(run_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    model = ConvAE(T=cfg["T"], bottleneck=cfg["bottleneck"], width=cfg.get("width", 32), bounded=cfg.get("bounded", True))
    model.load_state_dict(st["model"])
    model.eval()
    return run_dir, cfg, norm, model


def load_error_scale(run_dir: Path) -> np.ndarray:
    """Median per-channel error of the model on val_A normal windows (computed at the end of training)."""
    p = run_dir / "error_scale.json"
    if p.exists():
        return np.asarray(json.loads(p.read_text())["median_err_valA"], dtype=np.float32)
    return np.ones(len(TARGET_CHANNELS), dtype=np.float32)


@torch.no_grad()
def score_tracks(run_id: str, stride: int = 8, track_keys: Optional[Sequence[str]] = None, save: bool = True,
                 batch_size: int = 1024) -> pd.DataFrame:
    run_dir, cfg, norm, model = load_run(run_id)
    labels = load_labels()
    el = eligible_tracks(labels, cfg.get("min_valid_bins", 32), require_quality=False)   # gate applied in the evaluation
    keys = sorted(el["track_key"]) if track_keys is None else [k for k in track_keys if k in set(el["track_key"])]
    arrays = load_track_arrays(keys)
    ds = WindowDataset(arrays, keys, norm, cfg["T"], stride)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    scale = load_error_scale(run_dir)
    errs, per_ch, zs, tidx, starts = [], [], [], [], []
    for x, i, s in dl:
        z = model.encode(x)
        e, pc = masked_mse(model.decode(z), x, per_channel=True)
        errs.append(e.numpy()); per_ch.append(pc.numpy()); zs.append(z.numpy()); tidx.append(i.numpy()); starts.append(s.numpy())
    errs, per_ch, zs = np.concatenate(errs), np.concatenate(per_ch), np.concatenate(zs)
    tidx, starts = np.concatenate(tidx), np.concatenate(starts)
    win = pd.DataFrame({"track_key": np.array(ds.keys)[tidx], "start": starts, "err": errs})
    for j, c in enumerate(TARGET_CHANNELS):
        win[f"err_{c}"] = per_ch[:, j]
    for j in range(zs.shape[1]):
        win[f"z{j}"] = zs[:, j]

    g = win.groupby("track_key", sort=True)
    tracks = pd.DataFrame({
        "score": g["err"].quantile(TRACK_PERCENTILE / 100.0),
        "err_mean": g["err"].mean(),
        "err_max": g["err"].max(),
        "n_windows": g.size(),
    })
    for c in TARGET_CHANNELS:   # channel attribution: share of the error carried by each channel in the worst windows
        tracks[f"attr_{c}"] = g[f"err_{c}"].apply(lambda s: float(s.nlargest(max(1, int(np.ceil(0.05 * len(s))))).mean()))
    attr = tracks[[f"attr_{c}" for c in TARGET_CHANNELS]]
    tracks[[f"attr_{c}" for c in TARGET_CHANNELS]] = attr.div(attr.sum(axis=1).replace(0, np.nan), axis=0)
    tracks = tracks.reset_index().merge(labels, on="track_key", how="left")
    if save:
        win.to_parquet(run_dir / "scores_windows.parquet", index=False)
        tracks.to_parquet(run_dir / "scores_tracks.parquet", index=False)
    return tracks


def vehicle_scores(tracks: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """One row per (scenario, oracle vehicle): max over its tracks, role of the vehicle, strata for analysis."""
    agg = {score_col: "max", "n_windows": "sum", "track_key": "count"}
    for c in tracks.columns:
        if c.startswith("attr_"):
            agg[c] = "mean"
    veh = tracks.groupby(["scenario_id", "vehicle_id_oracle"], sort=True).agg(agg).rename(columns={"track_key": "n_tracks"})
    first = tracks.groupby(["scenario_id", "vehicle_id_oracle"]).first()
    for c in ("role", "anomaly_type", "level", "injection", "group", "region", "scenario_type", "split"):
        if c in first.columns:
            veh[c] = first[c]
    return veh.reset_index()
