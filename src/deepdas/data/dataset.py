"""
Windows for the autoencoder: robust normalisation, track eligibility, window index, torch Dataset.

* A track enters the dataset when it has at least ``MIN_VALID_BINS`` valid bins
  after the 1 s burn-in (same rule for normal and anomaly tracks).
* Normalisation statistics (median, 1.4826·MAD per value channel) are fitted on
  the valid bins of the TRAIN-split normal tracks only and stored with the run.
* Windows of ``T`` bins; a stride for training, a finer stride for scoring; a
  window aligned to the end of the track is always added so the tail is covered;
  tracks shorter than ``T`` give one zero-padded window (``m_valid`` = 0 on the pad).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .. import paths
from .channels import CHANNELS, MASK_CHANNELS, VALUE_CHANNELS

MIN_VALID_BINS = 32
CLIP_Z = 15.0        # standardised values are winsorised at ±15 robust-σ (Kalman glitches in the exports)
N_VALUE = len(VALUE_CHANNELS)
N_CH = len(CHANNELS)
IDX_VALID = CHANNELS.index("m_valid")
IDX_LEAD = CHANNELS.index("m_lead")
# Reconstruction targets: the value channels whose Kalman noise is small relative to their signal.
# a_lon and v_lat (differentiated Kalman states, noise-dominated) enter the encoder as inputs only.
TARGET_CHANNELS = ["v_lon", "d_lat", "r_lim", "g_lead", "th_lead", "dv_lead"]
TARGET_IDX = [CHANNELS.index(c) for c in TARGET_CHANNELS]
N_TARGET = len(TARGET_CHANNELS)
LEAD_CHANNEL_IDX = [VALUE_CHANNELS.index(c) for c in ("g_lead", "th_lead", "dv_lead")]
# minimum scale per value channel (physical units): v_lon, a_lon, d_lat, v_lat, r_lim, g_lead, th_lead, dv_lead
SCALE_FLOOR = np.array([0.5, 0.2, 0.1, 0.2, 0.05, 2.0, 0.2, 0.5])


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def load_labels() -> pd.DataFrame:
    return pd.read_parquet(paths.PROCESSED / "labels.parquet")


def load_splits() -> dict:
    return json.loads((paths.PROCESSED / "splits.json").read_text())


def load_track_arrays(track_keys: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
    """track_key -> float32 array [n_bins, N_CH] of post-burn-in bins (NaN where off-road)."""
    cols = ["track_key", "bin"] + CHANNELS + ["burn_in"]
    ch = pd.read_parquet(paths.PROCESSED / "channels.parquet", columns=cols)
    ch["track_key"] = ch["track_key"].astype(str)
    if track_keys is not None:
        ch = ch[ch["track_key"].isin(set(track_keys))]
    ch = ch[~ch["burn_in"]].sort_values(["track_key", "bin"])
    arrays: Dict[str, np.ndarray] = {}
    values = ch[CHANNELS].to_numpy(dtype=np.float32)
    keys = ch["track_key"].to_numpy()
    if len(keys) == 0:
        return arrays
    bounds = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        arrays[keys[a]] = values[a:b]
    return arrays


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
@dataclass
class RobustNorm:
    median: np.ndarray   # (N_VALUE,)
    scale: np.ndarray    # (N_VALUE,)  1.4826 * MAD, floored

    @classmethod
    def fit(cls, arrays: Dict[str, np.ndarray], keys: Sequence[str]) -> "RobustNorm":
        """Median and 1.4826·MAD per value channel over valid bins.

        The three leader channels sit at their cap whenever no leader is present (most bins),
        so their statistics are taken over bins WITH a leader; every scale is floored at a
        physically meaningful minimum (``SCALE_FLOOR``) so a degenerate MAD cannot blow up a channel.
        """
        chunks = [arrays[k] for k in keys if k in arrays]
        x = np.concatenate(chunks, axis=0)
        valid = x[:, IDX_VALID] > 0
        v = x[valid][:, :N_VALUE].astype(np.float64)
        has_lead = x[valid][:, IDX_LEAD] > 0
        med = np.nanmedian(v, axis=0)
        mad = np.nanmedian(np.abs(v - med), axis=0)
        for j in LEAD_CHANNEL_IDX:
            if has_lead.sum() > 100:
                med[j] = np.nanmedian(v[has_lead, j])
                mad[j] = np.nanmedian(np.abs(v[has_lead, j] - med[j]))
        scale = np.maximum(1.4826 * mad, SCALE_FLOOR)
        return cls(median=med, scale=scale)

    def apply(self, x: np.ndarray) -> np.ndarray:
        out = x.copy()
        out[:, :N_VALUE] = (out[:, :N_VALUE] - self.median) / self.scale
        out[:, :N_VALUE] = np.nan_to_num(out[:, :N_VALUE], nan=0.0, posinf=0.0, neginf=0.0)
        out[:, :N_VALUE] = np.clip(out[:, :N_VALUE], -CLIP_Z, CLIP_Z)
        return out.astype(np.float32)

    def to_json(self) -> dict:
        return {"channels": VALUE_CHANNELS, "median": self.median.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_json(cls, d: dict) -> "RobustNorm":
        return cls(median=np.array(d["median"]), scale=np.array(d["scale"]))


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #
def window_starts(n: int, T: int, stride: int) -> List[int]:
    if n <= T:
        return [0]
    starts = list(range(0, n - T + 1, stride))
    if starts[-1] != n - T:
        starts.append(n - T)
    return starts


def eligible_tracks(labels: pd.DataFrame, min_valid_bins: int = MIN_VALID_BINS, require_quality: bool = True) -> pd.DataFrame:
    """Tracks long enough to window; by default also passing the label-free quality gate (deepdas.data.quality)."""
    ok = labels["n_valid_bins"] >= min_valid_bins
    if require_quality and "quality_ok" in labels.columns:
        ok &= labels["quality_ok"].astype(bool)
    return labels[ok].reset_index(drop=True)


class WindowDataset(Dataset):
    """Windows [N_CH, T] from normalised track arrays."""

    def __init__(self, arrays: Dict[str, np.ndarray], keys: Sequence[str], norm: RobustNorm, T: int, stride: int):
        self.T = T
        self.keys = [k for k in keys if k in arrays]
        self.tracks = [norm.apply(arrays[k]) for k in self.keys]
        self.index: List[Tuple[int, int]] = []
        for i, x in enumerate(self.tracks):
            for s in window_starts(len(x), T, stride):
                self.index.append((i, s))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, j: int):
        i, s = self.index[j]
        x = self.tracks[i]
        seg = x[s:s + self.T]
        if len(seg) < self.T:
            pad = np.zeros((self.T - len(seg), N_CH), dtype=np.float32)   # zeros = channel medians, m_valid = 0
            seg = np.concatenate([seg, pad], axis=0)
        return torch.from_numpy(seg.T.copy()), i, s     # [N_CH, T]


def masked_mse(x_hat: torch.Tensor, x: torch.Tensor, per_channel: bool = False):
    """MSE over the target channels and valid steps.  x: [B, N_CH, T], x_hat: [B, N_TARGET, T]."""
    m = x[:, IDX_VALID:IDX_VALID + 1, :]                       # [B, 1, T]
    err = (x_hat - x[:, TARGET_IDX, :]) ** 2 * m              # [B, N_TARGET, T]
    denom = m.sum(dim=(1, 2)).clamp_min(1.0)                   # valid steps per window
    per_ch = err.sum(dim=2) / denom[:, None]                  # [B, N_TARGET]
    total = per_ch.mean(dim=1)                                 # [B]
    return (total, per_ch) if per_channel else total
