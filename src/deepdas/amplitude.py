"""
DAS amplitude channel: binning of detection amplitudes onto the 0.1 s track bins, the 14-channel array layout, its normalisation,
the amplitude-aware convolutional autoencoder used to initialise the detector encoder, and its training loss.

Array layout per track (post-burn-in bins):
    [0:13]  the 13 primary channels, unchanged (8 values + 5 masks, raw physical units)
    [13]    das_amp_log10  (raw log10 amplitude; NaN where not valid)
    [14]    amp_valid      (INTERNAL ONLY: 1 where m_valid > 0 and a valid amplitude exists; never a network input)
Network input = normalised [0:13] (RobustNorm of the 13-channel model) + normalised amplitude -> 14 channels.
Autoencoder targets = the 6 value targets + amplitude -> 7 channels.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from deepdas.data.channels import CHANNELS
from deepdas.data.dataset import (CLIP_Z, IDX_VALID, N_CH, N_VALUE, TARGET_CHANNELS, TARGET_IDX, RobustNorm, window_starts)
from deepdas.model.conv_ae import ConvAE

AMP = "das_amp_log10"
IDX_AMP = N_CH            # 13 (raw array)
IDX_AMPV = N_CH + 1       # 14 (raw array, internal)
IN_CH = N_CH + 1          # 14 network inputs
OUT_CH = len(TARGET_CHANNELS) + 1   # 7 targets
AMP_SCALE_FLOOR = 0.05    # log10 units; numerical safeguard only (train MAD-based scale is ~0.3)
INPUT_CHANNELS = list(CHANNELS) + [AMP]
TARGETS = list(TARGET_CHANNELS) + [AMP]


def attach_amplitude(ch: pd.DataFrame, amp: pd.DataFrame) -> pd.DataFrame:
    """ch: rows of channels (track_key, bin, 13 channels, burn_in); amp: (track_key, bin, amp_log10) on the SAME DL bins.
    Amplitude is valid only where m_valid > 0, m_das > 0 and amp_log10 is finite; otherwise NaN + amp_valid = 0."""
    x = ch.merge(amp[["track_key", "bin", "amp_log10"]], on=["track_key", "bin"], how="left")
    ok = (x["m_valid"] > 0) & (x["m_das"] > 0) & np.isfinite(x["amp_log10"].to_numpy(np.float64))
    x[AMP] = np.where(ok, x["amp_log10"], np.nan).astype(np.float32)
    x["amp_valid"] = ok.astype(np.float32)
    return x.drop(columns=["amp_log10"])


def to_arrays(x: pd.DataFrame) -> Dict[str, np.ndarray]:
    x = x[~x["burn_in"]].sort_values(["track_key", "bin"])
    vals = x[list(CHANNELS) + [AMP, "amp_valid"]].to_numpy(np.float32)
    keys = x["track_key"].to_numpy(); out = {}
    if len(keys):
        b = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])
        for a, c in zip(b[:-1], b[1:]):
            out[keys[a]] = vals[a:c]
    return out


class AmpNorm:
    """RobustNorm for the 13 channels + global robust scaling of log10 amplitude (fitted on valid training bins only)."""

    def __init__(self, base: RobustNorm, amp_median: float, amp_scale: float):
        self.base, self.amp_median, self.amp_scale = base, float(amp_median), float(amp_scale)

    @classmethod
    def fit(cls, arrays, keys):
        base = RobustNorm.fit({k: arrays[k][:, :N_CH] for k in keys if k in arrays}, keys)
        a = np.concatenate([arrays[k][arrays[k][:, IDX_AMPV] > 0, IDX_AMP] for k in keys if k in arrays]).astype(np.float64)
        med = float(np.median(a)); mad = float(np.median(np.abs(a - med)))
        return cls(base, med, max(1.4826 * mad, AMP_SCALE_FLOOR)), {"n_valid_train_bins": int(len(a)), "raw_mad": mad}

    def apply(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """-> (network input [n, 14] float32, amp_valid [n])"""
        x13 = self.base.apply(arr[:, :N_CH])
        v = arr[:, IDX_AMPV] > 0
        amp = np.zeros(len(arr), np.float32)
        amp[v] = np.clip((arr[v, IDX_AMP] - self.amp_median) / self.amp_scale, -CLIP_Z, CLIP_Z)   # missing -> 0 AFTER normalisation
        return np.concatenate([x13, amp[:, None]], axis=1).astype(np.float32), v.astype(np.float32)

    def to_json(self):
        return {**self.base.to_json(), "amp_channel": AMP, "amp_transform": "per detection log10(A) (A>0, finite), mean per DL 0.1 s bin",
                "amp_median": self.amp_median, "amp_scale": self.amp_scale, "amp_scale_rule": f"max(1.4826*MAD, {AMP_SCALE_FLOOR})",
                "amp_clip_z": CLIP_Z, "amp_missing_fill": 0.0}

    @classmethod
    def from_json(cls, d):
        return cls(RobustNorm.from_json(d), d["amp_median"], d["amp_scale"])


class AmpWindows(Dataset):
    """Windows: input [14, T], amp_valid [T]; zero padding for short tracks (m_valid = 0, amp_valid = 0 on the pad)."""

    def __init__(self, arrays, keys: Sequence[str], norm: AmpNorm, T: int, stride: int):
        self.T = T; self.keys = [k for k in keys if k in arrays]
        prep = [norm.apply(arrays[k]) for k in self.keys]
        self.tracks = [p[0] for p in prep]; self.valid = [p[1] for p in prep]
        self.index: List[Tuple[int, int]] = [(i, s) for i, x in enumerate(self.tracks) for s in window_starts(len(x), T, stride)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, j):
        i, s = self.index[j]
        seg = self.tracks[i][s:s + self.T]; v = self.valid[i][s:s + self.T]
        if len(seg) < self.T:
            seg = np.concatenate([seg, np.zeros((self.T - len(seg), IN_CH), np.float32)]); v = np.concatenate([v, np.zeros(self.T - len(v), np.float32)])
        return torch.from_numpy(seg.T.copy()), torch.from_numpy(v.copy()), i, s


def build_model(T=64, bottleneck=8, width=32, bounded=True) -> ConvAE:
    return ConvAE(in_ch=IN_CH, out_ch=OUT_CH, T=T, bottleneck=bottleneck, width=width, bounded=bounded)


def masked_mse_amp(x_hat, x, amp_valid, per_channel=False):
    """Loss on the 6 value targets (mask m_valid, per-channel mean over valid steps) + the amplitude target with the weight of one
    channel (mask m_valid * amp_valid, mean over its own valid steps).  A window without valid amplitude gets the 6-target loss:
    total = (sum_6 + has_amp * amp_term) / (6 + has_amp)."""
    m = x[:, IDX_VALID:IDX_VALID + 1, :]
    err6 = (x_hat[:, :6] - x[:, TARGET_IDX, :]) ** 2 * m
    per6 = err6.sum(dim=2) / m.sum(dim=(1, 2)).clamp_min(1.0)[:, None]
    ma = m[:, 0] * amp_valid
    na = ma.sum(dim=1)
    has = (na > 0).float()
    amp_t = ((x_hat[:, 6] - x[:, IN_CH - 1]) ** 2 * ma).sum(dim=1) / na.clamp_min(1.0)
    total = (per6.sum(dim=1) + has * amp_t) / (6.0 + has)
    if per_channel:
        return total, torch.cat([per6, amp_t[:, None]], dim=1), has
    return total


def amp_on_bins(rows50, aud, track_key, tid):
    """Mean log10 DAS amplitude per 0.1 s DL bin of one track.
    rows50: the 50 Hz track rows of the export (read_export); aud: DAS rows of the tracking audit (t, tid, amp).
    A detection is assigned to the bin of the DL DAS row with the same 0.02 s tick; if none, to the nearest DAS row within 0.015 s.
    Returns (per-bin frame [track_key, bin, amp_log10, n_det], counters)."""
    rr = rows50[rows50.track_key == track_key].sort_values("t"); t = rr.t.to_numpy(np.float64); t0 = float(t.min())
    dl = pd.DataFrame({"tick": np.round(t / 0.02).astype(np.int64), "bin": np.floor((t - t0) / 0.1 + 1e-6).astype(np.int32), "has_das": rr.has_das.to_numpy()})
    dd = dl[dl.has_das].drop_duplicates("tick")
    a = aud[aud.tid == tid].copy(); a["tick"] = np.round(a.t / 0.02).astype(np.int64)
    m = a.merge(dd[["tick", "bin"]], on="tick", how="left"); miss = m.bin.isna(); fb = 0
    if miss.any() and len(dd) >= 2:
        td = t[rr.has_das.to_numpy()]; bd = dl.bin.to_numpy()[rr.has_das.to_numpy()]; tm = m.loc[miss, "t"].to_numpy()
        j = np.clip(np.searchsorted(td, tm), 1, len(td) - 1); jj = np.where(np.abs(td[j - 1] - tm) <= np.abs(td[j] - tm), j - 1, j)
        near = np.abs(td[jj] - tm) <= 0.015; m.loc[m.index[miss][near], "bin"] = bd[jj[near]]; fb = int(near.sum())
    ok = np.isfinite(m.amp) & (m.amp > 0); mm = m[m.bin.notna() & ok]
    b = mm.assign(l=np.log10(mm.amp)).groupby("bin").agg(amp_log10=("l", "mean"), n_det=("l", "size")).reset_index()
    b.insert(0, "track_key", track_key); b["bin"] = b.bin.astype(np.int32)
    return b, {"track_key": track_key, "audit_das": len(m), "fallback": fb, "unmatched": int(m.bin.isna().sum()), "invalid_amp": int((~ok).sum())}
