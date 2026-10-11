"""
The detector network and its fixed preprocessing.

A track (or a group of tracks of one unit) is cut into 64-bin windows (stride 8). Every window is encoded by the ConvAE encoder into an
8-d code; the codes of a unit are pooled into mean and standard deviation (16 values), standardised with constants fitted on the training
normals (fmu, fsd), concatenated with the standardised tailgating feature, and passed to a small MLP head that outputs one logit.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..data.channels import CHANNELS
from . import phi_mode as _phi_mode

CI = {c: i for i, c in enumerate(CHANNELS)}
T, STRIDE, BIN_S = 64, 8, 0.1
MIN_DECISION_WINDOWS = 3          # fewer windows -> no decision


def clean_lead_artifacts(a: np.ndarray) -> np.ndarray:
    """Remove a tracker artefact: a 'leader' closer than 1 m while the vehicle drives faster than 5 m/s is a merged/duplicate track,
    not a real vehicle ahead. Such bins get the no-leader values (m_lead 0, gap 60 m, headway 10 s, dv 0)."""
    a = a.copy()
    bad = (a[:, CI["m_lead"]] > 0.5) & (a[:, CI["g_lead"]] < 1.0) & (a[:, CI["v_lon"]] > 5) & (a[:, CI["m_valid"]] > 0)
    a[bad, CI["m_lead"]] = 0.0; a[bad, CI["g_lead"]] = 60.0; a[bad, CI["th_lead"]] = 10.0; a[bad, CI["dv_lead"]] = 0.0
    return a


def tailgating_fraction(arrs) -> float:
    """Share of valid bins with a leader, time headway < 0.8 s and speed > 5 m/s (the one extra input of the head)."""
    a = np.concatenate(arrs) if arrs else np.zeros((0, len(CHANNELS) + 2))
    v = a[:, CI["m_valid"]] > 0
    if v.sum() == 0:
        return 0.0
    tg = v & (a[:, CI["m_lead"]] > 0.5) & (a[:, CI["th_lead"]] < 0.8) & (a[:, CI["v_lon"]] > 5)
    return float(tg.sum() / v.sum())


def pool(z, seg, n):
    """Mean and standard deviation of the window codes of each unit (seg = unit index of every window)."""
    cnt = torch.zeros(n).index_add_(0, seg, torch.ones(len(seg)))[:, None]
    mean = torch.zeros(n, z.shape[1]).index_add_(0, seg, z) / cnt
    var = torch.zeros(n, z.shape[1]).index_add_(0, seg, (z - mean[seg]) ** 2) / cnt
    return torch.cat([mean, torch.sqrt(var + 1e-8)], 1)


class Detector(nn.Module):
    """Encoder (from the amplitude ConvAE) + pooled-code standardisation + MLP head. n_extra = number of extra unit features (1: tailgating)."""

    def __init__(self, ae, n_extra: int):
        super().__init__()
        self.enc, self.to_z, self.K = ae.enc, ae.to_z, n_extra
        self.register_buffer("fmu", torch.zeros(16)); self.register_buffer("fsd", torch.ones(16))
        self.register_buffer("emu", torch.zeros(max(n_extra, 1))); self.register_buffer("esd", torch.ones(max(n_extra, 1)))
        self.head = nn.Sequential(nn.Linear(16 + n_extra, 16), nn.GELU(), nn.Linear(16, 1))
        self.phi_mode = _phi_mode.current()        # 'active' (as trained) or 'neutralized' (inference-time intervention, see phi_mode.py)

    def codes(self, x):
        return self.to_z(self.enc(x).flatten(1))

    def head_in(self, f, e):
        h = (f - self.fmu) / self.fsd
        if not self.K:
            return h
        x = (e - self.emu[:self.K]) / self.esd[:self.K]
        if self.phi_mode == "neutralized":        # standardised phi input held at 0; weights and stored constants unchanged
            x = torch.zeros_like(x)
        return torch.cat([h, x], 1)

    def forward(self, x, seg, n, e):
        return self.head(self.head_in(pool(self.codes(x), seg, n), e)).squeeze(1)

    def score(self, x, e):
        """Logit of ONE unit from all of its windows x [n_windows, 14, T] and its extra features e [1, K]."""
        f = pool(self.codes(x), torch.zeros(len(x), dtype=torch.long), 1)
        return float(self.head(self.head_in(f, e)).squeeze())


def load_detector(checkpoint_path, n_extra: int) -> Detector:
    from ..amplitude import build_model
    d = Detector(build_model(), n_extra); ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mk = d.load_state_dict(ck["model"], strict=False)
    assert not mk.unexpected_keys and set(mk.missing_keys) <= {"emu", "esd"}, mk
    return d.eval()
