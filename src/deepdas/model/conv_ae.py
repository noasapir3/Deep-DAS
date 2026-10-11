"""
1-D convolutional autoencoder for 10 Hz driving windows.

Input  [B, 13, T]  (8 value channels + 5 mask channels, T = 64 by default)
Output [B, 6, T]   (reconstruction of the six target channels v_lon, d_lat, r_lim, g_lead, th_lead, dv_lead;
                    the noise-dominated a_lon and v_lat are inputs only)

Encoder : Conv(13→32, k7) → Conv(32→64, k5, s2) → Conv(64→64, k5, s2) → Conv(64→128, k3, s2) → Linear(128·T/8 → d)
Decoder : mirror with transposed convolutions, final Conv(32→6, k7), then a per-channel
          range squash  y = lo_c + (hi_c − lo_c)·sigmoid(x)  with (lo_c, hi_c) the 0.5 / 99.5
          percentiles of the channel on the normal TRAINING windows (± a 0.5 margin).
GELU activations, no normalisation layers (they would make a window's error depend on the batch).

Why the range squash: a plain MSE autoencoder with a bottleneck of d ≥ number of channels can
pass each channel's window mean through the code and therefore reconstructs *level* anomalies
(a vehicle at twice the speed limit, a constant lateral offset) almost perfectly.  Bounding the
decoder to the range of normal data makes any value outside that range unreconstructable by
construction — the reconstruction error then grows quadratically with the excursion.
"""
from __future__ import annotations

import torch
from torch import nn


class RangeSquash(nn.Module):
    """y = lo + (hi - lo) * sigmoid(x), per channel; (lo, hi) are buffers set from training data."""

    def __init__(self, n_ch: int, margin: float = 0.5):
        super().__init__()
        self.register_buffer("lo", torch.full((1, n_ch, 1), -5.0))
        self.register_buffer("hi", torch.full((1, n_ch, 1), 5.0))
        self.margin = margin

    def set_range(self, lo, hi) -> None:
        lo = torch.as_tensor(lo, dtype=torch.float32).view(1, -1, 1) - self.margin
        hi = torch.as_tensor(hi, dtype=torch.float32).view(1, -1, 1) + self.margin
        self.lo.copy_(lo)
        self.hi.copy_(hi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lo + (self.hi - self.lo) * torch.sigmoid(x)


class ConvAE(nn.Module):
    def __init__(self, in_ch: int = 13, out_ch: int = 6, T: int = 64, bottleneck: int = 8, width: int = 32,
                 bounded: bool = True):
        super().__init__()
        if T % 8 != 0:
            raise ValueError("T must be a multiple of 8")
        w1, w2, w3 = width, width * 2, width * 4
        self.T, self.bottleneck, self.bounded = T, bottleneck, bounded
        self.enc = nn.Sequential(
            nn.Conv1d(in_ch, w1, 7, padding=3), nn.GELU(),
            nn.Conv1d(w1, w2, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(w2, w2, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(w2, w3, 3, stride=2, padding=1), nn.GELU(),
        )
        self.flat = w3 * (T // 8)
        self.to_z = nn.Linear(self.flat, bottleneck)
        self.from_z = nn.Linear(bottleneck, self.flat)
        self.dec = nn.Sequential(
            nn.GELU(),
            nn.ConvTranspose1d(w3, w2, 3, stride=2, padding=1, output_padding=1), nn.GELU(),
            nn.ConvTranspose1d(w2, w2, 5, stride=2, padding=2, output_padding=1), nn.GELU(),
            nn.ConvTranspose1d(w2, w1, 5, stride=2, padding=2, output_padding=1), nn.GELU(),
            nn.Conv1d(w1, out_ch, 7, padding=3),
        )
        self.squash = RangeSquash(out_ch)
        self._w3 = w3

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)
        return self.to_z(h.flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_z(z).view(z.shape[0], self._w3, self.T // 8)
        y = self.dec(h)
        return self.squash(y) if self.bounded else y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
