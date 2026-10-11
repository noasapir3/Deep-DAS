"""Unit tests: autoencoder shapes, masked loss, windows, normalisation, seed determinism."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from deepdas.data.channels import CHANNELS, VALUE_CHANNELS
from deepdas.data.dataset import (CLIP_Z, IDX_VALID, N_CH, N_TARGET, N_VALUE, TARGET_IDX, RobustNorm, WindowDataset, masked_mse,
                                  window_starts)
from deepdas.model.conv_ae import ConvAE
from deepdas.train.train import set_seed


def test_tensor_layout_is_the_13_channels():
    assert N_CH == 13 and N_VALUE == 8 and N_TARGET == 6 and CHANNELS[IDX_VALID] == "m_valid"
    assert CHANNELS[:8] == VALUE_CHANNELS
    assert "a_lon" not in [CHANNELS[i] for i in TARGET_IDX] and "v_lat" not in [CHANNELS[i] for i in TARGET_IDX]


@pytest.mark.parametrize("T,d", [(32, 8), (64, 8), (128, 8), (64, 4), (64, 16)])
def test_model_shapes_and_size(T, d):
    m = ConvAE(T=T, bottleneck=d)
    x = torch.randn(5, 13, T)
    y = m(x)
    assert y.shape == (5, 6, T)
    assert m.encode(x).shape == (5, d)
    if T == 64 and d == 8:
        assert 120_000 < m.n_params() < 150_000
    # bounded decoder: outputs stay inside the configured per-channel range
    m.squash.set_range([-3.0] * 6, [3.0] * 6)
    y = m(torch.randn(50, 13, T) * 20)
    assert y.min() >= -3.5 - 1e-6 and y.max() <= 3.5 + 1e-6


def test_masked_mse_ignores_invalid_steps_and_uses_target_channels():
    x = torch.zeros(2, 13, 8)
    x[:, IDX_VALID, :] = 1.0
    x[0, IDX_VALID, 4:] = 0.0                 # second half padded
    x[:, CHANNELS.index("a_lon"), :] = 7.0    # non-target channel: must not count
    x_hat = torch.zeros(2, N_TARGET, 8)
    x_hat[0, :, 4:] = 100.0                   # huge error only on padded steps
    x_hat[1, 0, :] = 1.0                      # unit error on one target channel, all steps
    err, per_ch = masked_mse(x_hat, x, per_channel=True)
    assert err[0].item() == 0.0
    assert abs(err[1].item() - 1.0 / N_TARGET) < 1e-6 and abs(per_ch[1, 0].item() - 1.0) < 1e-6


def test_window_starts_cover_the_tail():
    assert window_starts(10, 64, 16) == [0]
    assert window_starts(64, 64, 16) == [0]
    assert window_starts(100, 64, 16) == [0, 16, 32, 36]
    assert window_starts(96, 64, 16) == [0, 16, 32]


def test_window_padding_and_mask():
    arr = np.ones((20, N_CH), dtype=np.float32)
    arr[:, IDX_VALID] = 1.0
    norm = RobustNorm(median=np.zeros(N_VALUE), scale=np.ones(N_VALUE))
    ds = WindowDataset({"k": arr}, ["k"], norm, T=64, stride=16)
    x, i, s = ds[0]
    assert x.shape == (N_CH, 64) and len(ds) == 1
    assert x[IDX_VALID, :20].sum() == 20 and x[IDX_VALID, 20:].sum() == 0
    assert (x[:N_VALUE, 20:] == 0).all()


def test_robust_norm_fit_apply_and_clip():
    rng = np.random.default_rng(0)
    n = 5000
    arr = np.zeros((n, N_CH), dtype=np.float32)
    arr[:, :N_VALUE] = rng.normal(loc=np.arange(N_VALUE) * 3.0, scale=np.arange(1, N_VALUE + 1) * 2.0, size=(n, N_VALUE))
    arr[:, CHANNELS.index("m_lead")] = 1.0
    arr[:, IDX_VALID] = 1.0
    arr[:10, 1] = 1e6                          # gross outliers must not move the robust statistics much
    norm = RobustNorm.fit({"a": arr}, ["a"])
    z = norm.apply(arr)
    assert np.all(np.abs(np.median(z[:, :N_VALUE], axis=0)) < 0.1)
    assert np.all(np.abs(np.abs(z[10:, :N_VALUE]).mean(0) - 0.8) < 0.15)   # E|z| ≈ 0.8 for a unit normal
    assert z[:10, 1].max() <= CLIP_Z
    assert np.isfinite(z).all()


def test_seed_makes_training_step_deterministic():
    losses = []
    for _ in range(2):
        set_seed(123)
        m = ConvAE(T=32, bottleneck=4)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        x = torch.randn(16, 13, 32)
        x[:, IDX_VALID, :] = 1.0
        for _ in range(3):
            opt.zero_grad(); loss = masked_mse(m(x), x).mean(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[0] == losses[1]
