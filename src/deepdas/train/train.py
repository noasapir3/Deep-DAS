"""
Training loop for the convolutional autoencoder (normal windows only, early stopping on val_A).

Resumable: ``checkpoint_last.pt`` holds model, optimiser, scheduler, epoch, best loss
and the RNG state, so a run interrupted by ``--time-budget`` continues bit-for-bit.
"""
from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .. import paths
from ..data.dataset import (IDX_VALID, TARGET_CHANNELS, TARGET_IDX, RobustNorm, WindowDataset, eligible_tracks, load_labels,
                            load_track_arrays, masked_mse)
from ..model.conv_ae import ConvAE


@dataclass
class TrainConfig:
    run_id: str = "cae_primary"
    seed: int = 0
    T: int = 64
    stride_train: int = 16
    stride_val: int = 16
    bottleneck: int = 8
    width: int = 32
    epochs: int = 100
    patience: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    min_valid_bins: int = 32
    lr_factor: float = 0.5
    lr_patience: int = 3
    bounded: bool = True          # range-squashed decoder output (see deepdas.model.conv_ae)
    range_pct: float = 0.5        # decoder range = [p, 100-p] percentiles of the normal training bins
    notes: str = ""

    @property
    def run_dir(self) -> Path:
        return paths.EXPERIMENTS / self.run_id


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _split_keys(labels: pd.DataFrame, min_valid_bins: int) -> Dict[str, List[str]]:
    el = eligible_tracks(labels, min_valid_bins)
    normal = el[el["role"] == "normal"]
    return {name: sorted(normal.loc[normal["split"] == name, "track_key"]) for name in ("train", "val_A", "val_B", "test")}


@torch.no_grad()
def channel_error_scale(model: ConvAE, loader: DataLoader) -> np.ndarray:
    """Median per-channel window error on the loader's (normal) windows."""
    model.eval()
    parts = []
    for x, _, _ in loader:
        _, pc = masked_mse(model(x), x, per_channel=True)
        parts.append(pc.numpy())
    return np.maximum(np.median(np.concatenate(parts), axis=0), 1e-4)


@torch.no_grad()
def evaluate_loss(model: ConvAE, loader: DataLoader) -> float:
    model.eval()
    tot, n = 0.0, 0
    for x, _, _ in loader:
        err = masked_mse(model(x), x)
        tot += float(err.sum())
        n += len(err)
    return tot / max(n, 1)


def train(cfg: TrainConfig, time_budget_s: Optional[float] = None, verbose: bool = True) -> dict:
    t_start = time.time()
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    done_path = run_dir / "done.json"
    if done_path.exists():
        return json.loads(done_path.read_text())

    set_seed(cfg.seed)
    labels = load_labels()
    keys = _split_keys(labels, cfg.min_valid_bins)
    arrays = load_track_arrays(keys["train"] + keys["val_A"])

    norm_path = run_dir / "norm.json"
    if norm_path.exists():
        norm = RobustNorm.from_json(json.loads(norm_path.read_text()))
    else:
        norm = RobustNorm.fit(arrays, keys["train"])
        norm_path.write_text(json.dumps(norm.to_json(), indent=1))

    ds_train = WindowDataset(arrays, keys["train"], norm, cfg.T, cfg.stride_train)
    ds_val = WindowDataset(arrays, keys["val_A"], norm, cfg.T, cfg.stride_val)
    gen = torch.Generator().manual_seed(cfg.seed)
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, generator=gen, num_workers=0, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=1024, shuffle=False, num_workers=0)

    model = ConvAE(T=cfg.T, bottleneck=cfg.bottleneck, width=cfg.width, bounded=cfg.bounded)
    # decoder range = 0.5 / 99.5 percentiles of every target channel over the valid train bins (quality-gated tracks)
    train_valid = np.concatenate([t[t[:, IDX_VALID] > 0][:, TARGET_IDX] for t in ds_train.tracks], axis=0)
    range_lo, range_hi = np.percentile(train_valid, cfg.range_pct, axis=0), np.percentile(train_valid, 100.0 - cfg.range_pct, axis=0)
    model.squash.set_range(range_lo, range_hi)
    nj = json.loads(norm_path.read_text())
    if "range_lo" not in nj:
        nj.update({"range_lo": range_lo.tolist(), "range_hi": range_hi.tolist(), "range_margin": model.squash.margin})
        norm_path.write_text(json.dumps(nj, indent=1))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=1e-5)

    history: List[dict] = []
    epoch0, best_val, best_epoch, bad_epochs = 0, float("inf"), -1, 0
    ckpt_last = run_dir / "checkpoint_last.pt"
    if ckpt_last.exists():
        st = torch.load(ckpt_last, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"]); sched.load_state_dict(st["scheduler"])
        epoch0, best_val, best_epoch, bad_epochs = st["epoch"] + 1, st["best_val"], st["best_epoch"], st["bad_epochs"]
        history = st["history"]
        torch.set_rng_state(st["rng_torch"]); np.random.set_state(st["rng_numpy"]); random.setstate(st["rng_python"])
        gen.set_state(st["rng_loader"])
        if verbose:
            print(f"resumed {cfg.run_id} at epoch {epoch0} (best val {best_val:.5f} @ {best_epoch})")

    info = {"n_train_tracks": len(ds_train.keys), "n_val_tracks": len(ds_val.keys), "n_train_windows": len(ds_train),
            "n_val_windows": len(ds_val), "n_params": model.n_params()}
    if verbose:
        print(json.dumps(info))

    stop_reason = "max_epochs"
    for epoch in range(epoch0, cfg.epochs):
        te = time.time()
        model.train()
        tot, n = 0.0, 0
        for x, _, _ in dl_train:
            opt.zero_grad(set_to_none=True)
            err = masked_mse(model(x), x)
            loss = err.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(err.sum()); n += len(err)
        train_loss = tot / max(n, 1)
        val_loss = evaluate_loss(model, dl_val)
        sched.step(val_loss)
        improved = val_loss < best_val - 1e-6
        if improved:
            best_val, best_epoch, bad_epochs = val_loss, epoch, 0
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch, "val_loss": val_loss}, run_dir / "checkpoint_best.pt")
        else:
            bad_epochs += 1
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": opt.param_groups[0]["lr"],
                        "seconds": time.time() - te, "improved": int(improved)})
        with open(run_dir / "history.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(history[0].keys())); w.writeheader(); w.writerows(history)
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "epoch": epoch, "best_val": best_val, "best_epoch": best_epoch, "bad_epochs": bad_epochs, "history": history,
                    "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state(), "rng_python": random.getstate(),
                    "rng_loader": gen.get_state()}, ckpt_last)
        if verbose:
            print(f"epoch {epoch:3d} train {train_loss:.5f} val {val_loss:.5f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"{'*' if improved else ''} ({time.time() - te:.0f}s)", flush=True)
        if bad_epochs >= cfg.patience:
            stop_reason = "early_stopping"
            break
        if time_budget_s is not None and time.time() - t_start > time_budget_s:
            if verbose:
                print("time budget reached — run again to resume")
            return {"status": "paused", "epoch": epoch, "best_val": best_val}

    # per-channel error scale of the best model on val_A normal windows (used to standardise scores)
    best = torch.load(run_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    scale = channel_error_scale(model, dl_val)
    (run_dir / "error_scale.json").write_text(json.dumps({"channels": list(TARGET_CHANNELS), "median_err_valA": scale.tolist()}, indent=1))

    result = {"status": "done", "stop_reason": stop_reason, "epochs_run": len(history), "best_epoch": best_epoch,
              "best_val_loss": best_val, "final_train_loss": history[-1]["train_loss"] if history else None,
              "train_seconds_total": float(sum(h["seconds"] for h in history)), **info, "config": asdict(cfg)}
    done_path.write_text(json.dumps(result, indent=1))
    return result
