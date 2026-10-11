#!/usr/bin/env python3
"""
Pre-training of the amplitude-aware convolutional autoencoder whose encoder initialises DeepDAS Ensemble v1
(shipped result: models/deepdas_ensemble_v1/autoencoder_init.pt, autoencoder_config.json, autoencoder_training_history.csv).
Configuration = the 13-channel primary autoencoder of the unsupervised study (research/01_unsupervised_autoencoder/runs/cae_primary) plus the
DAS log-amplitude as a 14th input and 7th masked target (deepdas.amplitude). Trained on the windows of the normal-corpus train split,
early stopping on val_A. Resumable; one writer per output directory. Writes only into <out_dir> (never into models/).
Usage: python scripts/pretrain_autoencoder.py <out_dir> [time_budget_s]
"""
import csv, json, os, random, sys, time
from dataclasses import asdict
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
import _setup  # noqa: F401
from deepdas import paths  # noqa: E402
from deepdas.train.train import TrainConfig, set_seed, _split_keys  # noqa: E402
from deepdas.data.dataset import IDX_VALID, TARGET_IDX, load_labels  # noqa: E402
from deepdas.data.channels import CHANNELS  # noqa: E402
import deepdas.amplitude as L  # noqa: E402

if len(sys.argv) < 2:
    sys.exit(__doc__)
RUN = Path(sys.argv[1]).resolve(); RUN.mkdir(parents=True, exist_ok=True)
if paths.MODELS.resolve() in RUN.parents or RUN == paths.MODELS.resolve():
    sys.exit("refusing to write into models/")
lock = RUN / "WRITER.lock"
if lock.exists():
    pid = int(lock.read_text().split()[0])
    try:
        os.kill(pid, 0); sys.exit(f"another writer (pid {pid}) is active in {RUN}; refusing to start")
    except OSError:
        pass
lock.write_text(f"{os.getpid()} {time.ctime()}\n")
budget = float(sys.argv[2]) if len(sys.argv) > 2 else None
t_start = time.time()

primary = json.loads((paths.EXPERIMENTS / "cae_primary" / "config.json").read_text())
cfg = TrainConfig(**{**primary, "run_id": "amplitude_only_candidate", "notes": "primary config + das_amp_log10 input and masked target"})
(RUN / "config.json").write_text(json.dumps({**asdict(cfg), "input_channels": L.INPUT_CHANNELS, "targets": L.TARGETS,
                                              "internal_mask_not_input": "amp_valid", "optimizer": "AdamW (as primary)",
                                              "loss": "masked_mse_amp: (sum of 6 primary per-channel MSEs + has_amp*amp MSE) / (6 + has_amp)"}, indent=1))
if (RUN / "done.json").exists():
    print("already done"); lock.unlink(); sys.exit(0)

set_seed(cfg.seed)
labels = load_labels()
keys = _split_keys(labels, cfg.min_valid_bins)
need = set(keys["train"] + keys["val_A"])
ch = pd.read_parquet(paths.PROCESSED / "channels.parquet", columns=["track_key", "bin"] + CHANNELS + ["burn_in"]); ch["track_key"] = ch.track_key.astype(str)
ch = ch[ch.track_key.isin(need)]
amp = pd.read_parquet(paths.PROCESSED / "amplitude.parquet"); amp["track_key"] = amp.track_key.astype(str)
arrays = L.to_arrays(L.attach_amplitude(ch, amp[amp.track_key.isin(need)]))
del ch

norm_path = RUN / "norm.json"
if norm_path.exists():
    norm = L.AmpNorm.from_json(json.loads(norm_path.read_text()))
else:
    norm, fitinfo = L.AmpNorm.fit(arrays, keys["train"])
    prim = json.loads((paths.EXPERIMENTS / "cae_primary" / "norm.json").read_text())
    same = bool(np.allclose(norm.base.median, prim["median"], rtol=0, atol=1e-9) and np.allclose(norm.base.scale, prim["scale"], rtol=0, atol=1e-9))
    assert same, "13-channel normalisation differs from the primary"
    norm_path.write_text(json.dumps({**norm.to_json(), **fitinfo, "primary_13ch_norm_identical": same}, indent=1))

ds_train = L.AmpWindows(arrays, keys["train"], norm, cfg.T, cfg.stride_train)
ds_val = L.AmpWindows(arrays, keys["val_A"], norm, cfg.T, cfg.stride_val)
gen = torch.Generator().manual_seed(cfg.seed)
dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, generator=gen, num_workers=0)
dl_val = DataLoader(ds_val, batch_size=1024, shuffle=False, num_workers=0)

model = L.build_model(cfg.T, cfg.bottleneck, cfg.width, cfg.bounded)
# decoder output ranges: the 6 value targets as in the primary autoencoder (valid train bins); amplitude over valid-amplitude train bins
value_targets = np.concatenate([t[t[:, IDX_VALID] > 0][:, TARGET_IDX] for t in ds_train.tracks], axis=0)
av = np.concatenate([t[v > 0, L.IN_CH - 1] for t, v in zip(ds_train.tracks, ds_train.valid)])
lo = np.r_[np.percentile(value_targets, cfg.range_pct, axis=0), np.percentile(av, cfg.range_pct)]
hi = np.r_[np.percentile(value_targets, 100 - cfg.range_pct, axis=0), np.percentile(av, 100 - cfg.range_pct)]
model.squash.set_range(lo, hi)
nj = json.loads(norm_path.read_text())
if "range_lo" not in nj:
    prim = json.loads((paths.EXPERIMENTS / "cae_primary" / "norm.json").read_text())
    nj.update({"range_lo": lo.tolist(), "range_hi": hi.tolist(), "range_margin": model.squash.margin,
               "old6_ranges_identical_to_primary": bool(np.allclose(lo[:6], prim["range_lo"]) and np.allclose(hi[:6], prim["range_hi"])),
               "amp_range_from_n_valid_train_bins": int(len(av)),
               "amp_clip_fraction_train": float(np.mean(np.abs(av) >= L.CLIP_Z))})
    norm_path.write_text(json.dumps(nj, indent=1))
opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=1e-5)


@torch.no_grad()
def val_loss(m):
    m.eval(); tot = n = 0
    for x, v, _, _ in dl_val:
        e = L.masked_mse_amp(m(x), x, v); tot += float(e.sum()); n += len(e)
    return tot / max(n, 1)


history, epoch0, best, best_ep, bad = [], 0, float("inf"), -1, 0
last = RUN / "checkpoint_last.pt"
if last.exists():
    st = torch.load(last, map_location="cpu", weights_only=False)
    model.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"]); sched.load_state_dict(st["scheduler"])
    epoch0, best, best_ep, bad, history = st["epoch"] + 1, st["best_val"], st["best_epoch"], st["bad_epochs"], st["history"]
    torch.set_rng_state(st["rng_torch"]); np.random.set_state(st["rng_numpy"]); random.setstate(st["rng_python"]); gen.set_state(st["rng_loader"])
    print(f"resumed at epoch {epoch0}", flush=True)
info = {"n_train_tracks": len(ds_train.keys), "n_val_tracks": len(ds_val.keys), "n_train_windows": len(ds_train), "n_val_windows": len(ds_val),
        "n_params": model.n_params(), "train_windows_with_valid_amp": int(sum(1 for i, s in ds_train.index if ds_train.valid[i][s:s + cfg.T].sum() > 0))}
print(json.dumps(info), flush=True)
stop = "max_epochs"
for epoch in range(epoch0, cfg.epochs):
    te = time.time(); model.train(); tot = n = 0
    for x, v, _, _ in dl_train:
        opt.zero_grad(set_to_none=True)
        e = L.masked_mse_amp(model(x), x, v); e.mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        tot += float(e.sum()); n += len(e)
    vl = val_loss(model); sched.step(vl)
    imp = vl < best - 1e-6
    if imp:
        best, best_ep, bad = vl, epoch, 0
        torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch, "val_loss": vl}, RUN / "checkpoint_best.pt")
    else:
        bad += 1
    history.append({"epoch": epoch, "train_loss": tot / max(n, 1), "val_loss": vl, "lr": opt.param_groups[0]["lr"], "seconds": time.time() - te, "improved": int(imp)})
    with open(RUN / "history.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "epoch": epoch, "best_val": best,
                "best_epoch": best_ep, "bad_epochs": bad, "history": history, "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state(),
                "rng_python": random.getstate(), "rng_loader": gen.get_state()}, last)
    print(f"epoch {epoch:3d} train {history[-1]['train_loss']:.5f} val {vl:.5f} lr {opt.param_groups[0]['lr']:.1e} {'*' if imp else ''} ({time.time() - te:.0f}s)", flush=True)
    if bad >= cfg.patience:
        stop = "early_stopping"; break
    if budget and time.time() - t_start > budget:
        print("time budget reached; rerun to resume", flush=True); lock.unlink(); sys.exit(0)

# per-channel val_A error of the best model (diagnostic only; not used in any decision)
model.load_state_dict(torch.load(RUN / "checkpoint_best.pt", map_location="cpu", weights_only=False)["model"]); model.eval()
pcs = []
with torch.no_grad():
    for x, v, _, _ in dl_val:
        _, pc, has = L.masked_mse_amp(model(x), x, v, per_channel=True); pcs.append(pc.numpy())
pcs = np.concatenate(pcs)
(RUN / "val_A_per_channel_error_DIAGNOSTIC.json").write_text(json.dumps({"channels": L.TARGETS, "median": np.median(pcs, 0).tolist()}, indent=1))
(RUN / "done.json").write_text(json.dumps({"status": "done", "stop_reason": stop, "epochs_run": len(history), "best_epoch": best_ep, "best_val_loss": best,
                                            "train_seconds_total": float(sum(h["seconds"] for h in history)), **info, "config": asdict(cfg)}, indent=1))
lock.unlink(); print("done", flush=True)
