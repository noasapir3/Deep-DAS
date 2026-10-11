"""
Qualitative real-data check: the single Klausner bus track of the Final Project.

The export folder (``data/fp/klausner``) is processed with exactly the pipeline
used for the simulated exports (import → 10 Hz channels → windows → primary
autoencoder), and the shipped Z-score baseline's published score is shown next
to it.  One unlabeled track cannot validate a detector; the figure only shows
that the pipeline runs end-to-end on real fused data and where the error sits.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from .. import paths
from ..data.channels import VALUE_CHANNELS, build_channels, CHANNELS
from ..data.dataset import CHANNELS as DS_CHANNELS, IDX_VALID, WindowDataset, masked_mse, window_starts
from ..data.import_exports import ExportRef, read_export
from ..data.map_matching import LaneIndex
from .scoring import load_run
from .style import DL_COLOR, NEUTRAL, NEUTRAL_LIGHT, TEXT_SECONDARY, ZS_COLOR, apply_style

KL = paths.FP_DATA / "klausner"


def klausner_channels() -> tuple:
    ref = ExportRef(group="real", scenario_id="klausner/bus_real", region="klausner", scenario_type="real_bus",
                    export_dir=KL, scenario_json=KL / "klausner_bus_real.sim.json")
    rows, meta = read_export(ref)
    li = LaneIndex.from_scenario_json(KL / "klausner_bus_real.sim.json")
    ch = build_channels(rows, meta, {"klausner": li})
    return rows, meta, ch


def run(run_id: str = "cae_primary", thr_dl: float = None, stride: int = 8) -> dict:
    apply_style()
    rows, meta, ch = klausner_channels()
    _, cfg, norm, model = load_run(run_id)
    T = cfg["T"]
    out = {"n_tracks": int(len(meta)), "tracks": []}
    fig, axes = plt.subplots(5, 1, figsize=(8.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 0.7, 1.1]})
    for tk, g in ch.groupby("track_key", observed=True):
        g = g[~g["burn_in"]].sort_values("bin")
        arr = g[DS_CHANNELS].to_numpy(dtype=np.float32)
        x = norm.apply(arr)
        starts = window_starts(len(x), T, stride)
        errs, centres = [], []
        with torch.no_grad():
            for s in starts:
                seg = x[s:s + T]
                if len(seg) < T:
                    seg = np.concatenate([seg, np.zeros((T - len(seg), x.shape[1]), dtype=np.float32)])
                xt = torch.from_numpy(seg.T[None].copy())
                errs.append(float(masked_mse(model(xt), xt)[0]))
                centres.append(g["t_rel"].iloc[min(s + T // 2, len(g) - 1)])
        errs = np.array(errs)
        score = float(np.percentile(errs, 95)) if len(errs) else float("nan")
        m = meta[meta["track_key"] == tk].iloc[0]
        out["tracks"].append({"track_key": tk, "duration_s": float(m["duration_s"]), "n_windows": len(errs), "p95_error": score,
                              "max_error": float(errs.max()) if len(errs) else float("nan"), "flagged_p99": bool(thr_dl is not None and score > thr_dl),
                              "n_das_meas": int(m.get("n_das_meas", 0)), "n_cam_meas": int(m.get("n_cam_meas", 0))})
        t = g["t_rel"].to_numpy()
        axes[0].plot(t, g["v_lon"], color=DL_COLOR); axes[0].set_ylabel("v_lon (m/s)")
        axes[1].plot(t, g["a_lon"], color=DL_COLOR); axes[1].set_ylabel("a_lon (m/s²)")
        axes[2].plot(t, g["d_lat"], color=DL_COLOR); axes[2].set_ylabel("d_lat (m)")
        axes[3].fill_between(t, 0, g["m_das"], color=DL_COLOR, alpha=0.5, linewidth=0, label="DAS")
        axes[3].fill_between(t, 0, -g["m_cam"], color=ZS_COLOR, alpha=0.5, linewidth=0, label="camera")
        axes[3].set_ylabel("sensor\nfraction"); axes[3].legend(loc="upper right", fontsize=7)
        axes[4].plot(centres, errs, "o-", color=DL_COLOR, ms=3, label="window error")
        if thr_dl is not None:
            axes[4].axhline(thr_dl, color="#0b0b0b", lw=0.9, ls=(0, (4, 3)), label="p99 threshold")
        axes[4].set_ylabel("masked MSE"); axes[4].set_yscale("log"); axes[4].legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("time since first fused row (s)")
    fig.suptitle("Klausner real bus track through the same pipeline (qualitative; no label exists)", fontsize=10)
    fig.tight_layout()
    fig_dir = paths.OUTPUTS / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "fig_klausner_real_track.png"); fig.savefig(fig_dir / "fig_klausner_real_track.pdf"); plt.close(fig)
    # published Z-score of the same track
    feats = KL / "klausner_real_features.csv"
    if feats.exists():
        from ..baseline.zscore import shipped_model
        from ..fp_anomaly_model.scenario_scorer import score_dataframe
        df = pd.read_csv(feats, low_memory=False)
        sc = score_dataframe(df, shipped_model())
        out["zscore"] = sc[["track_id", "z_score_max", "flagged", "top_feature"]].to_dict("records")
    (paths.OUTPUTS / "klausner_check.json").write_text(json.dumps(out, indent=1))
    return out
