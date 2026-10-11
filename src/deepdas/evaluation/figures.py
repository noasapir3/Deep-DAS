"""Report figures (matplotlib, static).  Every figure reads only DL outputs (experiments/, outputs/tables/)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .. import paths
from ..data.channels import VALUE_CHANNELS
from ..data.dataset import TARGET_CHANNELS
from .metrics import DAS_SIGMA_LABELS, curves
from .style import (DL_COLOR, GRID, NEUTRAL, NEUTRAL_LIGHT, ORDINAL3, SEQ_BLUE, SERIES, TEXT_SECONDARY, THIRD, ZS_COLOR,
                    apply_style, type_label)

FIG = paths.OUTPUTS / "figures"
LEVEL_ORDER = ["mild", "moderate", "severe"]
TYPE_ORDER = ["anomaly_speeding", "anomaly_hard_braking", "anomaly_stalled_vehicle", "anomaly_lane_weaving", "anomaly_tailgating",
              "anomaly_lane_straddling", "anomaly_collision", "anomaly_overweight_vehicle", "anomaly_reckless_driver"]
CHANNEL_LABELS = {"v_lon": "v_lon", "a_lon": "a_lon", "d_lat": "d_lat", "v_lat": "v_lat", "r_lim": "r_lim",
                  "g_lead": "gap", "th_lead": "headway", "dv_lead": "closing v"}


def _save(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / name
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def _thr_line(ax, x, color, label, axis="v"):
    (ax.axvline if axis == "v" else ax.axhline)(x, color=color, lw=1.0, ls=(0, (4, 3)), alpha=0.9)
    if axis == "v":
        ax.text(x, ax.get_ylim()[1], f" {label}", color=color, fontsize=7, va="top", ha="left")


# --------------------------------------------------------------------------- #
def fig_training_curves(run_id: str = "cae_primary", runs: Optional[List[str]] = None) -> Path:
    apply_style()
    runs = runs or [run_id]
    fig, axes = plt.subplots(1, len(runs), figsize=(3.6 * len(runs), 2.9), squeeze=False)
    for ax, rid in zip(axes[0], runs):
        h = pd.read_csv(paths.EXPERIMENTS / rid / "history.csv")
        done = json.loads((paths.EXPERIMENTS / rid / "done.json").read_text())
        ax.plot(h["epoch"], h["train_loss"], color=DL_COLOR, label="train")
        ax.plot(h["epoch"], h["val_loss"], color=ZS_COLOR, label="val_A (early stopping)")
        be = done["best_epoch"]
        ax.plot([be], [h.loc[h["epoch"] == be, "val_loss"].iloc[0]], "o", color=ZS_COLOR, ms=6, mfc="white", mew=1.5)
        ax.annotate(f"best epoch {be}", (be, h.loc[h["epoch"] == be, "val_loss"].iloc[0]), textcoords="offset points",
                    xytext=(6, 8), fontsize=7, color=TEXT_SECONDARY)
        ax.set_xlabel("epoch"); ax.set_ylabel("masked MSE (robust-z units)")
        ax.set_title(rid if len(runs) > 1 else "Training curves (primary model)")
        ax.legend(loc="upper right")
    return _save(fig, "fig_training_curves.png")


def fig_score_distributions(veh: pd.DataFrame, thr_dl: float, thr_zs_fp: float, thr_zs_p99: float) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), sharey=True)
    cats = ["normal (test regions)"] + [type_label(t) for t in TYPE_ORDER] + ["background vehicles"]
    rng = np.random.default_rng(0)
    for ax, col, thr_main, thr_alt, title, color in (
        (axes[0], "dl_score", thr_dl, None, "Autoencoder: track-P95 reconstruction error", DL_COLOR),
        (axes[1], "z_score_max", thr_zs_fp, thr_zs_p99, "Z-score baseline: max |z| over 156 features", ZS_COLOR)):
        for i, cat in enumerate(cats):
            if cat.startswith("normal"):
                g = veh[(veh["role"] == "normal") & (veh["split"] == "test")]
            elif cat.startswith("background"):
                g = veh[veh["role"] == "background"]
            else:
                g = veh[(veh["role"] == "injected") & (veh["anomaly_type"] == TYPE_ORDER[i - 1])]
            x = g[col].to_numpy(dtype=float)
            x = np.clip(x, 1e-3, None)
            y = i + rng.uniform(-0.22, 0.22, len(x))
            c = NEUTRAL if cat.startswith(("normal", "background")) else color
            ax.scatter(x, y, s=9 if len(x) > 50 else 16, color=c, alpha=0.45 if len(x) > 50 else 0.85, linewidths=0, zorder=3)
            if len(x):
                ax.plot([np.median(x)] * 2, [i - 0.32, i + 0.32], color="#0b0b0b", lw=1.2, zorder=4)
        ax.set_xscale("log")
        ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats)
        ax.set_ylim(len(cats) - 0.5, -0.5)          # normal at the top, background at the bottom
        ax.axvline(thr_main, color="#0b0b0b", lw=1.0, ls=(0, (4, 3)))
        ax.text(thr_main, -0.6, " threshold", fontsize=7, color=TEXT_SECONDARY, va="bottom")
        if thr_alt is not None:
            ax.axvline(thr_alt, color=TEXT_SECONDARY, lw=0.9, ls=(0, (1, 2)))
            ax.text(thr_alt, len(cats) - 0.4, " matched-FAR (p99 val_B)", fontsize=7, color=TEXT_SECONDARY, va="top")
        ax.set_title(title); ax.set_xlabel("vehicle score (log scale)")
        ax.grid(True, axis="x"); ax.grid(False, axis="y")
    fig.suptitle("Vehicle scores by category — black tick = median; dashed = operating threshold", fontsize=10, y=1.01)
    return _save(fig, "fig_score_distributions.png")


def fig_roc_pr(veh: pd.DataFrame, metrics: dict, primary: str) -> Path:
    apply_style()
    neg = veh[(veh["role"] == "normal") & (veh["split"] == "test")]
    pos = veh[veh["role"] == "injected"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    for col, color, name, key in (("dl_score", DL_COLOR, "autoencoder", metrics["runs"][primary]),
                                  ("z_score_max", ZS_COLOR, "Z-score (shipped)", metrics["baseline"])):
        c = curves(pos[col].to_numpy(float), neg[col].to_numpy(float))
        axes[0].plot(c["fpr"], c["tpr"], color=color, label=f"{name}  AUROC {key['auroc_all']:.3f} [{key['auroc_all_ci'][0]:.2f}, {key['auroc_all_ci'][1]:.2f}]")
        axes[1].plot(c["recall"], c["precision"], color=color, label=f"{name}  AUPRC {key['auprc_all']:.3f} [{key['auprc_all_ci'][0]:.2f}, {key['auprc_all_ci'][1]:.2f}]")
    axes[0].plot([0, 1], [0, 1], color=NEUTRAL_LIGHT, lw=0.8)
    axes[0].set_xlabel("false-alarm rate (test normal vehicles)"); axes[0].set_ylabel("recall (injected vehicles)")
    axes[0].set_title("ROC")
    axes[0].axvline(0.01, color=NEUTRAL, lw=0.8, ls=(0, (2, 2))); axes[0].text(0.02, 0.97, "1 % FAR", fontsize=7, color=TEXT_SECONDARY, va="top")
    axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision"); axes[1].set_title("Precision–recall")
    axes[1].axhline(len(pos) / (len(pos) + len(neg)), color=NEUTRAL_LIGHT, lw=0.8)
    axes[1].text(0.99, len(pos) / (len(pos) + len(neg)) + 0.02, "prevalence", fontsize=7, color=TEXT_SECONDARY, ha="right")
    for ax in axes:
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.03)
    h0, l0 = axes[0].get_legend_handles_labels(); h1, l1 = axes[1].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.tight_layout()
    return _save(fig, "fig_roc_pr.png")


def fig_recall_vs_severity(recall_tbl: pd.DataFrame, sev: pd.DataFrame, fp_rows: pd.DataFrame) -> Path:
    apply_style()
    fig, axes = plt.subplots(3, 3, figsize=(9.5, 7.6), sharey=True)
    for ax, atype in zip(axes.flat, TYPE_ORDER):
        sub = recall_tbl[(recall_tbl["anomaly_type"] == atype) & (recall_tbl["level"].isin(LEVEL_ORDER))].set_index("level").reindex(LEVEL_ORDER)
        x = np.arange(3)
        for dx, col, color, marker, label in ((-0.12, "dl_recall", DL_COLOR, "o", "autoencoder (p99 val_B)"),
                                             (0.0, "zs_fp_recall", ZS_COLOR, "s", "Z-score, shipped threshold 15.39"),
                                             (0.12, "zs_p99_recall", ZS_COLOR, "D", "Z-score, matched FAR (p99 val_B)")):
            y = sub[col].to_numpy(float)
            lo, hi = sub[col.replace("recall", "ci_lo")].to_numpy(float), sub[col.replace("recall", "ci_hi")].to_numpy(float)
            mfc = "white" if marker == "D" else color
            ax.errorbar(x + dx, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)], fmt=marker, color=color, mfc=mfc, ecolor=color, elinewidth=0.8, capsize=2, ms=5, label=label)
        fp = fp_rows[fp_rows["anomaly_type"] == atype]
        if len(fp):
            for _, r in fp.iterrows():
                ax.text(-0.45, 0.98, f"FP scenario: AE {'✓' if r['dl_flag_p99'] else '✗'}  Z {'✓' if r['zs_flag_fp'] else '✗'}", fontsize=6.5,
                        color=TEXT_SECONDARY, ha="left", va="top")
        ax.set_xticks(x); ax.set_xticklabels(LEVEL_ORDER); ax.set_xlim(-0.5, 2.6); ax.set_ylim(-0.05, 1.08)
        ax.set_title(type_label(atype))
        ax.grid(False, axis="x")
    for ax in axes[:, 0]:
        ax.set_ylabel("recall (3 seeds)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Recall versus injected severity (Wilson 95 % intervals, n = 3 per point)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    return _save(fig, "fig_recall_vs_severity.png")


def fig_far_by_type(far: pd.DataFrame) -> Path:
    apply_style()
    far = far[far["scenario_type"] != "ALL"].copy()
    far["label"] = far["scenario_type"].str.replace("normal_", "").str.replace("_", " ")
    far = far.sort_values("dl_far", ascending=True)
    y = np.arange(len(far))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for dy, col, color, label in ((0.18, "dl", DL_COLOR, "autoencoder (p99 val_B)"), (-0.18, "zs_fp", ZS_COLOR, "Z-score, shipped threshold")):
        v = far[f"{col}_far"].to_numpy(float) * 100
        lo, hi = far[f"{col}_ci_lo"].to_numpy(float) * 100, far[f"{col}_ci_hi"].to_numpy(float) * 100
        ax.barh(y + dy, v, height=0.34, color=color, label=label, linewidth=0)
        ax.errorbar(v, y + dy, xerr=[np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)], fmt="none", ecolor="#0b0b0b", elinewidth=0.7, capsize=2)
    ax.set_yticks(y); ax.set_yticklabels(far["label"] + "  (n=" + far["n"].astype(str) + ")")
    ax.axvline(1.0, color=NEUTRAL, lw=0.8, ls=(0, (2, 2))); ax.text(1.05, len(far) - 0.5, "1 % target", fontsize=7, color=TEXT_SECONDARY)
    ax.set_xlabel("false-alarm rate on held-out normal vehicles (%)"); ax.set_title("False alarms by normal scenario type (test regions)")
    ax.grid(False, axis="y"); ax.legend(loc="lower right")
    return _save(fig, "fig_far_by_scenario_type.png")


def fig_das_sigma(strata: pd.DataFrame, veh: pd.DataFrame) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    x = np.arange(len(DAS_SIGMA_LABELS))
    normal = veh[veh["role"] == "normal"]
    ax = axes[0]
    ax.hist(normal["das_sigma_mean"].dropna(), bins=40, color=DL_COLOR, alpha=0.8, linewidth=0)
    ax.set_xlabel("mean DAS σ of the track (m)"); ax.set_ylabel("normal vehicles"); ax.set_title("DAS noise level across the corpus")
    ax = axes[1]
    for col, color, label in (("dl_normal_score_median", DL_COLOR, "autoencoder"), ("zs_normal_score_median", ZS_COLOR, "Z-score")):
        v = strata[col].to_numpy(float)
        ax.plot(x, v / np.nanmedian(v), "o-", color=color, label=label)
    ax.set_xticks(x); ax.set_xticklabels(DAS_SIGMA_LABELS, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("median normal score (relative)"); ax.set_title("Normal-vehicle scores versus DAS σ"); ax.legend()
    ax = axes[2]
    for col, color, marker, label in (("dl_far_test", DL_COLOR, "o", "AE false-alarm rate"), ("zs_far_test", ZS_COLOR, "s", "Z-score false-alarm rate"),
                                      ("dl_recall", DL_COLOR, "^", "AE recall"), ("zs_recall", ZS_COLOR, "v", "Z-score recall")):
        ax.plot(x, strata[col].to_numpy(float), marker=marker, color=color, ls="-" if "far" in col else (0, (3, 2)), label=label)
    ax.set_xticks(x); ax.set_xticklabels(DAS_SIGMA_LABELS, rotation=25, ha="right", fontsize=7)
    ax.set_ylim(-0.03, 1.05); ax.set_title("Detection versus DAS σ"); ax.legend(fontsize=6.5)
    for i, n in enumerate(strata["n_normal_test"]):
        ax.text(i, 1.0, f"n={n}/{strata['n_injected'].iloc[i]}", fontsize=6, ha="center", color=TEXT_SECONDARY)
    return _save(fig, "fig_das_sigma_strata.png")


def fig_channel_attribution(attr: pd.DataFrame) -> Path:
    apply_style()
    rows = [t for t in TYPE_ORDER if t in attr.index] + [r for r in attr.index if r.startswith("normal")]
    a = attr.loc[rows, [f"attr_{c}" for c in TARGET_CHANNELS]].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seqblue", ["#fcfcfb"] + SEQ_BLUE)
    im = ax.imshow(a, cmap=cmap, vmin=0, vmax=max(0.5, np.nanmax(a)), aspect="auto")
    ax.set_xticks(range(len(TARGET_CHANNELS))); ax.set_xticklabels([CHANNEL_LABELS[c] for c in TARGET_CHANNELS])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([type_label(r) if r.startswith("anomaly") else r for r in rows])
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i, j] >= 0.15:
                ax.text(j, i, f"{a[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if a[i, j] > 0.45 else "#0b0b0b")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("share of reconstruction error (worst 5 % windows)")
    ax.set_title("Which channels carry the error — mean over injected vehicles of each type")
    return _save(fig, "fig_channel_attribution.png")


def fig_reconstruction_examples(run_id: str, examples: List[Dict]) -> Path:
    """examples: [{title, track_key, start}] — windows drawn from experiments/<run>/scores_windows.parquet."""
    apply_style()
    import torch
    from ..data.dataset import WindowDataset, load_track_arrays
    from .scoring import load_run
    _, cfg, norm, model = load_run(run_id)
    T = cfg["T"]
    arrays = load_track_arrays([e["track_key"] for e in examples])
    show = ["v_lon", "d_lat", "r_lim", "g_lead", "th_lead", "dv_lead"]
    fig, axes = plt.subplots(len(show), len(examples), figsize=(3.3 * len(examples), 1.35 * len(show)), sharex=True)
    t = np.arange(T) * 0.1
    for j, e in enumerate(examples):
        x = norm.apply(arrays[e["track_key"]])
        seg = x[e["start"]:e["start"] + T]
        if len(seg) < T:
            seg = np.concatenate([seg, np.zeros((T - len(seg), x.shape[1]), dtype=np.float32)])
        with torch.no_grad():
            rec = model(torch.from_numpy(seg.T[None].copy()))[0].numpy()
        valid = seg[:, -1] > 0
        for i, c in enumerate(show):
            k = VALUE_CHANNELS.index(c)
            kt = TARGET_CHANNELS.index(c)
            ax = axes[i, j]
            ax.plot(t, seg[:, k], color=DL_COLOR, lw=1.3, label="input")
            ax.plot(t, rec[kt], color=ZS_COLOR, lw=1.2, ls=(0, (4, 2)), label="reconstruction")
            if (~valid).any():
                ax.fill_between(t, -20, 20, where=~valid, color=NEUTRAL_LIGHT, alpha=0.5, linewidth=0)
            err = float(np.mean((rec[kt] - seg[:, k])[valid] ** 2)) if valid.any() else float("nan")
            ax.text(0.99, 0.92, f"mse {err:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=TEXT_SECONDARY)
            ax.set_ylim(-6, 6)
            if j == 0:
                ax.set_ylabel(CHANNEL_LABELS[c] + "\n(robust z)", fontsize=7)
            if i == 0:
                ax.set_title(e["title"], fontsize=8.5)
        axes[-1, j].set_xlabel("time in window (s)")
    axes[0, 0].legend(loc="lower left", fontsize=6.5)
    fig.suptitle("Input windows and their reconstructions (grey = padded / invalid steps)", fontsize=10)
    fig.tight_layout()
    return _save(fig, "fig_reconstruction_examples.png")


def fig_sensitivity(runs_summary: pd.DataFrame, primary: str) -> Path:
    """Top: AUROC (bootstrap CI) and recall at 1 % FAR; bottom: normal-window reconstruction loss (val_A)."""
    apply_style()
    groups = [("seed", ["cae_primary", "cae_seed1", "cae_seed2"], lambda r: f"seed {int(r['seed'])}"),
              ("window length", ["cae_T32", "cae_primary", "cae_T128"], lambda r: f"T={int(r['T'])} ({r['T'] / 10:.1f} s)"),
              ("bottleneck width", ["cae_d4", "cae_primary", "cae_d16", "cae_d32"], lambda r: f"d={int(r['bottleneck'])}")]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.4), gridspec_kw={"height_ratios": [1.6, 1]})
    rs = runs_summary.set_index("detector")
    for j, (name, ids, lab) in enumerate(groups):
        ids = [i for i in ids if i in rs.index]
        x = np.arange(len(ids))
        au = rs.loc[ids, "auroc_all"].to_numpy(float)
        ci = [json.loads(v) if isinstance(v, str) else v for v in rs.loc[ids, "auroc_all_ci"]]
        lo, hi = np.array([c[0] for c in ci]), np.array([c[1] for c in ci])
        ax = axes[0, j]
        ax.errorbar(x, au, yerr=[np.clip(au - lo, 0, None), np.clip(hi - au, 0, None)], fmt="o", color=DL_COLOR, capsize=3, label="AUROC (95 % CI)")
        ax.axhline(0.5, color=NEUTRAL_LIGHT, lw=0.8)
        rec = rs.loc[ids, "recall_at_1pct_far_all"].to_numpy(float)
        for xi, r, a in zip(x, rec, au):
            ax.text(xi, hi[list(x).index(xi)] + 0.012, f"recall@1%: {100 * r:.0f} %", ha="center", fontsize=6.5, color=TEXT_SECONDARY)
        ax.set_xticks(x); ax.set_xticklabels([lab(rs.loc[i]) for i in ids]); ax.set_ylim(0.45, 0.80)
        ax.set_title(f"{name}"); ax.grid(False, axis="x")
        if j == 0:
            ax.set_ylabel("AUROC (91 injected vs 632 normal)")
        ax = axes[1, j]
        vl = rs.loc[ids, "best_val_loss"].to_numpy(float)
        ax.bar(x, vl, width=0.5, color=THIRD, linewidth=0)
        for xi, v in zip(x, vl):
            ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=7, color=TEXT_SECONDARY)
        ax.set_xticks(x); ax.set_xticklabels([lab(rs.loc[i]) for i in ids]); ax.set_ylim(0, max(vl) * 1.3); ax.grid(False, axis="x")
        if j == 0:
            ax.set_ylabel("val_A reconstruction loss")
    fig.suptitle("Sensitivity study — detection quality (top) is flat while reconstruction quality (bottom) varies four-fold", fontsize=10)
    fig.tight_layout()
    return _save(fig, "fig_sensitivity.png")


def fig_latent(run_id: str, labels: pd.DataFrame) -> Path:
    apply_style()
    win = pd.read_parquet(paths.EXPERIMENTS / run_id / "scores_windows.parquet")
    zcols = [c for c in win.columns if c.startswith("z") and c[1:].isdigit()]
    win = win.merge(labels[["track_key", "role", "anomaly_type", "split"]], on="track_key", how="left")
    normal = win[(win["role"] == "normal") & (win["split"] == "test")]
    Z = normal[zcols].to_numpy(float)
    mu = Z.mean(0); U, S, Vt = np.linalg.svd(Z - mu, full_matrices=False)
    proj = lambda M: (M - mu) @ Vt[:2].T
    pn = proj(Z)
    fig, axes = plt.subplots(3, 3, figsize=(9, 8.4), sharex=True, sharey=True)
    for ax, atype in zip(axes.flat, TYPE_ORDER):
        ax.scatter(pn[:, 0], pn[:, 1], s=3, color=NEUTRAL_LIGHT, linewidths=0, rasterized=True)
        g = win[(win["role"] == "injected") & (win["anomaly_type"] == atype)]
        if len(g):
            p = proj(g[zcols].to_numpy(float))
            ax.scatter(p[:, 0], p[:, 1], s=8, color=DL_COLOR, alpha=0.8, linewidths=0, rasterized=True)
        ax.set_title(f"{type_label(atype)}  (n windows = {len(g)})", fontsize=8.5)
    for ax in axes[-1]:
        ax.set_xlabel("latent PC 1")
    for ax in axes[:, 0]:
        ax.set_ylabel("latent PC 2")
    fig.suptitle(f"Bottleneck codes (PCA of the {len(zcols)}-d latent) — grey: normal test windows, blue: injected", fontsize=10)
    fig.tight_layout()
    return _save(fig, "fig_latent_pca.png")


def fig_severity_scores(sev: pd.DataFrame, thr_dl: float, thr_zs: float) -> Path:
    """Score against the *realised* severity, one panel per type, colour = nominal level (ordinal ramp)."""
    apply_style()
    metric = {"anomaly_speeding": ("realised_speed_ratio_max", "max speed / limit"), "anomaly_hard_braking": ("realised_decel_peak_mps2", "peak deceleration (m/s²)"),
              "anomaly_stalled_vehicle": ("realised_stop_longest_s", "longest stop (s)"), "anomaly_lane_weaving": ("realised_lateral_abs_max_m", "max |lateral offset| (m)"),
              "anomaly_tailgating": ("realised_headway_p10_s", "10th-pct time headway (s)"), "anomaly_lane_straddling": ("realised_lateral_abs_max_m", "max |lateral offset| (m)"),
              "anomaly_collision": ("realised_collision_rel_speed_mps", "impact speed (m/s)"), "anomaly_overweight_vehicle": ("realised_weight_kg", "weight (kg)"),
              "anomaly_reckless_driver": ("realised_speed_ratio_max", "max speed / limit")}
    fig, axes = plt.subplots(3, 3, figsize=(9.6, 7.8))
    for ax, atype in zip(axes.flat, TYPE_ORDER):
        g = sev[sev["anomaly_type"] == atype]
        col, xlabel = metric[atype]
        for lvl, color in zip(LEVEL_ORDER, ORDINAL3):
            gg = g[g["level"] == lvl]
            ax.scatter(gg[col], gg["dl_score"], color=color, s=28, label=lvl, zorder=3)
        ax.axhline(thr_dl, color="#0b0b0b", lw=0.9, ls=(0, (4, 3)))
        ax.set_yscale("log"); ax.set_xlabel(xlabel, fontsize=8); ax.set_title(type_label(atype))
        if atype == "anomaly_tailgating":
            ax.invert_xaxis()
    for ax in axes[:, 0]:
        ax.set_ylabel("autoencoder score")
    axes[0, 0].legend(title="nominal level", fontsize=7, title_fontsize=7)
    fig.suptitle("Autoencoder score versus realised severity of the injected vehicle (dashed = p99 threshold)", fontsize=10)
    fig.tight_layout()
    return _save(fig, "fig_severity_scores.png")


def fig_event_timelines(run_id: str, thr_dl: float, sev: pd.DataFrame, veh: pd.DataFrame) -> Path:
    """Window error along the injected vehicle's track for the triggered / severe runs (seed 1), against the normal band."""
    apply_style()
    win = pd.read_parquet(paths.EXPERIMENTS / run_id / "scores_windows.parquet")
    labels = pd.read_parquet(paths.PROCESSED / "labels.parquet")[["track_key", "role", "split", "quality_ok", "scenario_id", "vehicle_id_oracle"]]
    win = win.merge(labels, on="track_key", how="left")
    normal = win[(win["role"] == "normal") & (win["split"] == "test") & (win["quality_ok"].astype(bool))]["err"]
    band = (float(normal.median()), float(normal.quantile(0.95)), float(normal.quantile(0.99)))
    cases = [("anomaly_hard_braking", "hard braking (trigger at 6 s)", True), ("anomaly_stalled_vehicle", "stalled vehicle (trigger at 6 s)", True),
             ("anomaly_speeding", "speeding", False), ("anomaly_lane_weaving", "lane weaving", False),
             ("anomaly_tailgating", "tailgating", False), ("anomaly_collision", "collision", False)]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharey=True)
    T = json.loads((paths.EXPERIMENTS / run_id / "config.json").read_text())["T"]
    for ax, (atype, title, trig) in zip(axes.flat, cases):
        ax.axhspan(band[0], band[2], color=NEUTRAL_LIGHT, alpha=0.5, linewidth=0)
        ax.axhline(band[1], color=NEUTRAL, lw=0.8, ls=(0, (2, 2)))
        ax.axhline(thr_dl, color="#0b0b0b", lw=0.9, ls=(0, (4, 3)))
        for lvl, color in zip(LEVEL_ORDER, ORDINAL3):
            runs = sev[(sev["anomaly_type"] == atype) & (sev["level"] == lvl) & (sev["seed"] == 1)]
            for _, r in runs.iterrows():
                key_prefix = f"severity/{r['run_id']}"
                w = win[(win["scenario_id"] == key_prefix) & (win["vehicle_id_oracle"] == r["injected_vehicle"])].sort_values("start")
                if not len(w):
                    continue
                t = (w["start"].to_numpy() + T / 2) * 0.1 + 1.0     # window centre, seconds after the track start (burn-in = 1 s)
                ax.plot(t, w["err"], "o-", color=color, ms=3, lw=1.2, label=lvl)
        if trig:
            ax.axvline(6.0, color=THIRD, lw=1.0)
            ax.text(6.2, thr_dl * 1.1, "trigger", color=THIRD, fontsize=7)
        ax.set_yscale("log"); ax.set_title(title); ax.set_xlabel("time since track start (s)")
    axes[0, 0].set_ylabel("window error (masked MSE)"); axes[1, 0].set_ylabel("window error (masked MSE)")
    handles, labels_ = axes[0, 2].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01), title="nominal level (seed 1)", title_fontsize=8)
    fig.suptitle("Window error along the injected vehicle's track — grey band: median–p99 of normal test windows; dashed: p99 vehicle threshold", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return _save(fig, "fig_event_timelines.png")
