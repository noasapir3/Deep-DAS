#!/usr/bin/env python3
"""Deep DAS — publication-style figures (repository copy of the approved report-figure script, v3).

Repository changes relative to the original script: input paths are relative to the repository root, and each figure is written to the
gallery folder of its data role (figures/<role>/{png,pdf,data}). Plotting code and data selection are unchanged.
Figures produced here: V3 score distributions (final test), V5 controlled comparisons (validation), S1/S2 stage-1 autoencoder (archived values).

Every value is read from DeepDAS_Release_v1.0 files; nothing is simulated, smoothed or interpolated.
Study-01 values (autoencoder stage) come from the archived study report and are marked ARCHIVED.
Final-test plots are POST-HOC visual analysis of models frozen before the test; no threshold or model is selected from them.

Outputs: pdf/ (vector), png/ (300 dpi), data/ (the exact numbers drawn in each figure, as CSV).
Sizes are the final printed sizes: column width 3.34 in, text width 6.93 in (A4, 17 mm margins, two columns).
"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

REL = Path(__file__).resolve().parents[2]          # repository root
FIG = REL / "figures"
ROLE = {"V2_roc_pr_macro": "final_test", "V3_score_distributions": "final_test", "V4_detection_by_type": "final_test",
        "V5_controlled_comparisons_3panel": "validation", "S1_ae_sensitivity_archived": "archived_stage1",
        "S2_ae_severity_archived": "archived_stage1", "S3_roc_including_legit_stops": "final_test", "V1_architecture": "architecture"}


def fig_path(kind, fname):
    """figures/<role of the figure>/<kind>/<fname>; the role is looked up from the figure-name prefix."""
    role = next(r for k, r in ROLE.items() if fname.startswith(k.split("_")[0] + "_"))
    d = FIG / role / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / fname

COL_W, TXT_W = 3.34, 6.93
# one colour + one marker per model, used in every figure
M = {"E_DIV": dict(c="#1B4F9C", m="o", name="DeepDAS Ensemble v1"),
     "CR_P99": dict(c="#C8661A", m="s", name="Supervised single model"),
     "Z": dict(c="#6B6B6B", m="^", name="SimStudio Z-score")}
GREY, LIGHT, RED = "#9A9A9A", "#E6E6E6", "#B2182B"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.6,
    "axes.linewidth": 0.6, "axes.edgecolor": "#222222", "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.direction": "out", "ytick.direction": "out",
    "lines.linewidth": 1.1, "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.3,
    "grid.color": "#E3E3E3", "grid.linewidth": 0.45, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
TYPES = ["speeding", "reckless_driver", "hard_braking", "stalled_vehicle", "lane_weaving", "lane_straddling", "collision", "overweight_vehicle", "tailgating"]
TNAME = {"speeding": "Speeding", "reckless_driver": "Reckless driving", "hard_braking": "Hard braking", "stalled_vehicle": "Stalled vehicle",
         "lane_weaving": "Lane weaving", "lane_straddling": "Lane straddling", "tailgating": "Tailgating", "collision": "Collision", "overweight_vehicle": "Overweight"}
THR = {"E_DIV": 1.9726181803233118, "CR_P99": -5.523461642265342, "Z": 15.392482911229285}   # frozen operating thresholds (final_test.py)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def panel_label(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    fig.savefig(fig_path("pdf", f"{name}.pdf"), bbox_inches="tight", pad_inches=0.015)
    fig.savefig(fig_path("png", f"{name}.png"), bbox_inches="tight", pad_inches=0.015)
    plt.close(fig); print("saved", name)


# =========================================================================================== final-test data (common decided subset)
X = pd.read_csv(REL / "results/final_test/vehicle_outcomes.csv")
CORE = X[(X.block == "core") & (X.label == "anomaly_verified")]
NORM = X[X.label.isin(["normal_scene", "normal_unaffected_background", "normal_matched_control"])]   # pre-registered negatives (T2)
LEG = X[X.label == "normal_legit_stop"]
A, N, L = CORE[CORE.common], NORM[NORM.common], LEG[LEG.common]
assert (len(A), len(N), len(L)) == (101, 1008, 28)
EVAL = pd.concat([A.assign(y=1), N.assign(y=0)])


def metrics(det, df=EVAL):
    return roc_auc_score(df.y, df[f"{det}_score"]), average_precision_score(df.y, df[f"{det}_score"])


# published values on the common subset must be reproduced exactly
assert round(metrics("E_DIV")[0], 3) == 0.908 and round(metrics("E_DIV")[1], 3) == 0.742
assert round(metrics("CR_P99")[0], 3) == 0.769 and round(metrics("CR_P99")[1], 3) == 0.515


def operating_point(det):
    s = THR[det]; tp = (A[f"{det}_score"] > s).mean(); fp = (N[f"{det}_score"] > s).mean()
    prec = (A[f"{det}_score"] > s).sum() / ((A[f"{det}_score"] > s).sum() + (N[f"{det}_score"] > s).sum())
    return fp, tp, prec


def macro_recall_curve(det):
    """Exact macro recall (mean of the 9 per-type recalls) and false-alarm rate at every distinct threshold."""
    sc_a = A[f"{det}_score"].to_numpy(); ty = A.type.to_numpy(); sc_n = np.sort(N[f"{det}_score"].to_numpy())
    cuts = np.unique(np.concatenate([sc_a, sc_n]))[::-1]          # alarm if score >= cut (all distinct values)
    fa = 1 - np.searchsorted(sc_n, cuts, side="left") / len(sc_n)
    rec = np.array([np.mean([(sc_a[ty == t] >= c).mean() for t in TYPES]) for c in cuts])
    return np.concatenate([[0.0], fa]), np.concatenate([[0.0], rec])


# =========================================================================================== V1 architecture (compact, no decoration)
def fig_architecture():
    fig, ax = plt.subplots(figsize=(TXT_W, 1.08)); ax.set_xlim(0, 100); ax.set_ylim(2.6, 19.6); ax.axis("off")
    blocks = [  # (x, w, title, lines, tensor shape below)
        (0.0, 13.0, "Fused track", ["Kalman tracker", "50 Hz state rows"], "no truth channel"),
        (15.8, 14.6, "Input channels", ["8 kinematic + 5 mask", "+ 1 DAS amplitude"], r"$14 \times L$ (10 Hz)"),
        (33.2, 11.6, "Windows", ["$T$ = 64 (6.4 s)", "stride 8"], r"$W \times 14 \times 64$"),
        (47.6, 14.6, "Conv1d encoder", ["4 conv (14→128)", "linear 1024→8"], r"$W \times 8$"),
        (65.0, 14.0, "Pooling + MLP", ["mean ⊕ std ⊕ $\\phi_u$", "MLP 17→16→1"], r"logit $\ell_{u,k}$"),
        (81.8, 18.2, "5-seed ensemble", [r"$z_{u,k}=(\ell_{u,k}-\mu_k)/\sigma_k$", r"$s_u=\mathrm{mean}_k\, z_{u,k}$", r"ALARM if $s_u>\tau$"], r"$W<3$: no decision"),
    ]
    y0, h = 5.6, 10.4
    for x, w, title, lines, shape in blocks:
        ax.add_patch(Rectangle((x, y0), w, h, fill=False, lw=0.6, ec="#222222"))
        ax.text(x + w / 2, y0 + h - 1.75, title, ha="center", va="center", fontsize=6.9, fontweight="bold")
        n = len(lines); top = y0 + h - 4.4
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, top - i * (2.45 if n == 3 else 2.6), ln, ha="center", va="center", fontsize=6.1)
        if shape:
            ax.text(x + w / 2, y0 - 1.7, shape, ha="center", va="center", fontsize=6.1, color="#444444")
    for (x, w, *_), (x2, *__) in zip(blocks[:-1], blocks[1:]):
        ax.annotate("", xy=(x2 - 0.25, y0 + h / 2), xytext=(x + w + 0.25, y0 + h / 2), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#222222", mutation_scale=6))
    xa, xb = 47.6, 79.0
    ax.plot([xa, xa, xb, xb], [y0 + h + 0.5, y0 + h + 1.2, y0 + h + 1.2, y0 + h + 0.5], color="#222222", lw=0.5)
    ax.text((xa + xb) / 2, y0 + h + 1.5, "trained separately for each seed $k=1,\\dots,5$ (67,225 parameters each)", ha="center", va="bottom", fontsize=6.1)
    save(fig, "V1_architecture")


# =========================================================================================== V2 ROC + PR + macro recall vs false-alarm rate
def fig_roc_pr():
    fig, axs = plt.subplots(1, 3, figsize=(TXT_W, 2.35), gridspec_kw=dict(wspace=0.42))
    rows = []
    for det in ("Z", "CR_P99", "E_DIV"):
        st = M[det]; auc, ap = metrics(det)
        fpr, tpr, _ = roc_curve(EVAL.y, EVAL[f"{det}_score"], drop_intermediate=False)
        pre, rec, _ = precision_recall_curve(EVAL.y, EVAL[f"{det}_score"])
        fa_m, rec_m = macro_recall_curve(det)
        fp0, tp0, pr0 = operating_point(det)
        mac0 = np.mean([(A[A.type == t][f"{det}_score"] > THR[det]).mean() for t in TYPES])
        axs[0].plot(fpr, tpr, color=st["c"], lw=1.1, label=f"{st['name']}  (ROC-AUC {auc:.3f}, AUPRC {ap:.3f})")
        axs[1].step(rec, pre, where="post", color=st["c"], lw=1.1)
        axs[2].step(fa_m, rec_m, where="post", color=st["c"], lw=1.1, label=st["name"])
        for ax, (xv, yv) in zip(axs, ((fp0, tp0), (tp0, pr0), (fp0, mac0))):
            ax.plot(xv, yv, marker=st["m"], ms=4.2, mfc="white", mec=st["c"], mew=1.0, zorder=5, ls="none")
        rows.append(dict(detector=st["name"], n_anomalies=len(A), n_normals=len(N), ROC_AUC=round(auc, 4), AUPRC=round(ap, 4),
                         op_threshold=THR[det], op_FAR=round(fp0, 4), op_pooled_recall=round(tp0, 4), op_precision=round(pr0, 4), op_macro_recall=round(mac0, 4)))
    prev = len(A) / (len(A) + len(N))
    axs[0].plot([0, 1], [0, 1], color=GREY, lw=0.5, ls=":")
    axs[0].set(xlim=(0, 1), ylim=(0, 1.005), xlabel="False-positive rate (normal driving)", ylabel="True-positive rate (pooled)")
    axs[1].axhline(prev, color=GREY, lw=0.5, ls=":")
    axs[1].text(0.02, prev + 0.015, f"prevalence {prev:.3f}", ha="left", va="bottom", fontsize=6, color="#555555")
    axs[1].set(xlim=(0, 1), ylim=(0, 1.005), xlabel="Recall (pooled)", ylabel="Precision")
    axs[2].set_xscale("log"); axs[2].set_xlim(1 / len(N) * 0.8, 1); axs[2].set_ylim(0, 1.005)
    axs[2].axvline(0.01, color=GREY, lw=0.5, ls=":")
    axs[2].text(0.0108, 0.97, "nominal 1 %", fontsize=6, color="#555555", va="top")
    axs[2].set(xlabel="False-alarm rate, normal driving (log)", ylabel="Macro recall (9 types)")
    for ax, lab, ttl in zip(axs, "abc", ("ROC", "Precision–recall", "Macro recall vs. false alarms")):
        ax.grid(True, which="major"); ax.set_axisbelow(True); panel_label(ax, lab, x=-0.2, y=1.02)
        ax.set_title(ttl, fontsize=7.2, pad=3)
    h, l = axs[0].get_legend_handles_labels(); h, l = h[::-1], l[::-1]
    h += [Line2D([], [], ls="none", marker="o", mfc="white", mec="#222222", ms=4, mew=0.9)]; l += ["frozen operating point ($\\tau$)"]
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=2, columnspacing=1.6, handlelength=1.6, fontsize=6.6)
    pd.DataFrame(rows).to_csv(fig_path("data", "V2_roc_pr_operating_points.csv"), index=False)
    save(fig, "V2_roc_pr_macro")


# =========================================================================================== V3 score distributions (one panel per detector, own scale)
def fig_distributions():
    """Fraction of each class per bin; bins are equal-width in the plotted coordinate (log for the Z-score), so bar areas are comparable."""
    fig, axs = plt.subplots(1, 3, figsize=(TXT_W, 2.05), gridspec_kw=dict(wspace=0.26))
    rows = []
    spec = {"E_DIV": dict(xlabel="Ensemble score $s_u$ (z units)", log=False, bins=np.linspace(-6, 6, 31)),
            "CR_P99": dict(xlabel="Single-model logit", log=False, bins=np.linspace(-55, 42, 31)),
            "Z": dict(xlabel="Z-score, track maximum (log scale)", log=True, bins=np.logspace(np.log10(0.9), np.log10(700), 31))}
    for ax, det, lab in zip(axs, ("E_DIV", "CR_P99", "Z"), "abc"):
        st, sp = M[det], spec[det]; t = THR[det]
        sn, sl, sa = N[f"{det}_score"], L[f"{det}_score"], A[f"{det}_score"]
        fp, fl, tp = int((sn > t).sum()), int((sl > t).sum()), int((sa > t).sum())
        w = lambda x: np.full(len(x), 1 / len(x))
        ax.hist(sn, bins=sp["bins"], weights=w(sn), color=LIGHT, ec="none")
        ax.hist(sn, bins=sp["bins"], weights=w(sn), histtype="step", color=GREY, lw=0.6)
        ax.hist(sa, bins=sp["bins"], weights=w(sa), histtype="step", color=st["c"], lw=1.2)
        ymax = ax.get_ylim()[1]
        ax.plot(sl, np.full(len(sl), -0.06 * ymax), ls="none", marker="|", ms=5.5, mew=0.8, color=RED, clip_on=False)
        ax.axvline(t, color="#222222", lw=0.7, ls="--")
        ax.set_ylim(-0.11 * ymax, ymax * 1.05)
        if sp["log"]:
            ax.set_xscale("log")
        ax.set_xlabel(sp["xlabel"]); ax.yaxis.grid(True); ax.set_axisbelow(True)
        ax.set_title(f"{st['name']}\n" + rf"above $\tau$: anomalies {tp}/{len(sa)}" + "\n" + f"normal {fp}/{len(sn)}, legit. stops {fl}/{len(sl)}",
                     fontsize=6.6, linespacing=1.35, pad=3)
        panel_label(ax, lab, x=-0.13, y=1.17)
        rows.append(dict(detector=st["name"], threshold=t, normal_above=fp, n_normal=len(sn), legit_above=fl, n_legit=len(sl), anomalies_above=tp, n_anomalies=len(sa),
                         missed=len(sa) - tp))
    axs[0].set_ylabel("Fraction of class per bin")
    h = [Rectangle((0, 0), 1, 1, fc=LIGHT, ec=GREY, lw=0.6), Line2D([], [], color="#333333", lw=1.2), Line2D([], [], ls="none", marker="|", ms=6, mew=0.9, color=RED),
         Line2D([], [], color="#222222", lw=0.7, ls="--")]
    fig.legend(h, ["normal driving ($n$=1,008)", "verified anomalies ($n$=101, model colour)", "legitimate stops <10 s ($n$=28)", r"frozen threshold $\tau$"],
               loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=4, columnspacing=1.4, handlelength=1.5, fontsize=6.5)
    pd.DataFrame(rows).to_csv(fig_path("data", "V3_score_distributions_counts.csv"), index=False)
    save(fig, "V3_score_distributions")


# =========================================================================================== V4 per-type dot-and-whisker
def fig_per_type():
    rows = []
    for t in TYPES:
        a = A[A.type == t]; r = dict(type=t, n=len(a))
        for det in M:
            k = int(a[f"{det}_alarm"].sum()); lo, hi = wilson(k, len(a)); r |= {f"{det}_k": k, f"{det}_rec": k / len(a), f"{det}_lo": lo, f"{det}_hi": hi}
        rows.append(r)
    D = pd.DataFrame(rows)
    # check against the published table T1 (common subset)
    T1 = pd.read_csv(REL / "results/final_test/T1_per_type.csv").set_index("type")
    for det in M:
        assert np.allclose(D.set_index("type")[f"{det}_rec"].round(3), T1.loc[D.type, f"{det}_recall_common"].values, atol=1e-3)
    D["gain"] = D.E_DIV_rec - D[["CR_P99_rec", "Z_rec"]].max(axis=1)
    D = D.sort_values(["gain", "E_DIV_rec"], ascending=[True, True]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(COL_W, 2.55)); y = np.arange(len(D)); off = {"E_DIV": 0.24, "CR_P99": 0.0, "Z": -0.24}
    for det in ("Z", "CR_P99", "E_DIV"):
        st = M[det]; yy = y + off[det]
        ax.hlines(yy, D[f"{det}_lo"], D[f"{det}_hi"], color=st["c"], lw=0.8, alpha=0.85)
        ax.plot(D[f"{det}_rec"], yy, ls="none", marker=st["m"], ms=3.6, mfc=st["c"] if det == "E_DIV" else "white", mec=st["c"], mew=0.9, label=st["name"])
    grp = np.where(D.gain > 1e-9, 2, np.where(D.gain < -1e-9, 0, 1))
    for yi in y[:-1]:
        ax.axhline(yi + 0.5, color="#9A9A9A" if grp[yi] != grp[yi + 1] else "#EFEFEF", lw=0.6 if grp[yi] != grp[yi + 1] else 0.5, zorder=0)
    for g, txt in ((2, "ensemble\nbetter"), (1, "tie"), (0, "baseline\nbetter")):
        yy = y[grp == g]
        if len(yy):
            ax.text(1.035, yy.mean(), txt, transform=ax.get_yaxis_transform(), fontsize=5.8, color="#555555", va="center", ha="left", linespacing=1.1)
    ax.set_yticks(y); ax.set_yticklabels([f"{TNAME[t]} ({n})" for t, n in zip(D.type, D.n)])
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.6, len(D) - 0.4)
    ax.set_xlabel("Recall at the frozen threshold (Wilson 95 % CI)")
    ax.xaxis.grid(True); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
    h, l = ax.get_legend_handles_labels(); ax.legend(h[::-1], l[::-1], loc="upper center", bbox_to_anchor=(0.42, 1.13), ncol=3, columnspacing=0.8, handletextpad=0.3, fontsize=6.2)
    D.to_csv(fig_path("data", "V4_per_type_common_subset.csv"), index=False)
    save(fig, "V4_detection_by_type")


# =========================================================================================== V5 controlled comparisons (each panel isolates one change)
def _swarm(vals, width=0.16):
    """deterministic horizontal offsets so that equal values do not overlap"""
    vals = np.asarray(vals); off = np.zeros(len(vals))
    for v in np.unique(np.round(vals, 6)):
        idx = np.where(np.isclose(vals, v))[0]; k = len(idx)
        off[idx] = (np.arange(k) - (k - 1) / 2) * (width / max(k - 1, 1)) if k > 1 else 0
    return off


def fig_ablation():
    """Three validation-only comparisons, each changing one factor; every seed shown. (The ensemble-vs-members result is reported in the text.)"""
    S3 = pd.read_csv(REL / "research/03_tailgating_data/records/C_tailgating_data_20261008_per_seed_and_ensemble.csv")
    S4 = pd.read_csv(REL / "research/04_stall_diversity/records/stall_diversity_20261008_per_seed_and_ensemble.csv")
    S7 = pd.read_csv(REL / "research/07_targeted_data_study/results/metrics_all.csv")
    fig, axs = plt.subplots(1, 3, figsize=(TXT_W, 2.05), gridspec_kw=dict(wspace=0.45, width_ratios=[0.8, 0.8, 1.45]))
    def seeds_and_ens(ax, x, seeds, ens, c):
        ax.plot(x + _swarm(seeds), seeds, ls="none", marker="o", ms=3.3, mfc="white", mec=c, mew=0.8, zorder=3)
        ax.plot([x - 0.24, x + 0.24], [ens, ens], color=c, lw=1.7, zorder=4)
    out = []
    ax = axs[0]
    for j, arm in enumerate(("C_old", "C_new")):
        d = S3[S3.arm == arm]; seeds_and_ens(ax, j, d[d.seed != "ENSEMBLE"].P99_val_tailgating_recall.astype(float), float(d[d.seed == "ENSEMBLE"].P99_val_tailgating_recall.iloc[0]), M["E_DIV"]["c"])
        out += [dict(panel="a", group=arm, level=s_, value=v) for s_, v in zip(d.seed, d.P99_val_tailgating_recall)]
    ax.set_xticks([0, 1]); ax.set_xticklabels(["without", "with"]); ax.set_xlim(-0.55, 1.55); ax.set_ylim(0, 1.04)
    ax.set_xlabel("varied tailgating data"); ax.set_ylabel("Recall"); ax.set_title("Tailgating ($n$ = 13)", fontsize=7.2, pad=3)
    ax = axs[1]
    for j, arm in enumerate(("E_quantity", "E_diversity")):
        d = S4[S4.arm == arm]; seeds_and_ens(ax, j, d[d.seed != "ENSEMBLE"].stallval_macro3.astype(float), float(d[d.seed == "ENSEMBLE"].stallval_macro3.iloc[0]), M["E_DIV"]["c"])
        out += [dict(panel="b", group=arm, level=s_, value=v) for s_, v in zip(d.seed, d.stallval_macro3)]
    ax.set_xticks([0, 1]); ax.set_xticklabels(["1 mode ×3", "3 modes"]); ax.set_xlim(-0.55, 1.55); ax.set_ylim(0, 1.04)
    ax.set_xlabel("24 added stall slots"); ax.set_ylabel("Recall (3 stop modes)"); ax.set_title("Stalled vehicles ($n$ = 18)", fontsize=7.2, pad=3)
    ax = axs[2]; ref = int(S7[(S7.model == "E_DIV") & (S7.level == "ensemble")].FA_L.iloc[0]); lim = ref + 3
    ax.axvspan(lim + 0.5, 56, color="#F2F2F2", zorder=0, lw=0); ax.axvline(lim + 0.5, color=RED, lw=0.7, ls="--")
    ax.text(lim + 1.0, 0.22, "pre-registered\nlimit C5", color=RED, fontsize=6, va="bottom")
    cols = {"E_DIV": M["E_DIV"]["c"], "W1": "#2E7D4F", "W2": "#7A4FA0"}; mks = {"E_DIV": "o", "W1": "s", "W2": "^"}
    names = {"E_DIV": "DeepDAS v1 (frozen)", "W1": "W1: + targeted data", "W2": "W2: W1 + sampling"}
    for mdl in ("E_DIV", "W1", "W2"):
        d = S7[S7.model == mdl]; s_ = d[d.level != "ensemble"]; e = d[d.level == "ensemble"].iloc[0]
        ax.plot(s_.FA_L, s_.Weak6, ls="none", marker=mks[mdl], ms=3.0, mfc="white", mec=cols[mdl], mew=0.7)
        ax.plot(e.FA_L, e.Weak6, ls="none", marker=mks[mdl], ms=5.4, color=cols[mdl], label=names[mdl])
        out += [dict(panel="c", group=mdl, level=r.level, value=r.Weak6, FA_L=r.FA_L) for r in d.itertuples()]
    ax.set_xlim(29, 55); ax.set_ylim(0.2, 0.92)
    ax.set_xlabel("False alarms on legitimate stops (of 54)"); ax.set_ylabel("Recall, 6 weak types"); ax.set_title("Targeted data: gain vs. cost", fontsize=7.2, pad=3)
    ax.legend(loc="upper left", fontsize=6, handlelength=0.8, handletextpad=0.3, borderaxespad=0.2, labelspacing=0.3)
    for ax, lab in zip(axs, "abc"):
        ax.yaxis.grid(True); ax.set_axisbelow(True); panel_label(ax, lab, x=-0.42 if lab != "c" else -0.24, y=1.02)
    hh = [Line2D([], [], ls="none", marker="o", mfc="white", mec="#333333", ms=3.3, mew=0.8), Line2D([], [], color="#333333", lw=1.7),
          Line2D([], [], ls="none", marker="o", ms=5, color="#333333")]
    fig.legend(hh, ["individual seed", "5-seed ensemble (a, b)", "5-seed ensemble (c)"], loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=6.4)
    pd.DataFrame(out).to_csv(fig_path("data", "V5_controlled_comparisons_3panel.csv"), index=False)
    save(fig, "V5_controlled_comparisons_3panel")


# =========================================================================================== S1 supplementary: autoencoder sensitivity (ARCHIVED study-01 table)
def fig_ae_sensitivity():
    rows = [("primary ($T$=64, $d$=8)", 0.598, 0.52, 0.67), ("seed 1", 0.591, 0.52, 0.66), ("seed 2", 0.608, 0.53, 0.68),
            ("$T$ = 32 (3.2 s)", 0.597, 0.53, 0.66), ("$T$ = 128 (12.8 s)", 0.600, 0.52, 0.67),
            ("$d$ = 4", 0.610, 0.53, 0.68), ("$d$ = 16", 0.589, 0.52, 0.66), ("$d$ = 32", 0.577, 0.50, 0.64),
            ("unbounded decoder", 0.369, 0.24, 0.52)]
    fig, ax = plt.subplots(figsize=(COL_W, 2.2)); y = np.arange(len(rows))[::-1]
    ax.axvspan(0.843, 0.928, color="#EDEDED", lw=0); ax.axvline(0.887, color=M["Z"]["c"], lw=0.9)
    ax.text(0.887, len(rows) - 0.3, "Z-score\n0.887 [0.84, 0.93]", ha="center", va="bottom", fontsize=6, color="#444444")
    ax.axvline(0.5, color=GREY, lw=0.5, ls=":")
    for yi, (lab, v, lo, hi) in zip(y, rows):
        ax.hlines(yi, lo, hi, color="#7B3F99", lw=0.8); ax.plot(v, yi, "o", ms=3.4, color="#7B3F99")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(0.2, 1.0); ax.set_ylim(-0.7, len(rows) + 0.9)
    ax.set_xlabel("AUROC, reconstruction error (bootstrap 95 % CI)"); ax.xaxis.grid(True); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
    pd.DataFrame(rows, columns=["run", "AUROC", "lo", "hi"]).to_csv(fig_path("data", "S1_ae_sensitivity_ARCHIVED.csv"), index=False)
    save(fig, "S1_ae_sensitivity_archived")


# =========================================================================================== S2 supplementary: reconstruction error vs severity (ARCHIVED table 3)
def fig_ae_severity():
    t3 = {"Speeding": (0.08, 0.36, 5.01), "Reckless driving": (0.09, 0.49, 5.01), "Hard braking": (0.27, 0.48, 0.54), "Stalled (4/10/30 s)": (0.63, 0.57, 0.52),
          "Lane weaving": (0.09, 0.19, 0.68), "Lane straddling": (0.04, 0.04, 0.05), "Tailgating": (0.62, 0.46, 0.61), "Collision": (0.83, 0.71, 0.76),
          "Overweight (20/30/40 t)": (0.07, 0.07, 0.07)}
    fig, ax = plt.subplots(figsize=(COL_W, 2.2)); y = np.arange(len(t3))[::-1]
    ax.axvspan(0.19, 0.29, color="#EDEDED", lw=0); ax.axvline(5.20, color="#222222", lw=0.7, ls="--")
    mk = (("o", "mild"), ("s", "moderate"), ("^", "severe")); cc = ("#C9B3DB", "#9A6DBF", "#5B2A86")
    for yi, (n, v) in zip(y, t3.items()):
        ax.plot(v, [yi] * 3, color="#BBBBBB", lw=0.5, zorder=1)
        for j in range(3):
            ax.plot(v[j], yi, ls="none", marker=mk[j][0], ms=3.4, color=cc[j], zorder=2, label=mk[j][1] if yi == y[0] else None)
    ax.set_xscale("log"); ax.set_xlim(0.03, 8); ax.set_ylim(-0.7, len(t3) - 0.3); ax.set_yticks(y); ax.set_yticklabels(list(t3))
    ax.set_xlabel("Median reconstruction-error score (log)"); ax.xaxis.grid(True, which="major"); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
    ax.legend(title="severity", loc="lower right", fontsize=6, title_fontsize=6.2, handletextpad=0.2)
    save(fig, "S2_ae_severity_archived")


# =========================================================================================== S3 supplementary: ROC with legitimate stops counted as negatives
def fig_roc_with_legit():
    ev = pd.concat([A.assign(y=1), N.assign(y=0), L.assign(y=0)])
    fig, ax = plt.subplots(figsize=(COL_W * 0.62, COL_W * 0.62)); rows = []
    for det in ("Z", "CR_P99", "E_DIV"):
        fpr, tpr, _ = roc_curve(ev.y, ev[f"{det}_score"], drop_intermediate=False); auc = roc_auc_score(ev.y, ev[f"{det}_score"]); ap = average_precision_score(ev.y, ev[f"{det}_score"])
        ax.plot(fpr, tpr, color=M[det]["c"], label=f"{auc:.3f}"); rows.append(dict(detector=M[det]["name"], ROC_AUC=auc, AUPRC=ap, n_neg=int((ev.y == 0).sum())))
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.5, ls=":"); ax.set(xlim=(0, 1), ylim=(0, 1.005), xlabel="False-positive rate", ylabel="True-positive rate")
    ax.legend(title="ROC-AUC", loc="lower right", title_fontsize=6.4); ax.grid(True); ax.set_axisbelow(True)
    pd.DataFrame(rows).to_csv(fig_path("data", "S3_roc_with_legit_stops.csv"), index=False)
    save(fig, "S3_roc_including_legit_stops")


if __name__ == "__main__":
    fig_distributions(); fig_ablation(); fig_ae_sensitivity(); fig_ae_severity()
