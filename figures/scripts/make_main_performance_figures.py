#!/usr/bin/env python3
"""Main final-test figures with the three detectors that were frozen before the final test.

  V6_roc_pr_frozen_detectors             ROC and precision-recall, common decided subset (101 anomalies, 1,008 normals)
  V7_detection_by_type_frozen_detectors  recall per anomaly type at the frozen thresholds, Wilson 95 % intervals

Why this script exists: the report and the front page show only the detectors that were frozen before the final test
(DeepDAS Ensemble v1 as trained and tested, the supervised single model, the SimStudio Z-score). The post-hoc
phi_u-neutralized option stays in its own table and in the four-configuration figures G3/G4 of
research/08_tailgating_feature_intervention/ (code/make_figures.py), which are not changed.

Plotting style, colours, markers, data selection and thresholds are copied from G3/G4 (research/08 code/make_figures.py)
with the neutralized series left out; nothing is simulated, smoothed or re-thresholded. Every drawn value is asserted
against the published tables (results/final_test/T1_per_type.csv, T2_overall.csv) and against the G3/G4 data files.
Usage (from the repository root, no model is loaded): python figures/scripts/make_main_performance_figures.py
"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "figures" / "final_test"
G08 = ROOT / "research/08_tailgating_feature_intervention/figures/data"
for d in ("pdf", "png", "data"):
    (OUT / d).mkdir(parents=True, exist_ok=True)
TAU = json.loads((ROOT / "models/deepdas_ensemble_v1/ensemble.json").read_text())["decision"]["threshold"]
THR = {"E_DIV": TAU, "CR_P99": -5.523461642265342, "Z": 15.392482911229285}      # frozen operating thresholds
COL_W, TXT_W = 3.34, 6.93
M = {"E_DIV": dict(c="#1B4F9C", m="o", ls="-", name="DeepDAS Ensemble v1 (as trained and tested)", fill=True),
     "CR_P99": dict(c="#C8661A", m="s", ls="-", name="Supervised single model", fill=False),
     "Z": dict(c="#6B6B6B", m="^", ls="-", name="SimStudio Z-score", fill=False)}
G08_NAME = {"E_DIV": "Ensemble v1, phi_u active (hybrid)", "CR_P99": "Supervised single model", "Z": "SimStudio Z-score"}
GREY = "#9A9A9A"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.4, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.4,
                     "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "legend.frameon": False, "grid.color": "#E3E3E3", "grid.linewidth": 0.45,
                     "pdf.fonttype": 42, "savefig.dpi": 300})
TYPES = ["speeding", "reckless_driver", "hard_braking", "stalled_vehicle", "lane_weaving", "lane_straddling", "collision", "overweight_vehicle", "tailgating"]
TN = {"speeding": "Speeding", "reckless_driver": "Reckless driving", "hard_braking": "Hard braking", "stalled_vehicle": "Stalled vehicle", "lane_weaving": "Lane weaving",
      "lane_straddling": "Lane straddling", "tailgating": "Tailgating", "collision": "Collision", "overweight_vehicle": "Overweight"}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def save(fig, name, data: pd.DataFrame):
    fig.savefig(OUT / "pdf" / f"{name}.pdf", bbox_inches="tight", pad_inches=0.015); fig.savefig(OUT / "png" / f"{name}.png", bbox_inches="tight", pad_inches=0.015)
    data.to_csv(OUT / "data" / f"{name}.csv", index=False); plt.close(fig); print("saved", name)


def label(ax, s, x=-0.16, y=1.03):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom")


# ------------------------------------------------------------------ final-test vehicles, common decided subset (published active results only)
X = pd.read_csv(ROOT / "results/final_test/vehicle_outcomes.csv").set_index("vk")
CORE = X[(X.block == "core") & (X.label == "anomaly_verified") & X.common]
NORM = X[X.label.isin(["normal_scene", "normal_unaffected_background", "normal_matched_control"]) & X.common]
assert (len(CORE), len(NORM)) == (101, 1008)
EV = pd.concat([CORE.assign(y=1), NORM.assign(y=0)])


# ================================================================== V7 recall per type, three frozen detectors
def fig_per_type():
    rows = []
    for t in TYPES:
        a = CORE[CORE.type == t]; r = dict(type=t, n=len(a))
        for det in ("E_DIV", "CR_P99", "Z"):
            k = int(a[f"{det}_alarm"].sum()); lo, hi = wilson(k, len(a)); r |= {f"{det}_k": k, f"{det}_rec": k / len(a), f"{det}_lo": lo, f"{det}_hi": hi}
        rows.append(r)
    D = pd.DataFrame(rows); D["order"] = D.E_DIV_rec - D[["CR_P99_rec", "Z_rec"]].max(axis=1)
    D = D.sort_values(["order", "E_DIV_rec"]).reset_index(drop=True)
    # identical to the published table T1 and to the four-configuration figure G3
    T1 = pd.read_csv(ROOT / "results/final_test/T1_per_type.csv").set_index("type"); G3 = pd.read_csv(G08 / "G3_detection_by_type_four_configurations.csv").set_index("type")
    for det in ("E_DIV", "CR_P99", "Z"):
        assert np.allclose(D.set_index("type")[f"{det}_rec"].round(3), T1.loc[D.type, f"{det}_recall_common"].values, atol=1e-3)
        assert (D.set_index("type")[f"{det}_k"] == G3.loc[D.type, f"{det}_k"].values).all()
    fig, ax = plt.subplots(figsize=(COL_W, 2.7)); y = np.arange(len(D)); off = {"E_DIV": 0.24, "CR_P99": 0.0, "Z": -0.24}
    for det in ("Z", "CR_P99", "E_DIV"):
        st = M[det]; yy = y + off[det]
        ax.hlines(yy, D[f"{det}_lo"], D[f"{det}_hi"], color=st["c"], lw=0.75, alpha=0.85)
        ax.plot(D[f"{det}_rec"], yy, ls="none", marker=st["m"], ms=3.4, mfc=st["c"] if st["fill"] else "white", mec=st["c"], mew=0.9, label=st["name"])
    for yi in y[:-1]:
        ax.axhline(yi + 0.5, color="#EFEFEF", lw=0.5, zorder=0)
    ax.set_yticks(y); ax.set_yticklabels([f"{TN[t]} ({n})" for t, n in zip(D.type, D.n)]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.6, len(D) - 0.4); ax.set_xlabel("Recall at the frozen thresholds (Wilson 95 % CI)")
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels(); o = [2, 0, 1]          # plotted Z, single, ensemble -> legend rows: ensemble, single / Z-score
    ax.legend([h[i] for i in o], [l[i] for i in o], loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=2, fontsize=6.0, columnspacing=0.8, handletextpad=0.3)
    save(fig, "V7_detection_by_type_frozen_detectors", D.drop(columns="order"))


# ================================================================== V6 ROC / PR, three frozen detectors
def fig_roc():
    fig, axs = plt.subplots(1, 2, figsize=(TXT_W * 0.78, 2.35), gridspec_kw=dict(wspace=0.32)); rows = []
    G4 = pd.read_csv(G08 / "G4_roc_pr_four_configurations.csv").set_index("configuration")
    for det in ("Z", "CR_P99", "E_DIV"):
        st = M[det]; s = EV[f"{det}_score"]; auc, ap = roc_auc_score(EV.y, s), average_precision_score(EV.y, s)
        fpr, tpr, _ = roc_curve(EV.y, s, drop_intermediate=False); pre, rec, _ = precision_recall_curve(EV.y, s)
        axs[0].plot(fpr, tpr, color=st["c"], ls=st["ls"], lw=1.1, label=f"{st['name']}  (ROC-AUC {auc:.3f}, AUPRC {ap:.3f})")
        axs[1].step(rec, pre, where="post", color=st["c"], ls=st["ls"], lw=1.1)
        a_ = s[EV.y == 1] > THR[det]; n_ = s[EV.y == 0] > THR[det]; tp, fp = a_.mean(), n_.mean(); pr = a_.sum() / (a_.sum() + n_.sum())
        for ax, (xv, yv) in zip(axs, ((fp, tp), (tp, pr))):
            ax.plot(xv, yv, marker=st["m"], ms=4.0, mfc="white", mec=st["c"], mew=1.0, ls="none", zorder=5)
        row = dict(configuration=st["name"], ROC_AUC=round(auc, 4), AUPRC=round(ap, 4), op_FPR=round(fp, 4), op_TPR=round(tp, 4), op_precision=round(pr, 4),
                   n_anomalies=int(EV.y.sum()), n_normals=int((EV.y == 0).sum()))
        g = G4.loc[G08_NAME[det]]                      # identical to the four-configuration figure G4
        assert all(np.isclose(row[c], g[c]) for c in ("ROC_AUC", "AUPRC", "op_FPR", "op_TPR", "op_precision")), (det, row, g.to_dict())
        rows.append(row)
    assert (round(rows[2]["ROC_AUC"], 3), round(rows[2]["AUPRC"], 3)) == (0.908, 0.742) and (round(rows[1]["ROC_AUC"], 3), round(rows[1]["AUPRC"], 3)) == (0.769, 0.515)
    prev = EV.y.mean(); axs[0].plot([0, 1], [0, 1], color=GREY, lw=0.5, ls=":"); axs[1].axhline(prev, color=GREY, lw=0.5, ls=":")
    axs[1].text(0.02, prev + 0.015, f"prevalence {prev:.3f}", fontsize=6, color="#555555")
    axs[0].set(xlim=(0, 1), ylim=(0, 1.005), xlabel="False-positive rate (normal driving)", ylabel="True-positive rate (pooled)", title="ROC")
    axs[1].set(xlim=(0, 1), ylim=(0, 1.005), xlabel="Recall (pooled)", ylabel="Precision", title="Precision–recall")
    for ax, l in zip(axs, "ab"):
        ax.grid(True); ax.set_axisbelow(True); label(ax, l, x=-0.2)
    h, l = axs[0].get_legend_handles_labels(); h, l = h[::-1], l[::-1]
    h.append(Line2D([], [], ls="none", marker="o", mfc="white", mec="#222222", ms=4, mew=0.9)); l.append("frozen operating point")
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=2, fontsize=6.0, columnspacing=1.0)
    save(fig, "V6_roc_pr_frozen_detectors", pd.DataFrame(rows))


if __name__ == "__main__":
    fig_per_type(); fig_roc()
