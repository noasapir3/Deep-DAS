#!/usr/bin/env python3
"""Figures of the tailgating-feature intervention (phi_u ACTIVE = original hybrid vs phi_u NEUTRALIZED at inference).
Every figure: ../figures/pdf (vector), ../figures/png (300 dpi), ../figures/data (the exact numbers drawn).
Sources: results/ (hybrid, published) and results_phi_neutralized/ (intervention) of this release copy; nothing is simulated.
Final-test figures are post-hoc analyses of an already exposed test set."""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[3]; HERE = Path(__file__).resolve().parents[1] / "figures"
for d in ("pdf", "png", "data"):
    (HERE / d).mkdir(parents=True, exist_ok=True)
TAB = Path(__file__).resolve().parents[1] / "tables"
A, N = ROOT / "results", ROOT / "results_phi_neutralized"
TAU = json.loads((ROOT / "models/deepdas_ensemble_v1/ensemble.json").read_text())["decision"]["threshold"]
THR = {"E_DIV": TAU, "E_NEU": TAU, "CR_P99": -5.523461642265342, "Z": 15.392482911229285}
COL_W, TXT_W = 3.34, 6.93
M = {"E_DIV": dict(c="#1B4F9C", m="o", ls="-", name="Ensemble v1, $\\phi_u$ active (hybrid)", fill=True),
     "E_NEU": dict(c="#0F8B7D", m="D", ls="--", name="Ensemble v1, $\\phi_u$ neutralized (post hoc)", fill=False),
     "CR_P99": dict(c="#C8661A", m="s", ls="-", name="Supervised single model", fill=False),
     "Z": dict(c="#6B6B6B", m="^", ls="-", name="SimStudio Z-score", fill=False)}
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
    fig.savefig(HERE / "pdf" / f"{name}.pdf", bbox_inches="tight", pad_inches=0.015); fig.savefig(HERE / "png" / f"{name}.png", bbox_inches="tight", pad_inches=0.015)
    data.to_csv(HERE / "data" / f"{name}.csv", index=False); plt.close(fig); print("saved", name)


def label(ax, s, x=-0.16, y=1.03):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom")


# ------------------------------------------------------------------------ final-test vehicles: both modes on the same common subset
Xa = pd.read_csv(A / "final_test/vehicle_outcomes.csv").set_index("vk"); Xn = pd.read_csv(N / "final_test/vehicle_outcomes.csv").set_index("vk")
assert (Xa.common == Xn.common).all() and (Xa.E_DIV_decided == Xn.E_DIV_decided).all()      # identical decided populations
X = Xa.copy(); X["E_NEU_score"] = Xn.E_DIV_score; X["E_NEU_alarm"] = Xn.E_DIV_alarm
CORE = X[(X.block == "core") & (X.label == "anomaly_verified") & X.common]
NORM = X[X.label.isin(["normal_scene", "normal_unaffected_background", "normal_matched_control"]) & X.common]
assert (len(CORE), len(NORM)) == (101, 1008)
EV = pd.concat([CORE.assign(y=1), NORM.assign(y=0)])


# ======================================================================== G1 development cases: score with and without phi
def fig_dev():
    D = pd.read_csv(TAB / "development_cases_comparison.csv").iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(COL_W, 2.35)); y = np.arange(len(D))
    ax.axvline(TAU, color="#222222", lw=0.7, ls="--"); ax.text(TAU + 0.05, len(D) - 0.35, rf"frozen $\tau$ = {TAU:.3f}", fontsize=6.2, va="center")
    for yi, r in zip(y, D.itertuples()):
        tg = r.anomaly_type == "Tailgating"
        ax.annotate("", xy=(r.score_phi_neutralized, yi), xytext=(r.score_phi_active, yi),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8 if tg else 0.5, color="#B2182B" if tg else "#888888", mutation_scale=6, shrinkA=2.5, shrinkB=2.5))
        ax.plot(r.score_phi_active, yi, "o", ms=4.2, color=M["E_DIV"]["c"], zorder=5)
        ax.plot(r.score_phi_neutralized, yi, "D", ms=3.8, mfc="white", mec=M["E_NEU"]["c"], mew=0.9, zorder=6)
    ax.set_yticks(y); ax.set_yticklabels([f"{t}" + (f"  ($\\phi_u$={p:.2f})" if p > 0.01 else "") for t, p in zip(D.anomaly_type, D.phi_u)])
    for t in ax.get_yticklabels():
        if t.get_text().startswith("Tailgating"):
            t.set_fontweight("bold")
    ax.set_xlim(-0.6, 4.9); ax.set_ylim(-0.6, len(D) - 0.1); ax.set_xlabel("Ensemble score $s_u$ (development cases)")
    ax.xaxis.grid(True); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
    h = [Line2D([], [], ls="none", marker="o", ms=4.2, color=M["E_DIV"]["c"]), Line2D([], [], ls="none", marker="D", ms=3.8, mfc="white", mec=M["E_NEU"]["c"], mew=0.9)]
    ax.legend(h, [f"$\\phi_u$ active: {int(D.detected_phi_active.sum())}/9 detected", f"$\\phi_u$ neutralized: {int(D.detected_phi_neutralized.sum())}/9"],
              loc="upper center", bbox_to_anchor=(0.45, 1.15), ncol=2, fontsize=6.3, handletextpad=0.3, columnspacing=1.0)
    save(fig, "G1_development_cases_phi_dependence", D.iloc[::-1])


# ======================================================================== G2 tailgating detection: validation vs exposed final test
def fig_tailgating():
    V = pd.read_csv(TAB / "validation_comparison.csv").set_index("metric")
    va = V.loc["tailgating (12 v3 + 1 cross-region)", "phi active (original hybrid)"]; vn = V.loc["tailgating (12 v3 + 1 cross-region)", "phi neutralized (inference-time intervention)"]
    T = CORE[CORE.type == "tailgating"]; fa, fn, nt = int(T.E_DIV_alarm.sum()), int(T.E_NEU_alarm.sum()), len(T)
    rows = [("Validation", "active", *map(int, va.split("/"))), ("Validation", "neutralized", *map(int, vn.split("/"))),
            ("Exposed final test\n(post hoc)", "active", fa, nt), ("Exposed final test\n(post hoc)", "neutralized", fn, nt)]
    Dd = pd.DataFrame(rows, columns=["population", "phi_mode", "detected", "n"]); Dd["recall"] = Dd.detected / Dd.n
    Dd[["lo", "hi"]] = [wilson(k, n) for k, n in zip(Dd.detected, Dd.n)]
    fig, axs = plt.subplots(1, 2, figsize=(COL_W, 1.85), sharey=True, gridspec_kw=dict(wspace=0.12))
    for ax, pop in zip(axs, Dd.population.unique()):
        d = Dd[Dd.population == pop]
        for j, r in enumerate(d.itertuples()):
            st = M["E_DIV"] if r.phi_mode == "active" else M["E_NEU"]
            ax.bar(j, r.recall, width=0.62, color=st["c"] if r.phi_mode == "active" else "white", ec=st["c"], lw=1.0, hatch=None if r.phi_mode == "active" else "////")
            ax.errorbar(j, r.recall, yerr=[[r.recall - r.lo], [r.hi - r.recall]], fmt="none", ecolor="#333333", elinewidth=0.6, capsize=1.6)
            ax.text(j, 0.04, f"{r.detected}/{r.n}", ha="center", va="bottom", fontsize=6.6, color="white" if r.phi_mode == "active" else "#222222",
                    bbox=None if r.phi_mode == "active" else dict(fc="white", ec="none", pad=0.4))
        ax.set_xticks([0, 1]); ax.set_xticklabels(["$\\phi_u$ active", "$\\phi_u$ neutr."]); ax.set_title(pop, fontsize=7.0, pad=3)
        ax.set_ylim(0, 1.05); ax.yaxis.grid(True); ax.set_axisbelow(True); ax.set_xlim(-0.6, 1.6)
    axs[0].set_ylabel("Tailgating recall (Wilson 95 % CI)")
    save(fig, "G2_tailgating_detection_two_options", Dd)


# ======================================================================== G3 per type, four configurations, common subset (final test)
def fig_per_type():
    rows = []
    for t in TYPES:
        a = CORE[CORE.type == t]; r = dict(type=t, n=len(a))
        for det in ("E_DIV", "E_NEU", "CR_P99", "Z"):
            k = int(a[f"{det}_alarm"].sum()); lo, hi = wilson(k, len(a)); r |= {f"{det}_k": k, f"{det}_rec": k / len(a), f"{det}_lo": lo, f"{det}_hi": hi}
        rows.append(r)
    D = pd.DataFrame(rows); D["order"] = D.E_DIV_rec - D[["CR_P99_rec", "Z_rec"]].max(axis=1)
    D = D.sort_values(["order", "E_DIV_rec"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(COL_W, 2.95)); y = np.arange(len(D)); off = {"E_DIV": 0.3, "E_NEU": 0.1, "CR_P99": -0.1, "Z": -0.3}
    for det in ("Z", "CR_P99", "E_NEU", "E_DIV"):
        st = M[det]; yy = y + off[det]
        ax.hlines(yy, D[f"{det}_lo"], D[f"{det}_hi"], color=st["c"], lw=0.75, alpha=0.85, linestyles="dashed" if det == "E_NEU" else "solid")
        ax.plot(D[f"{det}_rec"], yy, ls="none", marker=st["m"], ms=3.4, mfc=st["c"] if st["fill"] else "white", mec=st["c"], mew=0.9, label=st["name"])
    for yi in y[:-1]:
        ax.axhline(yi + 0.5, color="#EFEFEF", lw=0.5, zorder=0)
    ax.set_yticks(y); ax.set_yticklabels([f"{TN[t]} ({n})" for t, n in zip(D.type, D.n)]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.6, len(D) - 0.4); ax.set_xlabel("Recall at the frozen thresholds (Wilson 95 % CI)")
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels(); ax.legend(h[::-1], l[::-1], loc="lower center", bbox_to_anchor=(0.36, 1.0), ncol=2, fontsize=6.0, columnspacing=0.8, handletextpad=0.3)
    save(fig, "G3_detection_by_type_four_configurations", D.drop(columns="order"))


# ======================================================================== G4 ROC / PR, four configurations, common subset (final test)
def fig_roc():
    fig, axs = plt.subplots(1, 2, figsize=(TXT_W * 0.78, 2.35), gridspec_kw=dict(wspace=0.32)); rows = []
    for det in ("Z", "CR_P99", "E_NEU", "E_DIV"):
        st = M[det]; s = EV[f"{det}_score"]; auc, ap = roc_auc_score(EV.y, s), average_precision_score(EV.y, s)
        fpr, tpr, _ = roc_curve(EV.y, s, drop_intermediate=False); pre, rec, _ = precision_recall_curve(EV.y, s)
        axs[0].plot(fpr, tpr, color=st["c"], ls=st["ls"], lw=1.1, label=f"{st['name']}  (ROC-AUC {auc:.3f}, AUPRC {ap:.3f})")
        axs[1].step(rec, pre, where="post", color=st["c"], ls=st["ls"], lw=1.1)
        a_ = s[EV.y == 1] > THR[det]; n_ = s[EV.y == 0] > THR[det]; tp, fp = a_.mean(), n_.mean(); pr = a_.sum() / (a_.sum() + n_.sum())
        for ax, (xv, yv) in zip(axs, ((fp, tp), (tp, pr))):
            ax.plot(xv, yv, marker=st["m"], ms=4.0, mfc="white", mec=st["c"], mew=1.0, ls="none", zorder=5)
        rows.append(dict(configuration=st["name"].replace("$\\phi_u$", "phi_u"), ROC_AUC=round(auc, 4), AUPRC=round(ap, 4), op_FPR=round(fp, 4), op_TPR=round(tp, 4), op_precision=round(pr, 4),
                         n_anomalies=int(EV.y.sum()), n_normals=int((EV.y == 0).sum())))
    prev = EV.y.mean(); axs[0].plot([0, 1], [0, 1], color=GREY, lw=0.5, ls=":"); axs[1].axhline(prev, color=GREY, lw=0.5, ls=":")
    axs[1].text(0.02, prev + 0.015, f"prevalence {prev:.3f}", fontsize=6, color="#555555")
    axs[0].set(xlim=(0, 1), ylim=(0, 1.005), xlabel="False-positive rate (normal driving)", ylabel="True-positive rate (pooled)", title="ROC")
    axs[1].set(xlim=(0, 1), ylim=(0, 1.005), xlabel="Recall (pooled)", ylabel="Precision", title="Precision–recall")
    for ax, l in zip(axs, "ab"):
        ax.grid(True); ax.set_axisbelow(True); label(ax, l, x=-0.2)
    h, l = axs[0].get_legend_handles_labels(); h, l = h[::-1], l[::-1]
    h.append(Line2D([], [], ls="none", marker="o", mfc="white", mec="#222222", ms=4, mew=0.9)); l.append("frozen operating point")
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=2, fontsize=6.0, columnspacing=1.0)
    save(fig, "G4_roc_pr_four_configurations", pd.DataFrame(rows))


if __name__ == "__main__":
    fig_dev(); fig_tailgating(); fig_per_type(); fig_roc()
