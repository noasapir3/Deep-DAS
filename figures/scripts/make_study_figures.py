#!/usr/bin/env python3
"""Study figures for the repository gallery, in the same style as the report figures (make_report_figures.py).

Every value is read from the study records of this repository; nothing is simulated, smoothed or estimated.
  S02_seed_dependence             research/02 seed-study summary (P99, per seed)            -> figures/development/
  S05_legitimate_stop_remedies    research/05 results table (validation pool, own val_B P99) -> figures/validation/
  S06_information_flow_weak_types research/06 information-flow table (exposed final test)    -> figures/post_hoc/
Usage: python figures/scripts/make_study_figures.py   (from the repository root; no model is loaded)
"""
import re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]; FIG = ROOT / "figures"
COL_W = 3.34
HYB, SINGLE, Z, GREY, RED = "#1B4F9C", "#C8661A", "#6B6B6B", "#9A9A9A", "#B2182B"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.6,
    "axes.linewidth": 0.6, "axes.edgecolor": "#222222", "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.1, "legend.frameon": False, "grid.color": "#E3E3E3", "grid.linewidth": 0.45,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})


def save(fig, role, name, data):
    for k in ("png", "pdf", "data"):
        (FIG / role / k).mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / role / "pdf" / f"{name}.pdf", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(FIG / role / "png" / f"{name}.png", bbox_inches="tight", pad_inches=0.015)
    data.to_csv(FIG / role / "data" / f"{name}.csv", index=False); plt.close(fig); print("saved", role, name)


# ------------------------------------------------------------------ study 02: development cases detected per seed (single models, P99)
def fig_seed_dependence():
    md = (ROOT / "research/02_supervised_single_model/records/seedstudy_20261007_SEED_SUMMARY.md").read_text()
    block = md.split("## P99: per seed")[1].split("## P95")[0]
    rows = []
    for line in block.splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) > 5 and c[0] in ("Baseline", "A", "B", "C") and c[1].isdigit():
            rows.append(dict(arm=c[0], seed=int(c[1]), detected9=int(c[4]), tailgating_case_detected="C3 tailgating" not in c[5], source="research/02/records/seedstudy_20261007_SEED_SUMMARY.md (P99)"))
    D = pd.DataFrame(rows); assert len(D) == 12 and not D.tailgating_case_detected.any()
    lab = {"Baseline": "Baseline\n(no cleaning,\nno $\\phi_u$)", "A": "A\n(cleaning,\nno $\\phi_u$)", "B": "B\n(cleaning\n+ osc)", "C": "C\n(cleaning\n+ $\\phi_u$)"}
    fig, ax = plt.subplots(figsize=(COL_W, 2.1)); x = np.arange(4)
    for i, a in enumerate(["Baseline", "A", "B", "C"]):
        v = D[D.arm == a].sort_values("seed").detected9.to_numpy()
        ax.scatter(np.full(3, i) + np.array([-0.12, 0, 0.12]), v, s=16, facecolor="white", edgecolor=HYB, lw=0.9, zorder=3)
        ax.hlines(v.mean(), i - 0.25, i + 0.25, color=HYB, lw=1.2, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([lab[a] for a in ["Baseline", "A", "B", "C"]], fontsize=6.4)
    ax.set_ylim(0, 9.4); ax.set_yticks(range(0, 10, 3)); ax.set_ylabel("Development cases detected (of 9)")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.text(3.45, 0.4, "tailgating case: 0 of 12 runs", ha="right", fontsize=6.2, color="#444444")
    ax.scatter([], [], s=16, facecolor="white", edgecolor=HYB, label="single seed (0, 1, 2)"); ax.plot([], [], color=HYB, lw=1.2, label="mean of 3 seeds")
    ax.legend(loc="upper left", fontsize=6.2)
    save(fig, "development", "S02_seed_dependence", D)


# ------------------------------------------------------------------ study 05: legitimate-stop alarms vs abnormal stops (validation pool)
def fig_legit_stop_remedies():
    md = (ROOT / "research/05_legitimate_stops/README.md").read_text()
    rows = []
    for line in md.splitlines():
        c = [x.strip().replace("*", "") for x in line.strip().strip("|").split("|")]
        if len(c) == 7 and re.match(r"^\d+", c[1]) and ("Ensemble" in c[0] or c[0].startswith(("F1", "F2", "Reference"))):
            rows.append(dict(variant=c[0].replace(" (reference)", ""), legit_stops_flagged_of_54=int(re.match(r"\d+", c[1]).group()),
                             abnormal_stops_detected_of_66=int(c[3]), passes_replacement_rule=c[6],
                             source="research/05_legitimate_stops/README.md, results table (validation pool, each ensemble at its own val_B P99)"))
    D = pd.DataFrame(rows); assert len(D) == 6, D
    fig, ax = plt.subplots(figsize=(COL_W, 2.05)); y = np.arange(len(D))[::-1]
    ax.barh(y, D.legit_stops_flagged_of_54, color=[HYB if v.startswith("DeepDAS") else "#A9BEDC" for v in D.variant], height=0.6, zorder=2)
    for yi, v, a in zip(y, D.legit_stops_flagged_of_54, D.abnormal_stops_detected_of_66):
        ax.text(v + 0.8, yi, f"{v}/54   (abnormal stops {a}/66)", va="center", fontsize=6.1)
    ax.axvline(54 * 0.2, color=RED, ls="--", lw=0.9, zorder=3)
    ax.text(54 * 0.2 + 0.6, -0.95, "replacement rule: $\\leq$20 % of 54 (and $\\leq$ half of the reference)", color=RED, fontsize=6.0, va="center")
    ax.set_ylim(-1.3, len(D) - 0.5)
    ax.set_yticks(y); ax.set_yticklabels(D.variant.str.replace("DeepDAS Ensemble v1", "Ensemble v1 (frozen)"), fontsize=6.6)
    ax.set_xlim(0, 75); ax.set_xlabel("Legitimate stops < 10 s flagged (false alarms, of 54)")
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    save(fig, "validation", "S05_legitimate_stop_remedies", D)


# ------------------------------------------------------------------ study 06: where the weak types are lost (exposed final test, post hoc)
def fig_information_flow():
    I = pd.read_csv(ROOT / "research/06_weak_type_diagnosis/records/weak_types_diag_20261010_results_information_flow_by_type.csv").set_index("type")
    weak = ["hard_braking", "speeding", "overweight_vehicle", "lane_straddling", "collision", "lane_weaving"]
    name = {"hard_braking": "Hard braking", "speeding": "Speeding", "overweight_vehicle": "Overweight", "lane_straddling": "Lane straddling",
            "collision": "Collision", "lane_weaving": "Lane weaving"}
    st = ["input_probe_AUC(13 stats)", "encoder_probe_AUC(E_DIV seeds, mean)", "model_score_AUC(E_DIV ens)"]
    D = I.loc[weak, st + ["recall_at_frozen_thr", "n_anomaly_tracks", "n_normal_tracks"]].reset_index()
    pal = ["#1B4F9C", "#C8661A", "#0F8B7D", "#6B6B6B", "#7B3F99", "#B2182B"]; mk = ["o", "s", "D", "^", "v", "P"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.3)); x = np.arange(3)
    for i, t in enumerate(weak):
        ax.plot(x, D.set_index("type").loc[t, st].to_numpy(float), marker=mk[i], ms=3.4, color=pal[i], lw=1.0,
                label=f"{name[t]} ({D.set_index('type').loc[t, 'recall_at_frozen_thr']})")
    ax.set_xticks(x); ax.set_xticklabels(["input statistics\n(logistic probe)", "pooled encoder code\n(linear probe)", "ensemble score\n(frozen head)"], fontsize=6.5)
    ax.set_ylabel("AUC against normal tracks"); ax.set_ylim(0.7, 1.01); ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(ncol=2, loc="lower left", fontsize=6.0, title="type (recall at frozen $\\tau$)", title_fontsize=6.0, columnspacing=0.8, handlelength=1.4)
    save(fig, "post_hoc", "S06_information_flow_weak_types", D)


if __name__ == "__main__":
    fig_seed_dependence(); fig_legit_stop_remedies(); fig_information_flow()
