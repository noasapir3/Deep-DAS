#!/usr/bin/env python3
"""Klausner experiment: the frozen DeepDAS Ensemble v1 on the one real DAS + camera recording of the SimStudio project.

What it does (inference only; nothing is trained, re-calibrated or changed):
  1. imports the real export (data/simstudio_inputs/klausner/) with the unchanged pipeline used for every simulated run
     (read_export -> map matching on the real street geometry -> 14 channels -> DAS amplitude from the Kalman audit sheet ->
     usability and label-free quality gate);
  2. scores every usable track with the five frozen checkpoints, the frozen per-seed statistics and the frozen threshold tau,
     in both inference options (phi_mode = active and neutralized), and the supervised single model for comparison;
  3. reports the SimStudio Z-score of the same track from the shipped model, and the archived stage-1 autoencoder result.

Status of the result: one real, UNLABELLED track. It is a qualitative observation, not an evaluation: there is no ground truth for
this vehicle, and the recording was not part of any training, selection or test set.
Usage (from the repository root): python models/deepdas_ensemble_v1/experiment_klausner/run_klausner_experiment.py
Writes next to this script: klausner_scores.csv, klausner_track_summary.json, figures/klausner_experiment.{png,pdf}.
"""
import json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from python_calamine import CalamineWorkbook

from deepdas import paths
from deepdas.amplitude import amp_on_bins
from deepdas.data.channels import build_channels
from deepdas.data.import_exports import ExportRef, read_export
from deepdas.data.map_matching import LaneIndex
from deepdas.data.quality import track_quality

KL = paths.FP_DATA / "klausner"; SIM = KL / "klausner_bus_real.sim.json"
AE_ARCHIVED = {"track_score_p95_window_error": 2.15, "threshold_p99": 5.20, "flagged": False,
               "source": "research/01_unsupervised_autoencoder/records/unsup_REPORT.md, section 4.6 (archived, not regenerated)"}


def import_klausner():
    ref = ExportRef(group="real", scenario_id="klausner/bus_real", region="klausner", scenario_type="real_bus", export_dir=KL, scenario_json=SIM)
    rows50, meta = read_export(ref); rows50["track_key"] = rows50.track_key.astype(str); meta["track_key"] = meta.track_key.astype(str)
    ch = build_channels(rows50, meta, {"klausner": LaneIndex.from_scenario_json(SIM)}); ch["track_key"] = ch.track_key.astype(str)
    vb = ch[ch.m_valid > 0].groupby("track_key").size(); q = track_quality(ch).set_index("track_key")
    meta["n_valid_bins"] = meta.track_key.map(vb).fillna(0).astype(int); meta["quality_ok"] = meta.track_key.map(q["quality_ok"]).fillna(True).astype(bool)
    meta["usable"] = (meta.n_valid_bins >= 32) & meta.quality_ok
    rws = CalamineWorkbook.from_path(str(KL / "tracking_audit.xlsx")).get_sheet_by_name("Kalman Audit").to_python(skip_empty_area=False)
    col = {str(h): i for i, h in enumerate(rws[0])}
    aud = pd.DataFrame([(x[col["Time (s)"]], str(x[col["Track ID"]]), x[col["DAS Amplitude A (raw units)"]]) for x in rws[1:] if x[col["Sensor Type"]] == "DAS"],
                       columns=["t", "tid", "amp"])
    aud["t"] = pd.to_numeric(aud.t); aud["amp"] = pd.to_numeric(aud.amp, errors="coerce")
    A = pd.concat([amp_on_bins(rows50, aud, tk, tid)[0] for tk, tid in zip(meta.track_key, meta.track_id)], ignore_index=True)
    return ch, A, meta


def score(ch, A, keys, mode):
    os.environ["DEEPDAS_PHI_MODE"] = mode                     # read when the detectors are constructed
    from deepdas.ensemble.scoring import Ensemble, score_units
    R = score_units(ch, A, pd.DataFrame({"unit": keys, "tracks": [[k] for k in keys]}), ensemble=Ensemble())
    R.insert(1, "phi_mode", mode); return R


def main():
    ch, A, meta = import_klausner()
    keys = meta[meta.usable].track_key.tolist()
    R = pd.concat([score(ch, A, keys, m) for m in ("active", "neutralized")], ignore_index=True)
    from deepdas.ensemble.detector import tailgating_fraction, clean_lead_artifacts
    from deepdas.amplitude import attach_amplitude, to_arrays
    raw = to_arrays(attach_amplitude(ch[ch.track_key.isin(keys)], A[A.track_key.isin(keys)]))
    R["phi_u"] = R.unit.map({k: tailgating_fraction([clean_lead_artifacts(raw[k])]) for k in keys})
    spec = json.loads((paths.ENSEMBLE / "ensemble.json").read_text()); tau = spec["decision"]["threshold"]
    R["frozen_tau"] = tau
    R.to_csv(HERE / "klausner_scores.csv", index=False)

    # Z-score of the same track, from the shipped SimStudio model (as published)
    from deepdas.baseline.zscore import shipped_model
    from deepdas.fp_anomaly_model.scenario_scorer import score_dataframe
    Zs = score_dataframe(pd.read_csv(KL / "klausner_real_features.csv", low_memory=False), shipped_model())
    z = Zs[["track_id", "z_score_max", "flagged", "top_feature"]].to_dict("records")

    g = ch[ch.track_key == keys[0]].sort_values("bin"); g = g[~g["burn_in"]] if "burn_in" in g else g
    tt = (g.bin.to_numpy() - g.bin.iloc[0]) * 0.1; slow = np.abs(g.v_lon.to_numpy()) < 1.0
    best, start = 0.0, None                       # longest time span with every tracked sample below 1 m/s (gaps between samples included)
    for ti, sl in zip(tt, slow):
        if sl:
            start = ti if start is None else start; best = max(best, ti - start)
        else:
            start = None
    summary = {"tracks_in_export": int(len(meta)), "usable_tracks": len(keys), "frozen_tau": tau,
               "deepdas": R.drop(columns=[c for c in R.columns if c.startswith("CR_P99")]).to_dict("records"),
               "supervised_single_model": R[R.phi_mode == "active"][["unit", "CR_P99_logit", "CR_P99_decision"]].to_dict("records"),
               "zscore_shipped": z, "stage1_autoencoder_archived": AE_ARCHIVED,
               "track_facts": {"duration_s": float(meta[meta.usable].duration_s.iloc[0]) if "duration_s" in meta else None,
                               "valid_bins": int(meta[meta.usable].n_valid_bins.iloc[0]),
                               "longest_span_below_1_mps_s": round(float(best), 1)},
               "status": "qualitative observation on one real, unlabelled track; not an evaluation"}
    (HERE / "klausner_track_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    figure(g, R, tau, z, summary)
    print(R.to_string(index=False)); print(json.dumps({k: summary[k] for k in ("zscore_shipped", "track_facts")}, indent=1, default=float))


def figure(g, R, tau, z, S):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 7.5,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42, "savefig.dpi": 300})
    HYB, NEU = "#1B4F9C", "#0F8B7D"
    fig = plt.figure(figsize=(6.93, 2.6)); gs = fig.add_gridspec(3, 2, width_ratios=[2.3, 1], hspace=0.25, wspace=0.35)
    t = (g.bin - g.bin.iloc[0]).to_numpy() * 0.1
    for i, (c, lab) in enumerate((("v_lon", "$v$ (m/s)"), ("a_lon", "$a$ (m/s$^2$)"), ("m_das", "DAS"))):
        ax = fig.add_subplot(gs[i, 0]); ax.plot(t, g[c], color=HYB, lw=0.8); ax.set_ylabel(lab, fontsize=7)
        if i < 2: ax.set_xticklabels([])
        else: ax.set_xlabel("time along the tracked bus (s)")
    ax = fig.add_subplot(gs[:, 1]); a = R[R.phi_mode == "active"].E_DIV_ens.iloc[0]; n = R[R.phi_mode == "neutralized"].E_DIV_ens.iloc[0]
    ax.bar([0, 1], [a, n], color=[HYB, "white"], edgecolor=[HYB, NEU], hatch=[None, "////"], width=0.6, lw=0.9)
    ax.axhline(tau, color="#B2182B", ls="--", lw=0.9); ax.text(-0.45, tau * 1.04, f"frozen $\\tau$ = {tau:.3f}", color="#B2182B", fontsize=6.3, va="bottom", ha="left", bbox=dict(fc="white", ec="none", pad=0.6))
    for x, val in ((0, a), (1, n)):
        ax.text(x, val + 0.05 * max(a, n, tau), f"{val:.2f}", ha="center", fontsize=6.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["$\\phi_u$ active\n(as trained)", "$\\phi_u$ neutralized"], fontsize=6.6)
    ax.set_ylabel("Ensemble score $s_u$"); ax.set_ylim(min(0, a, n) * 1.15, max(a, n, tau) * 1.25)
    zz = z[0]; ax.set_title(f"DeepDAS Ensemble v1 on the real bus\n(Z-score: max |z| = {zz['z_score_max']:.1f}, flagged)", fontsize=7)
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures/klausner_experiment.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(HERE / "figures/klausner_experiment.png", bbox_inches="tight", pad_inches=0.02); plt.close(fig)


if __name__ == "__main__":
    main()
