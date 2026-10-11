"""
Inference with the frozen models: DeepDAS Ensemble v1 and the single supervised baseline. No training, no threshold fitting and no ground
truth are involved.

A *unit* is a list of track keys scored together: one usable track (the deployment unit), or the tracks of one simulated vehicle
(the unit of the validation and development tables). Decision: fewer than 3 windows -> no decision; otherwise ALARM if score > threshold.

Column names of the score tables keep the identifiers used throughout the study (``E_DIV`` = DeepDAS Ensemble v1,
``CR_P99`` = single supervised model) so that every published table is reproduced byte for byte.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .. import paths
from ..amplitude import AmpNorm, attach_amplitude, to_arrays
from ..data.dataset import window_starts
from .detector import MIN_DECISION_WINDOWS, STRIDE, T, clean_lead_artifacts, load_detector, tailgating_fraction

torch.set_num_threads(2)


def load_norm(model_dir: Path = paths.ENSEMBLE) -> AmpNorm:
    return AmpNorm.from_json(json.loads((model_dir / "input_normalization.json").read_text()))


class Ensemble:
    """DeepDAS Ensemble v1 (or any 5-seed model directory with the same layout)."""

    def __init__(self, model_dir: Path = paths.ENSEMBLE, spec: dict | None = None):
        self.dir = Path(model_dir); self.spec = spec or json.loads((self.dir / "ensemble.json").read_text())
        self.models = [load_detector(self.dir / s["checkpoint"], 1) for s in self.spec["seeds"]]
        self.mean = [s["valB_mean"] for s in self.spec["seeds"]]; self.std = [s["valB_std_ddof0"] for s in self.spec["seeds"]]
        self.threshold = self.spec["decision"]["threshold"]
        self.seed_thresholds = [json.loads((self.dir / s["thresholds"]).read_text())["thresholds_logit"]["P99"] for s in self.spec["seeds"]]


class SingleModel:
    """The single supervised baseline (no artefact cleaning, no extra feature)."""

    def __init__(self, model_dir: Path = paths.BASELINES / "supervised_single_p99"):
        self.spec = json.loads((Path(model_dir) / "model.json").read_text())
        self.model = load_detector(Path(model_dir) / self.spec["checkpoint"], 0); self.threshold = self.spec["threshold"]


def _windows(X: Dict[str, torch.Tensor], ks: List[str]):
    xs = []
    for k in ks:
        for s in window_starts(X[k].shape[1], T, STRIDE):
            x = X[k][:, s:s + T]; xs.append(torch.cat([x, torch.zeros(x.shape[0], T - x.shape[1])], 1) if x.shape[1] < T else x)
    return torch.stack(xs) if xs else None


@torch.no_grad()
def score_units(ch: pd.DataFrame, amp: pd.DataFrame, units: pd.DataFrame, ensemble: Ensemble | None = None,
                single: SingleModel | None = None, norm: AmpNorm | None = None) -> pd.DataFrame:
    """units: DataFrame [unit, tracks (list of track keys)] -> per unit: n_windows, single-model logit, the 5 seed logits, the ensemble
    score and both decisions."""
    ensemble = ensemble or Ensemble(); single = single or SingleModel(); norm = norm or load_norm()
    keys = set(k for ks in units.tracks for k in ks)
    raw = to_arrays(attach_amplitude(ch[ch.track_key.isin(keys)], amp[amp.track_key.isin(keys)]))
    Xr = {k: torch.from_numpy(norm.apply(a)[0].T.copy()) for k, a in raw.items()}
    Xc = {k: torch.from_numpy(norm.apply(clean_lead_artifacts(a))[0].T.copy()) for k, a in raw.items()}
    out = []
    for u in units.itertuples():
        ks = [k for k in u.tracks if k in Xr]
        xr, xc = _windows(Xr, ks), _windows(Xc, ks); n = 0 if xr is None else len(xr); row = {"unit": u.unit, "n_windows": n}
        if n:
            row["CR_P99_logit"] = single.model.score(xr, None)
            e = torch.tensor([[tailgating_fraction([clean_lead_artifacts(raw[k]) for k in ks])]], dtype=torch.float32)
            zs = []
            for s, m in enumerate(ensemble.models):
                lg = m.score(xc, e); row[f"E_DIV_s{s}_logit"] = lg; zs.append((lg - ensemble.mean[s]) / ensemble.std[s])
            row["E_DIV_ens"] = float(np.mean(zs))
        out.append(row)
    R = pd.DataFrame(out)
    dec = lambda sc, thr: np.where(R.n_windows < MIN_DECISION_WINDOWS, "no decision", np.where(sc > thr, "ALARM", "no alarm"))
    R["CR_P99_decision"] = dec(R.get("CR_P99_logit", np.nan), single.threshold); R["E_DIV_decision"] = dec(R.get("E_DIV_ens", np.nan), ensemble.threshold)
    return R


class UnitScorer:
    """Seed-level logits of one or more ensembles on a fixed set of units (artefact-cleaned input, tailgating feature), used by the
    validation and research evaluations. Arrays and windows are prepared once."""

    def __init__(self, ch: pd.DataFrame, amp: pd.DataFrame, units: pd.DataFrame, norm: AmpNorm | None = None):
        norm = norm or load_norm(); keys = set(k for t in units.tracks for k in t)
        raw = to_arrays(attach_amplitude(ch[ch.track_key.isin(keys)], amp[amp.track_key.isin(keys)]))
        self.X = {k: torch.from_numpy(norm.apply(clean_lead_artifacts(a))[0].T.copy()) for k, a in raw.items()}
        self.win, self.ext = {}, {}
        for u in units.itertuples():
            ks = [k for k in u.tracks if k in self.X]
            self.win[u.unit] = [(k, s) for k in ks for s in window_starts(self.X[k].shape[1], T, STRIDE)]
            self.ext[u.unit] = tailgating_fraction([clean_lead_artifacts(raw[k]) for k in ks])
        self.units = units.unit.tolist()

    def n_windows(self) -> np.ndarray:
        return np.array([len(self.win[u]) for u in self.units])

    @torch.no_grad()
    def logits(self, models) -> np.ndarray:
        out = np.full((len(models), len(self.units)), np.nan)
        for j, u in enumerate(self.units):
            ws = self.win[u]
            if not ws:
                continue
            x = torch.stack([(lambda w: torch.cat([w, torch.zeros(w.shape[0], T - w.shape[1])], 1) if w.shape[1] < T else w)(self.X[k][:, s:s + T]) for k, s in ws])
            e = torch.tensor([[self.ext[u]]], dtype=torch.float32)
            for i, m in enumerate(models):
                out[i, j] = m.score(x, e)
        return out
