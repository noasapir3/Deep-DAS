"""
Equivalence of the release with the frozen model and the original outputs.

1. The model description (ensemble.json) carries exactly the frozen constants and checkpoint hashes of the freeze record.
2. The nine development cases, re-scored by the release code, reproduce the stored per-seed z-scores and ensemble scores (|Δ| <= 1e-5,
   criterion E2 v1.1), the window counts and every decision (8/9 for DeepDAS Ensemble v1, 7/9 for the single model), and the development
   tables byte for byte. Raw logits are not compared with a tolerance (docs/CROSS_PLATFORM.md).
3. The shipped results equal the original project outputs (results/ORIGINAL_OUTPUTS_SHA256.txt).
4. Every location used by the code lies inside the release.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from deepdas import paths

ROOT = Path(__file__).resolve().parents[1]


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_ensemble_spec_equals_freeze_record():
    spec = json.loads((paths.ENSEMBLE / "ensemble.json").read_text())
    frz = json.loads((paths.ENSEMBLE / "freeze_record.json").read_text())
    assert spec["decision"]["threshold"] == frz["thresholds_ensemble"]["P99"] == 1.9726181803233118
    assert spec["decision"]["min_windows"] == 3
    for k, s in enumerate(spec["seeds"]):
        f = frz["seeds"][str(k)]
        assert (s["valB_mean"], s["valB_std_ddof0"], s["best_epoch"]) == (f["valB_mean"], f["valB_std_ddof0"], f["best_epoch"])
        assert _sha(paths.ENSEMBLE / s["checkpoint"]) == s["checkpoint_sha256"] == f["checkpoint_sha256"]
        assert _sha(paths.ENSEMBLE / s["thresholds"]) == f["thresholds_sha256"]


def test_single_model_frozen():
    m = json.loads((paths.BASELINES / "supervised_single_p99" / "model.json").read_text())
    t = json.loads((paths.BASELINES / "supervised_single_p99" / "thresholds.json").read_text())
    assert m["threshold"] == t["thresholds_logit"]["P99"] == -5.523461642265342
    assert _sha(paths.BASELINES / "supervised_single_p99" / m["checkpoint"]) == m["checkpoint_sha256"]


@pytest.fixture(scope="module")
def development_scores():
    from deepdas.ensemble.scoring import Ensemble, SingleModel, load_norm, score_units
    from deepdas.evaluation.development import development_units
    cases, CHA, AMA, VEH = development_units()
    units = VEH.rename(columns={"vkey": "unit", "track_key": "tracks"})[["unit", "tracks"]]
    return cases, score_units(CHA, AMA, units, Ensemble(), SingleModel(), load_norm()).set_index("unit")


def test_development_cases_match_stored_scores(development_scores):
    from deepdas.ensemble.scoring import Ensemble
    cases, A = development_scores; ens = Ensemble()
    E = pd.read_csv(paths.RESULTS / "validation/ensemble_scores_selection_round.csv"); E = E[E.group == "valid9"].set_index("case")
    for k in range(5):
        S = pd.read_csv(paths.RESULTS / f"validation/seed_{k}/vehicle_scores.csv"); S = S[S.group == "valid9"].set_index("case")
        for u, c in zip(cases.unit_key, cases.internal_case_label):
            assert abs((A.loc[u, f"E_DIV_s{k}_logit"] - ens.mean[k]) / ens.std[k] - E.loc[c, f"z{k}"]) <= 1e-5
            assert int(A.loc[u, "n_windows"]) == int(S.loc[c, "n_windows"])
    for u, c in zip(cases.unit_key, cases.internal_case_label):
        assert abs(A.loc[u, "E_DIV_ens"] - E.loc[c, "ens"]) <= 1e-5
    for u, c in zip(cases.unit_key, cases.internal_case_label):
        assert (A.loc[u, "E_DIV_ens"] > ens.threshold) == (E.loc[c, "ens"] > ens.threshold)
    assert int((A.E_DIV_decision == "ALARM").sum()) == 8
    assert int((A.CR_P99_decision == "ALARM").sum()) == 7


def test_development_tables_byte_identical(tmp_path):
    from deepdas.evaluation.development import reproduce
    T, D = reproduce(); f, g = tmp_path / "d.csv", tmp_path / "e.csv"; T.to_csv(f, index=False); D.to_csv(g, index=False)
    assert f.read_bytes() == (paths.RESULTS / "development/development_reproduction.csv").read_bytes()
    assert g.read_bytes() == (paths.RESULTS / "development/development_cases.csv").read_bytes()


def test_results_equal_original_outputs():
    for line in (ROOT / "results/ORIGINAL_OUTPUTS_SHA256.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            h, rel = line.split("  ", 1)
            assert _sha(ROOT / rel) == h, rel


def test_all_locations_inside_release():
    for k, v in vars(paths).items():
        if k.isupper() and isinstance(v, Path):
            assert ROOT.resolve() in [v.resolve(), *v.resolve().parents], k
