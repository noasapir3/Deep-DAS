#!/usr/bin/env python3
"""
Release verification (docs/EVALUATION_PROTOCOL.md §5–6).

  E1  every file against MANIFEST_SHA256.txt; frozen checkpoints against the hashes recorded at freezing (ensemble.json, model.json)
  E2  the nine development cases re-scored with the frozen models (criterion v1.1, docs/EVALUATION_PROTOCOL.md §6): per-seed z and
      ensemble score |Δ| <= 1e-5 against the stored scores; identical window counts, decisions (both models, both unit definitions) and
      thresholds; 8/9 (DeepDAS Ensemble v1) and 7/9 (single model); byte-identical development tables. Raw logits are reported, not gating.
  E3  (full mode) validation and final-test results regenerated into outputs/verify/ and compared with results/: every table byte-identical;
      every score file byte-identical, or (other CPU architectures) equal in keys, window counts and every decision, with z and
      ensemble scores |Δ| <= 1e-5; the shipped results/ compared with the sha256 of the original outputs (results/ORIGINAL_OUTPUTS_SHA256.txt)
  E9  self-containment and hygiene of the release code

Usage: python scripts/verify_release.py [--quick] [--steps E1,E2,E3a,E3b,E3c,E9] [--workers 1]
       --quick: E1, E2, E9 (a few minutes). Default: all (about 10 minutes on 2 cores). --steps: only the listed checks.
       --workers 1: regenerate the validation seeds one after the other (machines with less than about 6 GB RAM).
Exit status 0 only if every check passes. A report is written to outputs/verify/VERIFY_REPORT.json.
"""
import _setup  # noqa: F401
import ast
import hashlib
import io
import json
import re
import subprocess
import sys
import tokenize
from pathlib import Path

import numpy as np
import pandas as pd

from deepdas import paths

ROOT = paths.ROOT
OUT = ROOT / "outputs" / "verify"
TOL = 1e-5
QUICK = "--quick" in sys.argv
STEPS = set(sys.argv[sys.argv.index("--steps") + 1].split(",")) if "--steps" in sys.argv else \
    ({"E1", "E2", "E9"} if QUICK else {"E1", "E2", "E3a", "E3b", "E3c", "E9"})
THR = 1.9726181803233118
REPORT = {}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def record(check, ok, **details):
    REPORT[check] = {"pass": bool(ok), **details}
    print(f"[{'PASS' if ok else 'FAIL'}] {check}  " + "  ".join(f"{k}={v}" for k, v in details.items() if not isinstance(v, (list, dict))), flush=True)


# ---------------------------------------------------------------------------------------------------------------- E1 manifest
def e1():
    man = ROOT / "MANIFEST_SHA256.txt"
    if not man.exists():
        record("E1_manifest", False, reason="MANIFEST_SHA256.txt missing"); return
    bad, n = [], 0
    for line in man.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        h, rel = line.split("  ", 1); n += 1; p = ROOT / rel
        if not p.exists():
            bad.append(("missing", rel))
        elif sha(p) != h:
            bad.append(("hash", rel))
    spec = json.loads((paths.ENSEMBLE / "ensemble.json").read_text())
    for s in spec["seeds"]:
        if sha(paths.ENSEMBLE / s["checkpoint"]) != s["checkpoint_sha256"]:
            bad.append(("frozen checkpoint", s["checkpoint"]))
    single = json.loads((paths.BASELINES / "supervised_single_p99" / "model.json").read_text())
    if "checkpoint_sha256" in single and sha(paths.BASELINES / "supervised_single_p99" / single["checkpoint"]) != single["checkpoint_sha256"]:
        bad.append(("frozen checkpoint", "supervised_single_p99"))
    record("E1_manifest", not bad, files=n, problems=len(bad), detail=bad[:20])


# ---------------------------------------------------------------------------------------------------------------- E2 development
def e2():
    from deepdas.ensemble.scoring import Ensemble, SingleModel, load_norm, score_units
    from deepdas.evaluation.development import development_units, reproduce
    cases, CHA, AMA, VEH = development_units(); ens, single, norm = Ensemble(), SingleModel(), load_norm()
    A = score_units(CHA, AMA, VEH.rename(columns={"vkey": "unit", "track_key": "tracks"})[["unit", "tracks"]], ens, single, norm).set_index("unit")
    lab = dict(zip(cases.unit_key, cases.internal_case_label))
    prob = []
    # stored per-seed logits and z / ensemble scores of the selection round (rows 'valid9')
    E = pd.read_csv(paths.RESULTS / "validation" / "ensemble_scores_selection_round.csv"); E = E[E.group == "valid9"].set_index("case")
    max_logit = max_z = max_ens = 0.0
    for k in range(5):
        S = pd.read_csv(paths.RESULTS / "validation" / f"seed_{k}" / "vehicle_scores.csv"); S = S[S.group == "valid9"].set_index("case")
        for u, c in lab.items():
            max_logit = max(max_logit, abs(A.loc[u, f"E_DIV_s{k}_logit"] - S.loc[c, "logit"]))           # reported only
            if int(A.loc[u, "n_windows"]) != int(S.loc[c, "n_windows"]):
                prob.append(f"windows {c}")
            z = (A.loc[u, f"E_DIV_s{k}_logit"] - ens.mean[k]) / ens.std[k]; dz = abs(z - E.loc[c, f"z{k}"]); max_z = max(max_z, dz)
            if dz > TOL:
                prob.append(f"seed {k} z {c}: {dz:.2e}")
    margins = []
    for u, c in lab.items():
        d = abs(A.loc[u, "E_DIV_ens"] - E.loc[c, "ens"]); max_ens = max(max_ens, d)
        if d > TOL:
            prob.append(f"ensemble {c}: {d:.2e}")
        if (A.loc[u, "E_DIV_ens"] > ens.threshold) != (E.loc[c, "ens"] > ens.threshold):
            prob.append(f"decision {c}")
        margins.append(abs(float(E.loc[c, "ens"]) - ens.threshold))
    # thresholds and constants: those of the frozen records, never recomputed
    frz = json.loads((paths.ENSEMBLE / "freeze_record.json").read_text())
    spec = json.loads((paths.ENSEMBLE / "ensemble.json").read_text())
    st = json.loads((paths.BASELINES / "supervised_single_p99" / "thresholds.json").read_text())
    thr_ok = ens.threshold == frz["thresholds_ensemble"]["P99"] == THR
    thr_ok &= single.threshold == st["thresholds_logit"]["P99"] == -5.523461642265342
    for k, s_ in enumerate(spec["seeds"]):
        f = frz["seeds"][str(k)]
        thr_ok &= (s_["valB_mean"], s_["valB_std_ddof0"], s_["checkpoint_sha256"]) == (f["valB_mean"], f["valB_std_ddof0"], f["checkpoint_sha256"])
        thr_ok &= (ens.mean[k], ens.std[k]) == (f["valB_mean"], f["valB_std_ddof0"])
        thr_ok &= sha(paths.ENSEMBLE / s_["thresholds"]) == f["thresholds_sha256"]
    if not thr_ok:
        prob.append("thresholds or constants differ from the frozen record")
    # the development tables: byte-identical, decisions and counts
    OUT.mkdir(parents=True, exist_ok=True); T, D = reproduce(); f = OUT / "development_reproduction.csv"; T.to_csv(f, index=False)
    g = OUT / "development_cases.csv"; D.to_csv(g, index=False)
    ref = paths.RESULTS / "development" / "development_reproduction.csv"
    byte = f.read_bytes() == ref.read_bytes() and g.read_bytes() == (paths.RESULTS / "development" / "development_cases.csv").read_bytes()
    R = pd.read_csv(ref)
    for col in ("E_DIV_alarm_original_units", "E_DIV_alarm_track_units", "CR_P99_alarm_original_units", "CR_P99_alarm_track_units", "n_windows", "n_tracks"):
        if not (T[col].values == R[col].values).all():
            prob.append(f"table column {col}")
    n_e, n_c = int(T.E_DIV_alarm_original_units.sum()), int(T.CR_P99_alarm_original_units.sum())
    if (n_e, n_c) != (8, 7):
        prob.append(f"detections {n_e}/9, {n_c}/9")
    if not byte:
        prob.append("development tables not byte-identical")
    record("E2_development", not prob, max_dz=f"{max_z:.3g}", max_d_ensemble=f"{max_ens:.3g}", max_d_logit_reported=f"{max_logit:.3g}",
           closest_case_to_threshold=f"{min(margins):.3g}", deepdas_detected=f"{n_e}/9", single_detected=f"{n_c}/9",
           tables_byte_identical=byte, problems=prob)


# ---------------------------------------------------------------------------------------------------------------- E3 tables
def _rd(p):
    return pd.read_csv(p, keep_default_na=True)


def _maxdiff(a, b):
    a, b = pd.to_numeric(a, errors="coerce").to_numpy(float), pd.to_numeric(b, errors="coerce").to_numpy(float)
    if not (np.isnan(a) == np.isnan(b)).all():
        return float("inf")
    d = np.abs(a - b)[~np.isnan(a)]
    return float(d.max()) if len(d) else 0.0


def equivalent(name: str, new: Path, ref: Path):
    """For a score file that is not byte-identical: same rows, keys, window counts and decisions; z / ensemble scores within TOL.
    Returns (ok, details). Raw logits and the single-model logit are reported, not gating."""
    a, b = _rd(new), _rd(ref); det = {"rows": f"{len(a)}/{len(b)}"}; ok = len(a) == len(b) and list(a.columns) == list(b.columns)
    if not ok:
        return False, det
    key = [c for c in ("group", "vkey", "key", "unit", "vk", "seed", "case") if c in a.columns]
    ok &= all(a[c].astype(str).equals(b[c].astype(str)) for c in key)
    tol_cols = [c for c in a.columns if re.fullmatch(r"z\d|ens|E_DIV_ens|E_DIV_score|E_DIV_orac_score", c)]
    rep_cols = [c for c in a.columns if c.endswith("logit") or c in ("CR_P99_score", "CR_orac_score")]
    for c in tol_cols:
        d = _maxdiff(a[c], b[c]); det[f"max_d_{c}"] = f"{d:.3g}"; ok &= d <= TOL
    for c in rep_cols:
        det[f"reported_max_d_{c}"] = f"{_maxdiff(a[c], b[c]):.3g}"
    if "E_DIV_s0_logit" in a.columns:                          # per-seed z derived with the frozen val_B constants
        seeds = json.loads((paths.ENSEMBLE / "ensemble.json").read_text())["seeds"]
        for k, sd in enumerate(seeds):
            za = (pd.to_numeric(a[f"E_DIV_s{k}_logit"]) - sd["valB_mean"]) / sd["valB_std_ddof0"]
            zb = (pd.to_numeric(b[f"E_DIV_s{k}_logit"]) - sd["valB_mean"]) / sd["valB_std_ddof0"]
            d = _maxdiff(za, zb); det[f"max_d_z{k}"] = f"{d:.3g}"; ok &= d <= TOL
    if name == "validation_metrics.csv":                      # counts, recalls and rates identical; continuous columns within TOL
        for c in a.columns:
            if pd.api.types.is_numeric_dtype(a[c]):
                d = _maxdiff(a[c], b[c])
                if d > 0:
                    det[f"max_d_{c}"] = f"{d:.3g}"; ok &= d <= TOL and not c.endswith("_n")
            else:
                ok &= a[c].astype(str).equals(b[c].astype(str))
        return bool(ok), det
    # everything that is not a score must be identical: decisions, window counts, labels, flags
    for c in a.columns:
        if c in tol_cols or c in rep_cols:
            continue
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            same = _maxdiff(a[c], b[c]) == 0
        else:
            same = a[c].astype(str).equals(b[c].astype(str))
        if not same:
            ok = False; det[f"differs_{c}"] = True
    # decisions implied by the ensemble score; distance of the closest unit to the threshold (reported)
    for c in ("ens", "E_DIV_ens", "E_DIV_score"):
        if c in a.columns:
            ok &= bool(((a[c] > THR) == (b[c] > THR)).all())
            det[f"closest_to_threshold_{c}"] = f"{float((pd.to_numeric(b[c]) - THR).abs().min()):.3g}"
    return bool(ok), det


TABLES = {"T1_per_type.csv", "T2_overall.csv", "T3_stops_tailgating.csv", "T4_secondary.csv", "T5_uncertainty.json", "T6_focus_runs.csv",
          "T7_zscore_top_features_on_anomalies.csv", "FINAL_TEST_REPORT.md"}


def compare_dirs(new: Path, ref: Path, files):
    res, details = {}, {}
    for f in files:
        a, b = new / f, ref / f
        if not (a.exists() and b.exists()):
            res[f] = "missing"; continue
        if a.read_bytes() == b.read_bytes():
            res[f] = "byte-identical"; continue
        if f in TABLES:
            res[f] = "FAIL (table not byte-identical)"; continue
        ok, det = equivalent(Path(f).name, a, b); details[f] = det
        res[f] = "equivalent (decisions identical, scores within 1e-5)" if ok else "FAIL"
    return res, details


def e3a():
    orig = ROOT / "results" / "ORIGINAL_OUTPUTS_SHA256.txt"; bad = []
    for line in orig.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            h, rel = line.split("  ", 1)
            if not (ROOT / rel).exists() or sha(ROOT / rel) != h:
                bad.append(rel)
    record("E3a_results_equal_original_outputs", not bad, problems=bad)


def _record_cmp(check, ok, r, d):
    nb = sum(v == "byte-identical" for v in r.values()); ne = sum(v.startswith("equivalent") for v in r.values())
    record(check, ok and r and all(v == "byte-identical" or v.startswith("equivalent") for v in r.values()),
           files=len(r), byte_identical=nb, equivalent=ne, result=r, details=d)


def e3b():
    vd = OUT / "validation"
    extra = ["--workers", sys.argv[sys.argv.index("--workers") + 1]] if "--workers" in sys.argv else []
    ok = subprocess.run([sys.executable, str(ROOT / "scripts/reproduce_validation.py"), str(vd), *extra]).returncode == 0
    files = [f"seed_{k}/{n}" for k in range(5) for n in ("vehicle_scores.csv", "vehicle_scores_legitimate_stop_set.csv")]
    files += ["ensemble_scores.csv", "ensemble_scores_legitimate_stop_set.csv", "validation_metrics.csv"]
    r, d = compare_dirs(vd, paths.RESULTS / "validation", files) if ok else ({}, {})
    _record_cmp("E3b_validation_regenerated", ok, r, d)


def e3c():
    fd = OUT / "final_test"
    ok = subprocess.run([sys.executable, str(ROOT / "scripts/reproduce_final_test.py"), str(fd)]).returncode == 0
    files = ["scores_dl_tracks.csv", "scores_dl_oracle_vehicles.csv", "vehicle_outcomes.csv", *sorted(TABLES)]
    r, d = compare_dirs(fd, paths.RESULTS / "final_test", files) if ok else ({}, {})
    _record_cmp("E3c_final_test_regenerated", ok, r, d)


# ---------------------------------------------------------------------------------------------------------------- E9 hygiene
VENDORED = ("src/deepdas/fp/", "src/deepdas/fp_anomaly_model/", "src/deepdas/batches/populate_vehicles.py")
FORBIDDEN_IMPORTS = {"simstudio", "anomaly_model"}
MACHINE = re.compile(r"/Users/|/home/|/sessions/|[A-Z]:\\\\")
TEMP_COMMENT = re.compile(r"(?i)\b(todo|fixme|hack|xxx|erratum|addendum|corrected|temporar\w*|old version|claude|chatgpt)\b|(?<!Ensemble )\bv[1-4]\b")
TEMP_NAMES = re.compile(r"^(p10|p11|p12|h01|h03|h04|h05|w03|w05|w07|tg3|tg4|TG3_TRAIN|P10_TYPES|V1)$")


def code_files():
    for sub in ("src", "scripts", "tests"):
        for p in sorted((ROOT / sub).rglob("*.py")):
            yield p
    yield from sorted((ROOT / "research").glob("*/code/*.py"))


def e9():
    prob = []
    for p in code_files():
        rel = str(p.relative_to(ROOT)); txt = p.read_text(encoding="utf-8")
        tree = ast.parse(txt)
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                    [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else []
            for n in names:
                if n.split(".")[0] in FORBIDDEN_IMPORTS and not rel.startswith(VENDORED):
                    prob.append(f"{rel}: imports {n}")
        if rel.startswith(VENDORED) or rel.endswith("verify_release.py") or rel.startswith("tests/"):
            continue
        if MACHINE.search(txt):
            prob.append(f"{rel}: machine-specific path")
        for t in tokenize.generate_tokens(io.StringIO(txt).readline):
            if t.type == tokenize.COMMENT and TEMP_COMMENT.search(t.string):
                prob.append(f"{rel}:{t.start[0]}: comment {t.string.strip()[:60]!r}")
            if t.type == tokenize.NAME and TEMP_NAMES.match(t.string):
                prob.append(f"{rel}:{t.start[0]}: name {t.string}")
    # every location used by the code lies inside the release
    for k, v in vars(paths).items():
        if k.isupper() and isinstance(v, Path) and ROOT.resolve() not in [v.resolve(), *v.resolve().parents]:
            prob.append(f"paths.{k} outside the release: {v}")
    record("E9_self_containment_hygiene", not prob, files=sum(1 for _ in code_files()), problems=prob)


if __name__ == "__main__":
    import platform
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT["platform"] = {"machine": platform.machine(), "system": platform.system(), "python": platform.python_version()}
    for step, fn in (("E3a", e3a), ("E3b", e3b), ("E3c", e3c), ("E1", e1), ("E2", e2), ("E9", e9)):   # E3 first: its subprocesses need the memory
        if step in STEPS:
            fn()
    REPORT["steps"] = sorted(STEPS)
    name = "VERIFY_REPORT.json" if "--steps" not in sys.argv else f"VERIFY_REPORT_{'_'.join(sorted(STEPS))}.json"
    (OUT / name).write_text(json.dumps(REPORT, indent=1, default=str))
    ok = all(v["pass"] for k, v in REPORT.items() if isinstance(v, dict) and "pass" in v)
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED", "->", OUT / name)
    sys.exit(0 if ok else 1)
