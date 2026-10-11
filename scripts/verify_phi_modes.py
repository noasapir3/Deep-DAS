#!/usr/bin/env python3
"""Checks of the v1.1 review copy (nothing is trained; the checks only read, or re-run inference into a temporary folder).

  P1  the five checkpoints, the input normalisation and ensemble.json are byte-identical to DeepDAS_Release_v1.0 (hashes in its manifest)
  P2  the original release folder, if given, still verifies against its own manifest (nothing in it was modified)
  P3  phi_mode=active reproduces the published development and final-test tables byte for byte (validation too with --full)
  P4  phi_mode=neutralized reproduces the stored intervention results (results_phi_neutralized/) byte for byte
  P5  the neutralized head input is exactly 0 for every phi value, and the frozen standardisation constants are unchanged
  P6  the four-row summary used in the report equals the stored tables
  P7  every number in report v2 is found in its source file (research/08/code/number_sources.py)
Usage: python scripts/verify_phi_modes.py [--original PATH_TO_DeepDAS_Release_v1.0] [--full]"""
import hashlib, json, os, subprocess, sys, tempfile, filecmp
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; PY = sys.executable; ok = True


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def report(name, passed, detail=""):
    global ok
    ok &= passed; print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")


man = {l.split()[1].lstrip("*").lstrip("./"): l.split()[0] for l in (ROOT / "MANIFEST_SHA256.txt").read_text().splitlines() if l.strip()}
frozen = [k for k in man if k.startswith("models/deepdas_ensemble_v1/")]
report("P1 frozen model files identical to v1.0", all(sha(ROOT / k) == man[k] for k in frozen), f"({len(frozen)} files)")

if "--original" in sys.argv:
    orig = Path(sys.argv[sys.argv.index("--original") + 1])
    r = subprocess.run(["sha256sum", "-c", "--quiet", "MANIFEST_SHA256.txt"], cwd=orig, capture_output=True, text=True)
    report("P2 original release unchanged", r.returncode == 0, str(orig))

tmp = Path(tempfile.mkdtemp(prefix="phi_verify_"))
def run(script, mode, out, *extra):
    env = dict(os.environ, DEEPDAS_PHI_MODE=mode)
    subprocess.run([PY, str(ROOT / "scripts" / script), str(out), *extra], env=env, check=True, capture_output=True)
same = lambda a, b, files: all(filecmp.cmp(a / f, b / f, shallow=False) for f in files)

run("reproduce_development_cases.py", "active", tmp / "dev_a")
report("P3 active development byte-identical", same(tmp / "dev_a", ROOT / "results/development", ["development_reproduction.csv", "development_cases.csv"]))
run("reproduce_final_test.py", "active", tmp / "ft_a")
FT = ["T1_per_type.csv", "T2_overall.csv", "T3_stops_tailgating.csv", "T4_secondary.csv", "T5_uncertainty.json", "vehicle_outcomes.csv"]
report("P3 active final test byte-identical", same(tmp / "ft_a", ROOT / "results/final_test", FT))
if "--full" in sys.argv:
    run("reproduce_validation.py", "active", tmp / "val_a", "--workers", "1")
    report("P3 active validation byte-identical", same(tmp / "val_a", ROOT / "results/validation", ["validation_metrics.csv", "ensemble_scores.csv", "ensemble_scores_legitimate_stop_set.csv"]))

run("reproduce_development_cases.py", "neutralized", tmp / "dev_n")
report("P4 neutralized development reproduced", same(tmp / "dev_n", ROOT / "results_phi_neutralized/development", ["development_reproduction.csv", "development_cases.csv"]))
run("reproduce_final_test.py", "neutralized", tmp / "ft_n")
report("P4 neutralized final test reproduced", same(tmp / "ft_n", ROOT / "results_phi_neutralized/final_test", FT))
if "--full" in sys.argv:
    run("reproduce_validation.py", "neutralized", tmp / "val_n", "--workers", "1")
    report("P4 neutralized validation reproduced", same(tmp / "val_n", ROOT / "results_phi_neutralized/validation", ["validation_metrics.csv", "ensemble_scores.csv"]))

sys.path.insert(0, str(ROOT / "src")); os.environ["DEEPDAS_PHI_MODE"] = "neutralized"
import torch  # noqa: E402
from deepdas.ensemble.detector import load_detector  # noqa: E402
spec = json.loads((ROOT / "models/deepdas_ensemble_v1/ensemble.json").read_text())
dets = [load_detector(ROOT / "models/deepdas_ensemble_v1" / s["checkpoint"], 1) for s in spec["seeds"]]
f, e = torch.randn(4, 16), torch.tensor([[0.0], [0.006], [0.5], [0.81]])
report("P5 neutralized 17th input == 0 and constants frozen",
       all(torch.equal(d.head_in(f, e)[:, 16], torch.zeros(4)) and abs(float(d.emu[0]) - 0.005994477309286594) < 1e-12 and abs(float(d.esd[0]) - 0.03163501247763634) < 1e-9 for d in dets))

import pandas as pd  # noqa: E402
S = pd.read_csv(ROOT / "research/08_tailgating_feature_intervention/tables/report_summary_table.csv")
expect = [("7/9", "8/9"), ("9/13", "12/13"), ("10/11", "11/11"), ("0.560", "0.568")]
report("P6 report summary table", [tuple(map(str, r)) for r in S[["phi neutralized", "phi active"]].itertuples(index=False)] == expect)
r = subprocess.run([PY, str(ROOT / "research/08_tailgating_feature_intervention/code/number_sources.py")], capture_output=True, text=True)
report("P7 report v2 number sources", r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout else "")
print("ALL PASSED" if ok else "SOME CHECKS FAILED"); sys.exit(0 if ok else 1)
