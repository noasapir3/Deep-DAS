#!/usr/bin/env python3
"""Reproduce the validation scores of DeepDAS Ensemble v1 with the frozen checkpoints and thresholds (nothing is trained or calibrated).
Per seed: <out>/seed_<k>/vehicle_scores.csv and vehicle_scores_legitimate_stop_set.csv; then the ensemble tables ensemble_scores.csv,
ensemble_scores_legitimate_stop_set.csv and validation_metrics.csv. Seeds run two at a time (about 3 GB RAM each); --workers 1 runs
them one after the other (for machines with less than about 6 GB RAM).
Usage: python scripts/reproduce_validation.py [out_dir] [--workers 1|2] [--phi-mode active|neutralized]
(default results/validation, 2 workers; neutralized: results_phi_neutralized/validation, frozen mu_k, sigma_k, tau)"""
import shutil, subprocess, sys
from pathlib import Path
import _setup  # noqa: F401
import _phi_io  # noqa: E402  (--phi-mode active|neutralized; exported to the seed workers via DEEPDAS_PHI_MODE)
from deepdas import paths

args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] not in ("--seed", "--workers")]
out = Path(args[0]) if args else _phi_io.default_out(paths, "validation"); out.mkdir(parents=True, exist_ok=True)
if "--seed" in sys.argv:                                  # worker: one seed
    from deepdas.train.supervised import evaluate_frozen_seed
    s = int(sys.argv[sys.argv.index("--seed") + 1]); work = out / f"_work_seed_{s}"
    evaluate_frozen_seed(s, work)
    (out / f"seed_{s}").mkdir(exist_ok=True)
    for f in ("vehicle_scores.csv", "vehicle_scores_legitimate_stop_set.csv"):
        shutil.move(str(work / "evaluation" / f), str(out / f"seed_{s}" / f))
    shutil.rmtree(work); sys.exit(0)

chain = lambda seeds: subprocess.Popen(" && ".join(f'"{sys.executable}" "{__file__}" "{out}" --seed {s}' for s in seeds), shell=True)
workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
procs = [chain((0, 1, 2)), chain((3, 4))] if workers > 1 else [chain((0, 1, 2, 3, 4))]
if not all(p.wait() == 0 for p in procs):
    sys.exit("a seed failed")
from deepdas.evaluation.validation import ensemble_tables  # noqa: E402
M, E, LE = ensemble_tables([out / f"seed_{s}" for s in range(5)], frozen=(_phi_io.MODE == "neutralized"))
E.to_csv(out / "ensemble_scores.csv", index=False); LE.to_csv(out / "ensemble_scores_legitimate_stop_set.csv", index=False); M.to_csv(out / "validation_metrics.csv", index=False)
_phi_io.stamp(out); print(f"[phi_mode={_phi_io.MODE}]")
print(M[["seed", "macro9", "val_AP", "stallval_macro3", "val_tailgating_n", "FA_valA_rate", "stallval_alarm_legit_stop_n", "diag9_detected"]].to_string(index=False))
