#!/usr/bin/env python3
"""Train the seeds of DeepDAS Ensemble v1 from the released training data into an output directory (never into models/).
Each seed: training, threshold calibration on val_B and validation scoring. With the same library versions and --threads 1 the shipped
checkpoints are reproduced exactly. This is for verification and study; the released model is models/deepdas_ensemble_v1.
Usage: python scripts/train_ensemble.py <out_dir> [--seeds 0,1,2,3,4] [--threads 1]"""
import sys
from pathlib import Path
import _setup  # noqa: F401
from deepdas import paths
from deepdas.train.supervised import train_seed

if len(sys.argv) < 2:
    sys.exit(__doc__)
out = Path(sys.argv[1]).resolve()
if out == paths.ENSEMBLE.resolve() or paths.MODELS.resolve() in out.parents:
    sys.exit("refusing to write into models/")
seeds = [int(s) for s in sys.argv[sys.argv.index("--seeds") + 1].split(",")] if "--seeds" in sys.argv else [0, 1, 2, 3, 4]
threads = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 1
for s in seeds:
    train_seed(s, out / f"seed_{s}", threads)
