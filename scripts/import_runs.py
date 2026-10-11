#!/usr/bin/env python3
"""Import simulated runs (folders written by deepdas.batches generators) into the processed format of the model.
Usage: python scripts/import_runs.py <sims_dir> <out_dir> <group_name> [run_id ...]
The region of each run is read from its run.json plan row."""
import json, sys
from pathlib import Path
import _setup  # noqa: F401
from deepdas.importing import import_run, write_run

if len(sys.argv) < 4:
    sys.exit(__doc__)
sims, out, group = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]; ids = set(sys.argv[4:])
for d in sorted(p for p in sims.iterdir() if p.is_dir() and (not ids or p.name in ids)):
    info = json.loads((d / "run.json").read_text()); region = str((info.get("plan_row") or {}).get("region") or info.get("region")).zfill(3)
    ch, amp, meta = import_run(d, d.name, region, group); write_run(out, d.name, ch, amp, meta)
    print(d.name, "tracks", len(meta), "usable", int(meta.usable.sum()))
