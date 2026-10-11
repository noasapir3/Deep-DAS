#!/usr/bin/env python3
"""W00 — hash guard of the frozen model E_stall_diversity_ENSEMBLE (read-only). Re-hashes every original source file listed in the backup's
SHA256_SOURCE_vs_BACKUP.tsv (5,414 files: checkpoints, thresholds, normalisation, inference code, data, final-test results) and every backup file
(SHA256_BACKUP.txt). Exit code 1 and 'GUARD_FAIL' on any difference. Appends one line to guard/GUARD_LOG.txt.  Usage: w00_hash_guard.py <stage_name>"""
import csv, hashlib, sys, time
from pathlib import Path
W = Path(__file__).resolve().parents[1]; DL = W.parents[1]; B = DL / "FROZEN_FINAL_MODEL__E_stall_diversity_ENSEMBLE__20261010"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
bad = []; n = 0
for r in csv.DictReader(open(B / "SHA256_SOURCE_vs_BACKUP.tsv"), delimiter="\t"):
    n += 1; p = DL / r["path_rel_to_DL"]
    if not p.exists() or sha(p) != r["sha256_source"]:
        bad.append("source:" + r["path_rel_to_DL"])
m = 0
for line in open(B / "SHA256_BACKUP.txt"):
    h, p = line.rstrip("\n").split("  ", 1); m += 1
    if sha(B / p) != h:
        bad.append("backup:" + p)
for line in open(W / "guard/BACKUP_TOPLEVEL_sha256.txt"):        # backup README / record / manifests / tools (hashed at the start of the round)
    h, p = line.rstrip("\n").split("  ", 1); m += 1
    if sha(B / p) != h:
        bad.append("backup_toplevel:" + p)
stage = sys.argv[1] if len(sys.argv) > 1 else "manual"
msg = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} stage={stage} originals={n} backup={m} differences={len(bad)} -> {'GUARD_OK' if not bad else 'GUARD_FAIL ' + ';'.join(bad[:10])}"
(W / "guard").mkdir(exist_ok=True); open(W / "guard/GUARD_LOG.txt", "a").write(msg + "\n"); print(msg)
sys.exit(1 if bad else 0)
