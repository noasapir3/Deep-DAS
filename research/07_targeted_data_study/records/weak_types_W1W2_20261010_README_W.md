# W1 / W2: research record of the last improvement round (not adopted)
**Status:** research experiments only. Neither arm passed the pre-registered rule (C5, false alarms on legitimate stops), so **E_stall_diversity_ENSEMBLE stays the final model.**
Nothing here is used by the final model, and nothing here may overwrite it.

## Contents

| Path | What it is |
|---|---|
| `plan/` | Draft, v2 and FROZEN plan (with erratum 1); freeze manifests (plan/data and code); diff of the training script vs the E_DIV script |
| `scripts/` | `w00` hash guard; `w01`–`w04` planning, batch generation and independence; `w05` import; `x_train_eval_w.py` training; `w07` frozen evaluation + rule; `w08` report |
| `data/src/` | North-TLV scenario files used (identical copies of the map scenarios) |
| `data/RUN_PLAN_W.csv`, `data/plan_checks/`, `data/route_survey.csv` | Run plan (314 runs, seeds, event windows) and checks before simulation |
| `sims/` | **All 314 raw simulation runs**: scenario, export, truth trail, run.json, realization.json |
| `sims_feasibility_DISCARDED/` | 14 feasibility dry runs, never used |
| `data/W_processed/`, `data/W_vehicles.csv`, `data/W_run_report.csv` | Imported channels / amplitude / track metadata and per-vehicle labels |
| `runs/W{1,2}_s{0..4}/model/` | Checkpoints (encoder + head + pooled-feature normalisation fmu/fsd + tg-feature constants emu/esd), per-seed FROZEN_THRESHOLDS.json, val_B logits, history, exposure logs, train logs |
| `results/` | `RESULTS_W.md`, `SELECTION_W.json` (decision), `metrics_all.csv`, `unit_scores.csv`, `calibration.json` (per-seed val_B mean/std, P99, ensemble P99), per-unit tables, bootstrap CI |
| `results_dryrun/` | Code test of w07 on the old validation only (E_DIV), before freezing |
| `guard/GUARD_LOG.txt` | Every hash-guard run of the frozen model and its backup (all 0 differences) |
| `logs_*.log`, `LAUNCH_NOTE.txt`, `aborted_launch_missing_env/` | Training logs and disclosed launch incidents |
| `REPRODUCTION_CHECK_W.txt` | Closing reproducibility check |
| `SHA256_W_ALL.txt` | sha256 of every file in this folder |

**Shared inputs, not copied here:** the input normalisation `norm.json`, the initial autoencoder, the channel / preprocessing code, the E_DIV training data and the inference code (`hs_scorer.py`). All of them are inside the frozen backup `DL/FROZEN_FINAL_MODEL__E_stall_diversity_ENSEMBLE__20261010/mirror/`, at the same relative paths.

## Reproducing the W results
1. Restore the frozen backup into an empty folder: `bash tools/b02_restore_and_reproduce.sh <dir>` from the backup.
2. Copy this whole folder to `<dir>/experiments_followup/weak_types_W1W2_20261010/`.
3. Run `python3 scripts/w07_evaluate.py`. This scores E_DIV, W1 and W2 on the validation data, re-derives every threshold, and applies C1–C9.
4. Compare `results/` with the stored copy. The files are byte-identical; this was done on 2026-10-10 (`REPRODUCTION_CHECK_W.txt`).

**Retraining** (not to be done; documented only): run `run_train.sh` with seeds 0–4, which is deterministic per seed.
