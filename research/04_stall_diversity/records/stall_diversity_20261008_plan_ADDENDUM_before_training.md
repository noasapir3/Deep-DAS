# Addendum — written during the simulation batch, before any training or scoring

1. **Recording length.**
   - The unchanged runner stops a run at the 120 s cap or earlier, when every vehicle has left the map (`all_exited`).
   - START and STOP_END runs always reach 120 s, because a vehicle stays on the map.
   - MSM, CONTROL and LEGIT_STOP runs may end earlier.
   - This is a property of the runner, not changed. The duration per run is reported in `run_tracking_report.csv` / the realization files.
2. **Background vehicles in training.** These are taken only from runs shared by both arms (`arms == both`: CONTROL, LEGIT_STOP and the group-A MSM runs), so the added normal set is identical in both arms.
   Backgrounds of arm-specific runs (the quantity-only MSM runs and the diversity-only START / STOP_END runs) are not used in training.
   They are near-copies of the same scenes.
3. **Checkpoint selection set.** The checkpoint is selected on the original C_new validation set; stallval is never part of it, as stated in the plan.
   The training script is `x_train_eval_s.py`. Its full diff to `x_train_eval_c.py` is in `data/DIFF_x_train_eval_s_vs_c.txt`.
   The only changes are the STALLARM data addition, an exposure log that consumes no random numbers, and the stallval scoring group.
4. **Pilot.** Four runs (098 slot 1 MSM / START / LEGIT_STOP, 015 slot 3 STOP_END) were run first as part of the batch, not as retries.
   All four were verified, and none was rerun.

## Erratum at launch (2026-10-08, after 3 epochs of seed 0 and before any evaluation)
- **What went wrong.** The first launch included unaffected backgrounds from arm-specific runs (78 in quantity, 75 in diversity), contrary to point 2. The logs showed different normal counts per arm (289 vs 285 added background normals).
- **What was done.** Both chains were stopped, and the partial runs were moved to `aborted_launch_wrong_background_filter/`. They were never evaluated or used.
- **The fix.** A one-line filter now takes normal vehicles only from runs with `arms == both`. Nothing else changed, and the relaunch starts from scratch.
- **The C_new re-score of seed 0** does not depend on this filter (STALLARM = none). It was kept.
