# Tailgating dataset — plan v4 (TRAINING ONLY): speed variety + matched normal controls
Written 2026-10-08, before the v4 survey, run plan or any v4 simulation.

**Purpose:**
- Widen the speed range of the training tailgaters. v3 train had only 6–7 m/s, because its routes crossed 30 km/h segments.
- Add normal examples that match the tailgating runs in scene, route, speed and duration, so a model has to learn the gap and headway rather than
  the scene type.

**Unchanged from v3:** injection code (shared planned route, leader pinned at v_L ± 0.5 m/s, follower 2.5 v_L desired speed with
`ignore_speed_limit` during the event, release at 8 s + D), the label (tailgating_v1, gap along the shared route, gap ≥ 1 m inside the episode,
no follower collision), seed 1, and the simulator.

**New in v4:**
- **Routes:** only segments with a limit ≥ 13.89 m/s (50 km/h), so the leader can drive at 9.0 m/s ("mid" band) or 11.8 m/s ("high" band, 0.85 × 13.89).
  The route search still stops at boundary exits.
- **Regions:** `train`-split regions only, excluding the 12 tg3 train regions and the earlier cross-region regions (131, 050, 090, 065).
  The survey goes into `region_survey_v4.csv`. Eligible regions need a route ≥ 40 + 11.8·(25 + 8 + 10) = 547.4 m.
  12 regions are drawn deterministically (numpy rng 20261008).
- **Runs:** 2 tailgating runs per region (24), covering band {mid, high} × T_f {0.20, 0.30} × D {8, 15, 25} cyclically, so each combination appears twice.
- **Matched controls:** every tailgating run gets a control run that is identical except that the follower keeps the simulator's default IDM headway
  (T 1.2 s, s0 2 m; steady headway ≈ 1.4–1.5 s). 24 control runs.
  - A control follower counts as a clean normal only if its longest continuous tailgating_v1 condition is < 1 s and it has no collision.
    Otherwise it is excluded and reported.
- **No validation or test data are created or moved.** Validation stays exactly the tg3 val_A runs plus the original val sets.

**Gap below 1 m — policy (from the tg3_026_0 audit):**
- A gap below 1 m is not marked as an artifact by itself.
- The simulator's hard-stop clamp (a speed loss > 2 m/s within one 0.02 s step) is recorded per run (`follower_hard_stop_clamp_t`) and reported.
  It does not change the label.

**Success criterion:**
- ≥ 80 % of tailgating runs realized and ≥ 80 % of controls clean.
- Realized tailgating episodes are present in both speed bands.

**Erratum (before any v4 simulation):**
- The first survey compared limits with `>= 13.89`, but 50 km/h is stored as 13.8889 m/s. Only 1 region qualified.
- The comparison now uses a numerical tolerance (`>= 13.88`), which matches the stated intent (50 km/h segments). Nothing else changed.
- The void outputs are kept as `*_VOID_threshold_bug.csv`.
