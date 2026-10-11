# Tailgating dataset — plan v3 (versioned revision; written after the v2 pilot and before any v3 run)

**v2 result (kept in `sims_v2/`, `data/realization_v2.csv`):** train 17/24 (71 %), validation 10/12 (83 %).
Duration control worked: D = 8 / 15 / 25 s gave mean episodes of 9.6 / 16.3 / 26.3 s.

**v2 failures, diagnosed from truth only:**
1. The follower's desired speed was 1.3 v_L. The IDM free-road term (v/v0)^4 inflated the equilibrium gap by about 30 %, so at ~6 m/s the
   headway settled at 0.83–0.93 s instead of ~0.55 s (031, 042, 025_1).
2. The route search continued past boundary-exit nodes. The leader left the map after ~7 s (067, 122).

**v3 changes (and nothing else):**
- The route search stops at boundary-exit nodes (region survey recomputed into `region_survey_v3.csv`).
- During the event the follower's desired speed is 2.5 v_L with `ignore_speed_limit`. While it follows, its speed is bounded by the leader.
  At release (t = 8 s + D) it gets back the default IDM headway, `ignore_speed_limit = False` and an ordinary speed profile at v_L ± 0.5 m/s,
  so it does not become a speeding vehicle after the episode. Speeding labels are not affected: the vehicle is labelled only as tailgating.
- Same regions, the same train / validation split, the same grid (v_L × T_f ∈ {0.20, 0.30} × D) and the same label as v2.

**Success criterion:** unchanged (≥ 80 % in train and in validation). If a region has no eligible route after the boundary rule, it is reported, not replaced.
