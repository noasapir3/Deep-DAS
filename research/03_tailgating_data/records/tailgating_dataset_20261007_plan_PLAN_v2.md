# Tailgating dataset — plan v2 (versioned revision; written after the v1 pilot and before any v2 run)

**v1 result (kept unchanged in `sims/`, `data/realization_v1.csv`):** 10/24 train and 7/12 validation realized. This is below the pre-registered
80 % bar, so v1 is reported as is.

**v1 diagnosis (truth only, no detector involved):**
1. The start gap (23.5 m) with a closing speed of ~1.5 m/s left too little shared route. The leader often left the follower's lane before contact.
2. T_f = 0.40 s gives a steady headway ≈ 0.73 s at 6–7 m/s, too close to 0.8 s: normal speed noise pushes it over (1/14 realized).
3. Duration was not controlled: after the route prefix both vehicles drew the same random turns (same seed), so pairs differing only in D were identical.
4. The label's "same lane id" rule lost the leader for a moment at every lane-segment boundary, splitting continuous episodes.

**v2 changes (and nothing else):**
- Follower start s = 25.5 m (bumper gap 10 m instead of 23.5 m).
- T_f ∈ {0.20, 0.30} s (steady headway 0.37–0.63 s over the realized speeds), with v_L and D as in v1.
- Duration control: at t = 8 s + D the follower is released to the simulator's default IDM headway (T 1.2 s, s0 2 m), so the episode ends near D.
  The shared route covers 8 s + D + 10 s.
- Label gap measured along the shared route when both vehicles are on it (crossing lane boundaries); otherwise the v1 same-lane rule.
  The thresholds are unchanged (headway < 0.8 s, speed ≥ 5 m/s, ≥ 5 s continuous, gap ≥ 1 m, no follower collision).
- Same regions and the same train / validation split as v1. The final test remains ungenerated (FINAL_HOLDOUT_PLAN.md).

**Success criterion:** unchanged (≥ 80 % realization in train and in validation).
