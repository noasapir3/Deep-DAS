# Tailgating dataset — pre-registered (written before the run plan and before any simulation)

**Goal:** tailgating events that are realized reliably, with varied speed, duration and region, split by region into train / validation / final test
before any training. Nothing here is tuned on C3 or on any detector output.

**Why earlier injections failed:** newscene realized 1/8 and the cross-region batch 3/6. Leader and follower had no shared route and split at the first
junction, and the start lanes were short.

## Injection (scripts/tg10_tailgating_batch.py; simulator unchanged)
- Leader veh002 at s = 40 m and follower veh001 at s = 12 m on the same start lane. The start lane has no other vehicle on it at t = 0.
- Both vehicles receive the SAME planned lane sequence (public `set_vehicle_route`, set by a hook at t = 0). This is the shortest prefix of the
  region's longest simple route that covers the target duration: length ≥ 40 m + v_L·(D + 8 s approach). After the prefix, routing is the
  simulator's normal random policy, so the episode ends naturally.
- Leader: speed pinned to v_L ± 0.5 m/s, with v_L = min(nominal, 0.85 × the lowest speed limit on the route).
- Follower: desired speed 1.3 v_L, IDM time headway T_f, IDM minimum gap s0 = 2 m, so the steady headway ≈ T_f + 2/v_L < 0.8 s and the gap stays physical.
- Grid: v_L ∈ {6, 9, 12} m/s × T_f ∈ {0.25, 0.40} s × D ∈ {8, 15, 25} s = 18 combinations, assigned cyclically to runs. Seed 1.

## Label (truth only; frozen before simulation)
`tailgating_v1` (truth headway < 0.8 s, own speed ≥ 5 m/s, continuously ≥ 5 s) AND bumper gap ≥ 1.0 m throughout that episode AND no collision of
the follower. The leader in the label is the nearest vehicle ahead on the same lane (any vehicle); the share that is veh002 is recorded.
Background vehicles: unaffected ones (no stop ≥ 2 s) are usable normals; the leader is never a normal.

## Split by region (fixed now, deterministic: numpy rng 20261007 over the eligible regions)
- **Train:** 12 regions of the `train` split, 2 runs each (24 runs).
- **Validation:** 6 regions of the `val_A` split, 2 runs each (12 runs).
- **Excluded for variety:** regions of the earlier cross-region batch (131, 050, 090, 065, 002, 005), and the 8 test-split regions
  (already used for analysis via newscene).
- **Final test:** NOT generated now. It is part of the final holdout (see `FINAL_HOLDOUT_PLAN.md`): scenes never opened, generated only once the
  architecture is frozen, with this same injection and label code.
- **Eligibility:** a free start lane, a route ≥ 6·(5 + 8) + 40 = 118 m, and 0.85 × the lowest route limit ≥ 6 m/s.

## Success criterion for the dataset (fixed now)
A realization rate ≥ 80 % in both train and validation, with episodes spread over speeds and durations. If it is lower, I report it and do not
change the parameters silently; any change becomes a new, versioned plan.
