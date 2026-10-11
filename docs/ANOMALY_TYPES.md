# Anomaly types, injections and realisation rules

The nine anomaly types are those of the SimStudio Final Project.

- **Labels come only from the simulator truth.** An injected vehicle is labelled anomalous only if the effect is **measured** there (the realisation rule). The rule is applied before any model sees the run.
- **Runs that do not realise are excluded and reported.** They are never relabelled, and never re-run to obtain a realisation.

## Types

| Type (display) | Identifier in data | Injection in the cross-region set and the final test | Realisation rule (truth of the injected vehicle) |
|---|---|---|---|
| Speeding | `speeding` | speed profile 1.4 × the lane limit, speed limit ignored | p90 of speed / limit ≥ 1.10 |
| Reckless driving | `reckless_driver` | 1.4 × limit + weaving (1.5 m) + short headway (0.6 s, min gap 1.5 m), collisions allowed | p90 of speed / limit ≥ 1.10 |
| Hard braking | `hard_braking` | brake action at 6 m/s², at t = 6 s | peak deceleration ≥ 4 m/s² |
| Stalled vehicle | `stalled_vehicle` | stall (freeze) for 30 s at t = 6 s; in the stall sets, also START, STOP_END and gentle-stop modes | longest stop (truth v < 0.1 m/s) ≥ 10.0 s |
| Lane weaving | `lane_weaving` | sinusoidal lateral motion, amplitude 1.5 m, period 2.5 s | lateral std ≥ 1.0 m |
| Lane straddling | `lane_straddling` | fixed lateral offset of 1.8 m (on the lane line) | median \|lateral offset\| ≥ 1.7 m |
| Tailgating | `tailgating` | short desired headway behind a pinned leader on a shared planned route (see below) | `tailgating_v1` (below) |
| Collision | `collision` | the vehicle drives at 8 m/s into a frozen target ahead, with IDM disabled | ≥ 1 truth contact |
| Overweight vehicle | `overweight_vehicle` | weight 30 t | weight ≥ 20 t |

**Tailgating (`tailgating_v1`).** All of the following must hold for the follower:
- truth time headway < 0.8 s, at its own speed ≥ 5 m/s, for ≥ 5 s continuously;
- bumper gap ≥ 1 m throughout that episode, measured along the shared route;
- no collision of the follower.

In the cross-region set the rule was the earlier `gap_min < 5 m and headway p10 < 0.5 s`. The dedicated tailgating sets and the final test use `tailgating_v1`.

## Stops: one rule, no overlap

| Class | Truth definition | Label |
|---|---|---|
| Stall (abnormal) | longest stop ≥ 10.0 s | anomaly |
| Legitimate stop | 1.0 s ≤ longest stop < 10.0 s, peak deceleration < 4 m/s², moves afterwards | **normal** |
| Normal driving | longest stop < 2 s (background / control rule) | normal |

- **A stop of exactly 10.0 s is abnormal.**
- **Design holds stay away from the boundary:** legitimate 3, 5 and 8 s; stalls 15 s, 30 s or the rest of the recording.
- **Gaps between the rules are excluded, never labelled.** A background vehicle with a stop between 2 s and 10 s, for example one queued behind a stall, is excluded, not labelled.

**Stop modes of the stall sets:**

| Mode | Description |
|---|---|
| MSM | moves, freezes 30 s, moves on |
| START | stands from t = 0 |
| STOP_END | moves, then stops until the end |
| GENTLE | brakes at 2 m/s², stops 15 s, moves on |

## Variation in the later sets
- **Final test.** The same injection parameters as the cross-region set, but every injected vehicle starts on a **new start lane** with a planned route. Collision targets are placed ≤ 130 m ahead.
- **Study 07.** Wider grids per weak type: hard braking 5–8 m/s², speeding ×1.3–1.8, weight 20–40 t, offset 1.8–2.1 m, weaving amplitude and period, collision speed and target motion. It also adds partial (segment) events, whose realisation statistic is measured over the planned event window (`research/07_targeted_data_study/README.md`).

## The nine development cases (`data/processed/development_cases/cases.csv`)
These are the cases used during development, one per type. They are **development data**, not a test.

| Case | Type | Source |
|---|---|---|
| DC-01 | Speeding | SimStudio anomaly scenario, region 003 |
| DC-02 | Hard braking | controlled scenario |
| DC-03 | Stalled vehicle | SimStudio anomaly scenario, region 020 |
| DC-04 | Collision | controlled scenario, striking vehicle |
| DC-05 | Reckless driving | SimStudio anomaly scenario, region 120 |
| DC-06 | Lane weaving | SimStudio anomaly scenario, region 037 |
| DC-07 | Overweight vehicle | SimStudio anomaly scenario, region 140 |
| DC-08 | Lane straddling | controlled scenario |
| DC-09 | Tailgating | controlled scenario |

**Why four cases are controlled scenarios.** In the original SimStudio scenarios of these types the anomaly is not realised:
- **Hard braking:** the vehicle is frozen from the start, so it is a stall.
- **Lane straddling:** the offset is 0 m.
- **Tailgating:** the vehicle never comes closer than 8.9 m.
- **Collision:** there is no contact.

The controlled scenarios (C1–C4 in the records) realise the anomaly in the same maps and are verified by truth. The original scenarios are kept in `data/simstudio_inputs/anomalies/`.
