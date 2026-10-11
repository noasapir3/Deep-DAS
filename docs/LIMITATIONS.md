# Limitations and methodological changes

## A. Limitations of DeepDAS Ensemble v1

### 1. False alarms on legitimate short stops (main limitation)
A vehicle that brakes gently, stands 1–10 s and drives on is normal by the project's definitions, but the model often flags it.

| Data | Flagged |
|---|---|
| Final test | **20 of 28**: 3 s 3/8, 5 s 8/11, 8 s 9/9 |
| Validation | 40 of 54 |

- **Cause** (study 05): mean/std pooling lets the share of stop windows decide the score. Legitimate stops had very low training exposure, and the tracker shortens stops in the input, so their duration is a weak signal.
- **What did not fix it:** controlled exposure, a temporal branch, checkpoint choice and targeted data (studies 05 and 07). None removed it without breaking another condition.
- **Consequence:** in a setting with many short legitimate stops (traffic lights, stop signs), alarms near stops need a second look. The simulator has no traffic lights, so the frequency of such stops in real traffic is not represented.

### 2. False alarms on tailgating matched controls
These are vehicles following a slow, pinned leader at the default headway.
- Final test: 7 of 11. Validation: 6 of 7.
- Within each matched pair the tailgater always scores higher (11/11), so the ranking is right but the threshold does not separate them.
- With these controls removed, the false-alarm rate on normal driving in the final test is 13/985 = 1.3 % instead of 2.0 %.

### 3. Weak anomaly types

| Type | Final-test recall |
|---|---|
| Hard braking | 1/11 |
| Speeding | 4/12 |
| Overweight | 4/12 |
| Lane straddling | 5/12 |

- **Cause** (study 06): the information is in the input and the encoder, but each type had only 4 homogeneous training examples.
- Targeted data doubled their validation recall (study 07), but at the cost of more legitimate-stop alarms, so it was not adopted.

### 4. Decision coverage
In the final test, 1,008 of 1,708 normal vehicles had a decided track. The others had:

| Reason | Vehicles |
|---|---|
| No track | 134 |
| Only tracks that were not usable | 320 |
| Fewer than 3 windows | 246 |

- 101 of 105 realised anomalies were decided.
- Undecided units are reported, never counted as normal. **Coverage is a property of the tracker and of the decision rule, not of the network.**

### 5. Speeding definition overlaps normal driving
The simulator's own normal traffic reaches true speed ratios of 1.10–1.12 (the cap is 1.12 × the limit). The project's speeding rule is a p90 ratio ≥ 1.10, so mild speeding (×1.1–1.25) overlaps normal driving by construction of the data.

### 6. Development case not detected
Overweight (DC-07) scores 1.865 against the threshold 1.973.

### 7. Simulation only
- Every result is on SimStudio North-TLV scenes with simulated DAS, cameras and GPS.
- There is no real-world evaluation; the single real bus track of the SimStudio project was used only qualitatively in study 01.
- The exported tracks inherit truth in the tracker: oracle-aided association, truth-seeded initialisation, and noise-free camera/GPS velocity. Tracks are therefore cleaner than a real system would produce.

### 8. Truth-dependent unit in the validation and development tables
These tables use the **simulated vehicle** (tracks grouped by the oracle id) as the unit. This unit does not exist in deployment. The final test uses the deployment unit, one track, and counts a vehicle as detected if any of its tracks alarms.

## B. Limitations of the comparison
- **Z-score baseline.** It is used as shipped. 13 of its 156 features use the truth position error, which is not available in deployment. A truth-free sensitivity check on the final test gives the same macro recall (0.231). The comparison is therefore not unfair to the Z-score.
- **Single model and Z-score.** Both were frozen before the final test. Neither received any tuning after it.
- **Exposed final test.** The final test was run once and then used for the weak-type diagnosis. It cannot serve as an independent test for any later model.

## C. Methodological changes during the project (disclosed)
| Change | When | Why | Effect on reported results |
|---|---|---|---|
| Four development cases replaced by controlled scenarios (C1–C4) | study 01 | the original SimStudio scenarios did not realise these anomalies | the original scenarios are kept; results on them are reported in study 01 |
| Lead-artefact cleaning and tailgating fraction added | study 02 | a duplicate-track artefact; a direct headway measure | part of the model from then on |
| Single seed → 5-seed ensemble | after the seed study (02) | strong seed dependence | every later comparison is ensemble-level |
| Validation expanded with tailgating (study 03), stallval (04), stopval2 (05) and targeted validation (07) | per study | the earlier validation set could not see these behaviours | pre-registered; the earlier sets are kept unchanged |
| Tailgating label `gap_min < 5 m & headway p10 < 0.5 s` → `tailgating_v1` | study 03 | the earlier rule did not require a sustained episode | the cross-region set keeps its original rule |
| Partial-event realisation measured over the planned event window | study 07 | segment events | applies only to study 07 data |
| Plan revisions and errata (tailgating v1→v4; launch errata in studies 04 and 07; import erratum in 07) | as recorded | see `research/*/records/` | each was written before the affected step |
| Final test run once; used afterwards for diagnosis | 2026-10-09 / 10 | — | final-test numbers apply only to the three frozen detectors |

## D. Limitations of the release itself
- **Byte-exact reproduction is platform-bound.**
  - **Build platform** (Linux x86-64, CPU, the pinned library versions of `requirements.txt`, one thread for training): byte equality of every file and of the retrained weights is verified.
  - **Linux aarch64:** every decision and every result table is identical, and z and ensemble scores agree within 1e-5. Raw logits differ by up to 3.4e-5.
  - **Retraining** was verified only on x86-64.
  - See `CROSS_PLATFORM.md`. The E2 criterion was revised once for this reason (v1.0 → v1.1, `EVALUATION_PROTOCOL.md` §6).
- **A native macOS run was not tested.** The tested platforms are Linux x86-64 and Linux aarch64; the second ran in the virtual machine of an Apple-silicon Mac.
- **The raw simulator output is not in the release** (about 0.9 GB for the final test alone). Reproduction from processed data is fully self-contained. Reproduction from raw data (re-import, Z-score recomputation, regeneration checks) needs the separate raw-data package (`RAW_DATA_PACKAGE.md`). **Without that package these steps are not available.**
- **Re-simulating every set from scratch** with the shipped generators is possible but slow (hours). The release verification regenerated a sample of runs of every set, not all of them.
- **Study 07 training was not re-run** in the release verification. Its evaluation was reproduced byte for byte. The training scripts are archived as run.
- **Records keep the internal identifiers and paths of the working tree** (`docs/IDENTIFIERS.md`).
