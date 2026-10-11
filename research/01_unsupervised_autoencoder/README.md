# Study 01: Unsupervised convolutional autoencoder

## At a glance

| | |
|---|---|
| **Research question** | Does a model trained on normal driving only flag anomalous vehicles by reconstruction error, as the project proposal planned? |
| **What was compared** | A 1-D convolutional autoencoder (13 channels, 6.4 s windows, 8-d bottleneck, P95 window error per track, P99 threshold) against the shipped SimStudio Z-score, on the same vehicles. |
| **Data** | The normal corpus with a spatial split, and the nine anomaly types injected at three severities (91 injected vs 632 normal vehicles in the comparison). <b>Development only:</b> the “test” regions here are a split of the normal corpus, not the final test. |
| **Result** | AUROC 0.598 [0.52, 0.67] vs 0.887 for the Z-score; 1 of 91 injected vehicles detected at a 1 % false-alarm rate. AUROC stays near 0.6 for every seed, window length and bottleneck width. |
| **Conclusion and limits** | Reconstruction error measures within-window variation, which normal driving has as much as most injected events. The simulated anomalies lie inside the normal envelope of a 6.4 s window. The encoder still learned a useful representation and initialises every later model. |
| **Reproduce** | Archived study: the numbers come from the original study report in <code>records/</code> and are not regenerated. The 14-channel encoder used later is retrained by <code>scripts/pretrain_autoencoder.py</code>. |

<p align="center"><img src="../../figures/archived_stage1/png/S2_ae_severity_archived.png" width="60%"></p>

Median reconstruction-error score per type and severity (log scale) against the operating threshold (dashed); grey band: median of normal vehicles. Only severe speeding and severe reckless driving come close to the threshold.

<p align="center"><img src="../../figures/archived_stage1/png/stage1_autoencoder_vs_zscore_roc_pr_ARCHIVED.png" width="95%"></p>

Archived original figure: ROC and precision–recall of the autoencoder against the Z-score.

More figures: [figure gallery](../../figures/) · All studies: [research index](../)

---

## Full record

**Status:** completed and superseded as a detector. Its encoder is the initialisation of every supervised model that followed.
**Data role:** development only. The "test" regions named below are the spatial test split of the normal corpus. They are **not** the independent final test of `results/final_test/`.

## Hypothesis
A model trained on **normal driving only** can flag anomalous vehicles by how badly it reconstructs their track windows (reconstruction error), or by how far their window codes lie from the normal codes (latent distance). It would do this without any anomaly labels and without simulator ground truth at inference.

## Method
- **Input.** 13 channels per 0.1 s bin, derived from the exported tracks only (no ground truth):
  - 8 value channels: v_lon, a_lon, d_lat, v_lat, r_lim, g_lead, th_lead, dv_lead;
  - 5 sensor-presence masks: m_das, m_cam, m_gps, m_lead, m_valid.
- **Windows.** 64 bins (6.4 s), stride 8.
- **Model.** A 1-D convolutional autoencoder (`ConvAE`).
  - Encoder: Conv 13→32 k7, 32→64 k5 s2, 64→64 k5 s2, 64→128 k3 s2, Linear 1024→d, with bottleneck d = 8.
  - The decoder is bounded to the normal range and reconstructs six of the value channels. The two noise-dominated derivative channels a_lon and v_lat are inputs only.
  - Trained on the train-split normal corpus with masked MSE.
- **Scorers compared:**
  - mean reconstruction error per vehicle;
  - **latent max-z**: mean and std of the 8-d window codes per vehicle, compared with train-normal statistics; score = max |z|.
- **Operating point.** Thresholds are the P99 and P95 of the sufficient-evidence val_B normals (1,242 vehicles). Fewer than 3 windows means no decision.
- **Evaluation:**
  - the nine original SimStudio anomaly scenarios;
  - corrected controlled scenarios C1–C4 (realised hard braking, lane straddling, tailgating, and a contact collision);
  - a severity study: 9 types × 3 levels × 3 seeds = 81 runs;
  - the normal vehicles of the val_A and test splits.
- **Baseline.** The shipped SimStudio Z-score model, unchanged.
- **Sensitivity.** Seeds, window length (3.2–12.8 s) and bottleneck width (4–32).

## Results

**Reconstruction error.**
- 91 injected vehicles vs 632 held-out normal vehicles:
  - autoencoder: AUROC **0.60** [0.52, 0.67], detecting **1 of 91** at 1 % false-alarm rate;
  - Z-score: AUROC **0.89** [0.84, 0.93], recall 44 %.
- The result is robust to every setting:
  - seeds give ±0.01;
  - window length leaves AUROC at 0.60;
  - bottleneck 4–32 gives 0.61–0.58.

**Latent max-z (the release of this study), at P99:**
- **5 of 9** original causing vehicles detected, plus the collision target. The Z-score detects 6 of 9 on the same recordings.
- **17 of 81** severity vehicles (Z-score 29 of 81).
- False alarms: 9 of 513 test normals (1.8 %).
- Corrected controlled scenarios: none of C1 (braking), C2 (straddling) or C3 (tailgating) is detected at P99. C4, the contact collision, is detected only through the stationary period after contact.

**The audit of the original scenarios.** The study found that four of them did not realise their anomaly:
- the hard-braking vehicle never moves;
- the collision has no contact;
- the tailgating and straddling events are not realised.

This led to the controlled development cases used from then on (`data/processed/development_cases/`).

## Conclusions
1. **The anomalies are inside the normal envelope.** Most of this simulator's anomalies (stops, moderate braking, 1–2 m lateral drift, close following) lie inside the span of its normal driving, so reconstruction error cannot separate them.
2. **Capacity does not help.** Widening the code reconstructs the anomalies as well as normal windows.
3. **The Z-score wins with track-level statistics** (stop counts and durations), which a 6.4 s window cannot see.
4. **Supervision is needed.** Unsupervised detection is not sufficient here, so the next step was a supervised head on top of this encoder (study 02). The encoder itself is a good representation (see study 06: probe AUC up to 0.96 on the initial autoencoder) and is kept.
5. **The 14-channel encoder used by DeepDAS Ensemble v1** is this autoencoder retrained with the DAS log-amplitude as a 14th input and a 7th masked target (`scripts/pretrain_autoencoder.py`, `models/deepdas_ensemble_v1/autoencoder_*`).

## Contents
| Path | Content |
|---|---|
| `runs/cae_primary/` | Primary 13-channel autoencoder: config, normalisation, training history, best checkpoint (seed 0, epoch 79) |
| `records/` | The original reports of this study, verbatim: the full study report, design, leakage audit, and the release report of the latent max-z detector |

Code: `src/deepdas/model/conv_ae.py` (ConvAE), `src/deepdas/train/train.py` (autoencoder training), `src/deepdas/evaluation/` (the study's scoring, metrics and figures, unchanged in logic).
