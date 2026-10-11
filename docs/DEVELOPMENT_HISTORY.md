# Development history

The project's decisions in time order. Each row gives the step, what was decided, and where it is documented (`research/<study>/`).

| Date (2026) | Step | Decision / outcome | Study |
|---|---|---|---|
| Sep | Data pipeline built from the SimStudio exports: channels, map matching, quality gate, region split; leakage audit | the model sees only exported tracks | 01, `LEAKAGE_AUDIT.md` |
| Sep | Unsupervised ConvAE, reconstruction error and latent max-z; severity study; audit of the original anomaly scenarios | AUROC 0.60 vs Z-score 0.89; four original scenarios do not realise their anomaly → controlled cases C1–C4 | 01 |
| 27 Sep | Latent max-z detector released as the unsupervised reference | 5/9 at P99 | 01 |
| 27 Sep | Amplitude-aware autoencoder (14th channel `das_amp_log10`) | becomes the encoder initialisation | 01 / 02 |
| 27 Sep | Supervised head with cross-region anomalies (54 runs) | 7/9; frozen as the supervised single model (`CR_P99`) | 02 |
| 7 Oct | Experiments A–D (cleaning, `osc`, `tg`) and the seed study; new-scene check | cleaning + `tg` adopted; strong seed dependence → 5-seed ensembles | 02 |
| 7–8 Oct | Tailgating data plans v1–v4 (each written before it ran) | realised tailgating sets with matched controls | 03 |
| 8 Oct | Training with and without the new tailgating data (5 seeds each) | with the data: tailgating validation 0.39 → 0.92; selected (`C_new`) | 03 |
| 8 Oct | Stalled-case regression diagnosed; EXP1 (balanced sampling) and EXP2 (frozen encoder) | neither qualifies | 04 |
| 8 Oct | Stall diversity vs stall quantity | diversity selected → **DeepDAS Ensemble v1** frozen (threshold 1.9726181803233118) | 04 |
| 9 Oct | Legitimate-stop diagnosis; stopval2; F1 (exposure) and F2 (temporal branch) | neither qualifies | 05 |
| 9 Oct | Checkpoint selection by a task rule (replay of 15 runs) | no qualifying checkpoint; model unchanged | 05 |
| 9 Oct | **Final test** planned, frozen and run once (228 runs) | macro recall 0.568, ROC-AUC 0.908, AUPRC 0.742 | `results/final_test/` |
| 10 Oct | Weak-type diagnosis on the (now exposed) final-test data | information present in input and encoder; the head lacks varied examples | 06 |
| 10 Oct | Frozen model fully backed up and restore-tested | all outputs reproduced | – |
| 10 Oct | W1/W2 targeted-data round (314 runs, 10 models) | both fail C5 (legitimate stops) → not adopted | 07 |
| 10 Oct | **Model development closed** | DeepDAS Ensemble v1 is final; no further training, recalibration or testing | – |
| 10 Oct | This release built from the frozen backup and the study records | verification in `RELEASE_VERIFICATION.md` | – |

## Principles kept throughout
- **One change per experiment.** Each arm differed from its reference in one stated respect.
- **Pre-registration.** The rule was written before training. A failing candidate was never "rescued" by changing the rule afterwards.
- **Labels from realisation, never from injection parameters.** Runs that did not realise were excluded and reported, not re-run.
- **Thresholds only from val_B,** and never copied between models.
- **Frozen artefacts hash-guarded** before and after every round. The originals were never modified, moved or deleted.
