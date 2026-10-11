# Development history

The project's decisions in time order. Each row gives the step, what was decided, and where it is documented (`research/<study>/`). The same studies grouped by research question (Q1–Q4) are in [research/README.md](../research/README.md).

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
| 8 Oct | Stalled-case regression diagnosed; type-balanced sampling (EXP1) and a frozen encoder (EXP2) | neither qualifies | 04 |
| 8–9 Oct | Stall diversity vs stall quantity | diversity selected (8 Oct) → **DeepDAS Ensemble v1** frozen (freeze record dated 9 Oct; threshold 1.9726181803233118) | 04 |
| 9 Oct | Legitimate-stop diagnosis; legitimate-stop validation set (stopval2); controlled exposure (F1) and a temporal branch (F2) | neither qualifies | 05 |
| 9 Oct | Checkpoint selection by a task rule (replay of 15 runs) | no qualifying checkpoint; model unchanged | 05 |
| 9 Oct | **Final test** planned, frozen and run once (228 runs) | macro recall 0.568, ROC-AUC 0.908, AUPRC 0.742 | `results/final_test/` |
| 10 Oct | Weak-type diagnosis on the (now exposed) final-test data | information present in input and encoder; the head lacks varied examples | 06 |
| 10 Oct | Frozen model fully backed up and restore-tested | all outputs reproduced | – |
| 10 Oct | Targeted training data for the weak types: data expansion (W1) and the same with controlled sampling (W2); 314 runs, 10 models | both exceed the pre-registered limit on legitimate-stop alarms (C5) → not adopted | 07 |
| 10 Oct | **Model development closed** | DeepDAS Ensemble v1 is final; no further training, recalibration or testing | – |
| 10 Oct | This release built from the frozen backup and the study records | verification in `RELEASE_VERIFICATION.md` | – |
| 10 Oct | Post-hoc check of the tailgating summary: the frozen model with φ<sub>u</sub> neutralized at inference; review copy `DeepDAS_Release_v1.1_review` with the `--phi-mode` option (default `active`) | dependence measured; nothing trained or re-calibrated | 08 |
| 11 Oct | Klausner experiment: the final model on the one real recording, inference only | the stalled bus is detected in both options | `models/deepdas_ensemble_v1/experiment_klausner/` |
| 11 Oct | Public repository and data package (`data-v1.0`); report v2.2 | same model, code and results | – |
| 11 Oct | Documentation round v3: report reorganised by four research questions, descriptive names, study summaries | presentation only; no model, threshold, data or result changed | `docs/CHANGES_DOCS_v3.md`, `report/CHANGES_v3.md` |

## Principles kept throughout
- **One change per experiment.** Each arm differed from its reference in one stated respect.
- **Pre-registration.** The rule was written before training. A failing candidate was never "rescued" by changing the rule afterwards.
- **Labels from realisation, never from injection parameters.** Runs that did not realise were excluded and reported, not re-run.
- **Thresholds only from val_B,** and never copied between models.
- **Frozen artefacts hash-guarded** before and after every round. The originals were never modified, moved or deleted.
