# Design documents

* `DESIGN_phase1_forensics.md` — Phase 1: repository forensics of the Final Project (executed
  pipeline versus documentation, dead code, truth entry points, the reverse-engineered Z-score
  training procedure, dataset facts, the mis-injected anomaly scenarios) and the original
  research design (sections A–L), including the answers to the four design questions.
* `DESIGN_phase2_plan.md` — the reduced Phase-2 plan agreed before implementation: the shipped
  baseline stays untouched, the exported Kalman trajectories are reused as they are, the
  severity study is the smallest targeted set (9 types × 3 levels × 3 seeds), the DAS-noise
  question is answered observationally, one autoencoder only.

Two decisions were taken during implementation and are documented in `PIPELINE.md` §3 and
in the report: (1) the decoder output is bounded to the range of normal training data and only
the six non-differentiated channels are reconstruction targets — a plain MSE autoencoder with
an 8-d code reproduced level anomalies (a vehicle at twice the speed limit) almost perfectly;
(2) a label-free track-quality gate removes tracker failures (physically impossible fused
kinematics for ≥ 1 s) from training and reports them separately — they dominated the tail of
normal scores and would have set the operating threshold far above every injected anomaly.
