"""deepdas — Deep DAS course project package.

Sub-packages
------------
fp                 Verbatim copies of the Final-Project SimStudio modules (never edited).
fp_anomaly_model   Verbatim copies of the Final-Project feature extractor and Z-score scorer.
sim                Headless runner and severity scenario builder (project code, wraps ``fp``).
data               Import of existing exports, map matching, channels, windows, splits.
model / train      1-D convolutional autoencoder and its trainer.
evaluation         Scoring, thresholds, metrics, figures.
baseline           Thin wrapper that runs the shipped Z-score model unchanged.
"""
__version__ = "0.1.0"
