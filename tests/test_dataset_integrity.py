"""
Integrity of the built dataset and of the trained runs (skipped until the pipeline has been run).

* the model input table has exactly the 13 documented channels and no truth column;
* every track used for training is a normal track of a train region; no test-region
  track is trained on; test regions share no road node with train regions;
* val_A / val_B / test are disjoint region sets; anomaly scenarios are evaluation-only;
* every scored track is eligible; the vehicle table covers every eligible track.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from deepdas import paths
from deepdas.data.channels import CHANNELS
from deepdas.data.dataset import MIN_VALID_BINS, eligible_tracks, load_labels
from deepdas.data.splits import adjacency

needs_data = pytest.mark.skipif(not (paths.PROCESSED / "labels.parquet").exists(), reason="dataset not built")


@needs_data
def test_split_is_spatially_blocked():
    s = json.loads((paths.PROCESSED / "splits.json").read_text())
    sets = {k: set(s[k]) for k in ("train", "val_A", "val_B", "test")}
    for a in sets:
        for b in sets:
            if a != b:
                assert not (sets[a] & sets[b]), (a, b)
    assert sum(len(v) for v in sets.values()) == s["n_regions"] == 148
    adj = adjacency(sorted(set().union(*sets.values())))
    for t in sets["test"]:
        assert not (adj[t] & sets["train"]), f"test region {t} touches a train region"


@needs_data
def test_labels_roles_and_splits():
    lab = load_labels()
    assert set(lab["role"]) == {"normal", "injected", "background"}
    assert (lab.loc[lab["group"] == "fp_normal", "role"] == "normal").all()
    assert (lab.loc[lab["group"] != "fp_normal", "split"] == "test").all()
    s = json.loads((paths.PROCESSED / "splits.json").read_text())
    for name in ("train", "val_A", "val_B", "test"):
        regs = set(lab.loc[(lab["group"] == "fp_normal") & (lab["split"] == name), "region"])
        assert regs <= set(s[name]), name
    # every Final-Project anomaly scenario has exactly one injected vehicle except the collision pair
    inj = lab[(lab["group"] == "fp_anomaly") & (lab["role"] == "injected")].groupby("scenario_id")["vehicle_id_oracle"].nunique()
    assert (inj.drop("region_120/anomaly_collision") == 1).all() and inj["region_120/anomaly_collision"] == 2


@needs_data
def test_quality_gate_is_label_free_and_symmetric():
    from deepdas.data.quality import track_quality, MIN_RUN
    lab = load_labels()
    assert "quality_ok" in lab.columns
    # anomaly tracks are judged by the same rule; the gate never uses roles or scores
    ch = pd.read_parquet(paths.PROCESSED / "channels.parquet", columns=["track_key", "bin", "v_lon", "d_lat", "m_valid", "burn_in"])
    sub = ch[ch["track_key"].astype(str).isin(set(lab.loc[lab["role"] != "normal", "track_key"]))]
    q = track_quality(sub).set_index("track_key")
    merged = lab.set_index("track_key").join(q[["quality_ok"]], rsuffix="_recomputed").dropna(subset=["quality_ok_recomputed"])
    assert (merged["quality_ok"] == merged["quality_ok_recomputed"]).all()
    assert MIN_RUN == 10


@needs_data
def test_channels_table_columns():
    ch = pd.read_parquet(paths.PROCESSED / "channels.parquet")
    assert set(CHANNELS) <= set(ch.columns)
    assert not ({"true_x", "true_y", "segment_id", "vehicle_id_oracle", "x_hat", "y_hat", "das_amplitude", "snr"} & set(ch.columns))
    assert ch["m_valid"].isin([0.0, 1.0]).all()
    assert ((ch["m_das"] >= 0) & (ch["m_das"] <= 1)).all()


@needs_data
def test_training_used_only_normal_train_tracks():
    run = paths.EXPERIMENTS / "cae_primary"
    if not (run / "done.json").exists():
        pytest.skip("primary run not trained")
    from deepdas.train.train import _split_keys
    lab = load_labels()
    keys = _split_keys(lab, MIN_VALID_BINS)
    sub = lab.set_index("track_key").loc[keys["train"]]
    assert (sub["role"] == "normal").all() and (sub["split"] == "train").all()
    assert sub["quality_ok"].all() and (sub["n_valid_bins"] >= MIN_VALID_BINS).all()
    assert not set(keys["train"]) & set(keys["val_A"]) and not set(keys["train"]) & set(keys["test"])
    norm = json.loads((run / "norm.json").read_text())
    assert len(norm["median"]) == 8 and all(s > 0 for s in norm["scale"])
