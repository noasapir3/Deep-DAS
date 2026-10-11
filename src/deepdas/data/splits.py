"""
Spatially-blocked region split.

* test   : the regions that host the Final Project's anomaly scenarios (their
           normal scenarios are the held-out negatives);
* buffer : every region that shares at least one road node with a test region.
           Adjacent tiles continue the same streets, so a random split would leak
           near-identical geometry.  Buffer regions are never trained on; they are
           divided into ``val_A`` (early stopping) and ``val_B`` (threshold).
* train  : all remaining regions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

from .. import paths


def region_nodes(region: str) -> Set[tuple]:
    d = json.loads(paths.region_geometry_json(region).read_text())
    return {(round(float(v["x"]), 2), round(float(v["y"]), 2)) for v in d["nodes"].values()}


def adjacency(regions: List[str]) -> Dict[str, Set[str]]:
    adj, _ = adjacency_with_shared(regions)
    return adj


def adjacency_with_shared(regions: List[str]):
    """Region adjacency from node positions shared by two or more regions (plus the number of such positions)."""
    owner: Dict[tuple, Set[str]] = {}
    for r in regions:
        for p in region_nodes(r):
            owner.setdefault(p, set()).add(r)
    adj: Dict[str, Set[str]] = {r: set() for r in regions}
    n_shared = 0
    for rs in owner.values():
        if len(rs) > 1:
            n_shared += 1
        for a in rs:
            adj[a] |= rs - {a}
    return adj, n_shared


def make_splits(test_regions: List[str], seed: int = 0) -> dict:
    regions = sorted(p.name for p in paths.REGIONS.iterdir() if p.is_dir())
    adj, n_shared = adjacency_with_shared(regions)
    test = sorted(set(test_regions))
    buffer = sorted({n for t in test for n in adj[t]} - set(test))
    rng = np.random.default_rng(seed)
    order = list(buffer)
    rng.shuffle(order)
    half = (len(order) + 1) // 2
    val_a, val_b = sorted(order[:half]), sorted(order[half:])
    train = sorted(set(regions) - set(test) - set(buffer))
    return {
        "seed": seed,
        "n_regions": len(regions),
        "test": test,
        "val_A": val_a,
        "val_B": val_b,
        "train": train,
        "buffer": buffer,
        "adjacency_degree": {r: len(adj[r]) for r in regions},
        "n_shared_node_positions": int(n_shared),
    }


def region_split_of(splits: dict) -> Dict[str, str]:
    out = {}
    for name in ("train", "val_A", "val_B", "test"):
        for r in splits[name]:
            out[r] = name
    return out
