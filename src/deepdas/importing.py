"""
Import of one simulated run into the model's processed format, with the unchanged data pipeline:
read_export (50 Hz track rows) -> build_channels (map matching on the region geometry) -> track_quality, and the DAS amplitude of every
track on the same 0.1 s bins (from the Kalman audit sheet of the export). A track is usable if it has >= 32 valid bins and passes the
label-free quality gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np  # noqa: F401  (used by amp_on_bins)
import pandas as pd
from python_calamine import CalamineWorkbook

from . import paths
from .amplitude import amp_on_bins
from .data.channels import build_channels
from .data.import_exports import ExportRef, read_export
from .data.map_matching import LaneIndex
from .data.quality import track_quality


def import_run(run_dir: Path, run_id: str, region: str, group: str, geometry_json: Path | None = None):
    """-> (channels, amplitude, track_meta) of one run folder written by a deepdas.batches generator."""
    run_dir = Path(run_dir); info = json.loads((run_dir / "run.json").read_text()); exp = run_dir / info["result"]["export_dir"]
    geometry_json = geometry_json or paths.scenario_region_geometry_json(region)
    ref = ExportRef(group=group, scenario_id=f"{group}/{run_id}", region=region, scenario_type=run_id, export_dir=exp, scenario_json=run_dir / f"{run_id}.sim.json")
    rows50, meta = read_export(ref); rows50["track_key"] = rows50.track_key.astype(str); meta["track_key"] = meta.track_key.astype(str)
    ch = build_channels(rows50, meta, {region: LaneIndex.from_scenario_json(geometry_json)}); ch["track_key"] = ch.track_key.astype(str)
    vb = ch[ch.m_valid > 0].groupby("track_key").size(); q = track_quality(ch).set_index("track_key")
    meta["n_valid_bins"] = meta.track_key.map(vb).fillna(0).astype(int); meta["quality_ok"] = meta.track_key.map(q["quality_ok"]).fillna(True).astype(bool)
    meta["usable"] = (meta.n_valid_bins >= 32) & meta.quality_ok
    rws = CalamineWorkbook.from_path(str(exp / "tracking_audit.xlsx")).get_sheet_by_name("Kalman Audit").to_python(skip_empty_area=False)
    col = {str(h): i for i, h in enumerate(rws[0])}
    aud = pd.DataFrame([(x[col["Time (s)"]], str(x[col["Track ID"]]), x[col["DAS Amplitude A (raw units)"]]) for x in rws[1:] if x[col["Sensor Type"]] == "DAS"], columns=["t", "tid", "amp"])
    aud["t"] = pd.to_numeric(aud.t); aud["amp"] = pd.to_numeric(aud.amp, errors="coerce")
    A = pd.concat([amp_on_bins(rows50, aud, tk, tid)[0] for tk, tid in zip(meta.track_key, meta.track_id)], ignore_index=True) if len(meta) else pd.DataFrame(columns=["track_key", "bin", "amp_log10"])
    return ch, A, meta


def write_run(out_dir: Path, run_id: str, ch, amp, meta):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ch.to_parquet(out_dir / f"{run_id}_channels.parquet", index=False); amp.to_parquet(out_dir / f"{run_id}_amp.parquet", index=False)
    meta.to_parquet(out_dir / f"{run_id}_track_meta.parquet", index=False)
