"""
Import the *existing* Final-Project simulation exports into DL/data/processed.

Input  : export folders written by the Final Project's ``scripts/run_all_simulations.py``
         (``<scenario>_export_<ts>_tracking_audit/tracking_audit.xlsx``), read in place.
Output : one parquet shard per region / anomaly scenario with the 50 Hz Kalman
         trajectory of every exported track, plus a per-track metadata table.

What is kept from the ``Track T*`` sheets (see ``TRACK_KEEP``):
    time, update type, the Kalman state (x, y, vx, vy, ax, ay) and *whether* a
    GPS / camera / DAS measurement entered the filter at that time step.

What is dropped and never reaches the processed data (see ``TRACK_TRUTH_DROPPED``):
    true position / distance, position error, the oracle road segment, all
    "distance along road" columns (they are computed on the oracle segment),
    innovations and NIS.  The oracle vehicle id is kept ONLY in the metadata
    table, as the label key for anomaly scenarios.

From the ``Kalman Audit`` sheet only per-track *sensor statistics* are taken
(number of accepted measurements per sensor, mean DAS/camera σ, mean DAS SNR).
They are evaluation strata (e.g. "performance versus DAS σ"), never model
inputs; ``docs/LEAKAGE_AUDIT.md`` lists them as truth-derived.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from python_calamine import CalamineWorkbook

from ..fp_anomaly_model.feature_extractor import _AUDIT_SHEET_COL_MAP, _TRACK_SHEET_COL_MAP

# --------------------------------------------------------------------------- #
# Column policy
# --------------------------------------------------------------------------- #
TRACK_KEEP = ["t", "update_kind", "x_hat", "y_hat", "vx_hat", "vy_hat", "ax_hat", "ay_hat"]
TRACK_PRESENCE = {"gps_x": "has_gps", "cam_x": "has_cam", "das_x": "has_das"}
TRACK_META_ONLY = ["vehicle_id_oracle"]
TRACK_TRUTH_DROPPED = [
    "true_x", "true_y", "true_s", "pos_err_m", "segment_id",
    "hat_s", "gps_s", "cam_s", "das_s", "das_fiber_position_m",
    "gps_y", "cam_y", "das_y",            # raw measurements are not inputs either
    "sigma_pos_m", "sigma_v_mps",         # functions of the idealized σ model
    "innovation_x", "innovation_y", "innovation_S_x", "innovation_S_y", "NIS_x",
]
UPDATE_KIND_CODE = {"prediction_only": 0, "position": 1, "position+velocity": 2}

EXPORT_DIR_RE = re.compile(r"^(?P<stem>.+)_export_(?P<ts>\d{8}_\d{6})_tracking_audit$")


@dataclass
class ExportRef:
    """One export folder to import."""
    group: str            # "fp_normal" | "fp_anomaly" | "severity"
    scenario_id: str      # Final-Project style id, e.g. "001/normal_medium_traffic"
    region: str           # "001"
    scenario_type: str    # "normal_medium_traffic" | "anomaly_speeding" | ...
    export_dir: Path
    scenario_json: Optional[Path]
    export_ts: str = ""
    extra: Dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _latest_export(dirs: List[Path]) -> Tuple[Path, str, int]:
    """Pick the most recent export folder when a scenario was run more than once."""
    stamped = []
    for d in dirs:
        m = EXPORT_DIR_RE.match(d.name)
        if m and (d / "tracking_audit.xlsx").exists():
            stamped.append((m.group("ts"), d))
    if not stamped:
        raise FileNotFoundError(f"no complete export in {dirs}")
    stamped.sort()
    return stamped[-1][1], stamped[-1][0], len(stamped)


def discover_fp_normal(tiles_root: Path, regions: Optional[Iterable[str]] = None) -> List[ExportRef]:
    """``<tiles_root>/NNN/simulations/<type>/<stem>_export_<ts>_tracking_audit``."""
    refs: List[ExportRef] = []
    region_dirs = sorted(d for d in Path(tiles_root).iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 3)
    if regions is not None:
        wanted = set(regions)
        region_dirs = [d for d in region_dirs if d.name in wanted]
    for rd in region_dirs:
        sim_root = rd / "simulations"
        if not sim_root.exists():
            continue
        for td in sorted(p for p in sim_root.iterdir() if p.is_dir()):
            exports = [p for p in td.iterdir() if p.is_dir() and EXPORT_DIR_RE.match(p.name)]
            if not exports:
                continue
            export_dir, ts, n = _latest_export(exports)
            jsons = sorted(td.glob("*.sim.json"))
            refs.append(ExportRef(
                group="fp_normal",
                scenario_id=f"{rd.name}/{td.name}",
                region=rd.name,
                scenario_type=td.name,
                export_dir=export_dir,
                scenario_json=jsons[0] if jsons else None,
                export_ts=ts,
                extra={"n_export_folders": n},
            ))
    return refs


def _fp_find_export_folder(region_dir: Path, true_stem: str) -> Optional[Path]:
    """Same rule as the Final Project's ``scripts/extract_anomaly_per_sim.py::_find_export_folder``:
    newest folder first (reverse name sort) over both naming patterns, must contain the workbook."""
    candidates = sorted(
        list(region_dir.glob(f"{true_stem}_export_*_tracking_audit")) +
        list(region_dir.glob(f"{true_stem}.sim_export_*_tracking_audit")),
        reverse=True,
    )
    for c in candidates:
        if c.is_dir() and (c / "tracking_audit.xlsx").exists():
            return c
    return None


def discover_fp_anomaly(anomaly_root: Path) -> List[ExportRef]:
    """``<anomaly_root>/region_NNN/<stem>[.sim]_export_<ts>_tracking_audit`` — the nine published scenarios,
    each resolved to the export folder the Final Project's extraction script would have used."""
    refs: List[ExportRef] = []
    for rd in sorted(Path(anomaly_root).glob("region_*")):
        if not rd.is_dir():
            continue
        region = rd.name.split("_")[1]
        for sim_json in sorted(rd.glob("*.sim.json")):
            true_stem = sim_json.name.replace(".sim.json", "")
            if true_stem.endswith("_clean") or true_stem.endswith("_sensors"):
                continue
            if true_stem == "region_140_anomaly_speeding":
                continue  # never part of the published anomaly set (no override flags, no features row)
            export_dir = _fp_find_export_folder(rd, true_stem)
            if export_dir is None:
                continue
            atype = true_stem.replace(f"region_{region}_", "", 1)      # anomaly_speeding
            refs.append(ExportRef(
                group="fp_anomaly",
                scenario_id=f"{rd.name}/{atype}",
                region=region,
                scenario_type=atype,
                export_dir=export_dir,
                scenario_json=sim_json,
                export_ts=EXPORT_DIR_RE.match(export_dir.name).group("ts"),
            ))
    return refs


def discover_severity(generated_root: Path) -> List[ExportRef]:
    """``<generated_root>/<run_id>/<stem>_export_<ts>_tracking_audit`` + ``run.json`` written by 05_simulate_severity."""
    refs: List[ExportRef] = []
    for run_dir in sorted(p for p in Path(generated_root).iterdir() if p.is_dir()):
        run_json = run_dir / "run.json"
        if not run_json.exists():
            continue
        info = json.loads(run_json.read_text())
        exports = [p for p in run_dir.iterdir() if p.is_dir() and EXPORT_DIR_RE.match(p.name)]
        if not exports:
            continue
        export_dir, ts, _ = _latest_export(exports)
        refs.append(ExportRef(
            group="severity",
            scenario_id=f"severity/{run_dir.name}",
            region=info["region"],
            scenario_type=info["anomaly_type"],
            export_dir=export_dir,
            scenario_json=run_dir / info["scenario_json"],
            export_ts=ts,
            extra=info,
        ))
    return refs


# --------------------------------------------------------------------------- #
# Reading one workbook
# --------------------------------------------------------------------------- #
def _sheet_to_df(wb: CalamineWorkbook, name: str, col_map: Dict[str, str]) -> pd.DataFrame:
    rows = wb.get_sheet_by_name(name).to_python(skip_empty_area=False)
    if len(rows) < 2:
        return pd.DataFrame()
    header = []
    for i, h in enumerate(rows[0]):
        h = str(h) if h not in (None, "") else f"col_{i}"
        if h in col_map:
            header.append(col_map[h])
        elif h.startswith("NIS_x"):
            header.append("NIS_x")
        else:
            header.append(h)
    data = [list(r[:len(header)]) for r in rows[1:]]
    return pd.DataFrame(data, columns=header)


def geometry_hash(scenario_json: Path) -> str:
    """SHA-256 of the map part of a scenario file (nodes, segments, lanes, boundary nodes)."""
    d = json.loads(Path(scenario_json).read_text())
    sub = {k: d.get(k) for k in ("nodes", "segments", "lanes", "boundary_exit_nodes")}
    return hashlib.sha256(json.dumps(sub, sort_keys=True).encode()).hexdigest()


def read_export(ref: ExportRef) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (rows50, meta) for one export folder.

    rows50 columns: track_key, t, update_kind (int8), x_hat.., has_gps, has_cam, has_das
    meta    : one row per track
    """
    wb_path = ref.export_dir / "tracking_audit.xlsx"
    wb = CalamineWorkbook.from_path(str(wb_path))
    names = wb.sheet_names
    track_sheets = [n for n in names if n.startswith("Track T")]

    # -- per-track sensor statistics from the Kalman Audit sheet -------------
    audit_stats: Dict[str, Dict[str, float]] = {}
    if "Kalman Audit" in names:
        aud = _sheet_to_df(wb, "Kalman Audit", _AUDIT_SHEET_COL_MAP)
        if len(aud):
            for c in ("sigma_m", "snr", "accepted"):
                aud[c] = pd.to_numeric(aud[c], errors="coerce")
            acc = aud[aud["accepted"] == 1]
            for tid, g in acc.groupby("global_track_id"):
                das = g[g["sensor_type"] == "DAS"]
                cam = g[g["sensor_type"] == "Camera"]
                gps = g[g["sensor_type"] == "GPS"]
                audit_stats[str(tid)] = {
                    "n_das_meas": int(len(das)), "n_cam_meas": int(len(cam)), "n_gps_meas": int(len(gps)),
                    "das_sigma_mean": float(das["sigma_m"].mean()) if len(das) else np.nan,
                    "das_sigma_max": float(das["sigma_m"].max()) if len(das) else np.nan,
                    "das_snr_mean": float(das["snr"].mean()) if len(das) else np.nan,
                    "cam_sigma_mean": float(cam["sigma_m"].mean()) if len(cam) else np.nan,
                }

    parts: List[pd.DataFrame] = []
    meta_rows: List[Dict] = []
    for sheet in track_sheets:
        tid = sheet.replace("Track ", "")
        df = _sheet_to_df(wb, sheet, _TRACK_SHEET_COL_MAP)
        if df.empty:
            continue
        for c in TRACK_KEEP[2:] + list(TRACK_PRESENCE):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["t"] = pd.to_numeric(df["t"], errors="coerce")
        df = df.dropna(subset=["t", "x_hat", "y_hat"]).sort_values("t").reset_index(drop=True)
        if df.empty:
            continue
        oracle = df["vehicle_id_oracle"].astype(str)
        oracle_major = oracle.mode().iloc[0]
        track_key = f"{ref.scenario_id}|{tid}"

        out = pd.DataFrame({
            "track_key": track_key,
            "t": df["t"].astype("float32"),
            "update_kind": df["update_kind"].map(UPDATE_KIND_CODE).fillna(-1).astype("int8"),
            "x_hat": df["x_hat"].astype("float32"),
            "y_hat": df["y_hat"].astype("float32"),
            "vx_hat": df["vx_hat"].astype("float32"),
            "vy_hat": df["vy_hat"].astype("float32"),
            "ax_hat": df["ax_hat"].astype("float32"),
            "ay_hat": df["ay_hat"].astype("float32"),
        })
        for src, dst in TRACK_PRESENCE.items():
            out[dst] = df[src].notna().to_numpy()
        parts.append(out)

        dt = np.diff(df["t"].to_numpy(dtype=float))
        meta = {
            "track_key": track_key,
            "group": ref.group,
            "scenario_id": ref.scenario_id,
            "region": ref.region,
            "scenario_type": ref.scenario_type,
            "export_dir": ref.export_dir.name,
            "export_ts": ref.export_ts,
            "track_id": tid,
            "vehicle_id_oracle": oracle_major,
            "oracle_purity": float((oracle == oracle_major).mean()),
            "n_rows": int(len(df)),
            "t_start": float(df["t"].iloc[0]),
            "t_end": float(df["t"].iloc[-1]),
            "duration_s": float(df["t"].iloc[-1] - df["t"].iloc[0]),
            "dt_median": float(np.median(dt)) if len(dt) else np.nan,
            "dt_max": float(dt.max()) if len(dt) else np.nan,
            "n_pred_only": int((out["update_kind"] == 0).sum()),
            "n_pos": int((out["update_kind"] == 1).sum()),
            "n_posvel": int((out["update_kind"] == 2).sum()),
            "frac_das_rows": float(out["has_das"].mean()),
            "frac_cam_rows": float(out["has_cam"].mean()),
            "frac_gps_rows": float(out["has_gps"].mean()),
        }
        meta.update(audit_stats.get(tid, {}))
        meta.update({k: v for k, v in ref.extra.items() if isinstance(v, (str, int, float, bool))})
        meta_rows.append(meta)

    rows50 = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    meta_df = pd.DataFrame(meta_rows)
    if ref.scenario_json is not None and ref.scenario_json.exists():
        meta_df["geometry_sha256"] = geometry_hash(ref.scenario_json)
        meta_df["scenario_json"] = ref.scenario_json.name
    return rows50, meta_df


# --------------------------------------------------------------------------- #
# Batch driver (resumable, one shard per region / scenario)
# --------------------------------------------------------------------------- #
def shard_name(ref: ExportRef) -> str:
    if ref.group == "fp_normal":
        return f"fp_normal_region_{ref.region}"
    if ref.group == "fp_anomaly":
        return "fp_anomaly_" + ref.scenario_id.replace("/", "_")
    return "severity_" + ref.scenario_id.split("/", 1)[1]


def import_shard(refs: List[ExportRef], out_dir: Path, name: str) -> Dict:
    """Import a list of exports into ``<out_dir>/<name>.parquet`` (+ ``.meta.parquet``)."""
    t0 = time.time()
    rows, metas = [], []
    for ref in refs:
        r, m = read_export(ref)
        if len(r):
            rows.append(r)
            metas.append(m)
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        all_rows = pd.concat(rows, ignore_index=True)
        all_rows["track_key"] = all_rows["track_key"].astype("category")
        all_rows.to_parquet(out_dir / f"{name}.parquet", index=False)
        pd.concat(metas, ignore_index=True).to_parquet(out_dir / f"{name}.meta.parquet", index=False)
    else:
        pd.DataFrame().to_parquet(out_dir / f"{name}.parquet")
        pd.DataFrame().to_parquet(out_dir / f"{name}.meta.parquet")
    return {"shard": name, "n_exports": len(refs), "n_tracks": int(sum(len(m) for m in metas)), "seconds": time.time() - t0}


def group_by_shard(refs: List[ExportRef]) -> Dict[str, List[ExportRef]]:
    shards: Dict[str, List[ExportRef]] = {}
    for r in refs:
        shards.setdefault(shard_name(r), []).append(r)
    return shards
