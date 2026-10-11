"""Locations inside the release directory. Nothing here points outside the release root."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FP_DATA = DATA / "simstudio_inputs"                      # SimStudio reference inputs (read-only)
PROCESSED = DATA / "processed" / "normal_corpus"         # normal driving corpus: channels, labels, splits, amplitude
DEVELOPMENT_CASES = DATA / "processed" / "development_cases"
ANOMALY_SETS = DATA / "processed" / "anomaly_sets"
FINAL_TEST = DATA / "processed" / "final_test"
SCENARIOS = DATA / "scenarios" / "north_tlv"
GENERATED = ROOT / "outputs" / "generated"               # new simulation runs written by the generators
MODELS = ROOT / "models"
ENSEMBLE = MODELS / "deepdas_ensemble_v1"
BASELINES = MODELS / "baselines"
EXPERIMENTS = ROOT / "research" / "01_unsupervised_autoencoder" / "runs"
RESEARCH = ROOT / "research"
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results"

ZSCORE_MODEL = BASELINES / "zscore" / "model_live.pkl"
REGIONS = FP_DATA / "regions"
ANOMALY_SCENARIOS = FP_DATA / "anomalies"


def region_geometry_json(region: str) -> Path:
    """Scenario JSON holding the region's map, lanes and full sensor layout."""
    return REGIONS / region / f"region_{region}_sensors.sim.json"


def scenario_region_geometry_json(region: str) -> Path:
    """Sensor-layout scenario of a North-TLV region in the scenario library (used for the simulated batches)."""
    return SCENARIOS / region / f"region_{region}_sensors.sim.json"


def ensure_dirs() -> None:
    for p in (GENERATED, OUTPUTS / "figures", OUTPUTS / "tables"):
        p.mkdir(parents=True, exist_ok=True)
