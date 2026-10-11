"""Severity variants change only the injected vehicle; the director acts through existing override fields;
the seeded runner is deterministic and writes the Final-Project export format."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from deepdas import paths
from deepdas.sim.anomaly_director import AnomalyDirector
from deepdas.sim.runner import run_scenario
from deepdas.sim.scenario_builder import BASES, LEVELS, PARAMS, SEEDS, all_variants, build_variant, write_variant

ANOMALY_FILES_PRESENT = all((paths.ANOMALY_SCENARIOS / b[0]).exists() for b in BASES.values())


@pytest.mark.skipif(not ANOMALY_FILES_PRESENT, reason="anomaly scenario files not vendored")
def test_variants_change_only_the_injected_vehicle():
    variants = all_variants()
    assert len(variants) == len(BASES) * len(LEVELS) * len(SEEDS) == 81
    assert len({v["run_id"] for v in variants}) == 81
    for v in variants:
        base = json.loads((paths.ANOMALY_SCENARIOS / v["base_scenario"]).read_text())
        sc = v["scenario"]
        for key in ("nodes", "segments", "lanes", "gps", "cameras", "das", "boundary_exit_nodes"):
            assert sc.get(key) == base.get(key), (v["run_id"], key)
        for vid in base["vehicles"]:
            if vid != v["injected_vehicle"]:
                assert sc["vehicles"][vid] == base["vehicles"][vid], (v["run_id"], vid)
        inj = sc["vehicles"][v["injected_vehicle"]]
        assert inj["lane_id"] == base["vehicles"][v["injected_vehicle"]]["lane_id"]
        assert inj["s"] == base["vehicles"][v["injected_vehicle"]]["s"]
        needs_hook = v["anomaly_type"] in ("anomaly_hard_braking", "anomaly_stalled_vehicle")
        assert bool(v["schedule"]) == needs_hook
        if v["anomaly_type"] == "anomaly_speeding":
            assert inj["ignore_speed_limit"] and abs(inj["target_speed_mps"] - v["nominal"]["ratio"] * v["injection"]["speed_limit_start_mps"]) < 1e-9
        if v["anomaly_type"] == "anomaly_lane_straddling":
            assert inj["lateral_mode"] == "straddle" and inj["lateral_fixed_offset_m"] == v["nominal"]["offset_m"]


def test_director_brake_and_stall_use_existing_fields():
    veh = SimpleNamespace(v=12.0, a_cmd_override_mps2=0.0, a_cmd_override_until_t=0.0, frozen=False)
    world = SimpleNamespace(vehicles={"veh001": veh})
    d = AnomalyDirector([{"t": 6.0, "vehicle": "veh001", "action": "brake", "decel_mps2": 6.0},
                         {"t": 20.0, "vehicle": "veh001", "action": "stall", "duration_s": 4.0}])
    d(None, world, 5.98)
    assert veh.a_cmd_override_mps2 == 0.0
    d(None, world, 6.0)
    assert veh.a_cmd_override_mps2 == -6.0 and abs(veh.a_cmd_override_until_t - (6.0 + 2.0 + 0.2)) < 1e-9
    d(None, world, 20.0)
    assert veh.frozen is True
    d(None, world, 23.99)
    assert veh.frozen is True
    d(None, world, 24.0)
    assert veh.frozen is False
    assert [e["action"] for e in d.log] == ["brake", "stall", "release"]


@pytest.mark.skipif(not (paths.ANOMALY_SCENARIOS / "region_003_anomaly_speeding.sim.json").exists(), reason="scenario not vendored")
def test_runner_is_deterministic_and_exports_fp_format(tmp_path):
    var = build_variant("speeding", "mild", 2)
    run_dir = write_variant(var, tmp_path)
    sim_json = run_dir / f"{var['run_id']}.sim.json"
    r1 = run_scenario(sim_json, tmp_path / "a", seed=2, record_truth=True, export=True)
    r2 = run_scenario(sim_json, tmp_path / "b", seed=2, record_truth=True, export=False)
    assert r1["events"] == r2["events"] and r1["steps"] == r2["steps"]
    t1 = [(round(r["t"], 3), r["vehicle_id"], round(r["x"], 6), round(r["v"], 6)) for r in r1["truth"]]
    t2 = [(round(r["t"], 3), r["vehicle_id"], round(r["x"], 6), round(r["v"], 6)) for r in r2["truth"]]
    assert t1 == t2
    export_dir = tmp_path / "a" / r1["export_dir"]
    assert {p.name for p in export_dir.iterdir()} == {"tracking_audit.xlsx", "kalman_measurement_audit.xlsx", "coverage_summary.xlsx"}
    r3 = run_scenario(sim_json, tmp_path / "c", seed=3, record_truth=True, export=False)
    assert [(round(r["x"], 6)) for r in r3["truth"][:50]] != [(round(r["x"], 6)) for r in r1["truth"][:50]]
