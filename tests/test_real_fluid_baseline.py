"""The real-data adapter must preserve measurement support, units and evidence."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from set_lcm.bridge.daf import BridgedSeries
from set_lcm.experiments import real_fluid_baseline as baseline
from set_lcm.experiments import real_water_balance as wb
from set_lcm.experiments.compare import reproduction_failure
from set_lcm.measurement import balance_residuals
from set_lcm.schema import Observation
from set_lcm.testbed.inputs import PublicInputs


def _toy():
    n = 4
    times = np.arange(n, dtype=float) * 86400.0
    net = np.array([1.0, 3.0, 2.0, 5.0])
    storage_start = 10000.0 + wb.CFS_DAY_TO_ACRE_FT * np.r_[0.0, np.cumsum(net[:-1])]
    storage_mean = storage_start + 0.5 * wb.CFS_DAY_TO_ACRE_FT * net
    covariance = np.diag([25.0, 0.04, 0.04, 0.04])
    covariance[0, 2] = covariance[2, 0] = 0.1
    obs = [Observation(t=t, arrival_t=t + 43200.0,
                       y=np.array([storage_mean[k], 1.0, net[k] + 1.0, 0.0]),
                       R=covariance.copy(), mask=np.ones(4, dtype=bool), source_ids=wb.SOURCE_IDS,
                       evidence_ids=tuple(f"{k}:{j}" for j in range(4))) for k, t in enumerate(times)]
    component = tuple(tuple((f"{k}:{j}",) for j in range(4)) for k in range(n))
    provenance = {"selectors": [s.describe() for s in wb.SELECTORS], "cadence_s": 86400.0,
                  "time": {"time_semantics": "calendar_day", "time_zone": "UTC",
                           "day_anchor": "midpoint", "day_anchor_offset_s": 43200.0,
                           "day_zone_citation": "Synthetic UTC days declared by this fixture."}}
    return BridgedSeries(PublicInputs(times, np.zeros(n)), obs, provenance, wb.SOURCE_IDS, component)


def test_adapter_preserves_exact_mean_support_and_all_contributing_evidence():
    bs = _toy()
    record = baseline.record_from_bridge(bs, 0, 4)
    adjacent = balance_residuals(record)
    cumulative = balance_residuals(record, cumulative=True)
    np.testing.assert_allclose(adjacent.residual, 0.0, atol=1e-7)
    np.testing.assert_allclose(cumulative.residual, 0.0, atol=1e-7)
    np.testing.assert_array_equal(record.edges, np.arange(5) * 86400.0 - 43200.0)
    assert record.covariance[0, 5] == pytest.approx(
        0.1 * baseline.ACRE_FT_TO_M3 * baseline.CFS_TO_M3_S)
    assert "0:0" in cumulative.evidence_ids[-1]
    assert "3:0" in cumulative.evidence_ids[-1]
    assert "1:0" not in cumulative.evidence_ids[-1]  # intermediate storage cancels
    assert all(f"{k}:{j}" in cumulative.evidence_ids[-1] for k in range(4) for j in (1, 2, 3))


def test_adapter_preserves_multiple_ids_for_one_equivalent_measurement():
    bs = _toy()
    components = list(bs.component_evidence)
    components[0] = (("original", "same-value-revision"),) + components[0][1:]
    bs = replace(bs, component_evidence=tuple(components))
    result = balance_residuals(baseline.record_from_bridge(bs, 0, 4), cumulative=True)
    assert {"original", "same-value-revision"}.issubset(result.evidence_ids[-1])


@pytest.mark.parametrize("change", ["missing", "order", "time", "bounds", "units", "anchor",
                                   "support", "zone", "citation", "declaration"])
def test_adapter_refuses_unrepresentable_records(change):
    bs = _toy()
    if change == "missing":
        bs.observations[1].mask[0] = False
    elif change == "order":
        bs = replace(bs, source_ids=tuple(reversed(bs.source_ids)))
    elif change == "time":
        obs = list(bs.observations)
        obs[1] = replace(obs[1], t=obs[1].t + 1.0)
        bs = replace(bs, observations=obs)
    elif change == "units":
        bs.provenance["selectors"][0]["match"]["unit"] = "m3"
    elif change == "anchor":
        bs.provenance["time"]["day_anchor"] = "start"
    elif change == "support":
        bs.provenance["time"]["time_semantics"] = "instant"
    elif change == "zone":
        bs.provenance["time"]["time_zone"] = "America/Denver"
    elif change == "citation":
        bs.provenance["time"]["day_zone_citation"] = ""
    elif change == "declaration":
        bs = replace(bs, provenance={})
    with pytest.raises(ValueError):
        baseline.record_from_bridge(bs, -1 if change == "bounds" else 0, 4)


@pytest.fixture(scope="module")
def report():
    return baseline.build_report()


@pytest.mark.parametrize("field,value", [
    ("y", np.ones(4, dtype=complex) + 1j), ("y", np.array(1.0)),
    ("R", np.eye(4, dtype=complex) + 1j), ("R", np.ones(4)),
    ("mask", np.ones(4, dtype=float)), ("mask", np.ones(1, dtype=bool)),
])
def test_adapter_refuses_implicit_broadcasting_and_complex_truncation(field, value):
    bs = _toy()
    observations = list(bs.observations)
    observations[0] = replace(observations[0], **{field: value})
    with pytest.raises(ValueError):
        baseline.record_from_bridge(replace(bs, observations=observations), 0, 4)


def test_real_replay_counts_all_inclusions_and_exclusions_and_preserves_joint_evidence(report):
    assert report["scope"].startswith("Real-record conditional consistency")
    assert report["declared"]["within_interval_model_status"].startswith("consumer assumption")
    assert len(report["sweeps"]) == len(wb.STORAGE_SIGMA_SWEEP)
    for sweep in report["sweeps"]:
        assert sweep["bridge_provenance"]["R_source"][wb.SOURCE_IDS[0]]["sigma"] == sweep["storage_sigma_acre_ft"]
        assert sum(w["n_days"] for w in sweep["windows"]) + sum(
            w["stop"] - w["start"] for w in sweep["skipped_windows"]) == sweep["n_days"]
        assert sweep["tested_adjacent_pairs"] + sweep["untested_adjacent_pairs"] == sweep["n_days"] - 1
        for window in sweep["windows"]:
            assert window["status"] in {"consistent", "unexplained"}
            assert window["evidence_ids"]
            assert window["joint_statistic"] == pytest.approx(window["adjacent_joint_statistic"], rel=1e-8, abs=1e-8)
            assert window["shared_initial_reading_variance_m6"] > 0.0
    json.dumps(report, allow_nan=False)


def test_real_baseline_report_reproduces(report):
    root = Path(__file__).resolve().parents[1] / "results"
    committed = json.loads((root / "real_fluid_baseline.json").read_text(encoding="utf-8"))
    failure = reproduction_failure("real_fluid_baseline.json", report, committed)
    assert failure is None, failure
    assert baseline.render(committed) == (root / "real_fluid_baseline.md").read_text(encoding="utf-8")
