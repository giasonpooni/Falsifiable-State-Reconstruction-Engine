"""The reservoir water balance: the filters, the constraint, and the report.

Reads the committed DAF observations (data/daf/usgs_ridgway_wy2023_2025.observations.json,
1,096 days of four USGS daily-mean series), so nothing here needs a network or a DAF
checkout. The report is regenerated and compared with the committed results value for
value, as tests/test_real_noaa.py does for the tide-gauge report.
"""
import json

import numpy as np
import pytest

from set_lcm.experiments import real_water_balance as wb
from set_lcm.experiments.compare import reproduction_failure
from set_lcm.experiments.provenance import REPO_ROOT
from set_lcm.lcm import detectability
from set_lcm.schema import Observation, Status
from set_lcm.testbed.estimators_balance import (
    CFS_DAY_TO_ACRE_FT, BalanceConfig, WaterBalanceAugmented, WaterBalanceClosed, WaterBalanceOpen,
)
from set_lcm.testbed.inputs import PublicInputs
from set_lcm.testbed.runner import EstimatorSpec, run

RESULTS = REPO_ROOT / "results"
COMMITTED = json.loads((RESULTS / "real_water_balance.json").read_text(encoding="utf-8"))
COMMITTED_MD = (RESULTS / "real_water_balance.md").read_text(encoding="utf-8")


def _body(md: str) -> list[str]:
    return [ln for ln in md.splitlines() if not ln.startswith("Generated with ")]


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    out = tmp_path_factory.mktemp("water_balance")
    wb.main(out, quiet=True)
    return (json.loads((out / "real_water_balance.json").read_text(encoding="utf-8")),
            (out / "real_water_balance.md").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bridged():
    return wb.load(wb.STORAGE_SIGMA_BASE)


# ---------------------------------------------------------------------------
# the unit conversion, which every volume in the report depends on
# ---------------------------------------------------------------------------

def test_the_cfs_day_conversion_is_the_definition_and_not_a_rounded_constant():
    assert CFS_DAY_TO_ACRE_FT == 86400.0 / 43560.0
    assert CFS_DAY_TO_ACRE_FT == pytest.approx(1.9834710743801653, rel=0, abs=1e-15)
    # one day of 1 ft^3/s is 86400 cubic feet, which is that many acre-ft
    assert 86400.0 / 43560.0 * 43560.0 == pytest.approx(86400.0)


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------

def test_the_committed_observations_bridge_to_a_complete_daily_grid(bridged):
    p = bridged.provenance
    assert p["n_grid"] == 1096                      # 2022-10-01 .. 2025-09-30, water years 2023-2025
    assert p["n_used"] == 4416 and p["n_in_conflict"] == 0
    assert p["n_deduplicated"] == 32                # DAF's trailing safety window, 4 series x 8 days
    assert set(p["n_missing"].values()) == {0}      # no gap in any series
    assert p["time"]["time_semantics"] == "calendar_day"
    assert p["time"]["day_anchor"] == "midpoint"
    assert p["time"]["day_zone_citation"] == wb.DAY_ZONE_CITATION
    assert p["cadence_s"] == 86400.0
    assert len(bridged.observations[0].y) == 4


def test_every_R_is_consumer_declared_and_carries_its_citation(bridged):
    """USGS states no per-value uncertainty, so the bridge required a citation for each one
    and the report prints it. A source-stated R here would mean the extractor invented one."""
    r_source = bridged.provenance["R_source"]
    assert set(r_source) == set(wb.SOURCE_IDS)
    for source_id, entry in r_source.items():
        assert entry["kind"] == "consumer-declared"
        assert entry["citation"].strip()
    assert r_source[wb.SOURCE_IDS[0]]["sigma"] == wb.STORAGE_SIGMA_BASE
    for source_id in wb.SOURCE_IDS[1:]:
        assert r_source[source_id]["relative"] == wb.FLOW_SIGMA_RELATIVE
        assert r_source[source_id]["sigma_floor"] == wb.FLOW_SIGMA_FLOOR
    assert "not acquired" in r_source[wb.SOURCE_IDS[1]]["citation"]
    assert "NOT a source statement" in r_source[wb.SOURCE_IDS[0]]["citation"]


def test_a_declared_sigma_without_a_citation_cannot_reach_the_bridge():
    from set_lcm.bridge.daf import DeclaredSigma

    with pytest.raises(ValueError, match="citation"):
        DeclaredSigma(sigma=200.0, citation="")


# ---------------------------------------------------------------------------
# the model-free closure residual
# ---------------------------------------------------------------------------

def test_the_closure_residual_is_arithmetic_on_the_readings(bridged):
    """Recomputed here from the observations, independently of experiments code."""
    Y = np.array([o.y for o in bridged.observations], dtype=float)
    expected = np.diff(Y[:, 0]) - CFS_DAY_TO_ACRE_FT * (Y[:-1, 2] + Y[:-1, 3] - Y[:-1, 1])
    c = wb.closure_residual(bridged)
    assert c["n_days"] == 1095
    assert c["mean"] == pytest.approx(float(expected.mean()), rel=0, abs=1e-9)
    assert c["sd"] == pytest.approx(float(expected.std(ddof=1)), rel=0, abs=1e-9)
    assert c["cumulative"] == pytest.approx(float(expected.sum()), rel=0, abs=1e-6)
    assert c["mean_as_cfs"] == pytest.approx(c["mean"] / CFS_DAY_TO_ACRE_FT, rel=0, abs=1e-12)


def test_the_report_states_the_closure_the_data_gives(fresh):
    data, _ = fresh
    c = data["closure_free"]
    # signs and magnitudes the report's prose commits to
    assert 0.0 < c["mean_as_cfs"] < 5.0                     # a small positive mean imbalance
    assert c["cumulative"] > 0.0
    assert abs(c["cumulative_as_fraction_of_gauged_inflow"]) < 0.01   # under 1% over three years
    assert c["sd"] > 10.0 * abs(c["mean"])                  # the daily scatter dwarfs the mean


def test_the_alignment_comparison_computes_all_three_and_prefers_none_by_fiat(fresh):
    data, md = fresh
    al = data["alignment"]
    assert set(al) >= {"same_day", "centred", "next_day", "smallest_sd", "note"}
    sds = {k: al[k]["sd"] for k in ("same_day", "centred", "next_day")}
    assert al["smallest_sd"] == min(sds, key=sds.get)
    assert "not by itself evidence" in al["note"]
    assert "It is not proof" in md


# ---------------------------------------------------------------------------
# what the constraint can and cannot see
# ---------------------------------------------------------------------------

def test_the_null_direction_of_the_balance_is_exactly_invisible():
    cs = wb.constraint_open(60000.0, 200.0 ** 2)
    P = np.diag([200.0 ** 2, 100.0 ** 2])
    f_null = np.array([1.0, 1.0]) / np.sqrt(2.0)            # storage and cumulative inflow together
    assert detectability(f_null, P, cs) == pytest.approx(0.0, abs=1e-30)
    f_row = np.array([1.0, -1.0]) / np.sqrt(2.0)
    assert detectability(f_row, P, cs) > 0.0


def test_the_constraint_declares_its_own_uncertainty_because_b_is_a_reading():
    cs = wb.constraint_open(60000.0, 200.0 ** 2)
    assert not cs.exact
    assert cs.b_var.tolist() == [200.0 ** 2]
    assert cs.rank == 1
    exact = wb.constraint_open(60000.0, 0.0)
    assert exact.b_var.tolist() == [0.0]                     # a declared zero is still a declaration


# ---------------------------------------------------------------------------
# the filters
# ---------------------------------------------------------------------------

def _toy(n=40, n_sensors=4):
    """A short synthetic record with a KNOWN ungauged term, used only to check that the
    filters implement the arithmetic they claim. Nothing here is evidence about a reservoir."""
    t = np.arange(n, dtype=float) * 86400.0
    inflow, outflow, ungauged = 100.0, 60.0, 10.0            # ft^3/s, ft^3/s, ft^3/s ungauged
    net_gauged = CFS_DAY_TO_ACRE_FT * (inflow - outflow)
    obs = []
    S = 50000.0
    for k in range(n):
        y = np.array([S, outflow, inflow, 0.0])
        R = np.diag([100.0 ** 2, 1.0, 1.0, 1.0])
        obs.append(Observation(t=t[k], arrival_t=t[k], y=y, R=R,
                               mask=np.ones(n_sensors, dtype=bool), source_ids=("s",) * n_sensors))
        S += net_gauged + CFS_DAY_TO_ACRE_FT * ungauged
    return PublicInputs(t=t, u_commanded=np.zeros(n)), obs, ungauged


def test_wb_closed_propagates_storage_by_the_measured_net_flow():
    inputs, obs, _ = _toy()
    cfg = BalanceConfig(q_storage=1.0, q_flow=1.0)
    est = WaterBalanceClosed((50000.0,), 1000.0, 86400.0, inputs.u_commanded, cfg, clock=inputs.t)
    # with no measurement at all, one predict step must add c * dt * (qin1 + qin2 - qout)
    x = np.array([50000.0, 60.0, 100.0, 0.0])
    x1, _ = est.predict(x, np.eye(4))
    assert x1[0] == pytest.approx(50000.0 + CFS_DAY_TO_ACRE_FT * 40.0)
    assert x1[1:].tolist() == x[1:].tolist()                 # rates are random walks


def test_wb_open_leaves_storage_and_cumulative_inflow_unlinked():
    inputs, obs, _ = _toy()
    cfg = BalanceConfig(q_storage=1.0, q_flow=1.0)
    est = WaterBalanceOpen((50000.0,), 1000.0, 86400.0, inputs.u_commanded, cfg, clock=inputs.t)
    x = np.array([50000.0, 0.0, 60.0, 100.0, 0.0])
    x1, _ = est.predict(x, np.eye(5))
    assert x1[0] == 50000.0                                  # S is a random walk: prediction moves it not at all
    assert x1[1] == pytest.approx(CFS_DAY_TO_ACRE_FT * 40.0)  # G integrates the rates


def test_the_declared_constraint_detects_a_balance_the_gauges_do_not_close():
    """On the toy record the ungauged term is known, so this checks the machinery reacts to
    it -- not that any real reservoir behaves this way."""
    inputs, obs, ungauged = _toy(n=60)
    cfg = BalanceConfig(q_storage=50.0, q_flow=1.0)
    cs = wb.constraint_open(50000.0, 100.0 ** 2)
    r = run(inputs, obs, cs, EstimatorSpec("wb_open", "wb_open", None, cusum=None),
            (50000.0,), 1000.0, est_cfg=cfg)
    res = r.res_pre[:, 0]
    assert np.isfinite(res).all()
    assert res[-1] > 0.0                                      # storage rose more than the gauges explain
    assert res[-1] == pytest.approx(CFS_DAY_TO_ACRE_FT * ungauged * 59, rel=0.15)
    assert np.nanmax(r.stat) > r.threshold                    # and the statistic says so


def test_wb_aug_is_identified_only_through_the_constraint(bridged):
    """The finding the report states: U is observed by nothing else, so without a projection
    it never leaves its prior, while +hard moves the reported U and +feedback moves the
    filter's own."""
    Y = np.array([o.y for o in bridged.observations], dtype=float)
    s0 = float(Y[0, 0])
    var = wb.STORAGE_SIGMA_BASE ** 2
    u = {}
    for name in ("wb_aug", "wb_aug+hard", "wb_aug+hard+feedback"):
        spec = next(s for s in wb.SPECS if s.name == name)
        u[name] = wb.run_one(spec, bridged, s0, var).x[:, 2]
    assert np.all(u["wb_aug"] == 0.0)
    assert abs(u["wb_aug+hard"][-1]) > 100.0
    assert abs(u["wb_aug+hard+feedback"][-1]) > 100.0
    assert not np.array_equal(u["wb_aug+hard"], u["wb_aug+hard+feedback"])


def test_feeding_the_constraint_back_lowers_the_statistic_that_tests_it(bridged):
    """The repository's standing warning, measured here: a constraint pushed into the filter
    is absorbed as if it were fresh evidence, and the test stops disagreeing with it."""
    Y = np.array([o.y for o in bridged.observations], dtype=float)
    s0, var = float(Y[0, 0]), wb.STORAGE_SIGMA_BASE ** 2
    plain = wb.run_one(next(s for s in wb.SPECS if s.name == "wb_aug+hard"), bridged, s0, var)
    fed = wb.run_one(next(s for s in wb.SPECS if s.name == "wb_aug+hard+feedback"), bridged, s0, var)
    assert np.nanmean(fed.stat) < np.nanmean(plain.stat)


def test_wb_closed_declares_no_constraint_so_nothing_can_reject_it(bridged):
    Y = np.array([o.y for o in bridged.observations], dtype=float)
    r = wb.run_one(next(s for s in wb.SPECS if s.name == "wb_closed"), bridged, float(Y[0, 0]),
                   wb.STORAGE_SIGMA_BASE ** 2)
    assert r.threshold is None
    assert not np.isfinite(r.stat).any()
    assert set(r.status) == {Status.SKIPPED}


def test_a_balance_filter_refuses_a_record_of_the_wrong_width():
    inputs, obs, _ = _toy()
    cfg = BalanceConfig(q_storage=1.0, q_flow=1.0)
    est = WaterBalanceOpen((50000.0,), 1000.0, 86400.0, inputs.u_commanded, cfg, clock=inputs.t)
    bad = Observation(t=0.0, arrival_t=0.0, y=np.zeros(2), R=np.eye(2),
                      mask=np.ones(2, dtype=bool), source_ids=("a", "b"))
    with pytest.raises(ValueError, match="4 gauges"):
        est.ingest(bad, 0)


def test_wb_aug_refuses_a_config_whose_ungauged_term_cannot_move():
    inputs, _, _ = _toy()
    with pytest.raises(ValueError, match="q_ungauged"):
        WaterBalanceAugmented((50000.0,), 1000.0, 86400.0, inputs.u_commanded,
                              BalanceConfig(q_storage=1.0, q_flow=1.0), clock=inputs.t)


# ---------------------------------------------------------------------------
# the report reproduces
# ---------------------------------------------------------------------------

def test_the_report_reproduces(fresh):
    data, md = fresh
    failure = reproduction_failure("real_water_balance.json", data, COMMITTED)
    assert failure is None, failure
    assert _body(md) == _body(COMMITTED_MD), \
        "results/real_water_balance.md does not match a fresh run of the source tree"


def test_the_report_says_what_it_cannot_validate(fresh):
    _, md = fresh
    assert "## What these numbers do not show" in md
    for phrase in ("Truth-free", "consumer-declared", "does not say which",
                   "equal-and-opposite gauge bias", "ESTIMATED"):
        assert phrase in md, phrase


def test_the_sweep_reports_every_conclusion_at_every_declared_storage_sigma(fresh):
    data, _ = fresh
    assert set(data["sweep"]) == {f"{s:g}" for s in wb.STORAGE_SIGMA_SWEEP}
    over = []
    for key in (f"{s:g}" for s in sorted(wb.STORAGE_SIGMA_SWEEP)):
        specs = data["sweep"][key]["specs"]
        assert {s.name for s in wb.SPECS} == set(specs)
        over.append(specs["wb_open"]["consistency_stat"]["fraction_over_threshold"])
    # a wider declared storage sigma can only make the balance easier to accept
    assert over == sorted(over, reverse=True), over
    # and the augmented filter is never rejected, at any declared sigma
    for key in data["sweep"]:
        assert data["sweep"][key]["specs"]["wb_aug"]["consistency_stat"]["fraction_over_threshold"] == 0.0
