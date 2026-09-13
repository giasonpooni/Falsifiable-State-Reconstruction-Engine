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
from set_lcm.experiments.real_water_balance_report import render_report
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


def test_finite_ungauged_variance_can_reject_a_large_storage_disagreement():
    """An algebraically feasible U does not guarantee a plausible finite-prior estimate."""
    clock = np.arange(8, dtype=float) * 86400.0
    obs = [Observation(t=t, arrival_t=t,
                       y=np.array([70000.0 if k == 0 else 75000.0, 0.0, 0.0, 0.0]),
                       R=np.diag([200.0 ** 2, 1.0, 1.0, 1.0]), mask=np.ones(4, dtype=bool),
                       source_ids=("storage", "outflow", "inflow1", "inflow2"))
           for k, t in enumerate(clock)]
    result = run(PublicInputs(clock, np.zeros(clock.size)), obs,
                 wb.constraint_aug(70000.0, 200.0 ** 2),
                 EstimatorSpec("wb_aug", "wb_aug", None, cusum=None),
                 (70000.0,), wb.PRIOR_STORAGE_STD, est_cfg=wb.CFG)
    assert np.all(result.x[:, 2] == 0.0)
    assert result.stat[1] > 100.0
    assert np.all(result.flag[3:])


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
    # This record is not rejected at these storage sigmas and the fixed declared q_U.
    # Finite U uncertainty does not guarantee acceptance for another record or q_U.
    for key in data["sweep"]:
        assert data["sweep"][key]["specs"]["wb_aug"]["consistency_stat"]["fraction_over_threshold"] == 0.0


# ---------------------------------------------------------------------------
# the cumulative is a sum of 1,095 noisy days, and must never be presented alone
# ---------------------------------------------------------------------------

def test_the_cumulative_carries_its_own_standard_error(bridged):
    """Recomputed here from the observations, independently of experiments code.

    The sum of n residuals of sd s has standard error s*sqrt(n) under independence
    -- so it grows with the record length even when the gauges close exactly. The
    AR(1) variance of the sum uses the measured lag-1 exactly for finite n.
    """
    Y = np.array([o.y for o in bridged.observations], dtype=float)
    r = np.diff(Y[:, 0]) - CFS_DAY_TO_ACRE_FT * (Y[:-1, 2] + Y[:-1, 3] - Y[:-1, 1])
    n = int(r.size)
    se = float(r.std(ddof=1)) * np.sqrt(n)
    c = wb.closure_residual(bridged)
    assert c["cumulative_se"] == pytest.approx(se, rel=1e-12)
    assert c["cumulative_in_se"] == pytest.approx(c["cumulative"] / se, rel=1e-12)

    rho = c["lag1_autocorr"]
    k = np.arange(1, n, dtype=float)
    factor = n + 2.0 * float(np.sum((n - k) * rho ** k))
    assert c["cumulative_se_ar1"] == pytest.approx(float(r.std(ddof=1)) * np.sqrt(factor), rel=1e-12)
    assert c["cumulative_se_ar1"] > c["cumulative_se"] > 0.0      # a positive lag-1 widens it
    lo, hi = c["cumulative_ci95_ar1"]
    assert lo == pytest.approx(c["cumulative"] - 1.96 * c["cumulative_se_ar1"], rel=1e-12)
    assert hi == pytest.approx(c["cumulative"] + 1.96 * c["cumulative_se_ar1"], rel=1e-12)
    # The measured record: the three-year total is NOT separated from zero.
    assert lo < 0.0 < hi
    assert abs(c["cumulative_in_se_ar1"]) < 1.96


def test_a_short_or_degenerate_record_reports_no_standard_error_rather_than_nan():
    """The result dict is serialized with allow_nan=False, so None is the only option."""
    for args in ((1.0, 1.0, 1, 0.0), (1.0, 0.0, 10, 0.0), (float("nan"), 1.0, 10, 0.0)):
        assert set(wb._cumulative_uncertainty(*args).values()) == {None}
    partial = wb._cumulative_uncertainty(1.0, 1.0, 10, float("nan"))
    assert partial["cumulative_se"] is not None          # independence needs no lag-1
    assert partial["cumulative_se_ar1"] is None          # the AR(1) model does
    assert partial["cumulative_ci95_ar1"] is None


def test_the_report_never_presents_the_cumulative_without_its_uncertainty(fresh):
    """The guard that matters: the artifact readers actually read.

    A three-year total of +1,526 acre-ft reads like a finding. It is 0.60 standard
    errors from zero. Whatever else the report says, the standard error must appear
    with it, the interval must be printed, and the magnitude must not be the
    emphasised number.
    """
    data, md = fresh
    c = data["closure_free"]
    cum = f"{c['cumulative']:,.0f}"
    assert f"**{cum} acre-ft**" not in md, "the cumulative is emphasised as if it were a finding"
    assert "not evidence of a net imbalance" in md
    for shown in (f"{c['cumulative_se']:,.0f}", f"{c['cumulative_se_ar1']:,.0f}",
                  f"{c['cumulative_in_se']:,.2f}", f"{c['cumulative_in_se_ar1']:,.2f}",
                  f"{c['cumulative_ci95_ar1'][0]:,.0f}", f"{c['cumulative_ci95_ar1'][1]:,.0f}"):
        assert shown in md, shown
    # and the daily statistic, which does reject, is kept distinct from the total
    assert "consistent with the gauges closing" in md
    assert "**daily** disagreement" in md


def test_the_augmented_route_reports_how_far_its_own_U_is_from_zero(fresh):
    """Two routes agreeing near zero is not corroboration of an imbalance."""
    data, md = fresh
    for key, cell in data["sweep"].items():
        for name, entry in cell["specs"].items():
            aug = entry.get("ungauged_cumulative")
            if aug is None:
                continue
            if aug["final_sd"] > 0.0:
                assert aug["final_in_sd"] == pytest.approx(aug["final"] / aug["final_sd"], rel=1e-12)
                assert f"{aug['final_in_sd']:,.2f}" in md, (key, name)
    assert "their agreement is not evidence of an imbalance" in md
    # The run whose ratio is largest is the one whose covariance absorbed the constraint,
    # and the report must say that where it prints it.
    per_sigma = [cell["specs"] for cell in data["sweep"].values()]
    for specs in per_sigma:
        ratios = {n: e["ungauged_cumulative"]["final_in_sd"] for n, e in specs.items()
                  if e.get("ungauged_cumulative")}
        assert max(ratios, key=ratios.get) == "wb_aug+hard+feedback", ratios
    assert "shrinks the very sd that column divides by" in md


def test_every_qualitative_sentence_of_the_report_is_a_computed_condition(fresh):
    """The report's claims about distance from zero are properties of this record.

    They are the sentences most likely to become false quietly if the evidence changed,
    so render_report() refuses to write when one of them fails.
    """
    data, _ = fresh
    assert data["claims"] and all(data["claims"].values()), \
        [k for k, v in data["claims"].items() if not v]
    for name in ("the_cumulative_interval_contains_zero",
                 "the_cumulative_is_under_two_standard_errors_from_zero",
                 "feedback_reports_the_largest_sd_ratio"):
        assert name in data["claims"], name
    broken = dict(data, claims=dict(data["claims"], the_cumulative_interval_contains_zero=False))
    with pytest.raises(RuntimeError, match="no longer true of the numbers"):
        render_report(broken)
