"""The second reservoir: what it measured, and what it cost to add.

Ridgway's site moved into a declaration on the claim that a second reservoir would cost a
declaration rather than a module. These tests pin the answer -- a declaration plus two
one-time costs -- and the finding that came with it, which is not Ridgway's finding.
"""
import json

import numpy as np
import pytest

from set_lcm.experiments import real_taylor_park as tp
from set_lcm.experiments import real_water_balance as wb
from set_lcm.experiments.balance_site import Site, site_from
from set_lcm.experiments.compare import reproduction_failure
from set_lcm.experiments.provenance import REPO_ROOT
from set_lcm.testbed.estimators_balance import (
    BalanceConfig, WaterBalanceAugmented, WaterBalanceClosed, WaterBalanceOpen, n_sensors,
)

RESULTS = REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# the filters stopped assuming Ridgway's shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_in", [1, 2, 3, 5])
def test_the_balance_filters_take_their_width_from_the_declaration(n_in):
    cfg = BalanceConfig(q_storage=500.0, q_flow=50.0, q_ungauged=200.0, n_inflows=n_in)
    clock = np.arange(4) * 86400.0
    for cls, extra_states in ((WaterBalanceOpen, 2), (WaterBalanceAugmented, 3), (WaterBalanceClosed, 1)):
        f = cls([1000.0], 20000.0, 86400.0, np.zeros((4, 1)), cfg, clock=clock)
        assert f.n_sensors == n_sensors(n_in) == 2 + n_in
        assert f.N == extra_states + 1 + n_in
        assert f.aug_names == ("qout",) + tuple(f"qin{i + 1}" for i in range(n_in))


def test_a_record_of_the_wrong_width_is_refused_by_name():
    """The failure that used to be a silent shape assumption is now a message naming both."""
    cfg = BalanceConfig(q_storage=500.0, q_flow=50.0, n_inflows=3)
    clock = np.arange(3) * 86400.0
    f = WaterBalanceOpen([1000.0], 20000.0, 86400.0, np.zeros((3, 1)), cfg, clock=clock)

    class Obs:
        y = np.zeros(4)
        mask = np.ones(4, dtype=bool)
        R = np.eye(4)
    with pytest.raises(ValueError, match="built for 5 gauges.*carries 4"):
        f.ingest(Obs(), 0)


def test_a_balance_with_no_gauged_inflow_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        BalanceConfig(q_storage=1.0, q_flow=1.0, n_inflows=0)


# ---------------------------------------------------------------------------
# the site view
# ---------------------------------------------------------------------------

def test_both_sites_resolve_from_their_declarations():
    ridgway, taylor = wb.SITE, tp.SITE
    assert (ridgway.n_inflows, taylor.n_inflows) == (2, 3)
    assert len(taylor.columns) == len(taylor.selectors) == 5
    assert taylor.cfg.n_inflows == 3
    assert taylor.volume_unit == ridgway.volume_unit == "acre-ft"


def test_a_declared_inflow_count_that_contradicts_the_sensors_is_refused():
    """Two ways of saying the same number is two ways for them to disagree."""
    text = (REPO_ROOT / "declarations" / "taylor_park.toml").read_text(encoding="utf-8")
    from set_lcm.declaration import loads
    bad = Site(loads(text.replace("value = 3.0", "value = 4.0", 1)))
    with pytest.raises(ValueError, match="declares n_inflows = 4 but carries 5 sensors"):
        _ = bad.n_inflows


def test_the_shared_study_runs_on_either_site_unchanged():
    """compute() is the Ridgway study. It is site-generic, and this is what says so."""
    cs_r = wb.constraint_open(1.0, 1.0, site=wb.SITE)
    cs_t = wb.constraint_open(1.0, 1.0, site=tp.SITE)
    assert cs_r.version != cs_t.version
    np.testing.assert_array_equal(cs_r.A, cs_t.A)   # same physics, different declaration

    # and the study really runs end to end on the site it is handed, not just builds its rows
    for site, expected in ((wb.SITE, 2), (tp.SITE, 3)):
        bridged = wb.load(site.storage_sigma_base, site=site)
        assert len(bridged.observations[0].y) == expected + 2
        closure = wb.closure_residual(bridged, site=site)
        assert closure["units"] == f"{site.volume_unit} per day"
        assert closure["n_days"] > 1000


# ---------------------------------------------------------------------------
# what the second site measured
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def report():
    return tp.compute()


def test_every_claim_the_report_makes_holds(report):
    failed = [name for name, held in report["claims"].items() if not held]
    assert not failed, failed


def test_render_refuses_to_print_a_sentence_that_stopped_being_true(report):
    broken = dict(report, claims=dict(report["claims"], the_balance_does_not_close_here=False))
    with pytest.raises(RuntimeError, match="no longer true of the numbers"):
        tp.render(broken)


def test_ridgways_own_claims_do_not_hold_on_this_record(report):
    """The guard fired on a site transfer, not on a wording slip. That is the finding."""
    ridgway_claims = wb.claims(report)
    assert not ridgway_claims["the_cumulative_interval_contains_zero"]
    assert not ridgway_claims["the_cumulative_is_under_two_standard_errors_from_zero"]


def test_the_imbalance_is_not_explained_by_the_absent_readings_alone(report):
    split = report["reporting_split"]
    full = split["every_gauge_reporting"]
    assert full["n_days"] > 500
    assert full["residual_over_gauged_inflow"] > 0.05
    assert split["a_gauge_absent"]["mean"] > full["mean"] > 0.0


def test_the_correspondence_with_ungauged_drainage_is_reported_but_not_concluded(report):
    split, cmp = report["reporting_split"], report["comparison_with_ridgway"]
    close = abs(split["every_gauge_reporting"]["residual_over_gauged_inflow"]
                - split["ungauged_drainage_fraction"])
    apart = abs(cmp["ridgway"]["residual_over_gauged_inflow"]
                - cmp["ridgway"]["ungauged_drainage_fraction"])
    assert close < 0.02 and apart > 0.05
    text = tp.render(report)
    assert "It is not concluded here" in text
    assert "two sites cannot tell which" in text


def test_the_seasonal_gauges_are_absent_not_zero(report):
    missing = sorted(report["record"]["n_missing"].values())
    assert missing[-1] == missing[-2] > 400
    assert "not a creek that is not flowing" in tp.render(report)


def test_the_report_states_what_the_second_site_actually_cost(report):
    cost = report["second_site_cost"]
    assert cost["declaration_only"] is False
    assert cost["paid_again_by_a_third_site"] is False
    assert any("two gauged inflows" in line for line in cost["what_it_did_not_cover"])


def test_a_day_with_no_computable_residual_is_null_not_nan(report):
    """json.dumps runs with allow_nan=False; a missing storage reading must not break it."""
    series = report["closure_series"]
    assert any(v is None for v in series), "this record has days the residual cannot be computed on"
    json.dumps(report, allow_nan=False)


def test_the_report_reproduces(report):
    committed = json.loads((RESULTS / "real_taylor_park.json").read_text(encoding="utf-8"))
    failure = reproduction_failure("real_taylor_park.json", report, committed)
    assert failure is None, failure
    assert tp.render(committed) == (RESULTS / "real_taylor_park.md").read_text(encoding="utf-8")
