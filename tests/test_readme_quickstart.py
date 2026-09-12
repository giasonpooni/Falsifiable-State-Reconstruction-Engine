"""The README's quickstart is a claim like any other, so it is checked like any other.

Every number the README prints in its two code blocks is recomputed here from the same
inputs. A change to the kernel that would make the front page wrong fails here first,
which is the only way a worked example stays true after the code moves under it.
"""
import re
from pathlib import Path

import numpy as np
import pytest

from set_lcm.fdi import isolability
from set_lcm.lcm import chi2_quantile, reconcile
from set_lcm.schema import ConstraintSet, Status

README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")

CS = ConstraintSet(version="total-v1", A=np.array([[1.0, 1.0]]), b=np.array([100.0]),
                   b_var=np.array([0.5 ** 2]), description="m1 + m2 = 100 kg")
X, P = np.array([52.0, 46.0]), np.diag([1.0, 1.0])


def test_the_reconcile_example_produces_the_values_the_readme_prints():
    est = reconcile(X, P, CS, mode="hard", threshold=chi2_quantile(CS.rank, 0.999))
    assert est.status is Status.OK
    assert est.consistency_stat == pytest.approx(1.78, abs=0.005)
    assert est.residual_pre == pytest.approx([-2.0])
    # NOT zero: b carries declared uncertainty, so the projection does not force the row
    assert est.residual_post == pytest.approx([-0.222], abs=0.0005)
    assert est.residual_post[0] != 0.0
    assert est.correction == pytest.approx([0.889, 0.889], abs=0.0005)
    # the unprojected estimate is kept, untouched
    assert est.x_unprojected == pytest.approx(X) and est.x_unprojected is not X
    for shown in ("[-2.0]", "1.78", "10.83", "[0.889, 0.889]", "[52.0, 46.0]"):
        assert shown in README, shown


def test_the_isolability_example_produces_the_values_the_readme_prints():
    r = isolability({"leak_from_tank_1": [1.0, 0.0],
                     "leak_from_tank_2": [0.0, 1.0],
                     "transfer_1_to_2": [1.0, -1.0]}, P, CS)
    assert r.d["leak_from_tank_1"] == pytest.approx(0.444, abs=0.0005)
    assert r.d["leak_from_tank_2"] == pytest.approx(0.444, abs=0.0005)
    assert r.d["transfer_1_to_2"] == pytest.approx(0.0, abs=1e-15)
    assert r.invisible == ["transfer_1_to_2"] and r.isolable == []
    pair = next(p for p in r.pairs if {p.a, p.b} == {"leak_from_tank_1", "leak_from_tank_2"})
    assert pair.cos == pytest.approx(1.0) and not pair.distinguishable
    for shown in ("'leak_from_tank_1': 0.444", "'transfer_1_to_2': 0.0",
                  "invisible ['transfer_1_to_2']", "cos=1.0", "distinguishable=False"):
        assert shown in README, shown


def test_the_threshold_the_readme_quotes_is_the_one_the_kernel_returns():
    assert chi2_quantile(CS.rank, 0.999) == 10.828
    assert "10.83" in README


def test_the_detection_delay_claim_comes_from_the_committed_grid():
    """This test exists because the README once said "9 steps where the constraint test took
    38" and neither number appeared anywhere in results/. Rule 5 says every reported number
    traces to results/; a front-page claim with no artifact behind it is the exact failure
    that rule forbids, so the claim is now pinned to the run that produces it."""
    import json
    s = json.loads((Path(__file__).resolve().parents[1] / "results"
                    / "summary.json").read_text(encoding="utf-8"))
    kf = s["scenarios"]["bias_quant_delay"]["aggregate"]["kf"]
    constraint, cusum = kf["detection"], kf["cusum"]
    # the biased sensor's own channel finds it, and finds it sooner than the constraint test
    assert cusum["s1"]["detection"]["detected_within"] == constraint["detected_within"] == 20
    assert cusum["s1"]["detection"]["median_delay"] == 13.5
    assert constraint["median_delay"] == 36.0
    assert cusum["s1"]["detection"]["median_delay"] < constraint["median_delay"]
    # and the UNbiased sensor's channel stays silent -- this is the localisation claim
    assert cusum["s2"]["detection"]["detected_within"] == 0
    assert cusum["s1"]["false_alarms"]["rate"] == cusum["s2"]["false_alarms"]["rate"] == 0.0
    for shown in ("median 13.5 steps", "36.0", "0/20", "bias_quant_delay"):
        assert shown in README, shown


def test_the_readme_does_not_claim_cusum_names_the_faulty_instrument():
    """The README once said the per-sensor channel "names the instrument". It does not, and
    the committed grid says so: leak_stale_constraint injects a 10 kg PROCESS leak with
    bias=None -- no sensor fault anywhere -- and CUSUM names sensor 2 in 20/20 seeds with the
    same confidence it has when it is right. It indicates a channel, not an instrument."""
    import json
    s = json.loads((Path(__file__).resolve().parents[1] / "results"
                    / "summary.json").read_text(encoding="utf-8"))
    leak = s["scenarios"]["leak_stale_constraint"]
    assert leak["meta"]["deg"]["bias"] is None, "this scenario must carry no sensor fault"
    kf = leak["aggregate"]["kf"]["cusum"]
    assert kf["s2"]["detection"]["detected_within"] == 20        # named, confidently
    assert kf["s2"]["detection"]["median_delay"] == 59.5
    assert kf["s1"]["detection"]["detected_within"] == 0
    # the README must carry that counterexample, not just the flattering one
    for shown in ("leak_stale_constraint", "no sensor fault at all", "median 59.5",
                  "not the faulty instrument"):
        assert shown in README, shown


def test_the_null_range_the_readme_quotes_is_the_measured_one():
    """Also once wrong, also in the flattering direction: the README quoted 0.57-0.64 for a
    null whose measured range is 0.463-0.644."""
    import json
    null = json.loads((Path(__file__).resolve().parents[1] / "results"
                       / "calibration.json").read_text(encoding="utf-8"))["null"]
    means = [v["mean"] for v in null.values()]
    lags = [v["lag1_autocorr"] for v in null.values()]
    taus = [v["autocorr_time"] for v in null.values()]
    assert f"{min(means):.3f}" == "0.463" and f"{max(means):.3f}" == "0.644"
    assert f"{min(lags):.3f}" == "0.761" and f"{max(lags):.3f}" == "0.935"
    assert f"{min(taus):.1f}" == "7.4" and f"{max(taus):.1f}" == "29.7"
    for shown in ("0.463–0.644", "0.761–0.935", "7.4–29.7"):
        assert shown in README, shown


def test_the_readme_does_not_present_the_cumulative_imbalance_as_a_finding():
    """The three-year total is 0.72 standard errors from zero, so it is not evidence of a net
    imbalance, and the front page must not lead with it as though it were. What rejects is the
    daily statistic."""
    import json
    import math
    w = json.loads((Path(__file__).resolve().parents[1] / "results"
                    / "real_water_balance.json").read_text(encoding="utf-8"))["closure_free"]
    se = w["sd"] * math.sqrt(w["n_days"])
    assert abs(w["cumulative"]) < se, "the cumulative is now outside 1 se: the README must be rewritten"
    assert f"{se:,.0f}" == "2,122"
    assert w["lag1_autocorr"] > 0, "positive lag-1 only widens the true standard error"
    assert "inside one standard error of zero" in README
    assert "2,122" in README and "64.12" in README
    assert "cumulative total is *not* the finding" in README


def test_the_headline_numbers_come_from_the_committed_results():
    """The front page quotes the Ridgway figures; they must be the ones in results/."""
    import json
    r = json.loads((Path(__file__).resolve().parents[1] / "results"
                    / "real_water_balance.json").read_text(encoding="utf-8"))
    c = r["closure_free"]
    assert f"{c['cumulative']:,.0f}" == "1,526"
    assert f"{c['cumulative_as_fraction_of_gauged_inflow'] * 100:+.2f}" == "+0.41"
    open_stats = r["sweep"][f"{r['declared']['storage_sigma_base']:g}"]["specs"]["wb_open"]["consistency_stat"]
    assert f"{open_stats['mean']:.2f}" == "9.69"
    assert f"{open_stats['fraction_over_threshold'] * 100:.1f}" == "38.6"
    assert open_stats["threshold"] == 10.828
    for shown in ("+1,526 acre-ft", "+0.41%", "9.69", "38.6%", "10.83"):
        assert shown in README, shown
    # and the sweep the README quotes as the sigma sensitivity
    rates = [r["sweep"][f"{s:g}"]["specs"]["wb_open"]["consistency_stat"]["fraction_over_threshold"]
             for s in sorted(r["declared"]["storage_sigma_sweep"])]
    assert [f"{x * 100:.1f}" for x in rates] == ["52.0", "38.6", "13.3"]
    assert "52.0% → 38.6% → 13.3%" in README
