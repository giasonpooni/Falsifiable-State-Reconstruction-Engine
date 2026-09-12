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
