"""Every number the prose quotes must be findable in the artifact it came from.

`docs/RESULTS.md` is written by hand and `results/` is generated, so nothing but a test
keeps them in step. The claims pinned here are the ones that were previously overstated:

  - a three-year cumulative imbalance of +1,526 acre-ft, quoted as a magnitude. It is
    0.60 standard errors from zero, and the interval contains zero.
  - a cross-check between two routes, described as independent. Both read the same
    bridged series, and the percentage divides by the other route's denominator.

A number that changes in `results/` must be changed in the prose, not tolerated by a
loose match here: every assertion below formats the artifact's own value and requires
that exact string in the document.
"""
import json
import re

import pytest

from set_lcm.experiments.provenance import REPO_ROOT

RESULTS = REPO_ROOT / "results"
WB = json.loads((RESULTS / "real_water_balance.json").read_text(encoding="utf-8"))
FLUID = json.loads((RESULTS / "real_fluid_baseline.json").read_text(encoding="utf-8"))


def _prose(path):
    """Wrapped markdown, flattened so a phrase assertion does not depend on line breaks.

    The typographic minus is folded to ASCII because the artifacts write ASCII.
    """
    text = path.read_text(encoding="utf-8").replace("\u2212", "-")
    return " ".join(text.split())


DOC = _prose(REPO_ROOT / "docs" / "RESULTS.md")
FLUID_DOC = _prose(REPO_ROOT / "docs" / "FLUID_BASELINE.md")
CLOSURE = WB["closure_free"]
MIDDLE = WB["sweep"]["200"]["specs"]


def _shown(x, n=0):
    return f"{x:,.{n}f}"


def test_the_cumulative_is_never_quoted_in_the_doc_without_its_standard_error():
    cum = _shown(CLOSURE["cumulative"])
    assert cum in DOC                                  # it is reported
    # ...and every value that says how far from zero it is, in the doc's own words
    for value in (_shown(CLOSURE["cumulative_se"]), _shown(CLOSURE["cumulative_se_ar1"]),
                  _shown(CLOSURE["cumulative_in_se"], 2), _shown(CLOSURE["cumulative_in_se_ar1"], 2),
                  _shown(CLOSURE["cumulative_ci95_ar1"][0]), _shown(CLOSURE["cumulative_ci95_ar1"][1])):
        assert value in DOC, value
    assert "standard errors from zero" in DOC
    # The one framing that must not come back: a bolded total presented as the result.
    assert f"**+{cum} acre-ft" not in DOC
    assert f"**{cum} acre-ft" not in DOC
    lo, hi = CLOSURE["cumulative_ci95_ar1"]
    assert lo < 0.0 < hi, "the doc's 'contains zero' claim is about this interval"
    assert "contains zero" in DOC


def test_the_doc_does_not_claim_the_two_routes_are_independent():
    assert "share no code path" not in DOC
    assert "not independent" in DOC
    # Both routes come from the same bridge call, and the doc must name it.
    assert "load(σ)" in DOC or "load(sigma)" in DOC
    # The augmented percentage divides by the arithmetic route's own denominator.
    assert "gauged_inflow_volume" in DOC
    aug = MIDDLE["wb_aug+hard"]["ungauged_cumulative"]
    assert aug["final_as_fraction_of_gauged_inflow"] * CLOSURE["gauged_inflow_volume"] == \
        pytest.approx(aug["final"], rel=1e-12)


def test_every_separation_from_zero_the_doc_quotes_is_the_artifacts_own():
    for name in ("wb_aug+hard", "wb_aug+hard+feedback"):
        aug = MIDDLE[name]["ungauged_cumulative"]
        for value in (_shown(aug["final"]), _shown(aug["final_sd"]), _shown(aug["final_in_sd"], 2)):
            assert value in DOC, (name, value)
    # the sweep endpoints the doc quotes as sensitivity
    for sigma in ("50", "800"):
        aug = WB["sweep"][sigma]["specs"]["wb_aug+hard"]["ungauged_cumulative"]
        assert _shown(aug["final_in_sd"], 2) in DOC, sigma


def test_the_rejection_the_doc_attributes_to_the_daily_statistic_is_the_daily_one():
    stat = MIDDLE["wb_open"]["consistency_stat"]
    assert f"{stat['mean']:,.2f}" in DOC
    assert f"{stat['fraction_over_threshold'] * 100:.1f}% of days" in DOC
    assert f"{stat['threshold']:.2f}" in DOC
    # and the doc must not attribute a rejection to the cumulative
    assert re.search(r"cumulative[^.]*\brejects\b", DOC) is None


def test_the_fluid_doc_quotes_the_window_sweep_the_artifact_measured():
    """The rejection rate depends on a consumer choice, so the doc must quote it as measured."""
    lengths = FLUID["declared"]["window_days_sensitivity"]
    cells = {(c["storage_sigma_acre_ft"], c["window_days"]): c for c in FLUID["window_sensitivity"]}
    tightest = min(sigma for sigma, _ in cells)
    for window in lengths:
        rate = cells[(tightest, window)]["rejected_fraction"]
        assert f"{rate * 100:.1f}%" in FLUID_DOC, (window, rate)
        assert str(window) in FLUID_DOC, window
    assert FLUID["declared"]["window_days"] in lengths
    assert "decides the rejection rate as much as the declared storage" in FLUID_DOC

    # the monotone claim the doc makes, checked at every declared sigma
    by_sigma = {}
    for (sigma, window), cell in cells.items():
        by_sigma.setdefault(sigma, []).append((window, cell["statistic_over_threshold"]["median"]))
    for sigma, points in by_sigma.items():
        medians = [median for _, median in sorted(points)]
        assert medians == sorted(medians), (sigma, medians)
