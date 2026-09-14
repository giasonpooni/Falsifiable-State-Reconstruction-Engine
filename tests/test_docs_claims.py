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
NOAA_MONTH = json.loads((RESULTS / "real_noaa_month.json").read_text(encoding="utf-8"))


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


def test_the_throughput_and_inflow_volume_the_doc_quotes_are_the_artifacts_own():
    """Both divide the closure residual, so both move when its day set does -- which is
    exactly what happened when the denominator was aligned with the numerator."""
    assert _shown(CLOSURE["mean_throughput_cfs"], 1) in DOC
    assert _shown(CLOSURE["gauged_inflow_volume"]) in DOC


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


def test_every_declared_citation_reaches_the_report_a_reader_reads():
    """A citation stored only in JSON is not a citation a reader of the table sees.

    Every sigma, interval and timing choice in this repository is consumer-declared,
    and the rule is that the declaration travels with the number. This walks each
    committed results JSON for any key naming a citation and requires the string in
    the sibling markdown -- so a citation cannot be recorded and then not shown.
    """
    unrendered = []
    for json_path in sorted(RESULTS.glob("*.json")):
        md_path = json_path.with_suffix(".md")
        if not md_path.exists():
            continue
        rendered = " ".join(md_path.read_text(encoding="utf-8").split())
        data = json.loads(json_path.read_text(encoding="utf-8"))

        def walk(node, path=""):
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value, f"{path}.{key}")
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    walk(value, f"{path}[{index}]")
            elif isinstance(node, str) and "citation" in path.rsplit(".", 1)[-1].lower():
                if " ".join(node.split()) not in rendered:
                    unrendered.append(f"{json_path.name}{path}")

        walk(data)
    assert not unrendered, f"declared citations stored but never rendered: {unrendered}"


def test_the_month_narrative_quotes_the_month_artifact():
    """P4c's numbers in docs/RESULTS.md, each formatted from the artifact it came from."""
    windows, fits, scored = NOAA_MONTH["windows"], NOAA_MONTH["fits"], NOAA_MONTH["scored"]
    assert f"{windows['month']['n_steps']:,}" in DOC
    for key in ("fit", "held_out", "month"):
        assert f"{windows[key]['n_evidence_ids']:,}" in DOC, key
        check = windows[key]["series_check"]
        assert f"{check['n_second_differences']:,}" in DOC, key
        assert f"{check['white_error_sigma_bound']:.4f}" in DOC, key
        assert f"{check['null_rel_sd_mean_square'] * 100:.2f}%" in DOC, key
        assert f"{windows[key]['stated_sigma_over_white_bound']['median']:.2f}" in DOC, key
        assert f"{windows[key]['resolution']['record_days']:.1f}" in DOC, key

    sigma = windows["month"]["stated_sigma"]
    assert f"{sigma['median']:.4f}" in DOC
    assert f"{sigma['max']:.3f}" in DOC
    worst = sigma["extreme"][0]
    assert worst["time"][:16] in DOC
    assert f"{worst['step_in_record']:,}" in DOC
    assert f"{worst['over_median']:.0f} times" in DOC
    assert f"{sigma['rms_without_extreme']:.4f}" in DOC and f"{sigma['rms']:.4f}" in DOC
    assert f"{sigma['n_stated_zero']} readings state" in DOC

    for kind, entry in scored.items():
        held = entry["held_out"]
        assert f"{held['fresh']['z_rms']:.3f}" in DOC, kind
        assert f"{held['z_rms_fresh_over_continued']:.4f}" in DOC, kind
        q_scale = fits[kind]["q_scale"]
        assert f"{q_scale:.2e}" in DOC or f"{q_scale:g}" in DOC, kind

    # the claim that carries the most weight, and its assumption
    assert "exceeds the bound on the median reading in **every** window" in DOC
    assert "that NOAA's stated" in DOC and "is wrong" in DOC
    assert "smoother than the declared R" in DOC
    # and the split's cost, which is the reason the month report exists
    unresolved = windows["fit"]["resolution"]["unresolved_among_modelled"]
    assert unresolved, "the narrative claims the fit window leaves a pair unresolved"
    for row in unresolved:
        assert f"{row['rayleigh_days']:.1f}" in DOC, row


# ---------------------------------------------------------------------------
# the errors-in-variables projection
#
# README, METHODS and ROADMAP all quote numbers from results/eiv_projection.json by hand.
# Those are exactly the sentences that go stale silently when a seed or a sweep moves, so
# every one of them is formatted from the artifact here and required verbatim.
# ---------------------------------------------------------------------------

PROJECTION = json.loads((RESULTS / "eiv_projection.json").read_text(encoding="utf-8"))
README = _prose(REPO_ROOT / "README.md")
METHODS = _prose(REPO_ROOT / "docs" / "METHODS.md")
ROADMAP = _prose(REPO_ROOT / "docs" / "ROADMAP.md")


def _arm(experiment, name):
    return experiment["by_arm"][name]


@pytest.mark.parametrize("doc,name", [(README, "README.md"), (METHODS, "METHODS.md"),
                                      (ROADMAP, "ROADMAP.md")])
def test_the_projection_numbers_the_prose_quotes_are_the_artifacts_own(doc, name):
    xs = PROJECTION["experiments"]
    widest = max(xs, key=lambda e: e["state_scale"])
    fit = PROJECTION["quadratic_fit"]

    # the coverage a reader acts on, at the operating point the prose names as the widest
    coverage = _arm(widest, "A treated as exact")["coverage"]
    assert f"{coverage * 100:.1f}%" in doc, (name, coverage)

    # the quadratic fit, which is the claim that the mechanism is the state-dependence
    assert f"{fit['worst_relative_error'] * 100:.2f}%" in doc, (name, fit["worst_relative_error"])


def test_the_prose_does_not_quote_the_fit_without_the_span_it_was_fitted_over():
    """`1 + c*scale^2` to 0.10% means nothing without the range it holds across."""
    scales = PROJECTION["declared"]["state_scale_swept"]
    span = f"{scales[-1] / scales[0]:.0f}x"
    for doc, name in ((METHODS, "METHODS.md"), (ROADMAP, "ROADMAP.md")):
        assert "1 + c" in doc, name
        assert span in doc, (name, span)


def test_the_prose_reports_the_one_step_gain_cost_it_was_measured_at():
    worst = max(abs(e["one_step_gain"]["reiterated_nees_change"])
                for e in PROJECTION["experiments"])
    dimension = PROJECTION["experiments"][0]["dimension"]
    for doc, name in ((METHODS, "METHODS.md"), (ROADMAP, "ROADMAP.md")):
        assert f"{worst:.2f}" in doc, (name, worst)
        assert f"target of {dimension}" in doc, name


def test_the_docs_do_not_still_say_the_projection_ignores_the_declaration():
    """The sentence this stage removed, kept out of every document that carried it."""
    for doc, name in ((README, "README.md"), (METHODS, "METHODS.md"), (ROADMAP, "ROADMAP.md")):
        for stale in ("solves against `A_bar` as though it were exact",
                      "The projection is unchanged",
                      "implemented for the test, not the projection"):
            assert stale not in doc, (name, stale)


# ---------------------------------------------------------------------------
# the project's own name
#
# It had drifted into three variants at once -- the repository called itself one thing, the
# README another, and the packaging a third -- and the README's clone command named a
# repository that does not exist. Nothing checked it, so nothing caught it. These derive every
# spelling from the packaging name and require the rest to agree.
# ---------------------------------------------------------------------------

import re as _re
import tomllib as _tomllib

import set_lcm as _set_lcm

_PYPROJECT = _tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
DIST_NAME = _PYPROJECT["project"]["name"]
README_RAW = (REPO_ROOT / "README.md").read_text(encoding="utf-8")


def test_the_clone_command_names_the_repository_that_exists():
    """A clone command is the one line a new reader runs first. It named a repository that is
    not there: the owner had moved, the name had not."""
    urls = [u.removesuffix(".git")
            for u in _re.findall(r"github\.com/[\w.-]+/([\w.-]+)", README_RAW)]
    assert urls, "the README should show how to clone the project"
    for slug in set(urls):
        if slug.lower().replace("-", "") == DIST_NAME.replace("-", ""):
            break
    else:
        raise AssertionError(
            f"no clone URL in the README matches the distribution name {DIST_NAME!r}; found {sorted(set(urls))}")


def test_the_package_docstring_and_the_readme_title_are_the_same_project():
    words = [w for w in DIST_NAME.split("-")]
    headline = _set_lcm.__doc__.splitlines()[0]
    title = README_RAW.splitlines()[0].lstrip("# ").strip()
    for w in words:
        assert w.lower() in headline.lower(), (w, headline)
        assert w.lower() in title.lower().replace("-", " ").replace(" ", "") or \
               w.lower() in title.lower(), (w, title)


def test_one_acronym_everywhere():
    acronyms = set(_re.findall(r"\b([A-Z]{4})\b", _set_lcm.__doc__.splitlines()[0]))
    assert len(acronyms) == 1, acronyms
    acronym = acronyms.pop()
    assert acronym in README_RAW
    initials = "".join(w[0] for w in DIST_NAME.split("-")).upper()
    assert acronym == initials, (acronym, initials)


def test_no_superseded_name_survives_anywhere():
    stale = ("FSRE", "Fluid-Sensor", "falsifiable-state-reconstruction")
    found = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in {".md", ".py", ".toml"}:
            continue
        if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
            continue
        if path.name == "test_docs_claims.py":
            continue                      # this file names them in order to forbid them
        text = path.read_text(encoding="utf-8", errors="ignore")
        found += [f"{path.relative_to(REPO_ROOT)}: {s}" for s in stale if s in text]
    assert not found, found
