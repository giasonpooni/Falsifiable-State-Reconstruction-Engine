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
# the README's own results table
#
# Each row summarises an artifact in a sentence carrying its headline number. Those numbers
# are copied by hand and nothing regenerates them, which is the same exposure that let
# ROADMAP keep saying "six perfectly confounded pairs" after the artifact said fifteen. The
# rows whose numbers are derived statistics -- the ones that actually move when a record,
# seed or sweep changes -- are pinned here.
# ---------------------------------------------------------------------------

TAYLOR = json.loads((RESULTS / "real_taylor_park.json").read_text(encoding="utf-8"))
MUSKINGUM = json.loads((RESULTS / "muskingum_reach.json").read_text(encoding="utf-8"))
EIV = json.loads((RESULTS / "errors_in_variables.json").read_text(encoding="utf-8"))


def test_the_second_sites_residual_is_quoted_on_the_subset_it_was_measured_on():
    """9.80% is the fully reported days, not the record. Quoting the record's 18.14% under
    the same words, or this number without them, would both be wrong."""
    split = TAYLOR["reporting_split"]
    complete = f"{split['every_gauge_reporting']['residual_over_gauged_inflow'] * 100:.2f}%"
    assert complete == "9.80%"
    assert f"{complete} of gauged inflow even on the days every gauge reports" in README
    # and the all-days figure is a different number, which is why the qualifier matters
    assert (f"{split['all_days']['residual_over_gauged_inflow'] * 100:.2f}%") == "18.14%"


def test_the_river_row_counts_the_faults_routing_actually_recovers():
    def fully_identified(variant):
        strong = {k: v for k, v in MUSKINGUM["evaluation"][variant].items()
                  if v.get("magnitude_multiple") == 4.0}
        return {k for k, v in strong.items()
                if v["outcomes"].get("identified correctly", 0) == v["n_seeds"]}

    routing, bare = fully_identified("continuity + routing"), fully_identified("continuity only")
    # The prose spells the count, so the word is derived from it rather than hard-coded:
    # a seventh recovered fault must change the sentence, not just satisfy a looser match.
    spelled = {5: "five", 6: "six", 7: "seven", 8: "eight"}[len(routing)]
    assert f"{spelled} declared faults recovered in 100% of records with routing" in README
    # "none of the storage ones without it" -- the half of the sentence that is easy to lose
    assert not {k for k in bare if k.startswith("storage")}
    assert "none of the storage ones without it" in README


def test_the_calibration_row_quotes_the_span_the_experiment_measured():
    rates = [arm["rejection_rate_over_nominal"]
             for experiment in EIV["experiments"]
             for name, arm in experiment["by_declaration"].items() if name == "A treated as exact"]
    assert f"{min(rates):.0f} to {max(rates):.0f} times the nominal false-alarm rate" in README


# ---------------------------------------------------------------------------
# the two design studies
#
# Correcting second_balance's confound selection changed the cooling loop's confounded-pair
# count from 6 to 15. The report followed and its claims() followed, because a report refuses
# to render prose its own numbers stopped supporting. ROADMAP stage 3a went on saying "six"
# through a full regeneration and a push, because nothing here pinned it. These do.
#
# TWO KINDS OF NUMBER ARE DELIBERATELY NOT PINNED, and the last test below keeps them from
# being silently deleted instead. ARCHIVAL values -- 2.21x, what the cooling loop's tightest
# pair read BEFORE the fix -- appear in no current artifact, so a later reader checking prose
# against results/ would find them unsupported and "correct" them, destroying the record of
# the correction. WALL-CLOCK numbers are machine facts and never claims.
# ---------------------------------------------------------------------------

BALANCE = json.loads((RESULTS / "second_balance.json").read_text(encoding="utf-8"))
COOLING = json.loads((RESULTS / "cooling_circuits.json").read_text(encoding="utf-8"))
TOPOLOGY = {t["key"]: {v["variant"]: v for v in t["variants"]} for t in BALANCE["topologies"]}
FULL = "energy + duty + header"


def test_the_isolable_counts_the_roadmap_quotes_per_topology():
    river, loop = TOPOLOGY["muskingum_two_reach"], TOPOLOGY["cooling_loop_mass_energy"]
    today = TOPOLOGY["ridgway_today"]["one closure"]
    for isolable, faults in ((today["n_isolable"], today["n_faults"]),
                             (river["continuity + declared routing"]["n_isolable"], 7),
                             (loop["mass + energy + declared duty"]["n_isolable"], 7)):
        assert f"**{isolable} of {faults}**" in ROADMAP
    # "1 of 7" is quoted twice, for two topologies that happen to agree.
    assert river["continuity only"]["n_isolable"] == loop["mass + energy only"]["n_isolable"] == 1
    assert ROADMAP.count("**1 of 7**") == 2


def test_the_confound_counts_the_selection_fix_changed():
    """The sentence that went stale, and the one that records why."""
    loop = TOPOLOGY["cooling_loop_mass_energy"]
    bare = len(loop["mass + energy only"]["confounded_pairs"])
    full = len(loop["mass + energy + declared duty"]["confounded_pairs"])
    assert (bare, full) == (15, 4)
    assert "leaving fifteen perfectly confounded pairs" in ROADMAP
    assert f"from 6 to {bare} and from 1 to {full}" in ROADMAP
    assert len(TOPOLOGY["muskingum_two_reach"]["continuity + declared routing"]
               ["confounded_pairs"]) == 1
    assert "The river's one remaining confound is physical" in ROADMAP


def test_the_tightest_separated_pair_the_fix_corrected_to():
    loop = TOPOLOGY["cooling_loop_mass_energy"]["mass + energy + declared duty"]
    tightest = next(c["tightest_separated_pair"] for c in loop["by_prior"] if not c["refused"])
    shown = f"{tightest['isolation_amplification']:.2f}x"
    assert shown == "1.38x"
    for doc, name in ((ROADMAP, "ROADMAP.md"), (METHODS, "METHODS.md")):
        assert shown in doc, name


def test_the_duty_row_count_both_documents_quote():
    last = COOLING["sweep"][-1]
    shown = f"{last['by_variant']['energy + duty']['n_isolable']} of {last['n_faults']}"
    assert shown == "18 of 19"
    for doc, name in ((README, "README.md"), (ROADMAP, "ROADMAP.md")):
        assert shown in doc, name


def test_the_two_levers_are_quoted_at_their_measured_ratios():
    """The comparison that replaced a hand-waved "roughly two orders of magnitude"."""
    amps = [e["by_variant"][FULL]["amplification_at_reference_prior"] for e in COOLING["sweep"]]
    over_prior = [p["isolation_amplification"] for e in COOLING["sweep"]
                  for p in e["by_variant"][FULL]["amplification_by_prior"]
                  if p["isolation_amplification"] is not None]
    circuits = f"{max(amps) / min(amps):.2f}"
    prior = f"{max(over_prior) / min(over_prior):.0f}"
    assert (circuits, prior) == ("1.14", "88")
    for doc, name in ((README, "README.md"), (ROADMAP, "ROADMAP.md")):
        assert f"factor of {circuits}" in doc, name
        assert prior in doc, name


def test_the_handover_circuit_counts_the_roadmap_quotes():
    def handover(scale):
        return next((e["circuits"] for e in COOLING["sweep"]
                     if next(p["binding_pair_kind"]
                             for p in e["by_variant"][FULL]["amplification_by_prior"]
                             if p["prior_scale"] == scale) == "within one circuit"), None)

    scales = COOLING["declared"]["prior_scales_swept"]
    assert (handover(1.0), handover(max(scales)), handover(min(scales))) == (4, 3, None)
    assert ("At the declared prior that handover falls at four circuits; a hundredfold looser "
            "it falls at three; a hundredfold tighter it never falls in this sweep.") in ROADMAP


def test_the_tie_family_size_the_roadmap_quotes():
    last = COOLING["sweep"][-1]
    assert last["by_variant"][FULL]["tightest_separated_pair"]["n_tied"] == last["circuits"] == 6
    assert "six tied pairs at six circuits" in ROADMAP


def test_the_pre_correction_value_is_kept_and_kept_labelled():
    """No artifact contains 2.21x, which is exactly why it needs its own test.

    A reader checking prose against `results/` would find it unsupported and "correct" it,
    erasing the record of what the confound-selection defect reported. It must stay, and it
    must stay marked as what it is.
    """
    for doc, name in ((ROADMAP, "ROADMAP.md"), (METHODS, "METHODS.md")):
        assert "2.21x" in doc, f"{name} dropped the pre-correction value"
        window = doc[max(0, doc.index("2.21x") - 400):doc.index("2.21x") + 200]
        assert re.search(r"confounded|did exactly that|reported", window), (
            f"{name} quotes 2.21x without saying it is what the defect reported")


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
    def squashed(text):
        return _re.sub(r"[^a-z0-9]", "", text.lower())

    headline = _set_lcm.__doc__.splitlines()[0]
    title = README_RAW.splitlines()[0].lstrip("# ").strip()
    for word in DIST_NAME.split("-"):
        assert word in squashed(headline), (word, headline)
        assert word in squashed(title), (word, title)


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


# ---------------------------------------------------------------------------
# the boundary between a name and an identifier
#
# The rename moved the project from FSRE to FSRT. It deliberately did NOT move the lowercase
# `fsre-` identifiers, because they are not prose: schema strings are version contracts,
# `fsre-frame-v1` is a hash domain separator mixed into every frame digest, and the DAF plan
# ids are recorded in committed evidence. Renaming them to match the project would invalidate
# committed declarations, every frame checksum, and the evidence replay -- for appearance.
#
# Which makes this the dangerous kind of inconsistency: the tidy-looking fix is the destructive
# one. These pin the identifiers so that fix fails loudly instead of quietly rewriting history.
# ---------------------------------------------------------------------------

import hashlib as _hashlib

import numpy as _np

from set_lcm import declaration as _declaration
from set_lcm import frame_quality as _frame_quality
from set_lcm import recording as _recording


def test_the_schema_identifiers_are_frozen_and_do_not_follow_the_project_name():
    assert _declaration.SCHEMA == "fsre-declaration-v1"
    assert _recording.SCHEMA == "fsre-tank-recording-v1"


def test_the_frame_digest_domain_separator_is_frozen():
    """Pinned by value, not by reading the source: any change to the separator OR to how the
    digest is built changes this number, and every checksum recorded against a real frame."""
    frame = _np.arange(6, dtype=_np.uint8).reshape(2, 3)
    expected = _hashlib.sha256(
        b"fsre-frame-v1\0"
        + b'{"dtype":"|u1","shape":[2,3]}'
        + b"\0" + frame.tobytes(order="C")).hexdigest()
    assert _frame_quality.frame_content_hash(frame) == expected


def test_every_artifact_declares_a_namespaced_schema_version():
    """It used to take three forms at once -- eight namespaced strings, three bare integers and
    five absent -- so a consumer could not read the field at all: an integer 1 does not say what
    it versions, and an absent one does not say what shape to expect. Every artifact now carries
    one, and it keeps the frozen prefix."""
    versions = {}
    for path in sorted((REPO_ROOT / "results").glob("*.json")):
        version = json.loads(path.read_text(encoding="utf-8")).get("schema_version")
        assert isinstance(version, str), (path.name, version)
        assert version.startswith("fsre-"), (path.name, version)
        versions[path.name] = version
    assert len(versions) == len(list((REPO_ROOT / "results").glob("*.json")))
    assert len(set(versions.values())) == len(versions), "two artifacts share a schema version"


def test_the_bundle_manifest_keeps_its_own_integer_contract():
    """Not every `schema_version` in the tree is an artifact version. camera_bundle writes one
    into its own manifest and validates it on the way back in -- a closed handshake between a
    writer and a reader in one module, not a contract with a consumer of results/. Normalising
    it would mean changing both halves for no reader's benefit, so it stays an integer and this
    says that is deliberate."""
    source = (REPO_ROOT / "src" / "set_lcm" / "experiments" / "camera_bundle.py").read_text(encoding="utf-8")
    assert '"schema_version": 1' in source
    assert 'manifest.get("schema_version") != 1' in source


def test_the_declarations_state_the_frozen_schema():
    for path in sorted((REPO_ROOT / "declarations").glob("*.toml")):
        assert f'schema = "{_declaration.SCHEMA}"' in path.read_text(encoding="utf-8"), path.name
