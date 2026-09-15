"""Headline numbers quoted in README.md and docs/ROADMAP.md, against the artifacts they come from.

Every report in `results/` refuses to render prose its own numbers stopped supporting --
that is what each experiment's `claims()` is for. The prose in `README.md` and
`docs/ROADMAP.md` had no such guard, and it cost exactly what one would expect: correcting
`second_balance`'s confound selection changed the cooling loop's confounded-pair count from
6 to 15, the report and its claims followed, and a sentence in ROADMAP stage 3a went on
saying "six" through a full regeneration and a push.

So the numbers those two documents quote from an artifact are pinned here. This is not a
general prose checker and does not try to be one: it is a short, explicit table of the
claims that would mislead a reader if they drifted, each tied to the value it came from.
Adding a headline number to either document means adding a line here.

TWO KINDS OF NUMBER ARE DELIBERATELY NOT PINNED, and they are listed in
`test_archival_numbers_are_still_present_and_still_labelled` so they cannot be silently
deleted either:

    ARCHIVAL -- a value from BEFORE a correction, quoted to say what the correction changed.
    2.21x is the cooling loop's tightest pair as the confound-selection defect reported it;
    no current artifact contains it, and a future reader who "fixes" it to match one would
    destroy the record of the fix.

    WALL-CLOCK -- latency and runtime, which are machine facts and never claims.
"""
import json
import pathlib
import re

import pytest

from set_lcm.experiments.provenance import REPO_ROOT

RESULTS = REPO_ROOT / "results"
FULL = "energy + duty + header"


def _read(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def docs():
    return {name: (REPO_ROOT / name).read_text(encoding="utf-8")
            for name in ("README.md", "docs/ROADMAP.md")}


@pytest.fixture(scope="module")
def second_balance():
    report = _read("second_balance.json")
    return {t["key"]: {v["variant"]: v for v in t["variants"]} for t in report["topologies"]}


@pytest.fixture(scope="module")
def cooling():
    return _read("cooling_circuits.json")


def _flat(text: str) -> str:
    """Collapse whitespace, so a rewrapped paragraph is not a test failure."""
    return re.sub(r"\s+", " ", text)


def _contains(docs, doc, phrase):
    assert _flat(phrase) in _flat(docs[doc]), f"{doc} no longer contains {phrase!r}"


# ---------------------------------------------------------------------------
# the second-balance design study
# ---------------------------------------------------------------------------

def test_the_isolable_counts_quoted_for_each_topology(docs, second_balance):
    river = second_balance["muskingum_two_reach"]
    loop = second_balance["cooling_loop_mass_energy"]
    today = second_balance["ridgway_today"]["one closure"]
    quoted = {
        "**0 of 4**": (today["n_isolable"], today["n_faults"]),
        "**5 of 7**": (river["continuity + declared routing"]["n_isolable"], 7),
        "**2 of 7**": (loop["mass + energy + declared duty"]["n_isolable"], 7),
    }
    for phrase, (isolable, faults) in quoted.items():
        assert phrase == f"**{isolable} of {faults}**", phrase
        _contains(docs, "docs/ROADMAP.md", phrase)
    # "1 of 7" is quoted twice, for two different topologies that happen to agree.
    assert river["continuity only"]["n_isolable"] == loop["mass + energy only"]["n_isolable"] == 1
    assert docs["docs/ROADMAP.md"].count("**1 of 7**") == 2


def test_the_confound_counts_that_the_selection_fix_changed(docs, second_balance):
    """The sentence that went stale. 6 -> 15 and 1 -> 4 when confounding stopped being read
    off a floating-point zero; the words in ROADMAP must track the artifact."""
    loop = second_balance["cooling_loop_mass_energy"]
    assert len(loop["mass + energy only"]["confounded_pairs"]) == 15
    _contains(docs, "docs/ROADMAP.md", "leaving fifteen perfectly confounded pairs")
    assert len(loop["mass + energy + declared duty"]["confounded_pairs"]) == 4
    _contains(docs, "docs/ROADMAP.md", "from 6 to 15 and from 1 to 4")
    river = second_balance["muskingum_two_reach"]["continuity + declared routing"]
    assert len(river["confounded_pairs"]) == 1
    _contains(docs, "docs/ROADMAP.md", "The river's one remaining confound is physical")


def test_the_tightest_separated_pair_the_fix_corrected_to(docs, second_balance):
    loop = second_balance["cooling_loop_mass_energy"]["mass + energy + declared duty"]
    tightest = next(c["tightest_separated_pair"] for c in loop["by_prior"] if not c["refused"])
    assert f"{tightest['isolation_amplification']:.2f}x" == "1.38x"
    for doc in ("docs/ROADMAP.md", "docs/METHODS.md"):
        assert "1.38x" in (REPO_ROOT / doc).read_text(encoding="utf-8"), doc


# ---------------------------------------------------------------------------
# the cooling-manifold study
# ---------------------------------------------------------------------------

def test_the_duty_row_count_quoted_in_both_documents(docs, cooling):
    last = cooling["sweep"][-1]
    phrase = f"{last['by_variant']['energy + duty']['n_isolable']} of {last['n_faults']}"
    assert phrase == "18 of 19"
    for doc in docs:
        _contains(docs, doc, phrase)


def test_the_two_levers_are_quoted_at_their_measured_ratios(docs, cooling):
    """The comparison that replaced "roughly two orders of magnitude"."""
    amps = [e["by_variant"][FULL]["amplification_at_reference_prior"] for e in cooling["sweep"]]
    over_prior = [p["isolation_amplification"] for e in cooling["sweep"]
                  for p in e["by_variant"][FULL]["amplification_by_prior"]
                  if p["isolation_amplification"] is not None]
    circuit_lever = max(amps) / min(amps)
    prior_lever = max(over_prior) / min(over_prior)
    assert f"{circuit_lever:.2f}" == "1.14"
    assert f"{prior_lever:.0f}" == "88"
    for doc in docs:
        _contains(docs, doc, "factor of 1.14")
        assert "88" in docs[doc], doc


def test_the_handover_circuit_counts_quoted_in_the_roadmap(docs, cooling):
    def handover(scale):
        return next((e["circuits"] for e in cooling["sweep"]
                     if next(p["binding_pair_kind"]
                             for p in e["by_variant"][FULL]["amplification_by_prior"]
                             if p["prior_scale"] == scale) == "within one circuit"), None)

    scales = cooling["declared"]["prior_scales_swept"]
    assert handover(1.0) == 4 and handover(max(scales)) == 3 and handover(min(scales)) is None
    _contains(docs, "docs/ROADMAP.md",
              "At the declared prior that handover falls at four circuits; a hundredfold "
              "looser it falls at three; a hundredfold tighter it never falls in this sweep.")


def test_the_tie_family_size_quoted_in_the_roadmap(docs, cooling):
    last = cooling["sweep"][-1]
    assert last["by_variant"][FULL]["tightest_separated_pair"]["n_tied"] == last["circuits"] == 6
    _contains(docs, "docs/ROADMAP.md", "six tied pairs at six circuits")


# ---------------------------------------------------------------------------
# what is deliberately not pinned
# ---------------------------------------------------------------------------

def test_archival_numbers_are_still_present_and_still_labelled(docs):
    """A value from before a correction, quoted to record what the correction changed.

    No current artifact contains 2.21x, so nothing can pin it -- which is exactly why it
    needs a test of its own: a future reader checking prose against `results/` would find it
    unsupported and "fix" it, destroying the record. It must stay, and it must stay marked as
    what it is.
    """
    for doc, before in (("docs/ROADMAP.md", "2.21x"), ("docs/METHODS.md", "2.21x")):
        text = (REPO_ROOT / doc).read_text(encoding="utf-8")
        assert before in text, f"{doc} dropped the pre-correction value {before}"
        window = text[max(0, text.index(before) - 400):text.index(before) + 200]
        assert re.search(r"confounded|did exactly that|reported", window), (
            f"{doc} quotes {before} without saying it is what the defect reported")
