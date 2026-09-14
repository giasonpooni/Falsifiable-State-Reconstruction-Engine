"""The diagnostic surface on real evidence, and the declared catalogue it judges.

Every record diagnose() had judged before this was one the repository generated, except
real_fluid_baseline's, which passes an EMPTY hypothesis set -- a consistency test with no
candidates. These tests pin the first real isolation result and the refusals around it.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from set_lcm.declaration import loads
from set_lcm.experiments import real_diagnosis as rd
from set_lcm.experiments.balance_site import site_from
from set_lcm.experiments.compare import reproduction_failure
from set_lcm.experiments.provenance import REPO_ROOT

RESULTS = REPO_ROOT / "results"
DECLARATIONS = REPO_ROOT / "declarations"
needs_daf = pytest.mark.skipif(not os.environ.get("DAF_ROOT"), reason="set DAF_ROOT to a DAF checkout")


# ---------------------------------------------------------------------------
# the declared catalogue
# ---------------------------------------------------------------------------

def test_both_sites_declare_their_candidates():
    for key, expected in (("ridgway", 4), ("taylor_park", 5)):
        faults = site_from(DECLARATIONS / f"{key}.toml").decl.faults
        assert len(faults) == expected
        assert all(f.amplitude_unit.strip() and f.profile.strip() for f in faults)


def test_only_the_site_with_gaps_declares_the_candidate_that_needs_them():
    ridgway = {f.name for f in site_from(DECLARATIONS / "ridgway.toml").decl.faults}
    taylor = {f.name for f in site_from(DECLARATIONS / "taylor_park.toml").decl.faults}
    assert "seasonal_creeks_unreported" in taylor
    assert "seasonal_creeks_unreported" not in ridgway
    assert ridgway < taylor


def test_a_fault_without_an_amplitude_unit_is_refused():
    """An amplitude with no unit is a number nobody can act on."""
    text = (DECLARATIONS / "ridgway.toml").read_text(encoding="utf-8")
    broken = text.replace('amplitude_unit = "acre-ft per day"\n', "", 1)
    with pytest.raises(ValueError, match="amplitude_unit must be a non-empty string"):
        loads(broken)


def test_a_repeated_fault_name_is_refused():
    text = (DECLARATIONS / "ridgway.toml").read_text(encoding="utf-8")
    dup = text + ('\n[[fault]]\nname = "ungauged_constant"\nlabel = "again"\n'
                  'profile = "constant_volume_per_day"\namplitude_unit = "x"\ndescription = "d"\n')
    with pytest.raises(ValueError, match="declares a fault name twice"):
        loads(dup)


def test_an_unknown_key_in_a_fault_table_is_refused():
    text = (DECLARATIONS / "ridgway.toml").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match=r"\[\[fault\]\] has unknown key"):
        loads(text.replace('profile = "constant_volume_per_day"',
                           'profile = "constant_volume_per_day"\nsignatur = "oops"', 1))


def test_a_profile_this_study_cannot_resolve_is_refused_by_name():
    """The declaration names a profile; the study owns the vocabulary and must say so."""
    text = (DECLARATIONS / "ridgway.toml").read_text(encoding="utf-8")
    site = site_from(DECLARATIONS / "ridgway.toml")
    bad = type(site)(loads(text.replace('profile = "constant_volume_per_day"',
                                        'profile = "vibes"', 1)))
    rec = {"windows": [{"days": 1, "storage_change": 0.0, "absent_days": 0, "inflow_cfs_days": 0.0}]}
    with pytest.raises(ValueError, match="profile 'vibes', which this study cannot resolve"):
        rd.signatures(bad, rec)


def test_an_unknown_fault_name_raises_and_lists_the_known_ones():
    with pytest.raises(KeyError, match="declares no fault"):
        site_from(DECLARATIONS / "ridgway.toml").decl.fault("gremlins")


# ---------------------------------------------------------------------------
# what it found
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def report():
    return rd.compute()


def test_every_claim_the_report_makes_holds(report):
    failed = [name for name, held in report["claims"].items() if not held]
    assert not failed, failed


def test_render_refuses_to_print_a_sentence_that_stopped_being_true(report):
    broken = dict(report, claims=dict(report["claims"],
                                      the_engine_never_names_a_single_cause_at_the_second_site=False))
    with pytest.raises(RuntimeError, match="no longer true of the numbers"):
        rd.render(broken)


def test_the_two_constant_rate_candidates_are_exactly_collinear_at_both_sites(report):
    """The rank-1 bound in a real catalogue: no covariance can separate these."""
    for key in ("ridgway", "taylor_park"):
        pair = next(p for p in report["sites"][key]["geometry"]
                    if {p["a"], p["b"]} == {"ungauged_constant", "outflow_reads_low"})
        assert pair["exactly_collinear"]
        assert abs(abs(pair["cos"]) - 1.0) < 1e-12


def test_the_engine_never_identifies_a_single_cause(report):
    """The honest outcome for a rank-1 constraint with two collinear candidates."""
    for key in ("ridgway", "taylor_park"):
        assert "identified" not in [c["status"] for c in report["sites"][key]["sweep"]]


def test_the_status_loosens_monotonically_as_error_is_admitted(report):
    for key in ("ridgway", "taylor_park"):
        chi2 = [c["null_statistic"] for c in report["sites"][key]["sweep"]]
        assert chi2 == sorted(chi2, reverse=True), (key, chi2)


def test_ambiguity_is_reached_only_where_the_record_does_not_close(report):
    statuses = {k: [c["status"] for c in report["sites"][k]["sweep"]] for k in report["sites"]}
    assert "ambiguous" in statuses["taylor_park"]
    assert "ambiguous" not in statuses["ridgway"]


def test_nothing_is_fitted_to_the_residuals_own_scatter(report):
    """The covariance comes from the declared sigmas; a circular one would be far larger."""
    assert "Nothing is fitted" in report["declared"]["covariance_source"]
    text = rd.render(report)
    assert "would be circular" in text
    assert "No fault is asserted to exist" in json.dumps(report["limitations"])


def test_the_report_reproduces(report):
    committed = json.loads((RESULTS / "real_diagnosis.json").read_text(encoding="utf-8"))
    failure = reproduction_failure("real_diagnosis.json", report, committed)
    assert failure is None, failure
    assert rd.render(committed) == (RESULTS / "real_diagnosis.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# the export filter that makes adding a site affordable
# ---------------------------------------------------------------------------

def test_a_session_filter_cannot_be_combined_with_a_fixtures_only_export():
    out = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "export_daf_fixtures.py"),
                          "--session", "x", "--only", "fixtures"],
                         capture_output=True, text=True)
    assert out.returncode != 0
    assert "excludes all of them" in out.stderr


@needs_daf
def test_exporting_one_session_leaves_every_other_manifest_entry_byte_identical(tmp_path):
    """Adding a site replayed every session: an hour of CPU for minutes of evidence. The merge
    must produce exactly the manifest a full export would, or it trades an hour for a doubt."""
    data = REPO_ROOT / "data" / "daf"
    before = (data / "manifest.json").read_text(encoding="utf-8")
    backup = tmp_path / "manifest.json"
    shutil.copy(data / "manifest.json", backup)
    try:
        out = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "export_daf_fixtures.py"),
                              "--session", "usgs_taylor_park_wy2023_2025"],
                             capture_output=True, text=True, env={**os.environ})
        assert out.returncode == 0, out.stderr
        assert (data / "manifest.json").read_text(encoding="utf-8") == before
    finally:
        shutil.copy(backup, data / "manifest.json")


def _exporter():
    """The export tool, imported for its pure logic. Its top-level imports are stdlib only, so
    this needs no DAF checkout -- which matters, because CI has none and the merge property is
    exactly the thing that must not silently stop holding."""
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        import export_daf_fixtures
        return export_daf_fixtures
    finally:
        sys.path.pop(0)


def test_merging_sessions_one_at_a_time_equals_one_full_export():
    merge = _exporter().merge_manifest_files
    fixtures = ["a.json", "b.json"]
    entries = {name: {"output": name, "sha": name.upper()} for name in
               ("a.json", "b.json", "s_one.json", "s_two.json", "s_three.json")}
    full = list(entries.values())

    incremental = merge(fixtures_and_two := [entries["a.json"], entries["b.json"],
                                             entries["s_one.json"], entries["s_three.json"]],
                        [entries["s_two.json"]], fixtures)
    one_pass = merge([], full, fixtures)
    assert incremental == one_pass
    assert [f["output"] for f in incremental] == ["a.json", "b.json",
                                                  "s_one.json", "s_three.json", "s_two.json"]
    assert fixtures_and_two[0] is incremental[0]        # untouched entries carried verbatim


def test_a_re_exported_session_replaces_its_predecessor_and_nothing_else():
    merge = _exporter().merge_manifest_files
    prior = [{"output": "a.json", "sha": "OLD-A"}, {"output": "s.json", "sha": "OLD-S"}]
    merged = merge(prior, [{"output": "s.json", "sha": "NEW-S"}], ["a.json"])
    assert merged == [{"output": "a.json", "sha": "OLD-A"}, {"output": "s.json", "sha": "NEW-S"}]


def test_a_session_new_to_the_manifest_lands_where_a_full_export_would_put_it():
    merge = _exporter().merge_manifest_files
    prior = [{"output": "a.json"}, {"output": "s_b.json"}, {"output": "s_d.json"}]
    merged = merge(prior, [{"output": "s_c.json"}], ["a.json"])
    assert [f["output"] for f in merged] == ["a.json", "s_b.json", "s_c.json", "s_d.json"]


def test_the_declared_covariance_is_positive_definite_by_construction():
    """Not by luck. Adjacent windows share a storage reading, which puts -sigma_S^2 on the
    off-diagonal and makes the storage part of each row exactly cancel; what leaves the matrix
    strictly diagonally dominant is the flow variance, which is positive on every real day.
    diagnose() would refuse a non-PD covariance, but far from the reason -- this states it."""
    for key in ("ridgway", "taylor_park"):
        site = site_from(REPO_ROOT / "declarations" / f"{key}.toml")
        rec = rd._record(site)
        for alignment_sd in rd.ALIGNMENT_SWEEP:
            cov = rd.covariance(rec, alignment_sd)
            off = np.abs(cov).sum(axis=1) - np.abs(np.diag(cov))
            assert (np.diag(cov) - off).min() > 0.0, (key, alignment_sd)
            assert np.linalg.eigvalsh(cov).min() > 0.0, (key, alignment_sd)
