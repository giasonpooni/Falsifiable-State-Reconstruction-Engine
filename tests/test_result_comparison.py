"""Cross-build roundoff allowances must not conceal changed decisions or invalid output."""
import pytest

from set_lcm.experiments import compare as comparison
from set_lcm.experiments.compare import EXCEPTIONS, compare


def test_only_the_declared_null_direction_has_an_absolute_allowance():
    before = {"blind": {"d": {"null_space (S + G)": 1.37e-37, "storage_only": 1.37e-37}}}
    after = {"blind": {"d": {"null_space (S + G)": 0.0, "storage_only": 0.0}}}
    diffs = compare(before, after, rel_tol=1e-8, exceptions=EXCEPTIONS["real_water_balance.json"])
    assert [d.path for d in diffs] == [".blind.d.storage_only"]
    # Same-build exact comparison remains exact, and a materially nonzero null fails.
    assert len(compare(before, after, rel_tol=0.0)) == 2
    after["blind"]["d"]["null_space (S + G)"] = 1e-12
    assert len(compare(before, after, rel_tol=1e-8,
                       exceptions=EXCEPTIONS["real_water_balance.json"])) == 2


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_are_not_accepted_as_roundoff(bad):
    assert compare({"score": 1.0}, {"score": bad}, rel_tol=1e-8)


def test_labels_counts_and_structure_remain_exact():
    before = {"claim": True, "count": 1_000_000_000, "label": "held"}
    after = {"claim": 1, "count": 1_000_000_001, "label": "applied"}
    assert len(compare(before, after, rel_tol=1e-6)) == 3
    assert compare({"a": [1.0]}, {"a": [1.0, 1.0]}, rel_tol=1e-6)


@pytest.mark.parametrize("before,after", [
    (1_000_000_000, 1_000_000_001.0),
    (1_000_000_000.0, 1_000_000_001),
])
def test_integer_involvement_keeps_count_changes_exact(before, after):
    assert compare({"count": before}, {"count": after}, rel_tol=1e-6)
    # Equal numeric values remain equal; the rule protects the value, not its JSON spelling.
    assert not compare({"count": 1}, {"count": 1.0}, rel_tol=1e-6)


@pytest.mark.parametrize("same_build", [True, False])
@pytest.mark.parametrize("before,after", [
    ({"claim": True}, {"claim": 1}),
    ({"count": 1_000_000_000}, {"count": 1_000_000_001.0}),
    ({"count": 1_000_000_000.0}, {"count": 1_000_000_001}),
    ({"score": float("inf")}, {"score": float("inf")}),
    ({"score": -float("inf")}, {"score": -float("inf")}),
    ({"score": float("nan")}, {"score": float("nan")}),
])
def test_reproduction_wrapper_rejects_changed_claims_counts_and_invalid_values(
    monkeypatch, same_build, before, after,
):
    monkeypatch.setattr(comparison, "same_build", lambda generation: same_build)
    failure = comparison.reproduction_failure("real_water_balance.json", after, before)
    assert failure is not None
    assert "1 value(s) differ" in failure


@pytest.mark.parametrize("same_build", [True, False])
def test_reproduction_wrapper_limits_roundoff_exception_to_other_builds(monkeypatch, same_build):
    monkeypatch.setattr(comparison, "same_build", lambda generation: same_build)
    before = {"blind": {"d": {"null_space (S + G)": 1.37e-37}}}
    after = {"blind": {"d": {"null_space (S + G)": 0.0}}}
    failure = comparison.reproduction_failure("real_water_balance.json", after, before)
    assert (failure is not None) == same_build


@pytest.mark.parametrize("same_build", [True, False])
def test_reproduction_wrapper_still_excludes_only_runtime_metadata(monkeypatch, same_build):
    monkeypatch.setattr(comparison, "same_build", lambda generation: same_build)
    before = {"provenance": {"git_head": "old"}, "latency_us_p50": 1.0,
              "claim": True, "count": 12, "score": 2.5}
    after = {**before, "provenance": {"git_head": "new"}, "latency_us_p50": 100.0}
    assert comparison.reproduction_failure("real_water_balance.json", after, before) is None
