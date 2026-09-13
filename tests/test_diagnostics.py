"""Declared finite-record diagnosis: statistical references, ambiguity and failure contracts."""
from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from set_lcm.diagnostics import _chi2_quantile, diagnose


# Independent references from scipy.stats.chi2.ppf, SciPy 1.18.1, in an ephemeral
# environment outside the project. SciPy is not a runtime or test dependency.
QUANTILES = [
    (1, .95, 3.841458820694124), (1, .99, 6.6348966010212145), (1, .999, 10.827566170662733),
    (2, .95, 5.991464547107979), (2, .99, 9.21034037197618), (2, .999, 13.815510557964274),
    (3, .95, 7.814727903251179), (3, .99, 11.344866730144373), (3, .999, 16.26623619623813),
    (10, .95, 18.307038053275146), (10, .99, 23.209251158954356), (10, .999, 29.58829844507442),
    (30, .95, 43.77297182574219), (30, .99, 50.89218131151707), (30, .999, 59.70306430442994),
    (100, .95, 124.34211340400407), (100, .99, 135.80672317102676), (100, .999, 149.44925277903886),
]


@pytest.mark.parametrize("dof,probability,reference", QUANTILES)
def test_chi_square_quantile_matches_independent_scipy_references(dof, probability, reference):
    assert _chi2_quantile(dof, probability) == pytest.approx(reference, rel=2e-11, abs=1e-12)


@pytest.mark.parametrize("dof,probability", [(0, .99), (-1, .99), (1.5, .99), (True, .99),
                                           (2, 0), (2, 1), (2, np.nan), (2, True)])
def test_quantile_refuses_invalid_arguments(dof, probability):
    with pytest.raises(ValueError):
        _chi2_quantile(dof, probability)


def test_two_strongly_separated_known_signatures_identify_conditionally():
    a, b = np.array([1., 1., 0., 0., 0., 0.]), np.array([0., 0., 1., 1., 0., 0.])
    result = diagnose(8 * a, np.eye(6), {"a": a, "b": b})
    assert result.status == "identified" and result.candidates == ("a",)
    assert result.null_dof == 6 and result.null_statistic == pytest.approx(128)
    fits = {fit.name: fit for fit in result.fits}
    assert fits["a"].amplitude == pytest.approx(8)
    assert fits["a"].amplitude_sd == pytest.approx(1 / np.sqrt(2))
    assert fits["a"].fit_dof == 5 and fits["a"].fit_statistic == pytest.approx(0, abs=1e-25)
    assert fits["a"].adequate and not fits["b"].adequate
    assert "conditional" in result.explanation.lower()
    assert any("not posterior" in assumption for assumption in result.assumptions)


def test_same_line_with_opposite_sign_and_scale_stays_ambiguous():
    h = np.arange(1., 7.)
    result = diagnose(4 * h, np.eye(6), {"one": h, "other": -7 * h})
    assert result.status == "ambiguous" and result.candidates == ("one", "other")
    one, other = result.fits
    assert one.amplitude == pytest.approx(-7 * other.amplitude)
    assert one.amplitude_sd == pytest.approx(7 * other.amplitude_sd)
    np.testing.assert_allclose(one.interval, -7 * np.array(other.interval)[::-1])
    assert one.fit_statistic == pytest.approx(other.fit_statistic, abs=1e-24)


def test_nuisance_absorbed_signature_is_unobservable_not_a_sensor_diagnosis():
    h = np.ones(6)
    nuisance = np.column_stack([h, -1e12 * h, np.zeros(6)])
    result = diagnose(8 * h, np.eye(6), {"bias": h}, nuisance=nuisance)
    assert result.status == "consistent" and result.candidates == ()
    assert result.nuisance_rank == 1 and result.null_dof == 5
    fit = result.fits[0]
    assert not fit.observable
    assert fit.amplitude is fit.amplitude_sd is fit.interval is None
    assert fit.fit_dof == result.null_dof and fit.fit_statistic == result.null_statistic
    assert "not evidence" in result.explanation


def test_nuisance_projection_matches_independent_generalized_least_squares():
    n = 7
    h, nuisance = np.arange(n, dtype=float) ** 2, np.ones((n, 1))
    covariance = .8 ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    residual = 3 + 2 * h + np.array([.2, -.4, .3, .8, -.1, .4, -.8])
    design = np.column_stack([nuisance, h])
    information = design.T @ np.linalg.solve(covariance, design)
    beta = np.linalg.solve(information, design.T @ np.linalg.solve(covariance, residual))
    error = residual - design @ beta
    result = diagnose(residual, covariance, {"quadratic": h}, nuisance=nuisance)
    fit = result.fits[0]
    assert fit.amplitude == pytest.approx(beta[1], rel=1e-12)
    assert fit.amplitude_sd ** 2 == pytest.approx(np.linalg.inv(information)[1, 1], rel=1e-12)
    assert fit.fit_statistic == pytest.approx(error @ np.linalg.solve(covariance, error), rel=1e-12)
    assert fit.fit_dof == n - 2


@pytest.mark.parametrize("units", [np.ones(3), np.array([1e8, 1e-6, 1e3])])
def test_small_independent_nuisance_direction_cannot_be_misattributed_to_a_fault(units):
    nuisance = np.array([[1., 1.], [0., 1e-13], [0., 0.]])
    residual = np.array([0., 10., 0.])
    np.testing.assert_allclose(nuisance @ np.array([-1e14, 1e14]), residual)
    result = diagnose(units * residual, np.diag(units ** 2), {"fault": units * [0., 1., 0.]},
                      nuisance=units[:, None] * nuisance)
    assert result.nuisance_rank == 2 and result.null_dof == 1
    assert result.status == "consistent" and result.candidates == ()
    assert result.null_statistic == pytest.approx(0, abs=1e-24)
    assert not result.fits[0].observable and result.fits[0].amplitude is None


@pytest.mark.parametrize("units", [np.ones(3), np.array([1e8, 1e-6, 1e3])])
def test_nuisance_below_rank_resolution_is_refused_not_discarded(units):
    nuisance = np.array([[1., 1.], [0., 1e-18], [0., 0.]])
    with pytest.raises(ValueError, match="nuisance rank cannot be resolved reliably"):
        diagnose(units * [0., 10., 0.], np.diag(units ** 2), {"fault": units * [0., 1., 0.]},
                 nuisance=units[:, None] * nuisance)


def test_small_real_signature_direction_after_nuisance_removal_is_not_dropped():
    result = diagnose([0., 10., 0.], np.eye(3),
                      {"nearly_confounded": [1., 1e-13, 0.], "direct": [0., 1., 0.]},
                      nuisance=np.array([[1.], [0.], [0.]]))
    assert result.status == "ambiguous"
    assert result.candidates == ("nearly_confounded", "direct")
    assert result.fits[0].observable and result.fits[0].amplitude == pytest.approx(1e14)


def test_unresolved_signature_is_refused_before_an_alternative_can_be_identified():
    with pytest.raises(ValueError, match="relative to nuisance rank cannot be resolved reliably"):
        diagnose([0., 10., 0.], np.eye(3),
                 {"unresolved": [1., 1e-18, 0.], "direct": [0., 1., 0.]},
                 nuisance=np.array([[1.], [0.], [0.]]))


def test_exact_linear_dependence_beyond_duplicate_columns_remains_supported():
    nuisance = np.array([[1., 0., 1.], [0., 1., 1.], [0., 0., 0.], [0., 0., 0.]])
    result = diagnose([3., 4., 0., 0.], np.eye(4), {"absorbed": [1., 1., 0., 0.]}, nuisance=nuisance)
    assert result.nuisance_rank == 2 and result.status == "consistent"
    assert not result.fits[0].observable


def test_large_unresolved_designs_refuse_instead_of_unbounded_exact_arithmetic():
    with pytest.raises(ValueError, match="exact rank verification size limit"):
        diagnose(np.zeros(130), np.eye(130), {}, nuisance=np.ones((130, 32)))


def test_unexplained_waveform_is_not_assigned_to_the_least_bad_candidate():
    residual = np.array([0., 0., 0., 0., 10., -10.])
    result = diagnose(residual, np.eye(6), {"first": [1, 0, 0, 0, 0, 0], "second": [0, 1, 0, 0, 0, 0]})
    assert result.status == "unexplained" and result.candidates == ()
    assert all(fit.adequate is False for fit in result.fits)


def test_null_adequacy_means_no_evidence_not_health_and_supports_no_catalogue():
    result = diagnose(np.zeros(6), np.eye(6), {})
    assert result.status == "consistent" and result.fits == () and result.candidates == ()
    assert result.null_statistic == 0 and "healthy" in result.explanation
    rejected = diagnose(np.full(6, 10.), np.eye(6), {})
    assert rejected.status == "unexplained" and rejected.fits == ()


def test_saturated_fit_and_full_nuisance_space_do_not_identify():
    saturated = diagnose([10.], [[1.]], {"anything": [1.]})
    assert saturated.status == "insufficient_evidence" and saturated.candidates == ()
    fit = saturated.fits[0]
    assert fit.observable and fit.fit_dof == 0 and fit.adequate is None
    assert fit.fit_statistic is fit.fit_threshold is None
    assert fit.amplitude == 10 and fit.interval is not None   # valid conditional estimation, no fit test
    absorbed = diagnose([10., -5.], np.eye(2), {"first": [1., 0.]}, nuisance=np.eye(2))
    assert absorbed.status == "insufficient_evidence" and absorbed.null_dof == 0
    assert absorbed.null_statistic is absorbed.null_threshold is None
    assert not absorbed.fits[0].observable


@pytest.mark.parametrize("factor", [1e-100, -1e-12, 1e12, -1e100])
def test_signature_rescaling_preserves_fit_and_transforms_amplitude(factor):
    h = np.arange(1., 7.)
    baseline = diagnose(4 * h, np.eye(6), {"fault": h})
    scaled = diagnose(4 * h, np.eye(6), {"fault": factor * h})
    before, after = baseline.fits[0], scaled.fits[0]
    assert scaled.status == baseline.status == "identified"
    assert after.observable and after.fit_dof == before.fit_dof
    assert after.fit_statistic == pytest.approx(before.fit_statistic, abs=1e-24)
    assert factor * after.amplitude == pytest.approx(before.amplitude, rel=1e-12)
    assert abs(factor) * after.amplitude_sd == pytest.approx(before.amplitude_sd, rel=1e-12)


def test_row_unit_changes_preserve_diagnostics_with_joint_covariance():
    h = np.arange(1., 7.)
    covariance = .7 ** np.abs(np.subtract.outer(np.arange(6), np.arange(6)))
    nuisance = np.ones((6, 1))
    residual = 3 * h + 2
    units = np.array([1e-8, 1e6, .001, 1000., 1e-4, 1e8])
    before = diagnose(residual, covariance, {"fault": h}, nuisance=nuisance)
    after = diagnose(units * residual, units[:, None] * covariance * units[None, :],
                     {"fault": units * h}, nuisance=units[:, None] * nuisance)
    assert before.status == after.status == "identified"
    assert before.nuisance_rank == after.nuisance_rank
    assert before.null_statistic == pytest.approx(after.null_statistic, rel=1e-12)
    assert before.fits[0].amplitude == pytest.approx(after.fits[0].amplitude, rel=1e-12)
    assert before.fits[0].amplitude_sd == pytest.approx(after.fits[0].amplitude_sd, rel=1e-12)


def test_conditional_amplitude_intervals_cover_independent_gaussian_records():
    """All records count, regardless of diagnosis: no selection-conditioned coverage claim."""
    rng = np.random.default_rng(1729)
    n, draws, amplitude = 8, 2000, 1.8
    h, nuisance = np.linspace(-1., 1., n), np.ones((n, 1))
    covariance = .16 * .6 ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    records = 2 + amplitude * h + rng.standard_normal((draws, n)) @ np.linalg.cholesky(covariance).T
    covered, z = 0, []
    for record in records:
        fit = diagnose(record, covariance, {"known_profile": h}, nuisance=nuisance).fits[0]
        covered += fit.interval[0] <= amplitude <= fit.interval[1]
        z.append((fit.amplitude - amplitude) / fit.amplitude_sd)
    assert .93 < covered / draws < .97
    assert abs(np.mean(z)) < .08 and .88 < np.var(z) < 1.12


def test_no_fault_rejection_rate_on_independent_joint_gaussian_records():
    rng = np.random.default_rng(2718)
    n, draws = 10, 1000
    covariance = .7 ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    records = rng.standard_normal((draws, n)) @ np.linalg.cholesky(covariance).T
    rejected = sum(diagnose(record, covariance, {}, alpha=.05).status == "unexplained" for record in records)
    assert .025 < rejected / draws < .075


@pytest.mark.parametrize("residual,covariance,hypotheses,kwargs", [
    ([], np.empty((0, 0)), {}, {}),
    ([[1., 2.]], np.eye(2), {}, {}),
    ([np.nan, 0], np.eye(2), {}, {}),
    ([1 + 2j, 0], np.eye(2), {}, {}),
    ([0, 0], np.eye(3), {}, {}),
    ([0, 0], [[1, .4], [.1, 1]], {}, {}),
    ([0, 0], [[1, 2], [2, 1]], {}, {}),
    ([0, 0], [[1, 1], [1, 1]], {}, {}),
    ([0, 0], [[-1, 0], [0, 1]], {}, {}),
    ([0, 0], [[1, 0], [0, np.inf]], {}, {}),
    ([0, 0], np.eye(2), [], {}),
    ([0, 0], np.eye(2), {"": [1, 0]}, {}),
    ([0, 0], np.eye(2), {"bad": [1]}, {}),
    ([0, 0], np.eye(2), {"bad": [[1, 0]]}, {}),
    ([0, 0], np.eye(2), {"bad": [1, np.inf]}, {}),
    ([0, 0], np.eye(2), {}, {"nuisance": [1, 1]}),
    ([0, 0], np.eye(2), {}, {"nuisance": np.ones((3, 1))}),
    ([0, 0], np.eye(2), {}, {"nuisance": [[1], [np.nan]]}),
])
def test_malformed_inputs_are_refused(residual, covariance, hypotheses, kwargs):
    with pytest.raises(ValueError):
        diagnose(residual, covariance, hypotheses, **kwargs)


@pytest.mark.parametrize("field", ["alpha", "interval_level"])
@pytest.mark.parametrize("value", [0, 1, -.01, 1.01, np.nan, np.inf, True, [0.01], 1e-20])
def test_invalid_probabilities_are_refused(field, value):
    with pytest.raises(ValueError):
        diagnose([0., 0.], np.eye(2), {}, **{field: value})


def test_outputs_are_frozen_json_safe_and_inputs_are_unchanged():
    residual, covariance, signature = np.array([10., 0.]), np.eye(2), np.array([1., 0.])
    copies = [a.copy() for a in (residual, covariance, signature)]
    result = diagnose(residual, covariance, {"one": signature, "zero": np.zeros(2)})
    for before, after in zip(copies, (residual, covariance, signature)):
        np.testing.assert_array_equal(before, after)
    with pytest.raises(FrozenInstanceError):
        result.status = "consistent"
    with pytest.raises(FrozenInstanceError):
        result.fits[0].amplitude = 0
    serialized = json.loads(json.dumps(result.as_dict(), allow_nan=False))
    assert serialized["fits"]["zero"]["amplitude"] is None
    assert serialized["fits"]["zero"]["interval"] is None
    full = diagnose(residual, covariance, {}, nuisance=np.eye(2))
    assert json.loads(json.dumps(full.as_dict(), allow_nan=False))["null_threshold"] is None


# ---------------------------------------------------------------------------
# an adequate candidate is not an actionable one
# ---------------------------------------------------------------------------

def _catalogue(n=8, epsilon=2e-3):
    """One direction, a near-collinear rival, and an orthogonal one."""
    a = np.zeros(n); a[:4] = 1.0
    near = a.copy(); near[4] = epsilon * np.linalg.norm(a)
    far = np.zeros(n); far[4:] = 1.0
    return a, near, far


def test_a_near_collinear_pair_is_reported_with_how_far_apart_it_is():
    """The pair is structurally distinct and practically inseparable; both are reported."""
    a, near, far = _catalogue()
    result = diagnose(6.0 * a, np.eye(8), {"a": a, "near": near, "far": far})
    assert result.status == "ambiguous"
    assert set(result.candidates) == {"a", "near"}
    fits = {fit.name: fit for fit in result.fits}
    assert fits["a"].nearest == "near" and fits["near"].nearest == "a"
    # sin(theta) for these two, and its reciprocal
    expected = 2e-3 / np.sqrt(1.0 + 2e-3 ** 2)
    assert fits["a"].nearest_orthogonal_fraction == pytest.approx(expected, rel=1e-6)
    assert fits["a"].nearest_isolation_amplification == pytest.approx(1.0 / expected, rel=1e-6)
    assert result.min_separation == pytest.approx(expected, rel=1e-6)
    assert "500 times the one that rejects no fault" in result.explanation


def test_every_verdict_states_the_closest_pair_including_an_identified_one():
    """`identified` from a barely separated catalogue is not the claim it looks like."""
    a, _, far = _catalogue()
    identified = diagnose(6.0 * a, np.eye(8), {"a": a, "far": far})
    assert identified.status == "identified"
    assert identified.min_separation == pytest.approx(1.0)      # orthogonal catalogue
    assert "Closest observable candidate pair" in identified.explanation
    consistent = diagnose(np.zeros(8), np.eye(8), {"a": a, "far": far})
    assert consistent.status == "consistent"
    assert "Closest observable candidate pair" in consistent.explanation


def test_a_lone_or_unobservable_candidate_has_nothing_to_be_separated_from():
    a, _, _ = _catalogue()
    lone = diagnose(6.0 * a, np.eye(8), {"a": a})
    assert lone.min_separation is None
    assert lone.fits[0].nearest is None
    assert lone.fits[0].nearest_orthogonal_fraction is None
    assert lone.as_dict()["min_separation"] is None
    # a candidate confounded with the nuisance space is unobservable and stays unpaired
    confounded = diagnose(6.0 * a, np.eye(8), {"a": a, "zero": np.zeros(8)})
    assert {fit.name: fit.observable for fit in confounded.fits} == {"a": True, "zero": False}
    assert next(f for f in confounded.fits if f.name == "zero").nearest is None
    assert confounded.min_separation is None       # one observable candidate, nothing to compare


def test_exactly_collinear_candidates_cannot_be_separated_at_any_amplitude():
    a, _, _ = _catalogue()
    result = diagnose(6.0 * a, np.eye(8), {"a": a, "twice": 2.0 * a})
    assert result.min_separation == 0.0
    fits = {fit.name: fit for fit in result.fits}
    # None rather than inf: there is no amplitude, and the dict must stay JSON-serializable
    assert fits["a"].nearest_isolation_amplification is None
    assert fits["a"].nearest_orthogonal_fraction == 0.0
    assert abs(fits["a"].nearest_cos) == pytest.approx(1.0)
    assert "no amplitude lets this record prefer one of them" in result.explanation
    json.dumps(result.as_dict(), allow_nan=False)     # the way every report writes it


def test_the_separation_is_archived_with_every_fit():
    a, near, far = _catalogue()
    archived = diagnose(6.0 * a, np.eye(8), {"a": a, "near": near, "far": far}).as_dict()
    assert archived["min_separation"] is not None
    for name, fit in archived["fits"].items():
        assert {"nearest", "nearest_cos", "nearest_orthogonal_fraction",
                "nearest_isolation_amplification"} <= set(fit), name
        assert fit["nearest"] is not None, name
    # and it survives a JSON round trip, which is how a report carries it
    assert json.loads(json.dumps(archived))["min_separation"] == archived["min_separation"]


def test_the_separation_uses_the_whitened_coordinates_the_decision_used():
    """A covariance that stretches one axis changes the separation, as it must.

    Two directions are geometrically orthogonal in the raw coordinates; under a
    covariance that makes one axis enormously noisy, the record cannot separate them.
    """
    raw_a = np.array([1.0, 0.0, 0.0, 0.0])
    raw_b = np.array([1.0, 1.0, 0.0, 0.0])
    residual = np.array([4.0, 0.0, 0.0, 0.0])
    plain = diagnose(residual, np.eye(4), {"a": raw_a, "b": raw_b})
    stretched = diagnose(residual, np.diag([1.0, 1e8, 1.0, 1.0]), {"a": raw_a, "b": raw_b})
    assert plain.min_separation > 0.5
    assert stretched.min_separation < 1e-3
    assert stretched.min_separation < plain.min_separation
