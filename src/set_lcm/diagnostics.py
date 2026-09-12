"""Finite-record Gaussian lack-of-fit tests for declared single-fault signatures.

The input is one stacked residual vector and its JOINT covariance, including temporal
and cross-channel correlation. Each candidate is one known signature with one unknown,
unrestricted signed amplitude. The onset/profile is supplied, never searched here. A
declared nuisance matrix spans arbitrary deterministic nuisance effects; no coefficient
prior is invented. Empty hypotheses request only the no-fault consistency test.

Whitening and orthogonal nuisance removal give independent unit-variance residual
coordinates under the declared Gaussian model. The no-fault and fitted-candidate squared
residual norms have chi-square distributions with their remaining residual degrees of
freedom. Thresholds use a converged incomplete-gamma calculation, not an asymptotic
quantile approximation. Existing reconciliation-kernel arithmetic is unchanged.

An adequate null means no departure was detected, not that the instruments are healthy.
Identification is conditional on the supplied candidates, nuisance space, known covariance
and single-fault model. Candidate adequacy is not a posterior probability. Intervals are
Gaussian intervals conditional on each fixed candidate/profile/covariance, without adjustment
for choosing a candidate or searching for an onset. No multiple-testing guarantee is made.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math
from statistics import NormalDist
from typing import Mapping

import numpy as np

__all__ = ["CandidateFit", "DiagnosticResult", "diagnose"]

_EXACT_RANK_MAX_ENTRIES = 4096
_EXACT_RANK_MAX_COLUMNS = 32
_EXACT_RANK_MAX_BITS = 16384
_GAMMA_EPS = 8 * np.finfo(float).eps
_GAMMA_MAX_ITER = 10000
_GAMMA_TINY = 1e-300


def _gamma_probabilities(a: float, x: float) -> tuple[float, float]:
    """Regularized lower/upper incomplete gamma, by convergent series or fraction."""
    if x == 0.0:
        return 0.0, 1.0
    prefactor = math.exp(a * math.log(x) - x - math.lgamma(a))
    if x < a + 1.0:
        term = total = 1.0 / a
        for i in range(1, _GAMMA_MAX_ITER + 1):
            term *= x / (a + i)
            total += term
            if abs(term) <= abs(total) * _GAMMA_EPS:
                p = min(1.0, max(0.0, total * prefactor))
                return p, 1.0 - p
    else:
        b = x + 1.0 - a
        c, d = 1.0 / _GAMMA_TINY, 1.0 / b
        fraction = d
        for i in range(1, _GAMMA_MAX_ITER + 1):
            numerator = -i * (i - a)
            b += 2.0
            d = b + numerator * d
            c = b + numerator / c
            if abs(d) < _GAMMA_TINY:
                d = math.copysign(_GAMMA_TINY, d)
            if abs(c) < _GAMMA_TINY:
                c = math.copysign(_GAMMA_TINY, c)
            d = 1.0 / d
            delta = d * c
            fraction *= delta
            if abs(delta - 1.0) <= _GAMMA_EPS:
                q = min(1.0, max(0.0, prefactor * fraction))
                return 1.0 - q, q
    raise ValueError("incomplete-gamma calculation did not converge")


def _probability(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite probability strictly between 0 and 1")
    try:
        p = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite probability strictly between 0 and 1") from None
    if not math.isfinite(p) or not 0.0 < p < 1.0:
        raise ValueError(f"{name} must be a finite probability strictly between 0 and 1")
    return p


@lru_cache(maxsize=256, typed=True)
def _chi2_quantile(dof: int, probability: float) -> float:
    """General chi-square quantile, bracketed against the regularized gamma CDF.

    The upper-tail comparison avoids subtracting its small probability from one during
    inversion. This private helper deliberately does not alter lcm.chi2_quantile's table.
    """
    if isinstance(dof, (bool, np.bool_)) or not isinstance(dof, (int, np.integer)) or dof < 1:
        raise ValueError("chi-square degrees of freedom must be a positive integer")
    p = _probability(probability, "chi-square probability")

    def below(x):
        lower, upper = _gamma_probabilities(dof / 2.0, x / 2.0)
        return lower < p if p < 0.5 else upper > 1.0 - p

    lo, hi = 0.0, float(max(1, dof))
    while below(hi):
        lo, hi = hi, 2.0 * hi
        if not math.isfinite(hi):
            raise ValueError("chi-square quantile exceeds the supported numeric range")
    for _ in range(2048):
        mid = lo + (hi - lo) / 2.0
        if mid == lo or mid == hi:
            return mid
        if below(mid):
            lo = mid
        else:
            hi = mid
        if hi - lo <= 2e-13 * max(hi, np.finfo(float).tiny):
            return (lo + hi) / 2.0
    raise ValueError("chi-square quantile inversion did not converge")


def _array(value, ndim: int, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    try:
        a = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite {ndim}-D numeric array") from None
    if a.ndim != ndim or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite {ndim}-D numeric array")
    return a


def _norm(value: np.ndarray) -> float:
    scale = float(np.max(np.abs(value))) if value.size else 0.0
    return 0.0 if scale == 0.0 else _finite(scale * float(np.linalg.norm(value / scale)), "vector norm")


def _finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} is outside the supported finite numeric range; rescale the inputs")
    return float(value)


def _verify_no_discarded_direction(matrix: np.ndarray, resolved_rank: int, name: str) -> None:
    """Distinguish exact dependence of supplied floats from an unresolved small direction.

    Arbitrary nuisance coefficients can amplify any nonzero direction, however small.
    A numerical rank cutoff alone cannot justify deleting it. Exact rational elimination
    is used only when SVD/projection proposes dropping a direction, and is deliberately
    bounded for this small-record diagnostic. It certifies dependencies of the supplied
    binary floats, not of an unobserved exact design before it was rounded by the caller.
    """
    advice = (f"{name} rank cannot be resolved reliably; supply a well-conditioned independent "
              "basis or reparameterize the declared design instead of dropping nuisance directions")
    if matrix.size > _EXACT_RANK_MAX_ENTRIES or matrix.shape[1] > _EXACT_RANK_MAX_COLUMNS:
        raise ValueError(advice + " (exact rank verification size limit)")
    rows = [[Fraction(float(value)) for value in row] for row in matrix]
    rank = 0
    for col in range(matrix.shape[1]):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][col]), None)
        if pivot is None:
            continue
        if rank == resolved_rank:
            raise ValueError(advice + " (a nonzero design direction is below numerical resolution)")
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for i in range(rank + 1, len(rows)):
            if not rows[i][col]:
                continue
            factor = rows[i][col] / rows[rank][col]
            for j in range(col + 1, matrix.shape[1]):
                value = rows[i][j] - factor * rows[rank][j]
                if max(value.numerator.bit_length(), value.denominator.bit_length()) > _EXACT_RANK_MAX_BITS:
                    raise ValueError(advice + " (exact rank verification arithmetic limit)")
                rows[i][j] = value
            rows[i][col] = Fraction(0)
        rank += 1
        if rank == len(rows):
            break


@dataclass(frozen=True)
class CandidateFit:
    name: str
    observable: bool
    fit_statistic: float | None
    fit_dof: int
    fit_threshold: float | None
    adequate: bool | None
    amplitude: float | None
    amplitude_sd: float | None
    interval: tuple[float, float] | None
    explanation: str

    def as_dict(self) -> dict:
        return {"name": self.name, "observable": self.observable,
                "fit_statistic": self.fit_statistic, "fit_dof": self.fit_dof,
                "fit_threshold": self.fit_threshold, "adequate": self.adequate,
                "amplitude": self.amplitude, "amplitude_sd": self.amplitude_sd,
                "interval": list(self.interval) if self.interval is not None else None,
                "explanation": self.explanation}


@dataclass(frozen=True)
class DiagnosticResult:
    status: str
    candidates: tuple[str, ...]
    null_statistic: float | None
    null_dof: int
    null_threshold: float | None
    fits: tuple[CandidateFit, ...]
    nuisance_rank: int
    alpha: float
    interval_level: float
    explanation: str
    assumptions: tuple[str, ...]

    def as_dict(self) -> dict:
        return {"status": self.status, "candidates": list(self.candidates),
                "null_statistic": self.null_statistic, "null_dof": self.null_dof,
                "null_threshold": self.null_threshold,
                "fits": {fit.name: fit.as_dict() for fit in self.fits},
                "nuisance_rank": self.nuisance_rank, "alpha": self.alpha,
                "interval_level": self.interval_level, "explanation": self.explanation,
                "assumptions": list(self.assumptions)}


def diagnose(residual, covariance, hypotheses: Mapping[str, object], *, nuisance=None,
             alpha: float = 0.01, interval_level: float = 0.95) -> DiagnosticResult:
    """Test a finite record against no fault and declared single-amplitude alternatives.

    residual is shape (n,), covariance (n,n), each signature (n,), and optional nuisance
    (n,k), with columns spanning arbitrary nuisance effects. Row i of every input must
    use the same physical units; covariance (i,j) uses their product. No units are inferred
    or converted. Diagonal standardization before Cholesky makes whitening insensitive to
    ordinary row-unit changes. Nuisance rank uses normalized design columns and a
    machine-precision SVD cutoff. A bounded exact-rank check permits truly dependent
    columns but raises ValueError when a real direction is below numerical resolution;
    it never bounds arbitrary nuisance coefficients implicitly. Unresolved designs above
    the exact-check size/arithmetic limits also raise and must be reparameterized.
    Zero/confounded fault signatures have
    no amplitude interval. A saturated candidate has no testable lack-of-fit and cannot
    produce an identified status. Empty hypotheses permit consistency-only use.
    """
    alpha = _probability(alpha, "alpha")
    interval_level = _probability(interval_level, "interval_level")
    q = 1.0 - alpha
    interval_q = 0.5 + interval_level / 2.0
    if not 0.0 < q < 1.0 or not 0.5 < interval_q < 1.0:
        raise ValueError("alpha or interval_level is too close to a boundary for float precision")
    zcrit = NormalDist().inv_cdf(interval_q)
    r = _array(residual, 1, "residual")
    n = r.size
    if n == 0:
        raise ValueError("residual must contain at least one sample")
    cov = _array(covariance, 2, "covariance")
    if cov.shape != (n, n):
        raise ValueError(f"covariance must have shape ({n}, {n})")
    if np.any(np.diag(cov) <= 0.0):
        raise ValueError("covariance must be positive definite")
    scales = np.sqrt(np.diag(cov))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        corr = (cov / scales[:, None]) / scales[None, :]
    if not np.all(np.isfinite(corr)) or not np.allclose(corr, corr.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance must be finite and symmetric in standardized row units")
    try:
        chol = np.linalg.cholesky(0.5 * corr + 0.5 * corr.T)
    except np.linalg.LinAlgError:
        raise ValueError("covariance must be positive definite") from None

    def whiten(v):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            out = np.linalg.solve(chol, v / scales)
        if not np.all(np.isfinite(out)):
            raise ValueError("whitened inputs exceed the supported numeric range")
        return out

    if not isinstance(hypotheses, Mapping):
        raise ValueError("hypotheses must map candidate names to finite 1-D signatures")
    signatures = {}
    for name, signature in hypotheses.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("hypothesis names must be non-empty strings")
        signature = _array(signature, 1, f"hypothesis {name!r}")
        if signature.shape != (n,):
            raise ValueError(f"hypothesis {name!r} must have shape ({n},)")
        signatures[name] = signature
    if nuisance is None:
        nuisance = np.empty((n, 0))
    nuisance = _array(nuisance, 2, "nuisance")
    if nuisance.shape[0] != n:
        raise ValueError(f"nuisance must have {n} rows, one per residual component")
    columns, raw_columns = [], []
    for column in nuisance.T:
        scale = float(np.max(np.abs(column)))
        if scale != 0.0:
            w = whiten(column / scale)
            columns.append(w / _norm(w))
            raw_columns.append(column)
    raw_design = np.column_stack(raw_columns) if raw_columns else np.empty((n, 0))
    if columns:
        design = np.column_stack(columns)
        U, singular, _ = np.linalg.svd(design, full_matrices=True)
        tolerance = max(design.shape) * np.finfo(float).eps * singular[0]
        nuisance_rank = int(np.sum(singular > tolerance))
        if nuisance_rank < min(design.shape):
            _verify_no_discarded_direction(raw_design, nuisance_rank, "nuisance")
        complement = U[:, nuisance_rank:]
    else:
        nuisance_rank, complement = 0, np.eye(n)
    y = complement.T @ whiten(r)
    null_dof = n - nuisance_rank

    def test(vector, dof):
        if dof == 0:
            return None, None, None
        length = _norm(vector)
        statistic = _finite(length * length, "lack-of-fit statistic")
        threshold = _chi2_quantile(dof, q)
        return statistic, threshold, bool(statistic <= threshold)

    null_statistic, null_threshold, null_adequate = test(y, null_dof)
    fits = []
    for name, signature in signatures.items():
        scale = float(np.max(np.abs(signature)))
        w = whiten(signature / scale) if scale != 0.0 else np.zeros(n)
        length = _norm(w)
        projected = complement.T @ (w / length) if length != 0.0 else np.zeros(null_dof)
        fraction = _norm(projected)
        observable = fraction > max(n, len(columns) + 1) * np.finfo(float).eps
        if not observable:
            if scale != 0.0 and null_dof > 0:
                _verify_no_discarded_direction(np.column_stack([raw_design, signature]), nuisance_rank,
                                              f"hypothesis {name!r} relative to nuisance")
            fits.append(CandidateFit(
                name, False, null_statistic, null_dof, null_threshold, null_adequate,
                None, None, None,
                "Signature is zero or numerically confounded with the declared nuisance space; "
                "its amplitude is unobservable. This model has the same lack-of-fit as no fault."))
            continue
        direction = projected / fraction
        coefficient = float(direction @ y)
        # Combine scales in log space: individually tiny/large signature and covariance
        # units can cancel, even when a sequence of divisions would under/overflow.
        try:
            sd = math.exp(-math.log(fraction) - math.log(length) - math.log(scale))
        except OverflowError:
            raise ValueError("amplitude standard deviation exceeds float range; rescale the signature") from None
        sd = _finite(sd, "amplitude standard deviation")
        if sd <= 0.0:
            raise ValueError("amplitude standard deviation underflows; rescale the signature")
        amplitude = _finite(coefficient * sd, "fitted amplitude")
        interval = (_finite(amplitude - zcrit * sd, "interval lower bound"),
                    _finite(amplitude + zcrit * sd, "interval upper bound"))
        fit_dof = null_dof - 1
        fit_statistic, fit_threshold, adequate = test(y - direction * coefficient, fit_dof)
        explanation = ("Conditional Gaussian amplitude interval for this fixed candidate/profile and "
                       "declared covariance; not adjusted for candidate or onset selection.")
        if fit_dof == 0:
            explanation += " The fit is saturated: no residual degrees of freedom remain to test it."
        fits.append(CandidateFit(name, True, fit_statistic, fit_dof, fit_threshold, adequate,
                                 amplitude, sd, interval, explanation))

    candidates = tuple(fit.name for fit in fits if fit.observable and fit.adequate is True)
    if null_dof == 0:
        status, explanation = ("insufficient_evidence", "The nuisance space spans the entire record; "
                               "no residual degrees of freedom remain for a consistency test.")
    elif null_adequate:
        status, explanation = ("consistent", "No departure from the declared no-fault model was detected "
                               "after nuisance removal. This is not evidence that the sensors are healthy.")
    elif any(fit.observable and fit.fit_dof == 0 for fit in fits):
        status, explanation = ("insufficient_evidence", "The no-fault model is rejected, but observable "
                               "candidates saturate the remaining record and cannot be checked for lack-of-fit.")
    elif len(candidates) > 1:
        status, explanation = ("ambiguous", "The no-fault model is rejected and multiple declared single-fault "
                               "candidates remain adequate. The record does not select one of them.")
    elif len(candidates) == 1:
        status, explanation = ("identified", "Only one declared single-fault candidate remains adequate after "
                               "rejecting no fault. Identification is conditional on this candidate catalogue, "
                               "the nuisance model and the declared joint covariance.")
    else:
        status, explanation = ("unexplained", "The no-fault model is rejected and no testable declared "
                               "single-fault candidate explains the residual. No fault classification is supported.")
    assumptions = (
        "Residual errors are jointly Gaussian with the supplied known covariance, including serial and cross-channel correlation.",
        "Each alternative has one unknown unrestricted signed amplitude and a supplied, fixed onset/profile; no onset search is performed.",
        "Declared nuisance columns have arbitrary deterministic coefficients and no invented priors.",
        "Per-model chi-square lack-of-fit tests are not posterior model probabilities or a multiple-testing error guarantee.",
        "Amplitude intervals condition on each candidate/profile/covariance and are not adjusted for model selection.",
        "Row units must agree across residual, signatures, nuisance and joint covariance; units are not inferred or converted.",
        "Rank uses machine-precision tolerances on normalized columns; unresolved nonzero directions or oversized exact-rank checks are refused, not silently discarded.",
    )
    return DiagnosticResult(status, candidates, null_statistic, null_dof, null_threshold, tuple(fits),
                            nuisance_rank, alpha, interval_level, explanation, assumptions)
