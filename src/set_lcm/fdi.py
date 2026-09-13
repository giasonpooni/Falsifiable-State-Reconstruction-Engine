"""Static single-fault geometry of the constraint RESIDUAL VECTOR.

Assume exactly one declared candidate fault is active, with unknown, unrestricted signed
amplitude, and a fixed residual map r = A x - b. With S = A P A^T + Sigma_b,

    signature(f) = S^(-1/2) A f          d(f) = || signature(f) ||^2.

Structural invisibility is membership in null(A); it does not depend on covariance or
the magnitude used to represent a direction. d(f), in contrast, measures the response at
that magnitude relative to uncertainty. Small positive d is weak detection power, not
structural blindness. Numerical null membership is checked on normalized directions in
an orthonormal row-space basis of A, independently of the whitened signature's length.

Two nonzero collinear signatures generate the same residual line when amplitude can have
either sign. Non-collinear lines separate single-fault candidates in the noiseless static
model; finite-noise classification is a separate problem. All visible pairs at rank(A) = 1
are collinear. This is not a theorem about every time record: known temporal fault profiles,
dynamic models, sign or amplitude restrictions, or changing residual maps may add information.

FaultPair also exposes the whitened orthogonal fraction sin(theta) and its reciprocal.
The reciprocal compares total signature length with the component an alternative fault
line cannot explain. It is a geometric ratio, not a calibrated ratio of operational
isolation and detection thresholds. These diagnostics depend on covariance and never
change the structural labels. They are undefined for pairs containing an invisible fault.

The scalar global statistic T = r^T S^-1 r discards vector direction. Even at rank 2,
isolation established here requires retaining r, not T alone. No simultaneous-fault or
operational diagnosis guarantee is made. A per-sensor innovation channel such as
testbed.cusum adds evidence, but its alarm alone does not prove which instrument failed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lcm import _S, _S_declared, _check_columns, _finite, _vector, check_spd, reduced
from .schema import ConstraintSet

__all__ = [
    "COLLINEAR_COS", "NULL_REL_TOL", "VISIBLE_D", "FaultPair", "Isolability", "residual_covariance",
    "whitened_signature", "isolability",
]

# Structural comparisons use normalized row-space coordinates, independent of covariance.
# The reported FaultPair.cos still describes the whitened signatures.
COLLINEAR_COS = 1.0 - 1e-9
NULL_REL_TOL = 1e-12
# Retained for import compatibility only. Visibility no longer thresholds d(f): detection
# power depends on covariance and amplitude, whereas nullspace membership does not.
VISIBLE_D = 1e-12


def residual_covariance(P: np.ndarray, cs: ConstraintSet) -> tuple[np.ndarray, np.ndarray]:
    """(A_r, S) for the independent row set of `cs`: S = A P A^T, plus Sigma_b where declared.

    The same reduction and the same S the kernel's own consistency_stat uses, so a signature
    computed here is a statement about the statistic actually reported, not a parallel one.
    """
    P = np.asarray(P, dtype=float)
    check_spd(P)
    A, _ = reduced(cs)
    _check_columns(A, P.shape[0])
    # Reuse the kernel's finite/conditioning guards: a signature must not claim to
    # describe a consistency calculation that the kernel itself would refuse.
    S = _S(A, P) if cs.b_var is None else _S_declared(A, P, cs.b_cov)
    return A, 0.5 * S + 0.5 * S.T


def whitened_signature(f, P: np.ndarray, cs: ConstraintSet) -> np.ndarray:
    """S^(-1/2) A f: what a unit fault along `f` does to the whitened residual.

    Its squared norm is lcm.detectability(f, P, cs) -- checked in the tests, not asserted --
    and its direction is everything else the residual says about the fault. The direction
    must be a finite 1-D vector matching the state, and the covariance must pass the same
    validation as lcm.detectability.
    """
    A, S = residual_covariance(P, cs)
    Af = A @ _vector(f, "f", A.shape[1])
    _finite(Af, "fault residual")
    w, V = np.linalg.eigh(S)
    _finite(w, "residual covariance eigenvalues")
    if w[0] <= 0.0:
        raise ValueError(f"the residual covariance is not positive definite (eigenvalues {w}); "
                         "neither P nor the declared b_var carries uncertainty along a constraint")
    signature = (V / np.sqrt(w)) @ (V.T @ Af)
    _finite(signature, "whitened signature")
    return signature


def _unit_direction(f, size: int) -> np.ndarray:
    """Normalize without under/overflow from squaring the supplied fault magnitude."""
    v = np.asarray(f, dtype=float).reshape(-1)
    if v.size != size or not np.all(np.isfinite(v)):
        raise ValueError(f"a fault direction must contain {size} finite values")
    scale = float(np.max(np.abs(v))) if v.size else 0.0
    if scale == 0.0:
        return v.copy()
    v = v / scale
    return v / np.linalg.norm(v)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _unit_direction(a, a.size), _unit_direction(b, b.size)
    return float(np.clip(a @ b, -1.0, 1.0))


def _orthogonal_fraction(a: np.ndarray, b: np.ndarray) -> float:
    """Whitened angular separation, avoiding cancellation in sqrt(1 - cos(theta)^2)."""
    a, b = _unit_direction(a, a.size), _unit_direction(b, b.size)
    # The denominator retains the true floating-point norm: projecting an identical
    # vector onto itself should leave exactly zero, not its normalization roundoff.
    orthogonal = a - b * float((a @ b) / (b @ b))
    return float(np.clip(np.linalg.norm(orthogonal) / np.linalg.norm(a), 0.0, 1.0))


@dataclass(frozen=True)
class FaultPair:
    """Static structural comparison, with optional whitened conditioning diagnostics.

    orthogonal_fraction is sin(theta); isolation_amplification is its reciprocal,
    a geometric length ratio rather than an operational fault-size threshold. Exact
    collinearity gives 0 and infinity. Both are NaN when either fault is invisible,
    or when a caller constructs a pair using the original five-argument interface.
    """
    a: str
    b: str
    cos: float                  # cosine between the two whitened signatures
    distinguishable: bool       # static row-space lines differ; not a finite-noise guarantee
    why: str
    orthogonal_fraction: float = float("nan")
    isolation_amplification: float = float("nan")


@dataclass(frozen=True)
class Isolability:
    """Static residual-vector geometry for one active fault of unrestricted signed amplitude.

    visible/invisible describe numerical nullspace membership, independently of d. The
    scalar global statistic alone does not retain the direction used by pairs/isolable.
    """
    residual_rank: int          # rank(A) after reduction: the dimension of the residual space
    d: dict[str, float]         # detectability per named direction
    signatures: dict[str, list] # the whitened signature per named direction
    visible: list[str]
    invisible: list[str]
    pairs: list[FaultPair]
    # STRUCTURALLY distinguishable from every other visible candidate -- a statement about
    # row-space geometry, NOT about whether two faults could be told apart at any noise level.
    # A pair can be structurally distinguishable and still nearly useless to separate: at an
    # orthogonal fraction of 0.002 the component that DIFFERS between the two signatures is
    # 1/500th of the component they share, so telling them apart works with 500x less signal
    # than merely seeing that something is wrong. Read `isolable` together with each pair's
    # orthogonal_fraction and isolation_amplification, which as_dict() archives beside cos for
    # exactly this reason; do not act on membership of this list alone.
    isolable: list[str]
    note: str

    def as_dict(self) -> dict:
        """Archive the whole comparison, including the angular diagnostics.

        `distinguishable` and `isolable` are structural; orthogonal_fraction and
        isolation_amplification are what say whether the structural separation is
        large enough to act on. Archiving the first without the second would record
        a yes/no that reads stronger than the geometry behind it, so both travel
        with every pair. They serialize as floats and may be NaN or infinite (an
        invisible member, or exact collinearity), which JSON renders as a literal;
        readers that need strict JSON must map the non-finite values themselves.
        """
        return {
            "residual_rank": self.residual_rank,
            "d": dict(self.d),
            "signatures": {k: list(v) for k, v in self.signatures.items()},
            "visible": list(self.visible),
            "invisible": list(self.invisible),
            "isolable": list(self.isolable),
            "pairs": [{"a": p.a, "b": p.b, "cos": p.cos, "distinguishable": p.distinguishable,
                       "why": p.why, "orthogonal_fraction": p.orthogonal_fraction,
                       "isolation_amplification": p.isolation_amplification}
                      for p in self.pairs],
            "note": self.note,
        }


def isolability(directions: dict, P: np.ndarray, cs: ConstraintSet) -> Isolability:
    """Compare static residual-vector lines under the module's single-fault assumptions.

    Visibility uses ||V_r^T f_unit|| > NULL_REL_TOL, where V_r spans row(A). Thus a small
    fault magnitude or a large covariance cannot turn weak detection power into structural
    blindness. Pair collinearity uses these same row-space coordinates and COLLINEAR_COS;
    pair.cos reports the whitened cosine separately. At numerical tolerances, near-null or
    near-collinear directions are treated as null or collinear, not certified exactly so.

    An isolable direction is separated from every other visible candidate. Invisible
    candidates remain unlocalisable against the no-fault case; pairs containing one are
    marked False because two visible fault lines are needed for this comparison. Neither
    temporal signatures nor simultaneous faults are evaluated here.
    """
    if not directions:
        raise ValueError("declare at least one named fault direction")
    A, _ = residual_covariance(P, cs)
    rank = int(A.shape[0])

    unit = {name: _unit_direction(f, A.shape[1]) for name, f in directions.items()}
    _, _, row_basis = np.linalg.svd(A, full_matrices=False)
    geometry = {name: row_basis @ f for name, f in unit.items()}
    sig = {name: whitened_signature(f, P, cs) for name, f in directions.items()}
    d = {name: float(s @ s) for name, s in sig.items()}
    visible = [n for n in directions if np.linalg.norm(geometry[n]) > NULL_REL_TOL]
    invisible = [n for n in directions if n not in visible]
    # Compare normalized directions, so tiny amplitudes cannot underflow the angle.
    unit_sig = {name: whitened_signature(unit[name], P, cs) for name in visible}

    pairs: list[FaultPair] = []
    names = list(directions)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a in invisible or b in invisible:
                dead = a if a in invisible else b
                pairs.append(FaultPair(a, b, float("nan"), False,
                                       f"{dead} is numerically invisible in null(A); it cannot be "
                                       "localized against no fault from this static residual. "
                                       "This pair does not contain two visible fault lines."))
                continue
            c = _cosine(unit_sig[a], unit_sig[b])
            same = abs(_cosine(geometry[a], geometry[b])) >= COLLINEAR_COS
            orthogonal_fraction = _orthogonal_fraction(unit_sig[a], unit_sig[b])
            amplification = float("inf") if orthogonal_fraction == 0.0 else 1.0 / orthogonal_fraction
            pairs.append(FaultPair(
                a, b, c, not same,
                ("the static residual signatures are numerically collinear: with unrestricted signed "
                 "unknown amplitude, either single fault can explain the same residual shift") if same else
                "the static residual vector has non-collinear fault signatures under the single-fault "
                "model; this does not imply separation by scalar T or reliable finite-noise diagnosis",
                orthogonal_fraction=orthogonal_fraction, isolation_amplification=amplification))

    isolable = [n for n in visible
                if all(p.distinguishable for p in pairs if n in (p.a, p.b) and p.a in visible and p.b in visible)]

    assumptions = (" Assumptions: one active candidate fault, unrestricted signed unknown amplitude, "
                   "and a fixed static residual map. The scalar global statistic T discards residual "
                   "direction; this result concerns the residual vector. Known temporal fault profiles "
                   "or dynamic models may add information and are not analyzed here. testbed.cusum "
                   "provides a separate per-sensor innovation channel, not proof of sensor failure.")
    if rank == 1 and len(visible) > 1:
        note = ("rank(A) = 1: every visible pair is collinear in the one-dimensional residual space. "
                "No visible candidate is isolatable from another by this static geometry.")
    elif not isolable:
        note = (f"rank(A) = {rank}. No declared visible direction is separated from every other visible "
                "candidate. Isolation needs signatures that are not collinear; rank alone is insufficient.")
    else:
        note = (f"rank(A) = {rank}. {len(isolable)} of {len(visible)} visible direction(s) are "
                "separated from every other visible candidate in the static residual vector. This is "
                "a structural condition, not a finite-noise detection or diagnosis guarantee.")
    note += assumptions
    return Isolability(residual_rank=rank, d=d, signatures={k: list(map(float, v)) for k, v in sig.items()},
                       visible=visible, invisible=invisible, pairs=pairs, isolable=isolable, note=note)
