"""Fault detection and ISOLATION: what the consistency statistic can tell apart.

`lcm.detectability` answers one question per fault direction -- can the constraint test see
this at all? -- and answers it exactly: d(f) = f^T A^T S^-1 A f, zero precisely on null(A).
That is necessary for a fault estimator and nowhere near sufficient, because a detector that
fires without saying WHICH instrument moved is an alarm, not a diagnosis.

This module answers the next question: given two candidate faults, can the statistic tell
them apart? The answer follows from one observation, and it is more restrictive than it
looks.

THE RESIDUAL IS THE ONLY CHANNEL. Everything the constraint test knows about a fault, it
knows through r = A x - b, whose covariance is S = A P A^T + Sigma_b. Whiten it:

    signature(f) = S^(-1/2) A f          d(f) = || signature(f) ||^2

so d(f) is the squared length of the signature, and the DIRECTION of the signature is
everything else the residual carries. Two faults f and g shift the whitened residual along
signature(f) and signature(g). If those point the same way, no amount of data separates
them -- observing a large residual is equally consistent with a small f and a large g. Only
their ratio is ever recoverable, never which one it was.

THE CONSEQUENCE, FOR EVERY TOPOLOGY IN THIS REPOSITORY. rank(A) = 1 for all of them: the
two-reservoir sum A = [1, 1], the reservoir closure A = [1, -1], the augmented closure
A = [1, -1, -1]. A rank-1 A makes the residual a SCALAR, so every signature is a number on
one axis and every pair of detectable faults is collinear by construction. Therefore:

    With rank(A) = 1, no fault is isolatable from any other. Not with a better
    covariance, not with more data, not with a longer record. The statistic has one
    number to report and cannot say which of many causes produced it.

That is not a limitation of this implementation; it is the dimension of the residual space.
It is also exactly the classical statement that a GLOBAL test on constraint residuals
detects a gross error without locating it, which the data-reconciliation literature has said
since Crowe (1985) and which this repository's README has always cited.

WHAT FOLLOWS ARCHITECTURALLY, and it is the useful part. Sensor-level localisation in this
tree cannot come from the constraint at all. It has to come from a channel that does not
pass through r -- which is precisely why `testbed.cusum` watches each sensor's own
normalised innovation and reads no constraint. The two channels are not redundant and not
alternatives: the constraint says the system is inconsistent, the per-sensor channel says
which instrument's own predictions went wrong, and only together do they distinguish
"evidence wrong" from "model wrong". Isolation through the constraint needs rank(A) >= 2
AND signatures that are not collinear; `isolability()` computes both and refuses to report
an isolation that the rank forbids.

Nothing here estimates a fault. It says what could be estimated, before anything is.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lcm import check_spd, reduced
from .schema import ConstraintSet

__all__ = [
    "COLLINEAR_COS", "VISIBLE_D", "FaultPair", "Isolability", "residual_covariance",
    "whitened_signature", "isolability",
]

# |cos| at or above this counts the two signatures as the same direction. At rank 1 every
# pair is exactly 1 up to floating point; the tolerance is for higher ranks, where a pair
# can be near-collinear without being exactly so.
COLLINEAR_COS = 1.0 - 1e-9
# d(f) below this counts the fault as invisible: the signature is numerically zero, so the
# fault moves the statistic by nothing a finite record could resolve.
VISIBLE_D = 1e-12


def residual_covariance(P: np.ndarray, cs: ConstraintSet) -> tuple[np.ndarray, np.ndarray]:
    """(A_r, S) for the independent row set of `cs`: S = A P A^T, plus Sigma_b where declared.

    The same reduction and the same S the kernel's own consistency_stat uses, so a signature
    computed here is a statement about the statistic actually reported, not a parallel one.
    """
    check_spd(P)
    A, _ = reduced(cs)
    S = A @ P @ A.T
    if cs.b_var is not None:
        S = S + cs.b_cov
    return A, 0.5 * (S + S.T)


def whitened_signature(f, P: np.ndarray, cs: ConstraintSet) -> np.ndarray:
    """S^(-1/2) A f: what a unit fault along `f` does to the whitened residual.

    Its squared norm is lcm.detectability(f, P, cs) -- checked in the tests, not asserted --
    and its direction is everything else the residual says about the fault.
    """
    A, S = residual_covariance(P, cs)
    Af = A @ np.asarray(f, dtype=float).reshape(-1)
    w, V = np.linalg.eigh(S)
    if w[0] <= 0.0:
        raise ValueError(f"the residual covariance is not positive definite (eigenvalues {w}); "
                         "neither P nor the declared b_var carries uncertainty along a constraint")
    return (V / np.sqrt(w)) @ (V.T @ Af)


@dataclass(frozen=True)
class FaultPair:
    a: str
    b: str
    cos: float                  # cosine between the two whitened signatures
    distinguishable: bool
    why: str


@dataclass(frozen=True)
class Isolability:
    """What the consistency statistic can and cannot tell apart, for one declared topology."""
    residual_rank: int          # rank(A) after reduction: the dimension of the residual space
    d: dict[str, float]         # detectability per named direction
    signatures: dict[str, list] # the whitened signature per named direction
    visible: list[str]
    invisible: list[str]
    pairs: list[FaultPair]
    isolable: list[str]         # directions distinguishable from EVERY other visible one
    note: str

    def as_dict(self) -> dict:
        return {
            "residual_rank": self.residual_rank,
            "d": dict(self.d),
            "signatures": {k: list(v) for k, v in self.signatures.items()},
            "visible": list(self.visible),
            "invisible": list(self.invisible),
            "isolable": list(self.isolable),
            "pairs": [{"a": p.a, "b": p.b, "cos": p.cos, "distinguishable": p.distinguishable,
                       "why": p.why} for p in self.pairs],
            "note": self.note,
        }


def isolability(directions: dict, P: np.ndarray, cs: ConstraintSet) -> Isolability:
    """For a declared set of named fault directions, what the constraint test can tell apart.

    A direction whose d(f) is numerically zero is INVISIBLE: the constraint is structurally
    blind to it and no covariance changes that. Two visible directions whose whitened
    signatures are collinear are INDISTINGUISHABLE: the residual moves the same way for
    both, so a reading of it is equally consistent with either and only their ratio is ever
    recoverable. A direction distinguishable from every other visible one is isolable.

    At rank(A) = 1 the residual is a scalar and every visible pair is collinear by
    construction, so `isolable` comes back empty and `note` says why. That is the state of
    every topology in this repository today, and it is the reason the per-sensor CUSUM
    channel exists.
    """
    if not directions:
        raise ValueError("declare at least one named fault direction")
    A, _ = residual_covariance(P, cs)
    rank = int(A.shape[0])

    sig = {name: whitened_signature(f, P, cs) for name, f in directions.items()}
    d = {name: float(s @ s) for name, s in sig.items()}
    visible = [n for n in directions if d[n] > VISIBLE_D]
    invisible = [n for n in directions if d[n] <= VISIBLE_D]

    pairs: list[FaultPair] = []
    names = list(directions)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a in invisible or b in invisible:
                dead = a if a in invisible else b
                pairs.append(FaultPair(a, b, float("nan"), False,
                                       f"{dead} is invisible (d = 0), so nothing distinguishes it from "
                                       "anything -- it never moves the residual at all"))
                continue
            na, nb = sig[a], sig[b]
            c = float(na @ nb / (np.linalg.norm(na) * np.linalg.norm(nb)))
            c = max(-1.0, min(1.0, c))
            same = abs(c) >= COLLINEAR_COS
            pairs.append(FaultPair(
                a, b, c, not same,
                ("the two whitened signatures are collinear: the residual moves the same way for "
                 "both, so observing it is equally consistent with either and only their ratio is "
                 "recoverable") if same else
                "the signatures point in different directions, so the residual distinguishes them"))

    isolable = [n for n in visible
                if all(p.distinguishable for p in pairs if n in (p.a, p.b) and p.a in visible and p.b in visible)]

    if rank == 1:
        note = ("rank(A) = 1: the residual is a scalar, so every visible fault's signature lies on one "
                "axis and every pair is collinear by construction. No fault is isolatable from any "
                "other through this constraint, at any covariance and over any record length. "
                "Sensor-level localisation has to come from a channel that does not pass through the "
                "residual -- in this tree, testbed.cusum, which reads each sensor's own normalised "
                "innovation and reads no constraint.")
    elif not isolable:
        note = (f"rank(A) = {rank}, so the residual could distinguish up to {rank} independent fault "
                "directions, but none of the declared ones is separated from every other visible one. "
                "Isolation needs signatures that are not collinear, not merely a residual with room "
                "for them.")
    else:
        note = (f"rank(A) = {rank}. {len(isolable)} of {len(visible)} visible direction(s) are "
                "distinguishable from every other visible one and can therefore be isolated by the "
                "constraint test alone; the rest are confusable with something and need the "
                "per-sensor channel to separate.")
    return Isolability(residual_rank=rank, d=d, signatures={k: list(map(float, v)) for k, v in sig.items()},
                       visible=visible, invisible=invisible, pairs=pairs, isolable=isolable, note=note)
