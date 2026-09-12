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
which CHANNEL's own predictions broke.

That is not the same as which instrument is faulty, and the committed grid says so: on
`leak_stale_constraint`, a 10 kg PROCESS leak carrying `bias=None` -- no sensor fault at all
-- the per-sensor channel names sensor 2 in 20/20 seeds at median 59.5 steps, the same
signature it gives when it is right about a real bias. So neither channel names an
instrument, and a fault estimator built here must say so.

Isolation through the constraint needs rank(A) >= 2, signatures that are not collinear, AND a
separation large enough to act on; `isolability()` computes all three and refuses to report
an isolation that any of them forbids.

Nothing here estimates a fault. It says what could be estimated, before anything is.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lcm import check_spd, reduced
from .schema import ConstraintSet

__all__ = [
    "COLLINEAR_COS", "MIN_SEPARATION", "VISIBLE_D", "FaultPair", "Isolability",
    "residual_covariance", "whitened_signature", "isolability",
]

# |cos| at or above this counts the two signatures as EXACTLY the same direction, to floating
# point. At rank 1 every pair is 1 up to rounding.
COLLINEAR_COS = 1.0 - 1e-9
# ...but a cosine gate alone is not an isolation criterion, and treating it as one is the
# mistake this constant exists to prevent. Two signatures at |cos| = 0.999998 are numerically
# distinct and operationally identical: the component of one orthogonal to the other is
# sin = 0.002, so telling them apart needs a fault 1/0.002 = 500x the size that detecting one
# needs. MIN_SEPARATION is that orthogonal fraction, declared rather than inherited from
# floating-point noise. The default says: an isolation requiring a fault ten times the
# detection threshold is not an isolation anyone can act on. Callers with a physical margin
# in the fault's own units should pass their own, computed from it.
MIN_SEPARATION = 0.1
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
    orthogonal_fraction: float  # sin: the part of one signature the other cannot explain
    isolation_amplification: float   # 1 / orthogonal_fraction: how much bigger a fault must be
                                     # to be ISOLATED than merely DETECTED. inf when collinear.
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
            "pairs": [{"a": p.a, "b": p.b, "cos": p.cos,
                       "orthogonal_fraction": p.orthogonal_fraction,
                       "isolation_amplification": p.isolation_amplification,
                       "distinguishable": p.distinguishable, "why": p.why} for p in self.pairs],
            "note": self.note,
        }


def isolability(directions: dict, P: np.ndarray, cs: ConstraintSet,
                *, min_separation: float = MIN_SEPARATION) -> Isolability:
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
                pairs.append(FaultPair(a, b, float("nan"), float("nan"), float("inf"), False,
                                       f"{dead} is invisible (d = 0), so nothing distinguishes it from "
                                       "anything -- it never moves the residual at all"))
                continue
            na, nb = sig[a], sig[b]
            c = float(na @ nb / (np.linalg.norm(na) * np.linalg.norm(nb)))
            c = max(-1.0, min(1.0, c))
            orth = float(np.sqrt(max(0.0, 1.0 - c * c)))
            amp = float("inf") if orth <= 0.0 else 1.0 / orth
            collinear = abs(c) >= COLLINEAR_COS
            separated = orth >= min_separation
            if collinear:
                why = ("the two whitened signatures are collinear: the residual moves the same way "
                       "for both, so observing it is equally consistent with either and only their "
                       "ratio is recoverable")
            elif not separated:
                why = (f"the signatures differ by an orthogonal fraction of {orth:.3g}, below the "
                       f"declared separation {min_separation:g}: isolating these two needs a fault "
                       f"{amp:.3g}x the size that detecting one needs, which is numerical residue "
                       "rather than structure")
            else:
                why = (f"the signatures are separated by an orthogonal fraction of {orth:.3g}, so a "
                       f"fault {amp:.3g}x the detection size is isolated")
            pairs.append(FaultPair(a, b, c, orth, amp, separated and not collinear, why))

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
